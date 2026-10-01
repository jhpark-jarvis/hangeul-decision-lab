import { applyAction } from "../game/actions";
import { validateGameState } from "../game/game-state";
import type { GameState } from "../game/types";
import { getValidPlacements } from "../pieces/placement";
import { compareEvaluations, evaluateState } from "./evaluator";
import { actionSequenceKey, createSearchKey } from "./search";
import {
  DEFAULT_SOLVER_CONFIG,
  type PathRewards,
  type SearchInfo,
  type SolverAction,
  type SolverCandidate,
  type SolverConfig,
  type SolverResult,
  type TurnSolution,
} from "./types";

function validateConfig(
  input: unknown,
): SolverResult<{ config: SolverConfig }> {
  if (!input || typeof input !== "object")
    return {
      ok: false,
      error: {
        code: "INVALID_SOLVER_CONFIG",
        message: "solver config가 필요합니다.",
      },
    };
  const { maxNodes, maxAlternatives, useMemoization } = input as Record<
    string,
    unknown
  >;
  if (
    typeof maxNodes !== "number" ||
    !Number.isSafeInteger(maxNodes) ||
    maxNodes < 1 ||
    typeof maxAlternatives !== "number" ||
    !Number.isInteger(maxAlternatives) ||
    maxAlternatives < 1 ||
    maxAlternatives > 3 ||
    typeof useMemoization !== "boolean"
  )
    return {
      ok: false,
      error: {
        code: "INVALID_SOLVER_CONFIG",
        message: "양의 node 예산, 1~3개 결과, memo boolean이 필요합니다.",
      },
    };
  return { ok: true, config: { maxNodes, maxAlternatives, useMemoization } };
}

/** Bounded DFS for ordinary pieces. Runtime is measured at the calling boundary. */
export function solveTurn(
  input: unknown,
  catalog: unknown,
  inputConfig: unknown = DEFAULT_SOLVER_CONFIG,
): SolverResult<{ result: TurnSolution }> {
  const validated = validateGameState(input);
  if (!validated.ok) return validated;
  const configured = validateConfig(inputConfig);
  if (!configured.ok) return configured;
  const initial = evaluateState(validated.state, catalog, {
    clearedRows: 0,
    acquiredItems: 0,
  });
  if (!initial.ok) return initial;
  const { config } = configured;
  const memo = new Set<string>();
  const candidates: SolverCandidate[] = [];
  const search: SearchInfo = {
    scope: "ordinary-pieces",
    searchComplete: true,
    optimalWithinScope: true,
    specialAbilitiesSearched: false,
    alternativesMayOmitEquivalentPaths: config.useMemoization,
    stopReason: "exhausted",
    visitedNodes: 0,
    memoPrunedNodes: 0,
    memoEntries: 0,
    discoveredCompletePaths: 0,
    evaluatedCandidates: 0,
    maxNodes: config.maxNodes,
  };
  function consider(
    state: GameState,
    actions: SolverAction[],
    rewards: PathRewards,
  ): SolverResult<object> {
    const evaluated = evaluateState(state, catalog, rewards);
    if (!evaluated.ok) return evaluated;
    search.evaluatedCandidates++;
    if (evaluated.evaluation.allCurrentPiecesPlaced)
      search.discoveredCompletePaths++;
    const candidate: SolverCandidate = {
      actions,
      evaluation: evaluated.evaluation,
      finalState: state,
    };
    const key = actionSequenceKey(actions);
    if (!candidates.some((entry) => actionSequenceKey(entry.actions) === key))
      candidates.push(candidate);
    candidates.sort((a, b) => compareEvaluations(b.evaluation, a.evaluation));
    candidates.splice(config.maxAlternatives);
    return { ok: true };
  }
  function visit(
    state: GameState,
    actions: SolverAction[],
    rewards: PathRewards,
  ): SolverResult<{ complete: boolean }> {
    search.visitedNodes++;
    if (state.pendingReroll || !state.remainingPieces.length) {
      const result = consider(state, actions, rewards);
      return result.ok ? { ok: true, complete: true } : result;
    }
    const key = createSearchKey(state, rewards);
    if (config.useMemoization && memo.has(key)) {
      search.memoPrunedNodes++;
      return { ok: true, complete: true };
    }
    const possible: SolverAction[] = [];
    for (const instance of [...state.remainingPieces].sort(
      (a, b) => a.pieceIndex - b.pieceIndex,
    )) {
      const placements = getValidPlacements(state.board, instance.piece);
      if (!placements.ok) return placements;
      possible.push(
        ...placements.placements.map((placement) => ({
          type: "place-piece" as const,
          ...placement,
          instanceId: instance.instanceId,
          pieceIndex: instance.pieceIndex,
        })),
      );
    }
    if (!possible.length) {
      const result = consider(state, actions, rewards);
      if (!result.ok) return result;
    }
    for (const action of possible) {
      if (search.visitedNodes >= config.maxNodes) {
        const result = consider(state, actions, rewards);
        return result.ok ? { ok: true, complete: false } : result;
      }
      const transition = applyAction(state, action);
      if (!transition.ok) return transition;
      const child = visit(transition.state, [...actions, action], {
        clearedRows: rewards.clearedRows + transition.info.clearedRows.length,
        acquiredItems:
          rewards.acquiredItems + transition.info.acquiredItems.length,
      });
      if (!child.ok) return child;
      if (!child.complete) return child;
    }
    if (config.useMemoization) memo.add(key);
    return { ok: true, complete: true };
  }
  const explored = visit(validated.state, [], {
    clearedRows: 0,
    acquiredItems: 0,
  });
  if (!explored.ok) return explored;
  search.searchComplete = explored.complete;
  search.optimalWithinScope = explored.complete;
  search.memoEntries = memo.size;
  if (!explored.complete) search.stopReason = "node-budget";
  else if (validated.state.pendingReroll)
    search.stopReason = "await-reroll-result";
  else if (!validated.state.remainingPieces.length)
    search.stopReason = "await-next-pieces";
  const [best, ...alternatives] = candidates;
  // Every visited subtree either supplies a leaf/frontier or reuses a completed one.
  return {
    ok: true,
    result: {
      ...structuredClone(best),
      alternatives: alternatives.map((entry) => structuredClone(entry)),
      search: { ...search },
    },
  };
}
