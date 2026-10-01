import { applyAction } from "../game/actions";
import { validateGameState } from "../game/game-state";
import type { GameState } from "../game/types";
import {
  getAbilityCandidates,
  getOrdinaryCandidates,
} from "./ability-candidates";
import { compareEvaluations, evaluateState } from "./evaluator";
import { actionSequenceKey, createSearchKey } from "./search";
import {
  DEFAULT_SOLVER_CONFIG,
  ORDINARY_PROBE_BUDGET_DIVISOR,
  type PathRewards,
  type SearchInfo,
  type SolverAction,
  type OrdinarySolverAction,
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

function runSearch<Action extends SolverAction>(
  input: unknown,
  catalog: unknown,
  inputConfig: unknown,
  getCandidates: (
    state: GameState,
  ) => SolverResult<{ actions: Action[]; excludedRerollTargets?: number }>,
  context: {
    abilitySearch?: NonNullable<SearchInfo["abilitySearch"]>;
    seeds?: SolverCandidate<Action>[];
    probeNodes?: number;
  } = {},
): SolverResult<{ result: TurnSolution<Action> }> {
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
  const candidates: SolverCandidate<Action>[] = [...(context.seeds ?? [])];
  const probeNodes = context.probeNodes ?? 0;
  const search: SearchInfo = {
    scope: context.abilitySearch ? "pieces-and-abilities" : "ordinary-pieces",
    searchComplete: true,
    optimalWithinScope: true,
    specialAbilitiesSearched: false,
    alternativesMayOmitEquivalentPaths: config.useMemoization,
    stopReason: "exhausted",
    visitedNodes: probeNodes,
    memoPrunedNodes: 0,
    memoEntries: 0,
    discoveredCompletePaths: 0,
    evaluatedCandidates: 0,
    maxNodes: config.maxNodes + probeNodes,
    ...(context.abilitySearch
      ? { abilitySearch: structuredClone(context.abilitySearch) }
      : {}),
  };
  function consider(
    state: GameState,
    actions: Action[],
    rewards: PathRewards,
  ): SolverResult<object> {
    const evaluated = evaluateState(state, catalog, rewards);
    if (!evaluated.ok) return evaluated;
    search.evaluatedCandidates++;
    if (evaluated.evaluation.allCurrentPiecesPlaced)
      search.discoveredCompletePaths++;
    const candidate: SolverCandidate<Action> = {
      actions,
      evaluation: evaluated.evaluation,
      finalState: state,
      usedAbilities: {
        singleCell: actions.filter((action) => action.type === "single-cell")
          .length,
        reroll: actions.filter((action) => action.type === "reroll").length,
      },
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
    actions: Action[],
    rewards: PathRewards,
  ): SolverResult<{ complete: boolean }> {
    search.visitedNodes++;
    if (search.abilitySearch)
      search.abilitySearch.maxActionDepth = Math.max(
        search.abilitySearch.maxActionDepth,
        actions.length,
      );
    if (state.pendingReroll || !state.remainingPieces.length) {
      const result = consider(state, actions, rewards);
      return result.ok ? { ok: true, complete: true } : result;
    }
    const key = createSearchKey(state, rewards);
    if (config.useMemoization && memo.has(key)) {
      search.memoPrunedNodes++;
      return { ok: true, complete: true };
    }
    const available = getCandidates(state);
    if (!available.ok) return available;
    const possible = available.actions;
    if (search.abilitySearch)
      search.abilitySearch.excludedRerollTargets +=
        available.excludedRerollTargets ?? 0;
    if (!possible.length) {
      const result = consider(state, actions, rewards);
      if (!result.ok) return result;
    }
    for (const action of possible) {
      if (search.visitedNodes >= search.maxNodes) {
        const result = consider(state, actions, rewards);
        return result.ok ? { ok: true, complete: false } : result;
      }
      const transition = applyAction(state, action);
      if (!transition.ok) return transition;
      if (action.type !== "place-piece") search.specialAbilitiesSearched = true;
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

/** Ordinary-only compatibility/reference search; no ability candidate traversal. */
export function solveOrdinaryTurn(
  input: unknown,
  catalog: unknown,
  inputConfig: unknown = DEFAULT_SOLVER_CONFIG,
): SolverResult<{ result: TurnSolution<OrdinarySolverAction> }> {
  return runSearch(input, catalog, inputConfig, getOrdinaryCandidates);
}

/** Bounded ordinary + ability search. Reroll ends at externally observed input. */
export function solveTurn(
  input: unknown,
  catalog: unknown,
  inputConfig: unknown = DEFAULT_SOLVER_CONFIG,
): SolverResult<{ result: TurnSolution }> {
  const validated = validateGameState(input);
  if (!validated.ok) return validated;
  const configured = validateConfig(inputConfig);
  if (!configured.ok) return configured;
  const { state } = validated;
  const { config } = configured;
  const abilitySearch: NonNullable<SearchInfo["abilitySearch"]> = {
    singleCellCandidates: "all-legal-cells",
    rerollCandidatesRestricted: true,
    excludedRerollTargets: 0,
    rerollFutureEvaluated: false,
    maxActionDepth: 0,
    ordinaryProbe: {
      visitedNodes: 0,
      searchComplete: false,
      discoveredCompletePaths: 0,
      noCompletePathProven: false,
    },
  };
  const seeds: SolverCandidate[] = [];
  const probeBudget =
    state.pendingReroll || !state.remainingPieces.length
      ? 0
      : Math.floor(config.maxNodes / ORDINARY_PROBE_BUDGET_DIVISOR);
  if (probeBudget > 0) {
    const probe = solveOrdinaryTurn(state, catalog, {
      ...config,
      maxNodes: probeBudget,
    });
    if (!probe.ok) return probe;
    const { search } = probe.result;
    abilitySearch.ordinaryProbe = {
      visitedNodes: search.visitedNodes,
      searchComplete: search.searchComplete,
      discoveredCompletePaths: search.discoveredCompletePaths,
      noCompletePathProven:
        search.searchComplete && search.discoveredCompletePaths === 0,
    };
    // An ordinary dead end with legal abilities is not an ability-search leaf.
    for (const entry of [probe.result, ...probe.result.alternatives])
      if (
        entry.evaluation.allCurrentPiecesPlaced ||
        entry.evaluation.phase === "gameover"
      )
        seeds.push({
          actions: entry.actions,
          evaluation: entry.evaluation,
          finalState: entry.finalState,
          usedAbilities: entry.usedAbilities,
        });
  }
  const foundNoCompletePath =
    probeBudget > 0 &&
    abilitySearch.ordinaryProbe.discoveredCompletePaths === 0;
  return runSearch(
    state,
    catalog,
    {
      ...config,
      maxNodes: config.maxNodes - abilitySearch.ordinaryProbe.visitedNodes,
    },
    (current) => getAbilityCandidates(current, catalog, foundNoCompletePath),
    {
      abilitySearch,
      seeds,
      probeNodes: abilitySearch.ordinaryProbe.visitedNodes,
    },
  );
}
