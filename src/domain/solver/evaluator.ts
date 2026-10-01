import {
  countEmptyCells,
  countIsolatedEmptyCells,
} from "../board/board-metrics";
import { getAvailableActions } from "../game/actions";
import { validateGameState } from "../game/game-state";
import { calculateMobility, countUnfillableGaps } from "./mobility";
import type { SolverEvaluation, SolverResult, PathRewards } from "./types";

export function evaluateState(
  input: unknown,
  catalog: unknown,
  rewards: PathRewards,
): SolverResult<{ evaluation: SolverEvaluation }> {
  const valid = validateGameState(input);
  if (!valid.ok) return valid;
  if (
    !rewards ||
    !Number.isSafeInteger(rewards.clearedRows) ||
    rewards.clearedRows < 0 ||
    !Number.isSafeInteger(rewards.acquiredItems) ||
    rewards.acquiredItems < 0
  )
    return {
      ok: false,
      error: {
        code: "INVALID_REWARDS",
        message: "누적 삭제/획득은 비음수 정수여야 합니다.",
      },
    };
  const phase = getAvailableActions(valid.state);
  if (!phase.ok) return phase;
  const mobility = calculateMobility(valid.state.board, catalog);
  if (!mobility.ok) return mobility;
  const gaps = countUnfillableGaps(valid.state.board, catalog);
  if (!gaps.ok) return gaps;
  const empty = countEmptyCells(valid.state.board);
  if (!empty.ok) return empty;
  const isolated = countIsolatedEmptyCells(valid.state.board);
  if (!isolated.ok) return isolated;
  return {
    ok: true,
    evaluation: {
      survivable: phase.phase !== "gameover",
      allCurrentPiecesPlaced: valid.state.remainingPieces.length === 0,
      phase: phase.phase,
      ...rewards,
      mobility: mobility.mobility,
      remainingAbilities: { ...valid.state.abilities },
      isolatedEmptyCells: isolated.count,
      unfillableGaps: gaps.count,
      emptyCells: empty.count,
    },
  };
}

/** Positive means a is better; no weighted score or future survival guarantee. */
export function compareEvaluations(
  a: SolverEvaluation,
  b: SolverEvaluation,
): number {
  const axes = (value: SolverEvaluation) => [
    Number(value.survivable),
    Number(value.allCurrentPiecesPlaced),
    value.mobility.playablePieceTypes,
    value.mobility.totalPlacements,
    value.clearedRows,
    value.acquiredItems,
    value.remainingAbilities.reroll + value.remainingAbilities.singleCell,
    -value.isolatedEmptyCells,
    -value.unfillableGaps,
    value.emptyCells,
  ];
  const left = axes(a);
  const right = axes(b);
  for (let index = 0; index < left.length; index++)
    if (left[index] !== right[index])
      return left[index] > right[index] ? 1 : -1;
  return 0;
}
