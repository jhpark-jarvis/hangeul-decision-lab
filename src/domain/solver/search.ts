import type { GameState } from "../game/types";
import type { PathRewards, SolverAction } from "./types";

/** Reward-sensitive key; only completed subtrees are memoized by solveTurn. */
export function createSearchKey(
  state: GameState,
  rewards: PathRewards,
): string {
  return JSON.stringify([
    state.board.map((row) => row.map(Number).join("")).join("/"),
    [...state.remainingPieces]
      .sort((a, b) => a.pieceIndex - b.pieceIndex)
      .map((entry) => [
        entry.instanceId,
        entry.pieceIndex,
        entry.piece.id,
        entry.piece.shape,
      ]),
    [...state.hiddenItems]
      .sort((a, b) => a.row - b.row || a.col - b.col)
      .map((item) => [item.row, item.col, item.type]),
    state.abilities.reroll,
    state.abilities.singleCell,
    state.pendingReroll,
    rewards.clearedRows,
    rewards.acquiredItems,
  ]);
}
export function actionSequenceKey(actions: SolverAction[]): string {
  return JSON.stringify(
    actions.map((action) => [
      action.pieceIndex,
      action.rotation,
      action.flipped,
      action.row,
      action.col,
      action.instanceId,
      action.pieceId,
      action.variant,
    ]),
  );
}
