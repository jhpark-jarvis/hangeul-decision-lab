import type { GameState } from "../../domain/game/types";
import type { RecognitionResult, ReviewState } from "./types";
import { updateReview } from "./review";

export type TurnComparison = {
  result: RecognitionResult;
  differences: { row: number; col: number }[];
  inherited: { row: number; col: number }[];
  matched: boolean;
};
export const MIN_CONTINUITY_READ_CELLS = 136;
/** Draft only. Current GameState already contains acquisition/retention after Apply. */
export function continueTurn(
  result: RecognitionResult,
  expected: GameState,
): TurnComparison {
  const draft = structuredClone(result);
  const differences = draft.board
    .flat()
    .filter(
      (c) =>
        c.status === "recognized" &&
        c.occupied !== null &&
        c.occupied !== expected.board[c.row][c.col],
    )
    .map(({ row, col }) => ({ row, col }));
  const read = draft.board
    .flat()
    .filter((c) => c.status === "recognized" && c.occupied !== null).length;
  const matched =
    read >= MIN_CONTINUITY_READ_CELLS &&
    differences.length === 0 &&
    !expected.pendingReroll;
  const inherited: { row: number; col: number }[] = [];
  if (matched)
    draft.board.flat().forEach((c) => {
      if (c.occupied === null || c.status !== "recognized") {
        c.occupied = expected.board[c.row][c.col];
        c.status = "recognized";
        delete c.confidence;
        inherited.push({ row: c.row, col: c.col });
      }
    });
  draft.hiddenItems = expected.hiddenItems.map((item) => ({
    ...item,
    status: matched ? "recognized" : "uncertain",
  }));
  draft.hiddenItemsStatus = "unknown";
  return { result: draft, differences, inherited, matched };
}
/** The caller must invoke only for an explicit whole-state confirmation action. */
export function confirmWholeReview(review: ReviewState): ReviewState {
  const confirmed = updateReview(review, (draft) => {
    draft.hiddenItemsStatus = "recognized";
    draft.hiddenItems.forEach((item) => {
      if (item.row !== null && item.col !== null && item.type !== null)
        item.status = "recognized";
    });
  });
  return { ...confirmed, confirmed: true };
}
