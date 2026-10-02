import { BOARD_HEIGHT, BOARD_WIDTH } from "../../domain/board/board";
import { updateReview } from "./review";
import type { RecognizedHiddenItem, ReviewState } from "./types";

/** Manual item labels never infer occupancy or completeness of the whole list. */
export function labelReviewItem(
  review: ReviewState,
  row: number,
  col: number,
  type: RecognizedHiddenItem["type"],
): { ok: true; review: ReviewState } | { ok: false; message: string } {
  if (
    !Number.isInteger(row) ||
    row < 0 ||
    row >= BOARD_HEIGHT ||
    !Number.isInteger(col) ||
    col < 0 ||
    col >= BOARD_WIDTH ||
    ![null, "reroll", "single-cell"].includes(type)
  )
    return {
      ok: false,
      message: "보드 안의 칸과 올바른 아이템 종류를 선택하세요.",
    };
  const others = review.draft.hiddenItems.filter(
    (item) => item.row !== row || item.col !== col,
  );
  if (type !== null && others.length >= 3)
    return {
      ok: false,
      message:
        "아이템은 최대 3개입니다. 잘못 표시한 아이템을 먼저 지운 뒤 다시 선택하세요.",
    };
  return {
    ok: true,
    review: updateReview(review, (draft) => {
      draft.hiddenItems = others.map((item) => ({ ...item }));
      if (type !== null)
        draft.hiddenItems.push({ row, col, type, status: "recognized" });
      draft.hiddenItemsStatus = "unknown";
    }),
  };
}
