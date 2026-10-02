import { describe, expect, it } from "vitest";
import {
  createSession,
  loadNextPieces,
} from "../../src/features/puzzle/session";
import { recognizeManualState } from "../../src/features/recognition/mock";
import {
  createReviewState,
  validateReviewedState,
} from "../../src/features/recognition/review";
import { labelReviewItem } from "../../src/features/recognition/item-label";
import { applyReviewedState } from "../../src/features/recognition/adapter";
import type { ReviewState } from "../../src/features/recognition/types";

const session = () => loadNextPieces(createSession(), ["DOT", "DOT", "DOT"]);
const review = () => ({
  ...createReviewState(recognizeManualState(session().game)),
  confirmed: true,
});
const label = (
  input: ReviewState,
  row: number,
  col: number,
  type: "single-cell" | "reroll" | null,
) => {
  const result = labelReviewItem(input, row, col, type);
  expect(result.ok).toBe(true);
  if (!result.ok) throw new Error(result.message);
  return result.review;
};

describe("manual item labels on review board", () => {
  it("does not infer unknown occupancy, mutate original, or confirm the whole item list", () => {
    const input = review();
    input.draft.board[12][7] = {
      row: 12,
      col: 7,
      occupied: null,
      status: "uncertain",
    };
    const snapshot = structuredClone(input);
    const output = label(input, 12, 7, "single-cell");
    expect(input).toEqual(snapshot);
    expect(output.original).toEqual(snapshot.original);
    expect(output.draft.board).toEqual(snapshot.draft.board);
    expect(output.draft.hiddenItems).toEqual([
      { row: 12, col: 7, type: "single-cell", status: "recognized" },
    ]);
    expect(output.confirmed).toBe(false);
    expect(output.draft.hiddenItemsStatus).toBe("unknown");
    expect(validateReviewedState(output).map((issue) => issue.path)).toEqual(
      expect.arrayContaining([
        "board.12.7",
        "hiddenItemsStatus",
        "confirmation",
      ]),
    );
    output.draft.board[0][0].occupied = true;
    expect(input).toEqual(snapshot);
  });

  it("keeps occupied cells independent from item labels and installs only after explicit confirmation", () => {
    const initial = session();
    const input = createReviewState(recognizeManualState(initial.game));
    input.draft.board[0][0].occupied = true;
    const output = label(input, 0, 0, "reroll");
    expect(output.draft.board[0][0].occupied).toBe(true);
    expect(
      applyReviewedState(initial, output, JSON.stringify(initial.game)).ok,
    ).toBe(false);
    output.draft.hiddenItemsStatus = "recognized";
    output.confirmed = true;
    const installed = applyReviewedState(
      initial,
      output,
      JSON.stringify(initial.game),
    );
    expect(installed.ok).toBe(true);
    if (!installed.ok) return;
    expect(installed.session.game.hiddenItems).toEqual([
      { row: 0, col: 0, type: "reroll" },
    ]);
    expect(installed.session.game.board[0][0]).toBe(true);
    expect(initial.game.hiddenItems).toEqual([]);
  });

  it("rejects a fourth item without state changes, allows replacement at capacity and removal/retry", () => {
    let input = review();
    for (const row of [0, 1, 2]) input = label(input, row, 0, "single-cell");
    input.confirmed = true;
    input.draft.hiddenItemsStatus = "recognized";
    const snapshot = structuredClone(input);
    expect(labelReviewItem(input, 3, 0, "reroll")).toEqual({
      ok: false,
      message: expect.stringContaining("최대 3개"),
    });
    expect(input).toEqual(snapshot);
    const changed = label(input, 1, 0, "reroll");
    expect(changed.draft.hiddenItems).toHaveLength(3);
    expect(changed.draft.hiddenItems.find((item) => item.row === 1)?.type).toBe(
      "reroll",
    );
    const removed = label(changed, 1, 0, null);
    expect(removed.draft.hiddenItems).toHaveLength(2);
    expect(removed.draft.board).toEqual(input.draft.board);
    expect(label(removed, 3, 0, "reroll").draft.hiddenItems).toHaveLength(3);
  });

  it("does not discard incomplete item placeholders to make space", () => {
    const input = review();
    input.draft.hiddenItems = Array.from({ length: 3 }, () => ({
      row: null,
      col: null,
      type: null,
      status: "unknown",
    }));
    const snapshot = structuredClone(input);
    expect(labelReviewItem(input, 0, 0, "single-cell").ok).toBe(false);
    expect(input).toEqual(snapshot);
  });

  it.each([
    [-1, 0],
    [16, 0],
    [0, -1],
    [0, 10],
    [0.5, 0],
    [0, NaN],
  ])("rejects out-of-board positions (%s,%s)", (row, col) => {
    const input = review();
    const snapshot = structuredClone(input);
    expect(labelReviewItem(input, row, col, "reroll").ok).toBe(false);
    expect(input).toEqual(snapshot);
  });
});
