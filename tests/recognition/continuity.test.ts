import { describe, it, expect } from "vitest";
import {
  createSession,
  loadNextPieces,
} from "../../src/features/puzzle/session";
import { applyAction } from "../../src/domain/game/actions";
import { recognizeManualState } from "../../src/features/recognition/mock";
import {
  createReviewState,
  validateReviewedState,
} from "../../src/features/recognition/review";
import {
  continueTurn,
  confirmWholeReview,
} from "../../src/features/recognition/continuity";

function game() {
  return loadNextPieces(createSession(), ["DOT", "DOT", "DOT"]).game;
}
describe("reviewed turn continuity", () => {
  it.each([0, 7])(
    "uses actual post-Apply item acquisition/retention at capacity %i without reintroducing consumed items",
    (capacity) => {
      const previous = game();
      previous.board[0] = Array(10).fill(true);
      previous.board[0][9] = false;
      previous.abilities.reroll = capacity;
      previous.hiddenItems = [
        { row: 0, col: 2, type: "single-cell" },
        { row: 15, col: 9, type: "reroll" },
      ];
      const instance = previous.remainingPieces[0];
      const applied = applyAction(previous, {
        type: "place-piece",
        ...instance,
        pieceId: "DOT",
        variant: instance.piece.shape,
        row: 0,
        col: 9,
      });
      expect(applied.ok).toBe(true);
      if (!applied.ok) return;
      const fresh = recognizeManualState(game());
      fresh.board = recognizeManualState(applied.state).board;
      fresh.board[15][9] = {
        row: 15,
        col: 9,
        occupied: null,
        status: "unknown",
      };
      fresh.abilities.reroll = { value: null, status: "unknown" };
      const before = structuredClone({ fresh, expected: applied.state });
      const next = continueTurn(fresh, applied.state);
      expect(next.matched).toBe(true);
      expect(next.inherited).toEqual([{ row: 15, col: 9 }]);
      expect(
        next.result.hiddenItems.map(({ row, col, type }) => ({
          row,
          col,
          type,
        })),
      ).toEqual(
        capacity === 7 ? previous.hiddenItems : [previous.hiddenItems[1]],
      );
      expect(next.result.abilities.reroll).toEqual({
        value: null,
        status: "unknown",
      });
      expect(next.result.pieces.map((p) => p.empty)).toEqual([
        false,
        false,
        false,
      ]);
      expect(next.result.hiddenItemsStatus).toBe("unknown");
      expect({ fresh, expected: applied.state }).toEqual(before);
      expect(
        validateReviewedState(
          confirmWholeReview(createReviewState(next.result)),
        ),
      ).not.toEqual([]);
    },
  );
  it("requires explicit whole confirmation, leaves original unconfirmed and does not bypass invalid fields", () => {
    const state = game();
    state.hiddenItems = [{ row: 5, col: 7, type: "reroll" }];
    const next = continueTurn(recognizeManualState(state), state);
    const review = createReviewState(next.result),
      confirmed = confirmWholeReview(review);
    expect(review.confirmed).toBe(false);
    expect(review.draft.hiddenItemsStatus).toBe("unknown");
    expect(confirmed.confirmed).toBe(true);
    expect(confirmed.revision).toBeGreaterThan(review.revision);
    expect(validateReviewedState(confirmed)).toEqual([]);
    confirmed.draft.board[0][0].occupied = null;
    expect(
      validateReviewedState(confirmed).some((i) => i.path.startsWith("board")),
    ).toBe(true);
    confirmed.draft.abilities.reroll.value = 8;
    expect(validateReviewedState(confirmed).length).toBeGreaterThan(0);
  });
  it.each(["different", "coverage", "pending"] as const)(
    "keeps prior items as uncertain drafts and unknown occupancy on %s",
    (condition) => {
      const state = game();
      state.hiddenItems = [{ row: 7, col: 8, type: "single-cell" }];
      const fresh = recognizeManualState(state);
      fresh.board[15][9].occupied = null;
      fresh.board[15][9].status = "unknown";
      if (condition === "different") fresh.board[0][0].occupied = true;
      if (condition === "coverage")
        fresh.board
          .flat()
          .slice(0, 25)
          .forEach((c) => {
            c.occupied = null;
            c.status = "unknown";
          });
      if (condition === "pending")
        state.pendingReroll = {
          instanceId: state.remainingPieces[0].instanceId,
          pieceIndex: 0,
        };
      const next = continueTurn(fresh, state);
      expect(next.matched).toBe(false);
      expect(next.inherited).toEqual([]);
      expect(next.result.board[15][9].occupied).toBeNull();
      expect(next.result.hiddenItems[0]).toEqual({
        ...state.hiddenItems[0],
        status: "uncertain",
      });
      expect(next.differences).toEqual(
        condition === "different" ? [{ row: 0, col: 0 }] : [],
      );
    },
  );
});
