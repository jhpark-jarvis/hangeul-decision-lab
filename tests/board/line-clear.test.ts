import { describe, expect, it } from "vitest";
import { createEmptyBoard } from "../../src/domain/board/board";
import { clearCompletedRows } from "../../src/domain/board/line-clear";
import { canPlace, placePiece } from "../../src/domain/pieces/placement";

describe("simultaneous horizontal row clearing", () => {
  it("leaves empty/partial boards unchanged but detached", () => {
    for (const row of [
      createEmptyBoard(),
      createEmptyBoard().map((cells, index) =>
        index === 0 ? cells.map((_, col) => col !== 9) : cells,
      ),
    ]) {
      const result = clearCompletedRows(row);
      if (!result.ok) throw new Error(result.error.message);
      expect(result.clearedRows).toEqual([]);
      expect(result.board).toEqual(row);
      result.board[0][0] = !result.board[0][0];
      expect(result.board).not.toEqual(row);
    }
  });

  it("clears adjacent and separated rows without shifting any other row", () => {
    const board = createEmptyBoard();
    for (const row of [0, 1, 8, 15]) board[row].fill(true);
    board[2][3] = true;
    board[7][7] = true;
    board[14][9] = true;
    const snapshot = structuredClone(board);
    board.forEach(Object.freeze);
    Object.freeze(board);
    const result = clearCompletedRows(board);
    if (!result.ok) throw new Error(result.error.message);
    expect(result.clearedRows).toEqual([0, 1, 8, 15]);
    for (let row = 0; row < 16; row++)
      expect(result.board[row]).toEqual(
        [0, 1, 8, 15].includes(row) ? Array(10).fill(false) : snapshot[row],
      );
    expect(board).toEqual(snapshot);
    const repeated = clearCompletedRows(result.board);
    if (!repeated.ok) throw new Error(repeated.error.message);
    expect(repeated.clearedRows).toEqual([]);
    expect(repeated.board).toEqual(result.board);
  });

  it("never clears a completed vertical column", () => {
    const board = createEmptyBoard();
    board.forEach((row) => {
      row[4] = true;
    });
    const result = clearCompletedRows(board);
    if (!result.ok) throw new Error(result.error.message);
    expect(result.clearedRows).toEqual([]);
    expect(result.board).toEqual(board);
    expect(result.board.flat().filter(Boolean)).toHaveLength(16);
  });

  it("clears all sixteen full rows and rejects malformed boards", () => {
    const result = clearCompletedRows(
      createEmptyBoard().map((row) => row.map(() => true)),
    );
    if (!result.ok) throw new Error(result.error.message);
    expect(result.clearedRows).toEqual(
      Array.from({ length: 16 }, (_, index) => index),
    );
    expect(result.board).toEqual(createEmptyBoard());
    expect(clearCompletedRows([])).toMatchObject({
      ok: false,
      error: { code: "INVALID_BOARD" },
    });
  });
});

describe("placement then clear then subsequent placement", () => {
  it("opens a previously blocked LINE_3 location and retains original and intermediate snapshots", () => {
    const original = createEmptyBoard();
    original[0] = Array.from({ length: 10 }, (_, col) => col !== 9);
    original[5][5] = true;
    const snapshot = structuredClone(original);
    const line = [[true, true, true]];
    expect(canPlace(original, line, 0, 0)).toBe(false);
    const first = placePiece(original, [[true]], 0, 9);
    if (!first.ok) throw new Error(first.error.message);
    expect(first.board[0].every(Boolean)).toBe(true); // placePiece itself does not clear.
    const cleared = clearCompletedRows(first.board);
    if (!cleared.ok) throw new Error(cleared.error.message);
    expect(cleared.clearedRows).toEqual([0]);
    expect(canPlace(cleared.board, line, 0, 0)).toBe(true);
    const second = placePiece(cleared.board, line, 0, 0);
    if (!second.ok) throw new Error(second.error.message);
    expect(second.board[0]).toEqual([
      true,
      true,
      true,
      false,
      false,
      false,
      false,
      false,
      false,
      false,
    ]);
    expect(second.board[5][5]).toBe(true);
    expect(original).toEqual(snapshot);
    expect(first.board[0].every(Boolean)).toBe(true);
    expect(cleared.board[0].some(Boolean)).toBe(false);
  });

  it("clears three rows completed by one vertical placement in the same step", () => {
    const board = createEmptyBoard();
    for (const row of [4, 5, 6])
      board[row] = Array.from({ length: 10 }, (_, col) => col !== 9);
    const placed = placePiece(board, [[true], [true], [true]], 4, 9);
    if (!placed.ok) throw new Error(placed.error.message);
    const result = clearCompletedRows(placed.board);
    if (!result.ok) throw new Error(result.error.message);
    expect(result.clearedRows).toEqual([4, 5, 6]);
    expect(result.board).toEqual(createEmptyBoard());
    expect(board[4][0]).toBe(true);
    expect(board[4][9]).toBe(false);
  });
});
