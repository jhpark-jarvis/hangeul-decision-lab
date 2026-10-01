import { describe, expect, it } from "vitest";
import {
  BOARD_CELL_COUNT,
  BOARD_HEIGHT,
  BOARD_WIDTH,
  createEmptyBoard,
  toggleCell,
  validateBoard,
  type Board,
  type BoardResult,
} from "../../src/domain/board/board";

function successfulBoard(result: BoardResult): Board {
  expect(result.ok).toBe(true);
  if (!result.ok) throw new Error(result.error.message);
  return result.board;
}

describe("board input contract", () => {
  it("creates 160 empty cells in 16 independent rows of 10 columns", () => {
    const board = createEmptyBoard();
    expect(board).toHaveLength(BOARD_HEIGHT);
    expect(board.every((row) => row.length === BOARD_WIDTH)).toBe(true);
    expect(board.flat()).toHaveLength(BOARD_CELL_COUNT);
    expect(board.flat().every((cell) => cell === false)).toBe(true);
    board[0][0] = true;
    expect(board.slice(1).every((row) => row[0] === false)).toBe(true);
    expect(createEmptyBoard()[0][0]).toBe(false);
  });

  it.each([
    [0, 0],
    [0, 9],
    [15, 0],
    [15, 9],
    [7, 4],
  ])("toggles only (%i, %i) and restores it on a second action", (row, col) => {
    const original = createEmptyBoard();
    const changed = successfulBoard(toggleCell(original, row, col));
    expect(changed[row][col]).toBe(true);
    expect(changed.flat().filter(Boolean)).toHaveLength(1);
    expect(successfulBoard(toggleCell(changed, row, col))).toEqual(original);
  });

  it("preserves frozen snapshots and keeps returned boards independently mutable", () => {
    const original = createEmptyBoard();
    original.forEach(Object.freeze);
    Object.freeze(original);
    const changed = successfulBoard(toggleCell(original, 0, 0));
    const next = successfulBoard(toggleCell(changed, 15, 9));
    next[0][0] = false;
    expect(original.flat().some(Boolean)).toBe(false);
    expect(changed[0][0]).toBe(true);
    expect(changed[15][9]).toBe(false);
    expect(next[15][9]).toBe(true);
  });

  it("detaches validated external input without discarding occupied cells", () => {
    const input = createEmptyBoard();
    input[15][9] = true;
    const validated = successfulBoard(validateBoard(input));
    input[15][9] = false;
    expect(validated[15][9]).toBe(true);
    validated[0][0] = true;
    expect(input[0][0]).toBe(false);
  });

  const malformed: unknown[] = [
    null,
    {},
    [],
    createEmptyBoard().slice(1),
    [...createEmptyBoard(), Array<boolean>(BOARD_WIDTH).fill(false)],
    createEmptyBoard().map((row, index) => (index === 0 ? row.slice(1) : row)),
    createEmptyBoard().map((row, index) =>
      index === 0 ? [...row, false] : row,
    ),
    createEmptyBoard().map((row, index) =>
      index === 0 ? [1, ...row.slice(1)] : row,
    ),
    createEmptyBoard().map((row, index) =>
      index === 0 ? ["false", ...row.slice(1)] : row,
    ),
    Array(BOARD_HEIGHT),
    Array.from({ length: BOARD_HEIGHT }, () => Array(BOARD_WIDTH)),
  ];
  it.each(malformed.map((input, index) => ({ input, index })))(
    "rejects malformed board fixture $index instead of treating it as empty",
    ({ input }) => {
      expect(validateBoard(input)).toMatchObject({
        ok: false,
        error: { code: "INVALID_BOARD" },
      });
      expect(toggleCell(input, 0, 0)).toMatchObject({
        ok: false,
        error: { code: "INVALID_BOARD" },
      });
    },
  );

  it.each([
    [-1, 0],
    [16, 0],
    [0, -1],
    [0, 10],
    [0.5, 0],
    [0, 0.5],
    [NaN, 0],
    [0, NaN],
    [Infinity, 0],
    [0, Infinity],
  ])(
    "rejects coordinate (%s, %s) and preserves the previous board",
    (row, col) => {
      const board = createEmptyBoard();
      board[7][4] = true;
      const snapshot = structuredClone(board);
      expect(toggleCell(board, row, col)).toMatchObject({
        ok: false,
        error: { code: "INVALID_COORDINATE" },
      });
      expect(board).toEqual(snapshot);
    },
  );
});
