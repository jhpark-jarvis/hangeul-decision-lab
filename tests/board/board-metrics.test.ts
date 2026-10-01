import { describe, expect, it } from "vitest";
import { createEmptyBoard } from "../../src/domain/board/board";
import {
  countEmptyCells,
  countOccupiedCells,
  countIsolatedEmptyCells,
  calculateRowProgress,
} from "../../src/domain/board/board-metrics";
import { freezeDeep } from "../game/fixtures";

function occupiedExcept(positions: number[][]) {
  const board = createEmptyBoard();
  board.forEach((row) => row.fill(true));
  for (const [row, col] of positions) board[row][col] = false;
  return board;
}

describe("independent board metrics", () => {
  it.each([
    { board: createEmptyBoard(), empty: 160, occupied: 0 },
    { board: occupiedExcept([]), empty: 0, occupied: 160 },
    {
      board: occupiedExcept([
        [0, 0],
        [15, 9],
        [5, 3],
      ]),
      empty: 3,
      occupied: 157,
    },
  ])(
    "counts empty and occupied cells on snapshot %#",
    ({ board, empty, occupied }) => {
      freezeDeep(board);
      expect(countEmptyCells(board)).toEqual({ ok: true, count: empty });
      expect(countOccupiedCells(board)).toEqual({ ok: true, count: occupied });
      expect(empty + occupied).toBe(160);
    },
  );
  it.each([
    { positions: [[7, 4]], expected: 1 },
    { positions: [[0, 0]], expected: 1 },
    {
      positions: [
        [0, 0],
        [15, 9],
        [0, 9],
        [15, 0],
      ],
      expected: 4,
    },
    {
      positions: [
        [4, 4],
        [5, 5],
      ],
      expected: 2,
    },
    {
      positions: [
        [4, 4],
        [4, 5],
      ],
      expected: 0,
    },
    {
      positions: [
        [4, 4],
        [5, 4],
      ],
      expected: 0,
    },
    {
      positions: [
        [0, 9],
        [1, 0],
      ],
      expected: 2,
    },
    {
      positions: [
        [0, 0],
        [0, 9],
      ],
      expected: 2,
    },
    { positions: [], expected: 0 },
  ])(
    "counts only orthogonally isolated empty cells %#",
    ({ positions, expected }) => {
      expect(
        countIsolatedEmptyCells(freezeDeep(occupiedExcept(positions))),
      ).toEqual({ ok: true, count: expected });
    },
  );
  it("an empty board has no isolated cell", () => {
    expect(countIsolatedEmptyCells(createEmptyBoard())).toEqual({
      ok: true,
      count: 0,
    });
  });
  it("reports all 16 original row indices and occupancy fractions without clearing", () => {
    const board = createEmptyBoard();
    board[0][0] = true;
    board[9].fill(true, 0, 9);
    board[15].fill(true);
    freezeDeep(board);
    const result = calculateRowProgress(board);
    if (!result.ok) throw new Error(result.error.message);
    expect(result.rows).toHaveLength(16);
    expect(result.rows[0]).toEqual({
      row: 0,
      occupiedCells: 1,
      emptyCells: 9,
      filledRatio: 0.1,
    });
    expect(result.rows[1]).toEqual({
      row: 1,
      occupiedCells: 0,
      emptyCells: 10,
      filledRatio: 0,
    });
    expect(result.rows[9]).toEqual({
      row: 9,
      occupiedCells: 9,
      emptyCells: 1,
      filledRatio: 0.9,
    });
    expect(result.rows[15]).toEqual({
      row: 15,
      occupiedCells: 10,
      emptyCells: 0,
      filledRatio: 1,
    });
    expect(result.rows.reduce((sum, row) => sum + row.occupiedCells, 0)).toBe(
      20,
    );
    result.rows[0].occupiedCells = 10;
    result.rows[0].filledRatio = 1;
    expect(board[0].filter(Boolean)).toHaveLength(1);
    const again = calculateRowProgress(board);
    if (!again.ok) throw new Error(again.error.message);
    expect(again.rows[0].filledRatio).toBe(0.1);
    expect(again.rows[15].filledRatio).toBe(1);
    expect(board[15].every(Boolean)).toBe(true);
  });
  it.each([
    { label: "countEmptyCells", metric: countEmptyCells },
    { label: "countOccupiedCells", metric: countOccupiedCells },
    { label: "countIsolatedEmptyCells", metric: countIsolatedEmptyCells },
    { label: "calculateRowProgress", metric: calculateRowProgress },
  ])(
    "$label rejects malformed boards and recovers on valid input",
    ({ metric }) => {
      for (const input of [
        null,
        [],
        {},
        Array(16),
        Array.from({ length: 16 }, () => Array(10)),
        [[false]],
        Array.from({ length: 16 }, () => Array(10).fill(0)),
      ]) {
        expect(metric(input)).toMatchObject({
          ok: false,
          error: { code: "INVALID_BOARD" },
        });
      }
      expect(metric(freezeDeep(createEmptyBoard()))).toMatchObject({
        ok: true,
      });
    },
  );
});
