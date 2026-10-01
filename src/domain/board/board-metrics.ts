import {
  BOARD_HEIGHT,
  BOARD_WIDTH,
  validateBoard,
  type BoardError,
} from "./board";

export type BoardMetricResult<T> =
  ({ ok: true } & T) | { ok: false; error: BoardError };
export type RowProgress = {
  row: number;
  occupiedCells: number;
  emptyCells: number;
  filledRatio: number;
};
const ORTHOGONAL_NEIGHBORS = [
  [-1, 0],
  [1, 0],
  [0, -1],
  [0, 1],
] as const;

export function countEmptyCells(
  input: unknown,
): BoardMetricResult<{ count: number }> {
  const result = validateBoard(input);
  if (!result.ok) return result;
  return {
    ok: true,
    count: result.board.reduce(
      (sum, row) => sum + row.filter((cell) => !cell).length,
      0,
    ),
  };
}

export function countOccupiedCells(
  input: unknown,
): BoardMetricResult<{ count: number }> {
  const result = validateBoard(input);
  if (!result.ok) return result;
  return {
    ok: true,
    count: result.board.reduce(
      (sum, row) => sum + row.filter(Boolean).length,
      0,
    ),
  };
}

/** Outside the board is a boundary, not an empty neighbor; diagonals do not connect. */
export function countIsolatedEmptyCells(
  input: unknown,
): BoardMetricResult<{ count: number }> {
  const result = validateBoard(input);
  if (!result.ok) return result;
  let count = 0;
  for (let row = 0; row < BOARD_HEIGHT; row++) {
    for (let col = 0; col < BOARD_WIDTH; col++) {
      if (result.board[row][col]) continue;
      if (
        ORTHOGONAL_NEIGHBORS.every(([dr, dc]) => {
          const neighborRow = row + dr;
          const neighborCol = col + dc;
          return (
            neighborRow < 0 ||
            neighborRow >= BOARD_HEIGHT ||
            neighborCol < 0 ||
            neighborCol >= BOARD_WIDTH ||
            result.board[neighborRow][neighborCol]
          );
        })
      )
        count++;
    }
  }
  return { ok: true, count };
}

/** Snapshot only: a full row is reported as 1, never cleared implicitly. */
export function calculateRowProgress(
  input: unknown,
): BoardMetricResult<{ rows: RowProgress[] }> {
  const result = validateBoard(input);
  if (!result.ok) return result;
  return {
    ok: true,
    rows: result.board.map((cells, row) => {
      const occupiedCells = cells.filter(Boolean).length;
      return {
        row,
        occupiedCells,
        emptyCells: BOARD_WIDTH - occupiedCells,
        filledRatio: occupiedCells / BOARD_WIDTH,
      };
    }),
  };
}
