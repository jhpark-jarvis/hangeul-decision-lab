import { validateBoard, type Board, type BoardError } from "./board";

export type ClearRowsResult =
  | { ok: true; board: Board; clearedRows: number[] }
  | { ok: false; error: BoardError };

export function clearCompletedRows(input: unknown): ClearRowsResult {
  const result = validateBoard(input);
  if (!result.ok) return result;
  const clearedRows: number[] = [];
  result.board.forEach((cells, row) => {
    if (cells.every(Boolean)) clearedRows.push(row);
  });
  for (const row of clearedRows) result.board[row].fill(false);
  return { ok: true, board: result.board, clearedRows };
}
