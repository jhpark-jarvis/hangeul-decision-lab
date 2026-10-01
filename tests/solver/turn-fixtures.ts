import { createEmptyBoard } from "../../src/domain/board/board";
import { game, instance } from "../game/fixtures";

export function success<T>(
  result: ({ ok: true } & T) | { ok: false; error: { message: string } },
): T {
  if (!result.ok) throw new Error(result.error.message);
  return result;
}

export function staggeredBoard() {
  const board = createEmptyBoard();
  board.forEach((row, index) => {
    row.fill(true);
    row[(index * 3) % 10] = false;
  });
  return board;
}
export function lineTurn() {
  const board = staggeredBoard();
  board[0].fill(false, 0, 9);
  return game({
    board,
    remainingPieces: [
      instance(0, "LINE_3"),
      instance(1, "LINE_3"),
      instance(2, "LINE_3"),
    ],
  });
}
export function rescueTurn() {
  const board = staggeredBoard();
  board[3].fill(true);
  board[3].fill(false, 2, 5);
  board[5].fill(true);
  board[5].fill(false, 2, 5);
  board[4].fill(true);
  board[4][9] = false;
  return game({
    board,
    remainingPieces: [instance(0, "MIEUM"), instance(2)],
    hiddenItems: [
      { row: 4, col: 1, type: "reroll" },
      { row: 3, col: 2, type: "single-cell" },
      { row: 5, col: 4, type: "reroll" },
    ],
  });
}
