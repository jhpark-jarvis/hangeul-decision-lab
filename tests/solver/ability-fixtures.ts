import { game, instance } from "../game/fixtures";
import { staggeredBoard } from "./turn-fixtures";

export function singleRescue(holes = 1) {
  const board = staggeredBoard();
  for (const row of [3, 5]) {
    board[row].fill(true);
    board[row].fill(false, 2, 5);
  }
  board[4].fill(true);
  board[4].fill(false, 10 - holes);
  return game({
    board,
    remainingPieces: [instance(2, "MIEUM")],
    abilities: { reroll: 0, singleCell: holes },
  });
}

export function reacquireRescue() {
  const board = staggeredBoard();
  for (const row of [3, 5, 9, 11]) {
    board[row].fill(true);
    board[row].fill(false, 2, 5);
  }
  for (const row of [4, 10]) {
    board[row].fill(true);
    board[row][9] = false;
  }
  return game({
    board,
    remainingPieces: [instance(0, "MIEUM"), instance(2, "MIEUM")],
    abilities: { reroll: 0, singleCell: 1 },
    hiddenItems: [{ row: 3, col: 0, type: "single-cell" }],
  });
}
