import {
  BOARD_HEIGHT,
  BOARD_WIDTH,
  validateBoard,
  type Board,
  type BoardError,
} from "../board/board";
import { getUniqueVariants, normalizeShape, validatePiece } from "./transforms";
import type { PieceError, Placement, Shape } from "./types";

type PlacementError =
  | BoardError
  | PieceError
  | { code: "OUT_OF_BOUNDS" | "COLLISION"; message: string };
export type PlacementResult<T> =
  ({ ok: true } & T) | { ok: false; error: PlacementError };

function coordinateIsValid(row: number, col: number): boolean {
  return Number.isInteger(row) && Number.isInteger(col);
}

function withinBounds(shape: Shape, row: number, col: number): boolean {
  return (
    row >= 0 &&
    col >= 0 &&
    row + shape.length <= BOARD_HEIGHT &&
    col + shape[0].length <= BOARD_WIDTH
  );
}

function overlaps(
  board: Board,
  shape: Shape,
  row: number,
  col: number,
): boolean {
  return shape.some((cells, offsetRow) =>
    cells.some(
      (cell, offsetCol) => cell && board[row + offsetRow][col + offsetCol],
    ),
  );
}

export function canPlace(
  input: unknown,
  inputShape: unknown,
  row: number,
  col: number,
): boolean {
  const board = validateBoard(input);
  const shape = normalizeShape(inputShape);
  return (
    board.ok &&
    shape.ok &&
    coordinateIsValid(row, col) &&
    withinBounds(shape.shape, row, col) &&
    !overlaps(board.board, shape.shape, row, col)
  );
}

export function placePiece(
  input: unknown,
  inputShape: unknown,
  row: number,
  col: number,
): PlacementResult<{ board: Board }> {
  const result = validateBoard(input);
  if (!result.ok) return result;
  const shape = normalizeShape(inputShape);
  if (!shape.ok) return shape;
  if (!coordinateIsValid(row, col)) {
    return {
      ok: false,
      error: {
        code: "INVALID_COORDINATE",
        message: "배치 좌표는 정수여야 합니다.",
      },
    };
  }
  if (!withinBounds(shape.shape, row, col)) {
    return {
      ok: false,
      error: {
        code: "OUT_OF_BOUNDS",
        message: "블록이 보드 범위를 벗어납니다.",
      },
    };
  }
  if (overlaps(result.board, shape.shape, row, col)) {
    return {
      ok: false,
      error: { code: "COLLISION", message: "블록이 기존 점유 칸과 겹칩니다." },
    };
  }
  shape.shape.forEach((cells, offsetRow) => {
    cells.forEach((cell, offsetCol) => {
      if (cell) result.board[row + offsetRow][col + offsetCol] = true;
    });
  });
  return { ok: true, board: result.board };
}

export function getValidPlacements(
  input: unknown,
  inputPiece: unknown,
): PlacementResult<{ placements: Placement[] }> {
  const board = validateBoard(input);
  if (!board.ok) return board;
  const piece = validatePiece(inputPiece);
  if (!piece.ok) return piece;
  const result = getUniqueVariants(piece.piece.shape);
  if (!result.ok) return result;
  const placements: Placement[] = [];
  for (const variant of result.variants) {
    for (let row = 0; row <= BOARD_HEIGHT - variant.shape.length; row++) {
      for (let col = 0; col <= BOARD_WIDTH - variant.shape[0].length; col++) {
        if (overlaps(board.board, variant.shape, row, col)) continue;
        placements.push({
          pieceId: piece.piece.id,
          variant: variant.shape.map((cells) => cells.slice()),
          rotation: variant.rotation,
          flipped: variant.flipped,
          row,
          col,
        });
      }
    }
  }
  return { ok: true, placements };
}
