import {
  BOARD_HEIGHT,
  BOARD_WIDTH,
  createEmptyBoard,
  validateBoard,
  type BoardError,
} from "../board/board";
import { getValidPlacements } from "../pieces/placement";
import { validatePiece } from "../pieces/transforms";
import type { Piece, PieceError } from "../pieces/types";

export type MobilityScore = {
  playablePieceTypes: number;
  totalPieceTypes: number;
  totalPlacements: number;
};
export type MobilityError = {
  code:
    | BoardError["code"]
    | PieceError["code"]
    | "INVALID_CATALOG"
    | "OUT_OF_BOUNDS"
    | "COLLISION";
  message: string;
};
export type MobilityResult<T> =
  ({ ok: true } & T) | { ok: false; error: MobilityError };

function analyzeCatalog(
  input: unknown,
  inputCatalog: unknown,
): MobilityResult<{ mobility: MobilityScore; uncoveredEmptyCells: number }> {
  const validated = validateBoard(input);
  if (!validated.ok) return validated;
  if (!Array.isArray(inputCatalog))
    return {
      ok: false,
      error: {
        code: "INVALID_CATALOG",
        message: "명시적인 Piece catalog 배열이 필요합니다.",
      },
    };
  const catalog: Piece[] = [];
  const ids = new Set<string>();
  for (const entry of inputCatalog) {
    const piece = validatePiece(entry);
    if (!piece.ok) return piece;
    if (ids.has(piece.piece.id))
      return {
        ok: false,
        error: {
          code: "INVALID_CATALOG",
          message: "catalog의 종류 ID가 중복됩니다.",
        },
      };
    ids.add(piece.piece.id);
    catalog.push(piece.piece);
  }
  const mobility: MobilityScore = {
    playablePieceTypes: 0,
    totalPieceTypes: catalog.length,
    totalPlacements: 0,
  };
  const covered = createEmptyBoard();
  for (const piece of catalog) {
    const result = getValidPlacements(validated.board, piece);
    if (!result.ok) return result;
    if (result.placements.length > 0) mobility.playablePieceTypes++;
    mobility.totalPlacements += result.placements.length;
    for (const placement of result.placements) {
      placement.variant.forEach((cells, dr) =>
        cells.forEach((cell, dc) => {
          if (cell) covered[placement.row + dr][placement.col + dc] = true;
        }),
      );
    }
  }
  let uncoveredEmptyCells = 0;
  for (let row = 0; row < BOARD_HEIGHT; row++)
    for (let col = 0; col < BOARD_WIDTH; col++) {
      if (!validated.board[row][col] && !covered[row][col])
        uncoveredEmptyCells++;
    }
  return { ok: true, mobility, uncoveredEmptyCells };
}

/** Per catalog ID, count unique orientation/coordinate placements, without probabilities. */
export function calculateMobility(
  input: unknown,
  catalog: unknown,
): MobilityResult<{ mobility: MobilityScore }> {
  const result = analyzeCatalog(input, catalog);
  if (!result.ok) return result;
  return { ok: true, mobility: result.mobility };
}

/** Number of empty cells not covered by ANY current legal placement of this catalog.
 * A snapshot heuristic, not a count of regions or proof of permanent impossibility.
 */
export function countUnfillableGaps(
  input: unknown,
  catalog: unknown,
): MobilityResult<{ count: number }> {
  const result = analyzeCatalog(input, catalog);
  if (!result.ok) return result;
  return { ok: true, count: result.uncoveredEmptyCells };
}
