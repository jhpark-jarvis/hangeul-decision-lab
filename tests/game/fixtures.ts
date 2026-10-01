import { createEmptyBoard } from "../../src/domain/board/board";
import { getInitialCatalog } from "../../src/domain/pieces/catalog";
import type { Piece } from "../../src/domain/pieces/types";
import type {
  GameAction,
  GameResult,
  GameState,
  PieceIndex,
  PieceInstance,
} from "../../src/domain/game/types";

export function piece(id = "DOT"): Piece {
  const found = getInitialCatalog().find((entry) => entry.id === id);
  if (!found) throw new Error("Unknown fixture piece");
  return found;
}

/** Original four-type oracle fixture; deliberately independent of default catalog growth. */
export function getRegressionCatalog(): Piece[] {
  return ["DOT", "LINE_3", "MIEUM", "L_3"].map((id) => piece(id));
}
export function instance(
  pieceIndex: PieceIndex,
  id = "DOT",
  instanceId = `set1-${pieceIndex}`,
): PieceInstance {
  return { instanceId, pieceIndex, piece: piece(id) };
}
export function game(overrides: Partial<GameState> = {}): GameState {
  return {
    board: createEmptyBoard(),
    remainingPieces: [instance(0), instance(1), instance(2)],
    hiddenItems: [],
    abilities: { reroll: 0, singleCell: 0 },
    pendingReroll: null,
    ...overrides,
  };
}
export function placement(
  target: PieceInstance,
  row: number,
  col: number,
): GameAction {
  return {
    type: "place-piece",
    instanceId: target.instanceId,
    pieceIndex: target.pieceIndex,
    pieceId: target.piece.id,
    variant: target.piece.shape,
    row,
    col,
  };
}
export function success<T>(result: GameResult<T>): T {
  if (!result.ok) throw new Error(result.error.message);
  return result;
}
export function freezeDeep<T>(value: T): T {
  if (value && typeof value === "object") {
    for (const field of Object.values(value)) freezeDeep(field);
    Object.freeze(value);
  }
  return value;
}
