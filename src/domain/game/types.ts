import type { Board, BoardError } from "../board/board";
import type { Piece, PieceError, Shape } from "../pieces/types";

export const MAX_ABILITY_COUNT = 7;
export const PIECES_PER_SET = 3;
export type PieceIndex = 0 | 1 | 2;
export type AbilityCounts = { reroll: number; singleCell: number };
export type HiddenItem = {
  row: number;
  col: number;
  type: "reroll" | "single-cell";
};
export type PieceInstance = {
  instanceId: string;
  pieceIndex: PieceIndex;
  piece: Piece;
};
export type PieceTarget = Pick<PieceInstance, "instanceId" | "pieceIndex">;
export type GameState = {
  board: Board;
  remainingPieces: PieceInstance[];
  hiddenItems: HiddenItem[];
  abilities: AbilityCounts;
  pendingReroll: PieceTarget | null;
};
export type GamePhase =
  "playing" | "gameover" | "await-next-pieces" | "await-reroll-result";
export type GameAction =
  | (PieceTarget & {
      type: "place-piece";
      pieceId: string;
      variant: Shape;
      row: number;
      col: number;
    })
  | { type: "single-cell"; row: number; col: number }
  | (PieceTarget & { type: "reroll" });
export type GameError = {
  code:
    | BoardError["code"]
    | PieceError["code"]
    | "OUT_OF_BOUNDS"
    | "COLLISION"
    | "INVALID_GAME_STATE"
    | "INVALID_ABILITIES"
    | "INVALID_HIDDEN_ITEMS"
    | "INVALID_CLEARED_ROWS"
    | "INVALID_PIECE_INSTANCE"
    | "INVALID_ACTION"
    | "MISSING_PIECE"
    | "INVALID_VARIANT"
    | "NO_ABILITY"
    | "AWAITING_REROLL"
    | "AWAITING_NEXT_PIECES"
    | "INVALID_LIFECYCLE";
  message: string;
};
export type GameResult<T> =
  ({ ok: true } & T) | { ok: false; error: GameError };
export type TransitionInfo = {
  clearedRows: number[];
  acquiredItems: HiddenItem[];
  discardedItems: HiddenItem[];
  spentAbility: HiddenItem["type"] | null;
  consumedPiece: PieceTarget | null;
};

export function gameError(
  code: GameError["code"],
  message: string,
): { ok: false; error: GameError } {
  return { ok: false, error: { code, message } };
}
