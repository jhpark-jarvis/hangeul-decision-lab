import { validateBoard } from "../board/board";
import { validatePiece } from "../pieces/transforms";
import { validateAbilities } from "./abilities";
import { validateHiddenItems } from "./hidden-items";
import {
  gameError,
  PIECES_PER_SET,
  type GameResult,
  type GameState,
  type PieceIndex,
  type PieceInstance,
  type PieceTarget,
} from "./types";

export function validatePieceInstances(
  input: unknown,
): GameResult<{ remainingPieces: PieceInstance[] }> {
  if (!Array.isArray(input) || input.length > PIECES_PER_SET)
    return gameError("INVALID_PIECE_INSTANCE", "남은 블록은 최대 3개입니다.");
  const remainingPieces: PieceInstance[] = [];
  const ids = new Set<string>();
  const slots = new Set<number>();
  for (const instance of input) {
    if (!instance || typeof instance !== "object")
      return gameError("INVALID_PIECE_INSTANCE", "블록 인스턴스가 필요합니다.");
    const { instanceId, pieceIndex, piece } = instance as Record<
      string,
      unknown
    >;
    if (
      typeof instanceId !== "string" ||
      !instanceId.trim() ||
      ids.has(instanceId) ||
      typeof pieceIndex !== "number" ||
      !Number.isInteger(pieceIndex) ||
      pieceIndex < 0 ||
      pieceIndex >= PIECES_PER_SET ||
      slots.has(pieceIndex)
    ) {
      return gameError(
        "INVALID_PIECE_INSTANCE",
        "고유 instanceId와 서로 다른 0~2 슬롯이 필요합니다.",
      );
    }
    const validated = validatePiece(piece);
    if (!validated.ok) return validated;
    ids.add(instanceId);
    slots.add(pieceIndex);
    remainingPieces.push({
      instanceId,
      pieceIndex: pieceIndex as PieceIndex,
      piece: validated.piece,
    });
  }
  return { ok: true, remainingPieces };
}

/** Validate every ingress field and detach nested arrays and objects. */
export function validateGameState(
  input: unknown,
): GameResult<{ state: GameState }> {
  if (!input || typeof input !== "object")
    return gameError("INVALID_GAME_STATE", "게임 상태가 필요합니다.");
  const source = input as Record<string, unknown>;
  const board = validateBoard(source.board);
  if (!board.ok) return board;
  const pieces = validatePieceInstances(source.remainingPieces);
  if (!pieces.ok) return pieces;
  const items = validateHiddenItems(source.hiddenItems);
  if (!items.ok) return items;
  const abilities = validateAbilities(source.abilities);
  if (!abilities.ok) return abilities;
  let pendingReroll: PieceTarget | null = null;
  if (source.pendingReroll !== null) {
    if (!source.pendingReroll || typeof source.pendingReroll !== "object")
      return gameError(
        "INVALID_GAME_STATE",
        "pendingReroll은 null 또는 남은 블록 대상이어야 합니다.",
      );
    const target = source.pendingReroll as Record<string, unknown>;
    const found = pieces.remainingPieces.find(
      (piece) =>
        piece.instanceId === target.instanceId &&
        piece.pieceIndex === target.pieceIndex,
    );
    if (!found)
      return gameError(
        "INVALID_GAME_STATE",
        "reroll 대기 대상이 남은 슬롯에 없습니다.",
      );
    pendingReroll = {
      instanceId: found.instanceId,
      pieceIndex: found.pieceIndex,
    };
  }
  return {
    ok: true,
    state: {
      board: board.board,
      remainingPieces: pieces.remainingPieces,
      hiddenItems: items.hiddenItems,
      abilities: abilities.abilities,
      pendingReroll,
    },
  };
}

export function enterNextPieces(
  input: unknown,
  inputPieces: unknown,
): GameResult<{ state: GameState }> {
  const validated = validateGameState(input);
  if (!validated.ok) return validated;
  if (
    validated.state.remainingPieces.length !== 0 ||
    validated.state.pendingReroll !== null
  )
    return gameError(
      "INVALID_LIFECYCLE",
      "현재 블록을 모두 소진한 뒤 다음 세트를 입력하세요.",
    );
  const pieces = validatePieceInstances(inputPieces);
  if (!pieces.ok) return pieces;
  if (pieces.remainingPieces.length !== PIECES_PER_SET)
    return gameError(
      "INVALID_LIFECYCLE",
      "다음 세트는 정확히 3개 블록이 필요합니다.",
    );
  return {
    ok: true,
    state: { ...validated.state, remainingPieces: pieces.remainingPieces },
  };
}

/** Accept the actual externally observed result; never sample a random piece. */
export function resolveReroll(
  input: unknown,
  inputPiece: unknown,
): GameResult<{ state: GameState }> {
  const validated = validateGameState(input);
  if (!validated.ok) return validated;
  const { state } = validated;
  if (!state.pendingReroll)
    return gameError(
      "INVALID_LIFECYCLE",
      "reroll 결과 입력 대기 상태가 아닙니다.",
    );
  const piece = validatePiece(inputPiece);
  if (!piece.ok) return piece;
  const target = state.remainingPieces.find(
    (entry) => entry.instanceId === state.pendingReroll?.instanceId,
  )!;
  if (piece.piece.id === target.piece.id)
    return gameError(
      "INVALID_LIFECYCLE",
      "reroll 결과는 다른 종류의 블록이어야 합니다.",
    );
  target.piece = piece.piece;
  state.pendingReroll = null;
  return { ok: true, state };
}
