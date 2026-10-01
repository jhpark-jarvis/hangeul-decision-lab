import { BOARD_HEIGHT, BOARD_WIDTH } from "../board/board";
import { clearCompletedRows } from "../board/line-clear";
import { getValidPlacements, placePiece } from "../pieces/placement";
import { getUniqueVariants, serializeShape } from "../pieces/transforms";
import { validateGameState } from "./game-state";
import { collectHiddenItems } from "./hidden-items";
import {
  gameError,
  type GameAction,
  type GamePhase,
  type GameResult,
  type GameState,
  type PieceTarget,
  type TransitionInfo,
} from "./types";

/** All legal actions, independent of any future solver candidate pruning. */
export function getAvailableActions(
  input: unknown,
): GameResult<{ actions: GameAction[]; phase: GamePhase }> {
  const validated = validateGameState(input);
  if (!validated.ok) return validated;
  const { state } = validated;
  if (state.pendingReroll)
    return { ok: true, actions: [], phase: "await-reroll-result" };
  if (!state.remainingPieces.length)
    return { ok: true, actions: [], phase: "await-next-pieces" };
  const actions: GameAction[] = [];
  for (const instance of state.remainingPieces) {
    const placements = getValidPlacements(state.board, instance.piece);
    if (!placements.ok) return placements;
    for (const placement of placements.placements) {
      actions.push({
        type: "place-piece",
        instanceId: instance.instanceId,
        pieceIndex: instance.pieceIndex,
        pieceId: placement.pieceId,
        variant: placement.variant,
        row: placement.row,
        col: placement.col,
      });
    }
  }
  if (state.abilities.singleCell > 0) {
    for (let row = 0; row < BOARD_HEIGHT; row++)
      for (let col = 0; col < BOARD_WIDTH; col++) {
        if (!state.board[row][col])
          actions.push({ type: "single-cell", row, col });
      }
  }
  if (state.abilities.reroll > 0) {
    for (const instance of state.remainingPieces)
      actions.push({
        type: "reroll",
        instanceId: instance.instanceId,
        pieceIndex: instance.pieceIndex,
      });
  }
  return { ok: true, actions, phase: actions.length ? "playing" : "gameover" };
}

/** Central transition for UI Apply and solver replay. Errors leave ingress untouched. */
export function applyAction(
  input: unknown,
  inputAction: unknown,
): GameResult<{ state: GameState; info: TransitionInfo }> {
  const validated = validateGameState(input);
  if (!validated.ok) return validated;
  const { state } = validated;
  if (state.pendingReroll)
    return gameError("AWAITING_REROLL", "실제 reroll 결과를 먼저 입력하세요.");
  if (!state.remainingPieces.length)
    return gameError(
      "AWAITING_NEXT_PIECES",
      "다음 블록 3개를 먼저 입력하세요.",
    );
  if (!inputAction || typeof inputAction !== "object")
    return gameError("INVALID_ACTION", "게임 action이 필요합니다.");
  const action = inputAction as Record<string, unknown>;
  const info: TransitionInfo = {
    clearedRows: [],
    acquiredItems: [],
    discardedItems: [],
    spentAbility: null,
    consumedPiece: null,
  };
  if (action.type === "reroll" || action.type === "place-piece") {
    const instance = state.remainingPieces.find(
      (entry) =>
        entry.instanceId === action.instanceId &&
        entry.pieceIndex === action.pieceIndex,
    );
    if (!instance)
      return gameError(
        "MISSING_PIECE",
        "action 대상 instanceId/slot이 남은 블록과 일치하지 않습니다.",
      );
    const target: PieceTarget = {
      instanceId: instance.instanceId,
      pieceIndex: instance.pieceIndex,
    };
    if (action.type === "reroll") {
      if (!state.abilities.reroll)
        return gameError("NO_ABILITY", "reroll 능력이 없습니다.");
      state.abilities.reroll--;
      state.pendingReroll = target;
      info.spentAbility = "reroll";
      return { ok: true, state, info };
    }
    if (action.pieceId !== instance.piece.id)
      return gameError(
        "MISSING_PIECE",
        "블록 종류가 현재 인스턴스와 다릅니다.",
      );
    const shape = serializeShape(action.variant);
    if (!shape.ok) return shape;
    const variants = getUniqueVariants(instance.piece.shape);
    if (!variants.ok) return variants;
    if (
      !variants.variants.some((variant) => {
        const key = serializeShape(variant.shape);
        return key.ok && key.key === shape.key;
      })
    )
      return gameError(
        "INVALID_VARIANT",
        "현재 블록의 회전/반전 모양이 아닙니다.",
      );
    if (typeof action.row !== "number" || typeof action.col !== "number")
      return gameError("INVALID_COORDINATE", "배치 좌표는 정수여야 합니다.");
    const placed = placePiece(
      state.board,
      action.variant,
      action.row,
      action.col,
    );
    if (!placed.ok) return placed;
    state.board = placed.board;
    state.remainingPieces = state.remainingPieces.filter(
      (entry) => entry.instanceId !== instance.instanceId,
    );
    info.consumedPiece = target;
  } else if (action.type === "single-cell") {
    if (!state.abilities.singleCell)
      return gameError("NO_ABILITY", "single-cell 능력이 없습니다.");
    if (typeof action.row !== "number" || typeof action.col !== "number")
      return gameError("INVALID_COORDINATE", "배치 좌표는 정수여야 합니다.");
    const placed = placePiece(state.board, [[true]], action.row, action.col);
    if (!placed.ok) return placed;
    state.board = placed.board;
    state.abilities.singleCell--;
    info.spentAbility = "single-cell";
  } else return gameError("INVALID_ACTION", "지원하지 않는 action 종류입니다.");
  const cleared = clearCompletedRows(state.board);
  if (!cleared.ok) return cleared;
  const collected = collectHiddenItems(
    state.hiddenItems,
    state.abilities,
    cleared.clearedRows,
  );
  if (!collected.ok) return collected;
  state.board = cleared.board;
  state.hiddenItems = collected.hiddenItems;
  state.abilities = collected.abilities;
  info.clearedRows = cleared.clearedRows;
  info.acquiredItems = collected.acquiredItems;
  info.discardedItems = collected.discardedItems;
  return { ok: true, state, info };
}
