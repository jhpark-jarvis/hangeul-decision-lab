import { validateGameState } from "../../domain/game/game-state";
import { getInitialCatalog } from "../../domain/pieces/catalog";
import { replaceGame, type PuzzleSession } from "../puzzle/session";
import type { FieldIssue, ReviewState } from "./types";
import { validateReviewedState } from "./review";

export function applyReviewedState(
  session: PuzzleSession,
  review: ReviewState,
  expectedGameSnapshot: string,
): { ok: true; session: PuzzleSession } | { ok: false; issues: FieldIssue[] } {
  const issues = validateReviewedState(review);
  if (JSON.stringify(session.game) !== expectedGameSnapshot)
    issues.push({
      path: "session",
      code: "STALE",
      message:
        "검토 중 수동 상태가 바뀌었습니다. 현재 입력으로 검토를 다시 시작하세요.",
    });
  if (session.game.pendingReroll)
    issues.push({
      path: "session",
      code: "PENDING",
      message: "기존 화면에서 실제 reroll 결과를 먼저 입력하세요.",
    });
  if (issues.length) return { ok: false, issues };
  const catalog = getInitialCatalog();
  const draft = review.draft;
  const candidate = {
    board: draft.board.map((cells) => cells.map((cell) => cell.occupied)),
    remainingPieces: draft.pieces
      .filter((slot) => !slot.empty)
      .map((slot) => ({
        instanceId:
          session.game.remainingPieces.find(
            (piece) =>
              piece.pieceIndex === slot.slot && piece.piece.id === slot.pieceId,
          )?.instanceId ?? `review-${session.version + 1}-${slot.slot}`,
        pieceIndex: slot.slot,
        piece: catalog.find((piece) => piece.id === slot.pieceId),
      })),
    hiddenItems: draft.hiddenItems.map(({ row, col, type }) => ({
      row,
      col,
      type,
    })),
    abilities: {
      reroll: draft.abilities.reroll.value,
      singleCell: draft.abilities.singleCell.value,
    },
    pendingReroll: null,
  };
  const valid = validateGameState(candidate);
  if (!valid.ok)
    return {
      ok: false,
      issues: [
        { path: "state", code: "INVALID", message: valid.error.message },
      ],
    };
  return {
    ok: true,
    session: replaceGame(
      session,
      valid.state,
      "검토한 상태를 반영했습니다. Analyze로 분석하세요.",
    ),
  };
}
