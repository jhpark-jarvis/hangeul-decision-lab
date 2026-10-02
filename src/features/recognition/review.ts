import { BOARD_HEIGHT, BOARD_WIDTH } from "../../domain/board/board";
import { getInitialCatalog } from "../../domain/pieces/catalog";
import type {
  Observation,
  RecognitionResult,
  FieldIssue,
  ReviewState,
} from "./types";

const ids = new Set(getInitialCatalog().map((piece) => piece.id));
const object = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === "object" && !Array.isArray(value);
function observation(
  value: unknown,
): value is Observation & Record<string, unknown> {
  return (
    object(value) &&
    ["recognized", "uncertain", "unknown"].includes(String(value.status)) &&
    (value.confidence === undefined ||
      (typeof value.confidence === "number" &&
        Number.isFinite(value.confidence) &&
        value.confidence >= 0 &&
        value.confidence <= 1))
  );
}

/** Structural contract only. Unknown values are legitimate recognition output. */
export function validateRecognitionResult(input: unknown): FieldIssue[] {
  const issues: FieldIssue[] = [];
  const invalid = (path: string) =>
    issues.push({
      path,
      code: "INVALID",
      message: "인식 결과의 형식이 올바르지 않습니다.",
    });
  if (!object(input)) {
    invalid("recognition");
    return issues;
  }
  if (!Array.isArray(input.board) || input.board.length !== BOARD_HEIGHT)
    invalid("board");
  else
    Array.from(input.board).forEach((cells, row) => {
      if (!Array.isArray(cells) || cells.length !== BOARD_WIDTH) {
        invalid(`board.${row}`);
        return;
      }
      Array.from(cells).forEach((cell, col) => {
        if (
          !observation(cell) ||
          !object(cell) ||
          cell.row !== row ||
          cell.col !== col ||
          (cell.occupied !== null && typeof cell.occupied !== "boolean")
        )
          invalid(`board.${row}.${col}`);
      });
    });
  if (!Array.isArray(input.pieces) || input.pieces.length !== 3)
    invalid("pieces");
  else
    Array.from(input.pieces).forEach((piece, slot) => {
      if (
        !observation(piece) ||
        !object(piece) ||
        piece.slot !== slot ||
        (piece.pieceId !== null && typeof piece.pieceId !== "string") ||
        (piece.empty !== null && typeof piece.empty !== "boolean")
      )
        invalid(`pieces.${slot}`);
    });
  if (!Array.isArray(input.hiddenItems) || input.hiddenItems.length > 3)
    invalid("hiddenItems");
  else
    Array.from(input.hiddenItems).forEach((item, index) => {
      if (
        !observation(item) ||
        !object(item) ||
        ![null, "reroll", "single-cell"].includes(item.type as null | string) ||
        (item.row !== null && typeof item.row !== "number") ||
        (item.col !== null && typeof item.col !== "number")
      )
        invalid(`hiddenItems.${index}`);
    });
  if (
    !["recognized", "uncertain", "unknown"].includes(
      String(input.hiddenItemsStatus),
    )
  )
    invalid("hiddenItemsStatus");
  for (const ability of ["reroll", "singleCell"]) {
    const count = object(input.abilities) ? input.abilities[ability] : null;
    if (
      !observation(count) ||
      !object(count) ||
      (count.value !== null && typeof count.value !== "number")
    )
      invalid(`abilities.${ability}`);
  }
  if (
    !object(input.source) ||
    ![
      "capture-stub",
      "manual-mock",
      "calibrated-board",
      "automatic-board",
      "automatic-game",
    ].includes(String(input.source.kind)) ||
    typeof input.source.timestamp !== "number" ||
    !Number.isFinite(input.source.timestamp)
  )
    invalid("source");
  else if (
    input.source.dimensions !== undefined &&
    (!object(input.source.dimensions) ||
      !Number.isInteger(input.source.dimensions.width) ||
      !Number.isInteger(input.source.dimensions.height) ||
      Number(input.source.dimensions.width) <= 0 ||
      Number(input.source.dimensions.height) <= 0)
  )
    invalid("source.dimensions");
  if (
    !object(input.summary) ||
    ![
      "unimplemented",
      "manual-copy",
      "rgb-samples",
      "grid-tiles",
      "grid-panel",
    ].includes(String(input.summary.engine))
  )
    invalid("summary");
  return issues;
}

export function createReviewState(recognition: RecognitionResult): ReviewState {
  if (validateRecognitionResult(recognition).length)
    throw new Error("인식 결과 형식이 올바르지 않습니다.");
  return {
    original: structuredClone(recognition),
    draft: structuredClone(recognition),
    revision: 0,
    confirmed: false,
  };
}
export function updateReview(
  review: ReviewState,
  change: (draft: RecognitionResult) => void,
): ReviewState {
  const draft = structuredClone(review.draft);
  change(draft);
  return { ...review, draft, revision: review.revision + 1, confirmed: false };
}
export function validateReviewedState(
  review: ReviewState,
  requireConfirmation = true,
): FieldIssue[] {
  const issues = validateRecognitionResult(review.draft);
  if (issues.length) return issues;
  const { draft } = review;
  const unresolved = (path: string) =>
    issues.push({
      path,
      code: "UNRESOLVED",
      message: "미확정 또는 불확실한 값을 직접 확인하세요.",
    });
  const invalid = (path: string, message: string) =>
    issues.push({ path, code: "INVALID", message });
  draft.board.forEach((cells, row) =>
    cells.forEach((cell, col) => {
      if (cell.status !== "recognized" || cell.occupied === null)
        unresolved(`board.${row}.${col}`);
    }),
  );
  draft.pieces.forEach((piece, slot) => {
    if (
      piece.status !== "recognized" ||
      piece.empty === null ||
      (!piece.empty && piece.pieceId === null)
    )
      unresolved(`pieces.${slot}`);
    else if (
      (piece.empty && piece.pieceId !== null) ||
      (!piece.empty && !ids.has(piece.pieceId!))
    )
      invalid(
        `pieces.${slot}`,
        "목록에 있는 블록 또는 이미 사용/없음을 선택하세요.",
      );
  });
  if (draft.hiddenItemsStatus !== "recognized") unresolved("hiddenItemsStatus");
  const positions = new Set<string>();
  draft.hiddenItems.forEach((item, index) => {
    const path = `hiddenItems.${index}`;
    if (
      item.status !== "recognized" ||
      item.type === null ||
      item.row === null ||
      item.col === null
    ) {
      unresolved(path);
      return;
    }
    if (!Number.isInteger(item.row) || item.row < 0 || item.row >= BOARD_HEIGHT)
      invalid(`${path}.row`, "행은 0~15 정수입니다.");
    if (!Number.isInteger(item.col) || item.col < 0 || item.col >= BOARD_WIDTH)
      invalid(`${path}.col`, "열은 0~9 정수입니다.");
    const key = `${item.row},${item.col}`;
    if (positions.has(key)) invalid(path, "아이템 좌표가 중복됩니다.");
    positions.add(key);
  });
  for (const key of ["reroll", "singleCell"] as const) {
    const count = draft.abilities[key];
    if (count.status !== "recognized" || count.value === null)
      unresolved(`abilities.${key}`);
    else if (
      !Number.isInteger(count.value) ||
      count.value < 0 ||
      count.value > 7
    )
      invalid(`abilities.${key}`, "능력 수는 0~7 정수입니다.");
  }
  if (
    Number(draft.abilities.reroll.value) +
      Number(draft.abilities.singleCell.value) >
    7
  )
    invalid("abilities", "능력 보유 합계는 7 이하여야 합니다.");
  if (requireConfirmation && !review.confirmed)
    issues.push({
      path: "confirmation",
      code: "UNCONFIRMED",
      message: "전체 검토 확인이 필요합니다.",
    });
  return issues;
}
