import type { Piece } from "../../domain/pieces/types";

export const HANGEUL_PIECE_LABELS = [
  "ㄱ",
  "ㄴ",
  "ㄷ",
  "ㄹ",
  "ㅁ",
  "ㅂ",
  "ㅅ",
  "ㅇ",
  "ㅈ",
  "ㅊ",
  "ㅋ",
  "ㅌ",
  "ㅍ",
  "ㅎ",
  "ㅡ",
  "ㅣ",
  "ㅏ",
  "ㅑ",
  "점",
] as const;

export type PieceLabelMapping = {
  pieceId: string;
  label: string;
  cells: number;
}[];

export const LABEL_CELL_COUNTS: Readonly<Record<string, number>> = {
  ㅡ: 3,
  ㅣ: 5,
  점: 1,
};

/** Text-only user mapping; never changes catalog or infers an unassigned label. */
export function completePieceLabels(
  catalog: readonly Piece[],
  labels: readonly string[],
): { ok: true; mapping: PieceLabelMapping } | { ok: false; message: string } {
  if (
    catalog.length !== HANGEUL_PIECE_LABELS.length ||
    labels.length !== catalog.length ||
    new Set(catalog.map((piece) => piece.id)).size !== catalog.length
  )
    return { ok: false, message: "19개 도형의 이름을 모두 선택하세요." };
  if (
    Array.from(labels).some(
      (label) => !HANGEUL_PIECE_LABELS.some((allowed) => allowed === label),
    )
  )
    return { ok: false, message: "아직 이름을 고르지 않은 도형이 있습니다." };
  if (new Set(labels).size !== labels.length)
    return { ok: false, message: "같은 이름을 두 도형에 지정할 수 없습니다." };
  if (
    catalog.some((piece, index) => {
      const expected = LABEL_CELL_COUNTS[labels[index]];
      return (
        expected !== undefined &&
        piece.shape.flat().filter(Boolean).length !== expected
      );
    })
  )
    return {
      ok: false,
      message: "ㅡ는 3칸, ㅣ는 5칸, 점은 1칸 도형에 지정하세요.",
    };
  return {
    ok: true,
    mapping: catalog.map((piece, index) => ({
      pieceId: piece.id,
      label: labels[index],
      cells: piece.shape.flat().filter(Boolean).length,
    })),
  };
}
