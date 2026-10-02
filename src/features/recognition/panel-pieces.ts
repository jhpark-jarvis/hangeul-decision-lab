import type { CapturedFrame } from "../capture/capture";
import type { PixelRegion } from "./calibration";
import type { Piece } from "../../domain/pieces/types";
import { getUniqueVariants } from "../../domain/pieces/transforms";
import type { RecognizedPieceSlot } from "./types";
import {
  panelCards,
  validPanelFrame,
  contained,
  relative,
  PANEL_PROFILE as C,
} from "./panel-profile";
import {
  readMask,
  maskBounds,
  matchText,
  whitePixel,
  cyanPixel,
  tilePixel,
  type TextTemplate,
} from "./panel-mask";

export type PanelPiecesResult = {
  pieces: RecognizedPieceSlot[];
  reasons: string[];
};
/** Same frame only; no retained image, guessed names, slot reuse or empty-by-absence. */
export function recognizePanelPieces(
  frame: CapturedFrame,
  board: PixelRegion,
  catalog: readonly Piece[],
  templates: readonly TextTemplate[] = [],
): PanelPiecesResult {
  const cards = panelCards(frame, board);
  return recognizePieceCards(frame, cards ?? [], catalog, templates);
}

/** Explicit card regions are also useful for independent partial-panel fixtures. */
export function recognizePieceCards(
  frame: CapturedFrame,
  cards: readonly PixelRegion[],
  catalog: readonly Piece[],
  templates: readonly TextTemplate[] = [],
): PanelPiecesResult {
  const pieces: RecognizedPieceSlot[] = Array.from(
    { length: 3 },
    (_, slot) => ({
      slot: slot as 0 | 1 | 2,
      pieceId: null,
      empty: null,
      status: "unknown",
    }),
  );
  const reasons = pieces.map(() => "패널을 찾지 못했습니다.");
  if (
    !validPanelFrame(frame) ||
    cards.length !== 3 ||
    catalog.length < 1 ||
    new Set(catalog.map((p) => p.id)).size !== catalog.length
  )
    return { pieces, reasons };
  const variants = catalog.flatMap((piece) => {
    const value = getUniqueVariants(piece.shape);
    return value.ok
      ? value.variants.map(({ shape }) => ({ id: piece.id, shape }))
      : [];
  });
  cards.forEach((card, slot) => {
    if (!contained(frame, card)) {
      reasons[slot] = "패널이 화면 밖으로 잘렸습니다.";
      return;
    }
    const white = readMask(
      frame,
      relative(card, 0.04, 0.08, 0.92, 0.84),
      whitePixel,
    );
    if (!white) {
      reasons[slot] = "패널 픽셀을 읽지 못했습니다.";
      return;
    }
    const fraction = white.data.reduce((s, v) => s + v, 0) / white.data.length;
    white.data.fill(0);
    if (fraction < 0.35) {
      const bg = readMask(
        frame,
        relative(card, 0.05, 0.08, 0.9, 0.84),
        cyanPixel,
      );
      const word = readMask(
        frame,
        relative(card, 0.14, 0.3, 0.72, 0.4),
        // Video capture chroma conversion lowers edge brightness of tiny glyphs.
        // The cyan-card guard below excludes other background layouts.
        (r, g, b) => Math.min(r, g, b) >= C.glyphMinChannel,
      );
      try {
        if (
          bg &&
          word &&
          bg.data.reduce((s, v) => s + v, 0) / bg.data.length > 0.7
        ) {
          const text = matchText(
            word,
            templates.filter((t) => t.value === "사용 완료"),
            "word",
          );
          if (text.status === "recognized") {
            pieces[slot] = {
              slot: slot as 0 | 1 | 2,
              pieceId: null,
              empty: true,
              status: "recognized",
            };
            reasons[slot] = "사용 완료 표시를 읽었습니다.";
            return;
          }
        }
        reasons[slot] = "조각 또는 사용 완료 표시를 확인하지 못했습니다.";
      } finally {
        bg?.data.fill(0);
        word?.data.fill(0);
      }
      return;
    }
    const mask = readMask(
      frame,
      relative(card, 0.055, 0.08, 0.405, 0.84),
      tilePixel,
    );
    if (!mask) return;
    try {
      const box = maskBounds(mask);
      if (!box) {
        reasons[slot] = "조각 도형이 보이지 않습니다.";
        return;
      }
      if (
        box.x <= 0 ||
        box.y <= 0 ||
        box.x + box.width >= mask.width ||
        box.y + box.height >= mask.height
      ) {
        reasons[slot] = "조각 도형이 잘렸습니다.";
        return;
      }
      const candidates = new Set<string>();
      for (const { id, shape } of variants) {
        // Bounds cover colored tile interiors; up to two edge pixels are absent
        // after fractional scaling/rasterization, especially on a one-cell axis.
        const pitchX = (box.width + 2) / shape[0].length,
          pitchY = (box.height + 2) / shape.length;
        if (
          pitchX / card.width < C.miniaturePitchMin ||
          pitchX / card.width > C.miniaturePitchMax ||
          pitchY / card.width < C.miniaturePitchMin ||
          pitchY / card.width > C.miniaturePitchMax ||
          Math.abs(pitchX / pitchY - 1) > C.pitchMismatch
        )
          continue;
        const matches = shape.every((line, r) =>
          line.every((filled, c) => {
            let hit = 0;
            for (let y = 0; y < 5; y++)
              for (let x = 0; x < 5; x++) {
                const ix = Math.floor(box.x - 1 + (c + 0.3 + x * 0.1) * pitchX),
                  iy = Math.floor(box.y - 1 + (r + 0.3 + y * 0.1) * pitchY);
                hit += mask.data[iy * mask.width + ix];
              }
            return filled
              ? hit / 25 >= C.tileFraction
              : hit / 25 <= C.emptyFraction;
          }),
        );
        if (matches) candidates.add(id);
      }
      if (candidates.size === 1) {
        pieces[slot] = {
          slot: slot as 0 | 1 | 2,
          pieceId: [...candidates][0],
          empty: false,
          status: "recognized",
        };
        reasons[slot] = "도형과 타일 크기가 일치합니다.";
      } else {
        pieces[slot].status = "uncertain";
        reasons[slot] = candidates.size
          ? "도형 후보가 여러 개입니다."
          : "등록된 도형과 일치하지 않습니다.";
      }
    } finally {
      mask.data.fill(0);
    }
  });
  return { pieces, reasons };
}
