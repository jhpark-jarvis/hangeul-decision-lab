import type { CapturedFrame } from "../../src/features/capture/capture";
import type { PixelRegion } from "../../src/features/recognition/calibration";
import type { BinaryMask } from "../../src/features/recognition/panel-mask";

// Independent synthetic UI renderer; no sampled game pixels.
export const syntheticUsedRows = [
  "1001010011101000110",
  "0111010010101110101",
  "1001010010101000110",
  "1001111011101110101",
];
export function rowMask(rows: readonly string[]): BinaryMask {
  return {
    width: rows[0].length,
    height: rows.length,
    data: Uint8Array.from(rows.join(""), Number),
  };
}
export function panelFixture(
  scale = 1,
  shift = 0,
): { frame: CapturedFrame; board: PixelRegion; cards: PixelRegion[] } {
  const p = 26 * scale,
    board = {
      x: 20 * scale + shift,
      y: 30 * scale + shift,
      width: p * 10,
      height: p * 16,
    };
  const width = Math.ceil(board.x + p * 16),
    height = Math.ceil(board.y + p * 17);
  const frame = {
    width,
    height,
    timestamp: 123,
    pixels: new Uint8ClampedArray(width * height * 4),
  };
  paint(frame, { x: 0, y: 0, width, height }, [25, 35, 45]);
  const cards = Array.from({ length: 3 }, (_, i) => ({
    x: board.x + p * 10.5,
    y: board.y + p * (0.9 + 2.9 * i),
    width: p * 4.7,
    height: p * 2.7,
  }));
  cards.forEach((card) => paint(frame, card, [244, 250, 252]));
  return { frame, board, cards };
}
export function paint(
  frame: CapturedFrame,
  region: PixelRegion,
  color: readonly number[],
) {
  for (
    let y = Math.ceil(region.y);
    y < Math.floor(region.y + region.height);
    y++
  )
    for (
      let x = Math.ceil(region.x);
      x < Math.floor(region.x + region.width);
      x++
    ) {
      if (x >= 0 && y >= 0 && x < frame.width && y < frame.height)
        frame.pixels.set([...color, 255], (y * frame.width + x) * 4);
    }
}
export function drawPiece(
  frame: CapturedFrame,
  card: PixelRegion,
  rows: readonly string[],
  color = [229, 69, 175],
) {
  const pitch = card.width / 15,
    left = card.x + card.width * 0.25 - (pitch * rows[0].length) / 2,
    top = card.y + card.height / 2 - (pitch * rows.length) / 2;
  for (const [r, row] of rows.entries())
    for (const [c, cell] of [...row].entries())
      if (cell === "1")
        paint(
          frame,
          {
            x: left + c * pitch,
            y: top + r * pitch,
            width: pitch - 1,
            height: pitch - 1,
          },
          color,
        );
}
export function drawUsed(frame: CapturedFrame, card: PixelRegion) {
  paint(frame, card, [0, 184, 215]);
  drawGlyph(frame, card, syntheticUsedRows, [255, 255, 255], 0.035);
}
export function drawGlyph(
  frame: CapturedFrame,
  region: PixelRegion,
  rows: readonly string[],
  color: readonly number[],
  pixelFraction: number,
) {
  const p = region.width * pixelFraction,
    left = region.x + (region.width - rows[0].length * p) / 2,
    top = region.y + (region.height - rows.length * p) / 2;
  for (const [r, row] of rows.entries())
    for (const [c, cell] of [...row].entries())
      if (cell === "1")
        paint(
          frame,
          { x: left + c * p, y: top + r * p, width: p, height: p },
          color,
        );
}
