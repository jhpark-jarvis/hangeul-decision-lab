import type { CapturedFrame } from "../../src/features/capture/capture";

export const referenceRows = [
  "0010000000",
  "0101000000",
  "0010000000",
  "1000000000",
  "1100000111",
  "1101000011",
  "1101111111",
  "1001101000",
  "0001100000",
  "0001000000",
  "0001000000",
  "0111110000",
  "0010101000",
  "0001111110",
  "0000010100",
  "0000001000",
];
/** Synthetic gradient/tile fixture; no actual screenshot or sampled RGB. */
export function automaticFixture(
  pitch = 26,
  x = 20,
  y = 30,
  rows: readonly string[] = referenceRows,
): CapturedFrame {
  const width = Math.ceil(x + pitch * 10 + 20),
    height = Math.ceil(y + pitch * 16 + 20);
  const pixels = new Uint8ClampedArray(width * height * 4);
  for (let py = 0; py < height; py++)
    for (let px = 0; px < width; px++) {
      const index = (py * width + px) * 4;
      const col = Math.floor((px - x) / pitch),
        row = Math.floor((py - y) / pitch);
      let color = [30, 30, 30];
      if (col >= 0 && col < 10 && row >= 0 && row < 16) {
        const fx = (px - x) / pitch - col,
          fy = (py - y) / pitch - row;
        color = [50, 165 + row * 0.7, 185 + row * 0.2];
        if (fx < 0.06 || fy < 0.06) color = color.map((c) => c - 12);
        if (
          rows[row][col] === "1" &&
          fx > 0.08 &&
          fx < 0.92 &&
          fy > 0.08 &&
          fy < 0.92
        ) {
          const base = [
            [60, 160, 220],
            [130, 190, 40],
            [210, 100, 180],
            [220, 150, 30],
          ][(row + col) % 4];
          const highlight = 100 * (1 - (fx + fy) / 2);
          color = base.map((c) => Math.min(255, c + highlight));
        }
      }
      pixels.set([...color.map(Math.round), 255], index);
    }
  return { width, height, timestamp: 1, pixels };
}

/** Connected cyan game surround and high-contrast panel, independent of photos. */
export function surroundedBoardFixture(
  pitch = 26,
  empty = false,
): CapturedFrame {
  const x = 37,
    y = 51;
  const board = automaticFixture(
    pitch,
    x,
    y,
    empty ? Array(16).fill("0000000000") : referenceRows,
  );
  const width = x + 16 * pitch + 23,
    height = y + 17 * pitch + 29;
  const pixels = new Uint8ClampedArray(width * height * 4);
  for (let py = 0; py < height; py++)
    for (let px = 0; px < width; px++) {
      let color = [30, 30, 30, 255];
      if (px >= x && px < x + 16 * pitch && py >= y && py < y + 17 * pitch)
        color = [50, 175, 190, 255];
      if (px >= x && px < x + 10 * pitch && py >= y && py < y + 16 * pitch) {
        const i = (py * board.width + px) * 4;
        color = Array.from(board.pixels.subarray(i, i + 4));
      }
      // Strong panel seams persist through the whole component and used cards.
      if (
        px >= x + 10 * pitch &&
        px < x + 16 * pitch &&
        py >= y &&
        py < y + 17 * pitch
      ) {
        const fx = (px - x) % pitch;
        if (fx < 2) color = [20, 95, 160, 255];
        if (
          py < y + 9 * pitch &&
          px > x + 10.5 * pitch &&
          px < x + 15.5 * pitch &&
          (py - y) % (3 * pitch) < 2.5 * pitch
        )
          color = [245, 250, 252, 255];
      }
      pixels.set(color, (py * width + px) * 4);
    }
  board.pixels.fill(0);
  return { width, height, pixels, timestamp: 1 };
}
