import type { CapturedFrame } from "../capture/capture";
import type { PixelRegion } from "./calibration";
import { contained, PANEL_PROFILE } from "./panel-profile";

export type BinaryMask = { width: number; height: number; data: Uint8Array };
export type TextTemplate = { value: string; mask: BinaryMask };
export type PixelTest = (r: number, g: number, b: number) => boolean;
export const whitePixel: PixelTest = (r, g, b) => Math.min(r, g, b) >= 210;
export const cyanPixel: PixelTest = (r, g, b) => r < 90 && g >= 120 && b >= 140;
export const tilePixel: PixelTest = (r, g, b) =>
  (r > 160 && b > 110 && r - g > 28 && b - g > 18) ||
  (b > 130 && g > 70 && g - r > 30 && b - r > 45) ||
  (g > 100 && g - r > 30 && g - b > 30) ||
  (r > 150 && g > 100 && r - b > 45 && g - b > 30);

export function readMask(
  frame: CapturedFrame,
  region: PixelRegion,
  test: PixelTest,
): BinaryMask | null {
  if (!contained(frame, region)) return null;
  const x0 = Math.ceil(region.x),
    y0 = Math.ceil(region.y),
    width = Math.floor(region.x + region.width) - x0,
    height = Math.floor(region.y + region.height) - y0;
  if (width < 1 || height < 1) return null;
  const data = new Uint8Array(width * height);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const i = ((y0 + y) * frame.width + x0 + x) * 4;
      if (frame.pixels[i + 3] !== 255) {
        data.fill(0);
        return null;
      }
      data[y * width + x] = Number(
        test(frame.pixels[i], frame.pixels[i + 1], frame.pixels[i + 2]),
      );
    }
  return { width, height, data };
}
export function maskBounds(mask: BinaryMask): PixelRegion | null {
  let x0 = mask.width,
    y0 = mask.height,
    x1 = -1,
    y1 = -1;
  for (let y = 0; y < mask.height; y++)
    for (let x = 0; x < mask.width; x++)
      if (mask.data[y * mask.width + x]) {
        x0 = Math.min(x0, x);
        y0 = Math.min(y0, y);
        x1 = Math.max(x1, x);
        y1 = Math.max(y1, y);
      }
  return x1 < 0
    ? null
    : { x: x0, y: y0, width: x1 - x0 + 1, height: y1 - y0 + 1 };
}
function normalized(
  mask: BinaryMask,
  width: number,
  height: number,
): Uint8Array | null {
  const box = maskBounds(mask);
  if (!box) return null;
  return Uint8Array.from({ length: width * height }, (_, i) => {
    const x = Math.min(
      mask.width - 1,
      Math.floor(box.x + (((i % width) + 0.5) / width) * box.width),
    );
    const y = Math.min(
      mask.height - 1,
      Math.floor(box.y + ((Math.floor(i / width) + 0.5) / height) * box.height),
    );
    return mask.data[y * mask.width + x];
  });
}
/** Symmetric foreground coverage, plus aspect and uniqueness guards. No probabilities. */
export function matchText(
  mask: BinaryMask,
  templates: readonly TextTemplate[],
  kind: "number" | "word",
): { value: string | null; status: "recognized" | "uncertain" | "unknown" } {
  const box = maskBounds(mask),
    w = kind === "number" ? 20 : 64,
    h = kind === "number" ? 28 : 16;
  if (!box || box.width < 2 || box.height < 4)
    return { value: null, status: "unknown" };
  const input = normalized(mask, w, h)!;
  const coverage = (a: Uint8Array, b: Uint8Array) => {
    let hit = 0,
      total = 0;
    for (let i = 0; i < a.length; i++)
      if (a[i]) {
        total++;
        const x = i % w,
          y = Math.floor(i / w);
        let found = false;
        for (let dy = -1; dy <= 1; dy++)
          for (let dx = -1; dx <= 1; dx++)
            if (
              x + dx >= 0 &&
              x + dx < w &&
              y + dy >= 0 &&
              y + dy < h &&
              b[(y + dy) * w + x + dx]
            )
              found = true;
        hit += Number(found);
      }
    return total ? hit / total : 0;
  };
  const scores = new Map<string, number>();
  try {
    for (const template of templates) {
      const bounds = maskBounds(template.mask);
      if (
        !bounds ||
        Math.abs(box.width / box.height / (bounds.width / bounds.height) - 1) >
          0.3
      )
        continue;
      const expected = normalized(template.mask, w, h)!;
      const score = Math.min(
        coverage(input, expected),
        coverage(expected, input),
      );
      expected.fill(0);
      scores.set(
        template.value,
        Math.max(scores.get(template.value) ?? 0, score),
      );
    }
    const ranked = [...scores].sort((a, b) => b[1] - a[1]);
    if (!ranked.length) return { value: null, status: "unknown" };
    const [best, score] = ranked[0];
    return score >= PANEL_PROFILE.textMatch &&
      score - (ranked[1]?.[1] ?? 0) >= PANEL_PROFILE.textGap
      ? { value: best, status: "recognized" }
      : { value: null, status: "uncertain" };
  } finally {
    input.fill(0);
  }
}
