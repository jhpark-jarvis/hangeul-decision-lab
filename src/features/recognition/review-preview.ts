import type { CapturedFrame } from "../capture/capture";
import type { PixelRegion } from "./calibration";

export const REVIEW_PREVIEW_MAX_WIDTH = 520;
export const REVIEW_PREVIEW_MAX_HEIGHT = 832;

/** Bounded board-only display copy; never changes recognition or input pixels. */
export function cropReviewPreview(
  frame: CapturedFrame,
  region: PixelRegion,
): CapturedFrame {
  if (
    ![
      frame.width,
      frame.height,
      region.x,
      region.y,
      region.width,
      region.height,
    ].every(Number.isFinite) ||
    !Number.isInteger(frame.width) ||
    !Number.isInteger(frame.height) ||
    frame.width <= 0 ||
    frame.height <= 0 ||
    frame.pixels.length !== frame.width * frame.height * 4 ||
    region.x < 0 ||
    region.y < 0 ||
    region.width <= 0 ||
    region.height <= 0 ||
    region.x + region.width > frame.width ||
    region.y + region.height > frame.height
  )
    throw new Error("검토 보드 영역이 유효하지 않습니다.");
  const scale = Math.min(
    1,
    REVIEW_PREVIEW_MAX_WIDTH / region.width,
    REVIEW_PREVIEW_MAX_HEIGHT / region.height,
  );
  const width = Math.max(1, Math.floor(region.width * scale));
  const height = Math.max(1, Math.floor(region.height * scale));
  const pixels = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const sx = Math.floor(region.x + ((x + 0.5) / width) * region.width);
      const sy = Math.floor(region.y + ((y + 0.5) / height) * region.height);
      const i = (sy * frame.width + sx) * 4;
      pixels.set(frame.pixels.subarray(i, i + 4), (y * width + x) * 4);
    }
  return { width, height, pixels, timestamp: frame.timestamp };
}
