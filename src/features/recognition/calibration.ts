import type { CapturedFrame } from "../capture/capture";
import type { BoardRecognitionConfig, RGB } from "./board";

export type PixelPoint = { x: number; y: number };
export type PixelRegion = BoardRecognitionConfig["region"];
export const MAX_PREVIEW_EDGE = 1024;
/** Generic sampling defaults, not a game palette or an accuracy claim. */
export const CALIBRATION_DEFAULTS = {
  maxDistance: 12,
  minDistanceGap: 20,
  minMatchedFraction: 0.8,
  sampleFraction: 0.6,
  samplesPerAxis: 3,
};

export function pointerToPixel(
  frame: Pick<CapturedFrame, "width" | "height">,
  bounds: { left: number; top: number; width: number; height: number },
  pointer: { clientX: number; clientY: number },
): PixelPoint | null {
  const x = pointer.clientX - bounds.left;
  const y = pointer.clientY - bounds.top;
  if (
    ![
      frame.width,
      frame.height,
      bounds.left,
      bounds.top,
      bounds.width,
      bounds.height,
      x,
      y,
    ].every(Number.isFinite) ||
    !Number.isInteger(frame.width) ||
    !Number.isInteger(frame.height) ||
    frame.width <= 0 ||
    frame.height <= 0 ||
    bounds.width <= 0 ||
    bounds.height <= 0 ||
    x < 0 ||
    y < 0 ||
    x >= bounds.width ||
    y >= bounds.height
  )
    return null;
  return {
    x: Math.floor((x / bounds.width) * frame.width),
    y: Math.floor((y / bounds.height) * frame.height),
  };
}

export function regionFromCorners(a: PixelPoint, b: PixelPoint): PixelRegion {
  return {
    x: Math.min(a.x, b.x),
    y: Math.min(a.y, b.y),
    width: Math.abs(b.x - a.x) + 1,
    height: Math.abs(b.y - a.y) + 1,
  };
}

export function sampleFrameColor(frame: CapturedFrame, point: PixelPoint): RGB {
  if (
    !Number.isInteger(point.x) ||
    !Number.isInteger(point.y) ||
    point.x < 0 ||
    point.y < 0 ||
    point.x >= frame.width ||
    point.y >= frame.height
  )
    throw new Error("표본 좌표가 프레임 범위 밖입니다.");
  const i = (point.y * frame.width + point.x) * 4;
  if (frame.pixels[i + 3] !== 255)
    throw new Error("불투명한 픽셀에서 표본을 선택하세요.");
  return [frame.pixels[i], frame.pixels[i + 1], frame.pixels[i + 2]];
}

/** Creates a bounded display-only buffer. Recognition still reads original pixels. */
export function previewFrame(frame: CapturedFrame): CapturedFrame {
  const scale = Math.min(
    1,
    MAX_PREVIEW_EDGE / frame.width,
    MAX_PREVIEW_EDGE / frame.height,
  );
  const width = Math.max(1, Math.floor(frame.width * scale));
  const height = Math.max(1, Math.floor(frame.height * scale));
  const pixels = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const sourceX = Math.floor(((x + 0.5) / width) * frame.width);
      const sourceY = Math.floor(((y + 0.5) / height) * frame.height);
      const i = (sourceY * frame.width + sourceX) * 4;
      pixels.set(frame.pixels.subarray(i, i + 4), (y * width + x) * 4);
    }
  }
  return { width, height, pixels, timestamp: frame.timestamp };
}

/** Called only by the owner after all reads; not by the pure recognizer. */
export function releaseFrame(frame: CapturedFrame | null): void {
  frame?.pixels.fill(0);
}
