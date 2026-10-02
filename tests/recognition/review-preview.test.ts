import { describe, expect, it } from "vitest";
import {
  cropReviewPreview,
  REVIEW_PREVIEW_MAX_WIDTH,
  REVIEW_PREVIEW_MAX_HEIGHT,
} from "../../src/features/recognition/review-preview";
import { releaseFrame } from "../../src/features/recognition/calibration";

describe("bounded board-only review preview", () => {
  it("copies only the requested crop with original coordinates and alpha", () => {
    const pixels = new Uint8ClampedArray(8 * 10 * 4);
    for (let y = 0; y < 10; y++)
      for (let x = 0; x < 8; x++)
        pixels.set([x, y, 20, x === 3 ? 0 : 255], (y * 8 + x) * 4);
    const before = pixels.slice();
    const crop = cropReviewPreview(
      { width: 8, height: 10, pixels, timestamp: 5 },
      { x: 2, y: 3, width: 4, height: 6 },
    );
    expect([crop.width, crop.height, crop.timestamp]).toEqual([4, 6, 5]);
    expect(Array.from(crop.pixels.slice(0, 8))).toEqual([
      2, 3, 20, 255, 3, 3, 20, 0,
    ]);
    expect(Array.from(crop.pixels.slice(-4))).toEqual([5, 8, 20, 255]);
    releaseFrame(crop);
    expect(crop.pixels.every((v) => v === 0)).toBe(true);
    expect(pixels).toEqual(before);
  });
  it("bounds display allocation while preserving the source region ratio", () => {
    const frame = {
      width: 1000,
      height: 1700,
      timestamp: 1,
      pixels: new Uint8ClampedArray(1000 * 1700 * 4),
    };
    const crop = cropReviewPreview(frame, {
      x: 100,
      y: 100,
      width: 800,
      height: 1280,
    });
    expect([crop.width, crop.height]).toEqual([
      REVIEW_PREVIEW_MAX_WIDTH,
      REVIEW_PREVIEW_MAX_HEIGHT,
    ]);
    expect(crop.pixels.length).toBe(520 * 832 * 4);
  });
  it("rejects malformed and out-of-frame crops before allocation", () => {
    const frame = {
      width: 8,
      height: 10,
      timestamp: 1,
      pixels: new Uint8ClampedArray(320),
    };
    for (const region of [
      { x: -1, y: 0, width: 4, height: 6 },
      { x: 2, y: 3, width: 9, height: 6 },
      { x: 2, y: 3, width: 4, height: NaN },
      { x: 2, y: 3, width: 0, height: 6 },
    ])
      expect(() => cropReviewPreview(frame, region)).toThrow("검토 보드 영역");
    expect(() =>
      cropReviewPreview(
        { ...frame, pixels: new Uint8ClampedArray(1) },
        { x: 0, y: 0, width: 4, height: 6 },
      ),
    ).toThrow();
  });
});
