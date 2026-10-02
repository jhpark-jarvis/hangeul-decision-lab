import { describe, expect, it } from "vitest";
import {
  pointerToPixel,
  previewFrame,
  regionFromCorners,
  releaseFrame,
  sampleFrameColor,
} from "../../src/features/recognition/calibration";
import type { CapturedFrame } from "../../src/features/capture/capture";

function frame(width = 4, height = 2): CapturedFrame {
  const pixels = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++)
      pixels.set([x % 256, y % 256, 30, 255], (y * width + x) * 4);
  return { width, height, pixels, timestamp: 123 };
}
const bounds = { left: 100, top: 200, width: 500, height: 250 };
describe("owned calibration frame and coordinates", () => {
  it("maps CSS scaling and page offsets to original integer pixels", () => {
    expect(
      pointerToPixel({ width: 2000, height: 1000 }, bounds, {
        clientX: 350,
        clientY: 325,
      }),
    ).toEqual({ x: 1000, y: 500 });
    expect(
      pointerToPixel({ width: 2000, height: 1000 }, bounds, {
        clientX: 599.99,
        clientY: 449.99,
      }),
    ).toEqual({ x: 1999, y: 999 });
  });
  it.each([
    { clientX: 99, clientY: 300 },
    { clientX: 600, clientY: 300 },
    { clientX: 300, clientY: 199 },
    { clientX: 300, clientY: 450 },
    { clientX: NaN, clientY: 300 },
  ])(
    "rejects outside or nonfinite pointer %j rather than silently clamping",
    (pointer) => {
      expect(
        pointerToPixel({ width: 2000, height: 1000 }, bounds, pointer),
      ).toBeNull();
    },
  );
  it("rejects zero display/frame dimensions", () => {
    expect(
      pointerToPixel({ width: 0, height: 1000 }, bounds, {
        clientX: 350,
        clientY: 325,
      }),
    ).toBeNull();
    expect(
      pointerToPixel(
        { width: 2000, height: 1000 },
        { ...bounds, width: 0 },
        { clientX: 350, clientY: 325 },
      ),
    ).toBeNull();
  });
  it("includes the selected end pixel and handles either corner order", () => {
    const a = { x: 20, y: 20 };
    const b = { x: 219, y: 339 };
    expect(regionFromCorners(a, b)).toEqual({
      x: 20,
      y: 20,
      width: 200,
      height: 320,
    });
    expect(regionFromCorners(b, a)).toEqual(regionFromCorners(a, b));
    expect(regionFromCorners(a, a)).toEqual({
      x: 20,
      y: 20,
      width: 1,
      height: 1,
    });
  });
  it("reads RGB from original pixels without mutating or retaining the frame", () => {
    const input = frame();
    const copy = input.pixels.slice();
    const color = sampleFrameColor(input, { x: 2, y: 1 });
    expect(color).toEqual([2, 1, 30]);
    expect(input.pixels).toEqual(copy);
    releaseFrame(input);
    expect(color).toEqual([2, 1, 30]);
  });
  it.each([
    { x: -1, y: 0 },
    { x: 4, y: 0 },
    { x: 0, y: 2 },
    { x: 1.5, y: 0 },
    { x: 0, y: NaN },
  ])("rejects bad sample point %j", (point) => {
    expect(() => sampleFrameColor(frame(), point)).toThrow("표본 좌표");
  });
  it("rejects transparent pixels as calibration evidence", () => {
    const input = frame();
    input.pixels[3] = 254;
    expect(() => sampleFrameColor(input, { x: 0, y: 0 })).toThrow("불투명");
  });
  it("keeps small previews independent and never enlarges the source", () => {
    const input = frame();
    const preview = previewFrame(input);
    expect(preview).toEqual(input);
    expect(preview.pixels).not.toBe(input.pixels);
    releaseFrame(preview);
    expect(input.pixels[3]).toBe(255);
  });
  it.each([
    [2048, 2, 1024, 1],
    [2, 2048, 1, 1024],
  ])(
    "bounds preview dimensions for %i×%i",
    (width, height, outputWidth, outputHeight) => {
      const input = frame(width, height);
      const preview = previewFrame(input);
      expect([preview.width, preview.height]).toEqual([
        outputWidth,
        outputHeight,
      ]);
      expect([...preview.pixels.slice(0, 4)]).toEqual([1, 1, 30, 255]);
      expect(preview.timestamp).toBe(input.timestamp);
      expect(input.pixels[3]).toBe(255);
    },
  );
  it("zeroes the owned payload idempotently, including lingering array references", () => {
    const input = frame();
    const lingering = input.pixels;
    releaseFrame(input);
    releaseFrame(input);
    releaseFrame(null);
    expect(lingering.every((value) => value === 0)).toBe(true);
  });
});
