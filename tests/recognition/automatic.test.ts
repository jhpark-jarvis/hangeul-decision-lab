import { describe, expect, it } from "vitest";
import { recognizeAutomaticBoard } from "../../src/features/recognition/automatic";
import {
  validateRecognitionResult,
  createReviewState,
  validateReviewedState,
} from "../../src/features/recognition/review";
import { automaticFixture, referenceRows } from "./automatic-fixture";
describe("automatic game board boundary", () => {
  it.each([18, 26, 39, 52])(
    "finds a translated %ipx grid and reads all tiles without samples",
    (pitch) => {
      const frame = automaticFixture(pitch, 37, 51),
        before = frame.pixels.slice();
      const found = recognizeAutomaticBoard(frame);
      expect(found.ok).toBe(true);
      if (!found.ok) return;
      expect(found.region.x).toBeGreaterThanOrEqual(
        37 - Math.ceil(pitch * 0.08),
      );
      expect(found.region.x).toBeLessThanOrEqual(37 + Math.ceil(pitch * 0.08));
      expect(found.region.y).toBeGreaterThanOrEqual(
        51 - Math.ceil(pitch * 0.08),
      );
      expect(found.region.y).toBeLessThanOrEqual(51 + Math.ceil(pitch * 0.08));
      expect(
        found.result.board.map((row) =>
          row
            .map((c) => (c.occupied ? "1" : c.occupied === false ? "0" : "?"))
            .join(""),
        ),
      ).toEqual(referenceRows);
      expect(frame.pixels.every((v, i) => v === before[i])).toBe(true);
      expect(validateRecognitionResult(found.result)).toEqual([]);
      expect(
        validateReviewedState(createReviewState(found.result), false).length,
      ).toBeGreaterThan(0);
      expect(found.result.pieces.every((p) => p.status === "unknown")).toBe(
        true,
      );
      expect(found.result.hiddenItemsStatus).toBe("unknown");
      expect(found.result.source).not.toHaveProperty("pixels");
      expect(found.result.board.flat().every((c) => !("confidence" in c))).toBe(
        true,
      );
    },
  );
  it("rejects uniform cyan without a lattice and recovers on the next frame", () => {
    const frame = automaticFixture();
    for (let i = 0; i < frame.pixels.length; i += 4)
      frame.pixels.set([50, 170, 180, 255], i);
    expect(recognizeAutomaticBoard(frame)).toEqual({
      ok: false,
      reason: "NOT_FOUND",
    });
    expect(recognizeAutomaticBoard(automaticFixture()).ok).toBe(true);
  });
  it("rejects malformed frame and tiny grids", () => {
    const frame = automaticFixture();
    expect(
      recognizeAutomaticBoard({ ...frame, pixels: new Uint8ClampedArray(1) }),
    ).toEqual({ ok: false, reason: "INVALID_FRAME" });
    expect(recognizeAutomaticBoard({ ...frame, timestamp: NaN })).toEqual({
      ok: false,
      reason: "INVALID_FRAME",
    });
    expect(recognizeAutomaticBoard(automaticFixture(6))).toEqual({
      ok: false,
      reason: "NOT_FOUND",
    });
  });
  it("leaves opaque foreign occlusion and transparent tiles unresolved", () => {
    const frame = automaticFixture(),
      p = 26;
    for (const [row, col, alpha] of [
      [0, 2, 0],
      [8, 8, 255],
    ]) {
      for (let y = 30 + row * p + 4; y < 30 + (row + 1) * p - 3; y++)
        for (let x = 20 + col * p + 4; x < 20 + (col + 1) * p - 3; x++)
          frame.pixels.set([255, 255, 255, alpha], (y * frame.width + x) * 4);
    }
    const found = recognizeAutomaticBoard(frame);
    expect(found.ok).toBe(true);
    if (found.ok) {
      expect(found.result.board[0][2].occupied).toBeNull();
      expect(found.result.board[8][8].occupied).toBeNull();
    }
  });
  it("rejects two matching boards instead of selecting one", () => {
    const first = automaticFixture(),
      width = first.width * 2 + 40,
      pixels = new Uint8ClampedArray(width * first.height * 4);
    for (let y = 0; y < first.height; y++)
      for (const offset of [0, first.width + 40])
        pixels.set(
          first.pixels.subarray(y * first.width * 4, (y + 1) * first.width * 4),
          (y * width + offset) * 4,
        );
    expect(recognizeAutomaticBoard({ ...first, width, pixels })).toEqual({
      ok: false,
      reason: "AMBIGUOUS",
    });
  });
});
