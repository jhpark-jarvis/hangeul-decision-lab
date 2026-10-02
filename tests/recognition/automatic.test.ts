import { describe, expect, it } from "vitest";
import { recognizeAutomaticBoard } from "../../src/features/recognition/automatic";
import {
  validateRecognitionResult,
  createReviewState,
  validateReviewedState,
} from "../../src/features/recognition/review";
import {
  automaticFixture,
  referenceRows,
  surroundedBoardFixture,
} from "./automatic-fixture";
describe("automatic game board boundary", () => {
  it.each([18, 26, 39, 52])(
    "finds an empty board inside a 1920x1080 sharing frame, pitch=%i",
    (pitch) => {
      const original = surroundedBoardFixture(pitch, true);
      const width = 1920,
        height = 1080,
        dx = 400,
        dy = 100;
      const pixels = new Uint8ClampedArray(width * height * 4);
      for (let i = 0; i < pixels.length; i += 4)
        pixels.set([30, 30, 30, 255], i);
      for (let y = 0; y < original.height; y++)
        pixels.set(
          original.pixels.subarray(
            y * original.width * 4,
            (y + 1) * original.width * 4,
          ),
          ((y + dy) * width + dx) * 4,
        );
      const found = recognizeAutomaticBoard({
        width,
        height,
        timestamp: 1,
        pixels,
      });
      expect(found.ok).toBe(true);
      if (!found.ok) return;
      const tolerance = Math.ceil(pitch * 0.08);
      expect(Math.abs(found.region.x - (37 + dx))).toBeLessThanOrEqual(
        tolerance,
      );
      expect(Math.abs(found.region.y - (51 + dy))).toBeLessThanOrEqual(
        tolerance,
      );
      expect(found.region.width).toBe(pitch * 10);
      expect(found.result.board.flat().every((c) => c.occupied === false)).toBe(
        true,
      );
    },
  );
  it.each(
    [18, 26, 39, 52].flatMap((pitch) =>
      [false, true].map((empty) => ({ pitch, empty })),
    ),
  )(
    "excludes the connected right panel at $pitch pixels, empty=$empty",
    ({ pitch, empty }) => {
      const frame = surroundedBoardFixture(pitch, empty);
      const found = recognizeAutomaticBoard(frame);
      expect(found.ok).toBe(true);
      if (!found.ok) return;
      expect(found.region.x).toBeCloseTo(37, -1);
      expect(found.region.y).toBeCloseTo(51, -1);
      expect(found.region.width).toBeCloseTo(pitch * 10, -1);
      expect(found.region.height).toBeCloseTo(pitch * 16, -1);
      expect(
        found.result.board.map((row) =>
          row
            .map((c) => (c.occupied === null ? "?" : Number(c.occupied)))
            .join(""),
        ),
      ).toEqual(empty ? Array(16).fill("0000000000") : referenceRows);
    },
  );
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
  it("rejects panel seams when there is no board lattice", () => {
    const frame = surroundedBoardFixture(26, true);
    for (let y = 51; y < 51 + 16 * 26; y++)
      for (let x = 37; x < 37 + 10 * 26; x++)
        frame.pixels.set([50, 175, 190, 255], (y * frame.width + x) * 4);
    expect(recognizeAutomaticBoard(frame)).toEqual({
      ok: false,
      reason: "NOT_FOUND",
    });
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
