import { describe, it, expect } from "vitest";
import { createHash } from "node:crypto";
import { automaticFixture, referenceRows } from "./automatic-fixture";
import { paint, drawPiece, drawGlyph, rowMask } from "./panel-fixture";
import {
  suggestedRegions,
  regionsFit,
  pieceCardRegions,
  countRegions,
  recognizeConfiguredGame,
} from "../../src/features/recognition/regions";
import { recognizeBoardRegion } from "../../src/features/recognition/automatic";
import { panelCards } from "../../src/features/recognition/panel-profile";
import { abilityRegions } from "../../src/features/recognition/panel-abilities";
const zero = ["01110", "11011", "11011", "11011", "11011", "11011", "01110"];
describe("session region contract", () => {
  it("suggests the approved automatic panel profile with identical card/count coordinates", () => {
    const frame = automaticFixture(),
      board = { x: 20, y: 30, width: 260, height: 416 };
    const regions = suggestedRegions(frame, board);
    pieceCardRegions(regions.pieces).forEach((r, i) =>
      Object.keys(r).forEach((k) =>
        expect(r[k as keyof typeof r]).toBeCloseTo(
          panelCards(frame, board)![i][k as keyof typeof r],
          8,
        ),
      ),
    );
    const counts = countRegions(regions.abilities),
      previous = abilityRegions(board);
    for (const key of ["singleCell", "reroll", "total"] as const)
      for (const k of ["x", "y", "width", "height"] as const)
        expect(counts[key][k]).toBeCloseTo(previous[key][k], 8);
  });
  it("reads independently placed panel regions and preserves frame bytes", () => {
    const small = automaticFixture(),
      frame = {
        ...small,
        width: 900,
        height: 800,
        pixels: new Uint8ClampedArray(900 * 800 * 4),
      };
    paint(frame, { x: 0, y: 0, width: 900, height: 800 }, [30, 30, 30]);
    for (let y = 0; y < small.height; y++)
      frame.pixels.set(
        small.pixels.subarray(y * small.width * 4, (y + 1) * small.width * 4),
        y * 900 * 4,
      );
    const regions = {
      dimensions: { width: 900, height: 800 },
      board: { x: 20, y: 30, width: 260, height: 416 },
      pieces: { x: 560, y: 80, width: 122.2, height: 221 },
      abilities: { x: 350, y: 570, width: 122.2, height: 156 },
    };
    for (const r of pieceCardRegions(regions.pieces)) {
      paint(frame, r, [244, 250, 252]);
      drawPiece(frame, r, ["1"]);
    }
    for (const [key, r] of Object.entries(countRegions(regions.abilities))) {
      paint(
        frame,
        { x: r.x - 4, y: r.y - 4, width: r.width + 8, height: r.height + 8 },
        key === "total" ? [230, 251, 252] : [20, 102, 190],
      );
      drawGlyph(
        frame,
        r,
        zero,
        key === "total" ? [20, 130, 160] : [255, 255, 255],
        key === "total" ? 0.16 : 0.11,
      );
    }
    const before = createHash("sha256").update(frame.pixels).digest("hex"),
      result = recognizeConfiguredGame(frame, regions, [
        { value: "0", mask: rowMask(zero) },
      ]);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(
      result.result.board.map((r) =>
        r.map((c) => (c.occupied ? "1" : "0")).join(""),
      ),
    ).toEqual(referenceRows);
    expect(result.result.pieces.map((p) => p.pieceId)).toEqual([
      "DOT",
      "DOT",
      "DOT",
    ]);
    expect(result.result.abilities.reroll.value).toBe(0);
    expect(result.result.abilities.singleCell.value).toBe(0);
    expect(createHash("sha256").update(frame.pixels).digest("hex")).toBe(
      before,
    );
  });
  it("rejects resized/outside regions and refuses to relocate a saved board", () => {
    const frame = automaticFixture(),
      board = { x: 20, y: 30, width: 260, height: 416 },
      regions = suggestedRegions({ ...frame, width: 500 }, board);
    expect(regionsFit(frame, regions)).toBe(false);
    expect(recognizeConfiguredGame(frame, regions, []).ok).toBe(false);
    expect(recognizeBoardRegion(frame, { ...board, x: board.x + 13 }).ok).toBe(
      false,
    );
    expect(recognizeBoardRegion(frame, { ...board, width: NaN }).ok).toBe(
      false,
    );
    expect(recognizeBoardRegion({ ...frame, timestamp: NaN }, board)).toEqual({
      ok: false,
      reason: "INVALID_FRAME",
    });
  });
});
