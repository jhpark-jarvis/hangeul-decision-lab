import { describe, expect, it } from "vitest";
import {
  candidateFootprint,
  encodeFootprints,
} from "../../src/research/spatial-footprint";
const key = (...a: unknown[]) => JSON.stringify(a);
describe("spatial candidate geometry", () => {
  it("uses asymmetric shape cells at absolute coordinates, including holes", () => {
    const shape = [
      [true, false, true],
      [false, true, false],
    ];
    const mask = candidateFootprint(
      key("place-piece", "held", 2, "shape", 14, 7, shape),
    );
    for (let row = 0; row < 16; row++)
      for (let col = 0; col < 10; col++)
        expect(mask[row][col]).toBe(
          Number(
            row >= 14 && col >= 7 && (shape[row - 14]?.[col - 7] ?? false),
          ),
        );
  });
  it("marks point bottom-right and reroll no cells, preserving full key order", () => {
    const keys = [
      key("reroll", "held", 0),
      key("single-cell", 15, 9),
      key("single-cell", 0, 0),
    ];
    const { footprints, candidates } = encodeFootprints(keys, 3);
    expect(candidates).toBe(3);
    expect(footprints).toEqual([
      "0".repeat(160),
      "0".repeat(159) + "1",
      "1" + "0".repeat(159),
    ]);
    expect(keys).toEqual([
      key("reroll", "held", 0),
      key("single-cell", 15, 9),
      key("single-cell", 0, 0),
    ]);
  });
  it.each([
    "null",
    "{}",
    "bad json",
    key("unknown"),
    key("single-cell", 16, 0),
    key("single-cell", 0, -1),
    key("single-cell", 0.5, 0),
    key("single-cell", 0, 0, 1),
    key("reroll", "", 0),
    key("reroll", "held", 3),
    key("place-piece", "held", 0, "X", 15, 9, [[true, true]]),
    key("place-piece", "held", 0, "X", 0, 0, [[1]]),
    key("place-piece", "held", 0, "X", 0, 0, [[false]]),
    key("place-piece", "held", 0, "X", 0, 0, [[true], [true, false]]),
  ])("rejects malformed or out-of-board geometry: %s", (bad) => {
    expect(() => candidateFootprint(bad)).toThrow();
  });
  it("rejects duplicates, empty/oversized mask and budget exhaustion before conversion", () => {
    const point = key("single-cell", 0, 0);
    for (const [keys, budget] of [
      [[], 1],
      [[point, point], 2],
      [[point], 0],
      [[point], 0.5],
    ] as [string[], number][])
      expect(() => encodeFootprints(keys, budget)).toThrow();
  });
});
