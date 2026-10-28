import { describe, expect, it } from "vitest";
import {
  fitSubsetIndices,
  uniqueFitLabel,
} from "../../src/research/controlled-point-fit";
import type { PointReference } from "../../src/research/point-simulation";
const reference: PointReference = {
  complete: true,
  visitedNodes: 40,
  leaves: 38,
  maxDepth: 2,
  bestObserved: null,
  optimalFirstActions: ["target"],
};
describe("controlled fitting contract without materializing a dataset", () => {
  it("fixes all sixteen rows and all ten columns", () => {
    const indices = fitSubsetIndices();
    expect(indices).toEqual([
      0, 13, 26, 39, 42, 55, 68, 71, 84, 97, 100, 113, 126, 139, 142, 155,
    ]);
    expect(new Set(indices.map((i) => i % 10)).size).toBe(10);
  });
  it("uses original legal order rather than first or sorted index", () => {
    expect(uniqueFitLabel(reference, ["z", "target", "a"])).toBe(1);
    expect(uniqueFitLabel(reference, ["target", "z"])).toBe(0);
  });
  it("rejects incomplete and ambiguous references", () => {
    for (const r of [
      { ...reference, complete: false },
      { ...reference, optimalFirstActions: null },
      { ...reference, optimalFirstActions: [] },
      { ...reference, optimalFirstActions: ["a", "b"] },
    ])
      expect(() => uniqueFitLabel(r, ["a", "b"])).toThrow();
  });
  it("rejects missing roots and duplicate candidates", () => {
    for (const keys of [["a"], ["target", "target"], ["target", "a", "a"]])
      expect(() => uniqueFitLabel(reference, keys)).toThrow();
  });
});
