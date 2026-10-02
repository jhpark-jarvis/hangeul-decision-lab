import { describe, expect, it } from "vitest";
import { getInitialCatalog } from "../../src/domain/pieces/catalog";
import {
  completePieceLabels,
  HANGEUL_PIECE_LABELS,
} from "../../src/features/pieces/labels";

// Arbitrary synthetic permutation for contract testing; not the user's real mapping.
const synthetic = () => {
  const rest = HANGEUL_PIECE_LABELS.filter(
    (label) => !["점", "ㅡ", "ㅣ"].includes(label),
  );
  return getInitialCatalog().map((piece) =>
    piece.id === "DOT"
      ? "점"
      : piece.id === "LINE_3"
        ? "ㅡ"
        : piece.id === "LINE_5"
          ? "ㅣ"
          : rest.shift()!,
  );
};
describe("user-defined piece label completion", () => {
  it("requires every label once without inferring missing names", () => {
    const names = synthetic();
    for (const bad of [
      names.slice(1),
      names.map(() => ""),
      names.map((name, i) => (i === 5 ? names[6] : name)),
      names.map((name, i) => (i === 3 ? "A" : name)),
    ])
      expect(completePieceLabels(getInitialCatalog(), bad).ok).toBe(false);
  });
  it("checks the three user-specified size constraints", () => {
    for (const label of ["점", "ㅡ", "ㅣ"] as const) {
      const names = synthetic();
      const position = names.indexOf(label);
      [names[position], names[12]] = [names[12], names[position]];
      expect(completePieceLabels(getInitialCatalog(), names)).toEqual({
        ok: false,
        message: expect.stringContaining("1칸"),
      });
    }
  });
  it("creates detached text results with original IDs and no catalog changes", () => {
    const catalog = getInitialCatalog();
    const names = synthetic();
    const before = structuredClone(catalog);
    const result = completePieceLabels(catalog, names);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.mapping).toHaveLength(19);
    expect(result.mapping[0]).toEqual({
      pieceId: "DOT",
      label: "점",
      cells: 1,
    });
    result.mapping[0].label = "changed";
    expect(catalog).toEqual(before);
    expect(names[0]).toBe("점");
  });
  it("rejects an incomplete or duplicated catalog", () => {
    const catalog = getInitialCatalog();
    expect(completePieceLabels(catalog.slice(1), synthetic()).ok).toBe(false);
    catalog[1] = catalog[0];
    expect(completePieceLabels(catalog, synthetic()).ok).toBe(false);
  });
  it("rejects a sparse label list instead of skipping an unassigned entry", () => {
    const names = synthetic();
    delete names[3];
    expect(completePieceLabels(getInitialCatalog(), names).ok).toBe(false);
  });
});
