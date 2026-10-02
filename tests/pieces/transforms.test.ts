import { describe, expect, it } from "vitest";
import { getInitialCatalog } from "../../src/domain/pieces/catalog";
import reference from "../fixtures/pieces/catalog-v2.json";
import {
  flipHorizontal,
  getUniqueVariants,
  normalizeShape,
  rotate90,
  serializeShape,
  validatePiece,
  validateShape,
} from "../../src/domain/pieces/transforms";
import type { Shape, ShapeResult } from "../../src/domain/pieces/types";

function matrix(rows: string): Shape {
  return rows.split("/").map((row) => Array.from(row, (cell) => cell === "1"));
}
function shape(result: ShapeResult): Shape {
  if (!result.ok) throw new Error(result.error.message);
  return result.shape;
}
function rows(input: Shape): string {
  return input.map((row) => row.map(Number).join("")).join("/");
}

describe("piece inputs and canonical catalog", () => {
  it("matches all nineteen supplied image footprints and returns detached data", () => {
    const catalog = getInitialCatalog();
    expect(
      catalog.map((piece) => [piece.id, piece.name, rows(piece.shape)]),
    ).toEqual(
      reference.pieces.map((piece) => [
        piece.id,
        piece.name,
        piece.rows.join("/"),
      ]),
    );
    expect(new Set(catalog.map((piece) => piece.id)).size).toBe(19);
    const ring = catalog.find((piece) => piece.id === "MIEUM")!;
    catalog[0].id = "changed";
    ring.shape[1][1] = true;
    catalog.pop();
    const fresh = getInitialCatalog();
    expect(fresh).toHaveLength(19);
    expect(fresh[0].id).toBe("DOT");
    expect(fresh.find((piece) => piece.id === "MIEUM")!.shape[1][1]).toBe(
      false,
    );
  });
  it.each(
    [
      null,
      {},
      [],
      [[]],
      [[false]],
      [[true], []],
      [[true], [true, false]],
      [[1]],
      [["true"]],
      Array(2),
      [Array(2)],
      [[true, , false]],
    ].map((input, index) => ({ input, index })),
  )(
    "rejects malformed shape $index consistently at every public boundary",
    ({ input }) => {
      for (const operation of [
        validateShape,
        normalizeShape,
        rotate90,
        flipHorizontal,
        serializeShape,
        getUniqueVariants,
      ]) {
        expect(operation(input)).toMatchObject({
          ok: false,
          error: { code: "INVALID_SHAPE" },
        });
      }
    },
  );

  it.each([
    null,
    {},
    [],
    { id: "", name: "x", shape: [[true]] },
    { id: "x", name: " ", shape: [[true]] },
    { id: 1, name: "x", shape: [[true]] },
  ])("rejects invalid piece metadata %j", (input) => {
    expect(validatePiece(input)).toMatchObject({
      ok: false,
      error: { code: "INVALID_PIECE" },
    });
  });

  it("validates piece shape and detaches accepted input", () => {
    expect(validatePiece({ id: "x", name: "x", shape: [] })).toMatchObject({
      ok: false,
      error: { code: "INVALID_SHAPE" },
    });
    const input = { id: "x", name: "x", shape: matrix("10/11") };
    const result = validatePiece(input);
    if (!result.ok) throw new Error(result.error.message);
    result.piece.shape[0][0] = false;
    expect(input.shape[0][0]).toBe(true);
  });
});

describe("shape transforms", () => {
  it("rotates a rectangular asymmetric shape clockwise with exact occupied positions", () => {
    const original = matrix("100/111");
    expect(rows(shape(rotate90(original)))).toBe("11/10/10");
    expect(rows(shape(flipHorizontal(original)))).toBe("001/111");
    let rotated = original;
    for (let turn = 0; turn < 4; turn++) rotated = shape(rotate90(rotated));
    expect(rotated).toEqual(original);
    expect(shape(flipHorizontal(shape(flipHorizontal(original))))).toEqual(
      original,
    );
  });

  it("trims all empty edges without filling a hole or mutating frozen input", () => {
    const input = matrix("00000/01110/01010/01110/00000");
    input.forEach(Object.freeze);
    Object.freeze(input);
    const normalized = shape(normalizeShape(input));
    expect(rows(normalized)).toBe("111/101/111");
    expect(shape(normalizeShape(normalized))).toEqual(normalized);
    normalized[1][1] = true;
    expect(input[2][2]).toBe(false);
    expect(rows(shape(normalizeShape(input)))).toBe("111/101/111");
  });

  it("uses unambiguous normalized row keys", () => {
    expect(serializeShape(matrix("000/010/000"))).toEqual({
      ok: true,
      key: "1x1:1",
    });
    expect(serializeShape(matrix("11"))).toEqual({ ok: true, key: "1x2:11" });
    expect(serializeShape(matrix("1/1"))).toEqual({ ok: true, key: "2x1:1/1" });
  });

  it.each([
    ["DOT", 1],
    ["LINE_3", 2],
    ["MIEUM", 1],
    ["L_3", 4],
  ])("deduplicates symmetric %s to %i orientations", (id, count) => {
    const piece = getInitialCatalog().find((entry) => entry.id === id);
    const result = getUniqueVariants(piece?.shape);
    if (!result.ok) throw new Error(result.error.message);
    expect(result.variants).toHaveLength(count);
    expect(result.variants.every((variant) => !variant.flipped)).toBe(true);
    expect(
      new Set(result.variants.map((variant) => rows(variant.shape))).size,
    ).toBe(count);
  });

  it("retains all eight chiral orientations with deterministic flip-then-rotation metadata", () => {
    const input = matrix("011/110/010"); // Synthetic F fixture, not an added catalog entry.
    const result = getUniqueVariants(input);
    if (!result.ok) throw new Error(result.error.message);
    expect(
      result.variants.map((variant) => [
        rows(variant.shape),
        variant.rotation,
        variant.flipped,
      ]),
    ).toEqual([
      ["011/110/010", 0, false],
      ["010/111/001", 90, false],
      ["010/011/110", 180, false],
      ["100/111/010", 270, false],
      ["110/011/010", 0, true],
      ["001/111/010", 90, true],
      ["010/110/011", 180, true],
      ["010/111/100", 270, true],
    ]);
    expect(getUniqueVariants(input)).toEqual(result);
  });

  it("detaches transformations and each returned variant from frozen snapshots", () => {
    const input = matrix("10/11");
    input.forEach(Object.freeze);
    Object.freeze(input);
    const result = getUniqueVariants(input);
    if (!result.ok) throw new Error(result.error.message);
    const second = structuredClone(result.variants[1]);
    result.variants[0].shape[0][0] = false;
    expect(rows(input)).toBe("10/11");
    expect(result.variants[1]).toEqual(second);
    const validated = shape(validateShape(input));
    validated[0][0] = false;
    expect(input[0][0]).toBe(true);
    expect(shape(rotate90(input))).toEqual(matrix("11/10"));
    expect(shape(flipHorizontal(input))).toEqual(matrix("01/11"));
  });
});
