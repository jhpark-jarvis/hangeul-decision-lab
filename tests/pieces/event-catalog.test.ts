import { describe, expect, it } from "vitest";
import reference from "../fixtures/pieces/catalog-v2.json";
import previous from "../fixtures/pieces/catalog-v1.json";
import userLabels from "../fixtures/pieces/labels.json";
import { createEmptyBoard, type Board } from "../../src/domain/board/board";
import { getInitialCatalog } from "../../src/domain/pieces/catalog";
import { getUniqueVariants } from "../../src/domain/pieces/transforms";
import { getValidPlacements } from "../../src/domain/pieces/placement";
import { calculateMobility } from "../../src/domain/solver/mobility";
import { freezeDeep, success } from "../game/fixtures";

// Coordinate dihedral oracle, independent of matrix rotation/placement functions.
function footprints(rows: string[]) {
  const cells = rows.flatMap((row, r) =>
    [...row].flatMap((cell, c) => (cell === "1" ? [[r, c]] : [])),
  );
  const unique = new Map<string, number[][]>();
  for (const reflected of [false, true]) {
    let transformed = cells.map(([r, c]) => [r, reflected ? -c : c]);
    for (let rotation = 0; rotation < 4; rotation++) {
      const minR = Math.min(...transformed.map(([r]) => r));
      const minC = Math.min(...transformed.map(([, c]) => c));
      const normalized = transformed
        .map(([r, c]) => [r - minR, c - minC])
        .sort(([r, c], [s, d]) => r - s || c - d);
      unique.set(JSON.stringify(normalized), normalized);
      transformed = transformed.map(([r, c]) => [c, -r]);
    }
  }
  return [...unique.values()];
}
function oracle(board: Board, rows: string[]) {
  const keys: string[] = [];
  for (const cells of footprints(rows))
    for (let r = 0; r < 16; r++)
      for (let c = 0; c < 10; c++)
        if (
          cells.every(
            ([dr, dc]) => r + dr < 16 && c + dc < 10 && !board[r + dr][c + dc],
          )
        )
          keys.push(JSON.stringify(cells.map(([dr, dc]) => [r + dr, c + dc])));
  return keys.sort();
}
const empty = createEmptyBoard();
const striped = createEmptyBoard().map((row, r) =>
  row.map((_, c) => (r * 7 + c * 3) % 11 < 3),
);
const narrow = createEmptyBoard().map((row, r) =>
  row.map((_, c) => !(r >= 3 && r <= 8 && c >= 2 && c <= 6)),
);

describe("2026 supplied catalog and placements", () => {
  it.each(reference.pieces)(
    "image $sourceIndex $id: exact cells, variants and legal footprints",
    (ref) => {
      const piece = getInitialCatalog().find((entry) => entry.id === ref.id)!;
      expect(piece.shape.flat().filter(Boolean)).toHaveLength(ref.cells);
      expect(piece.name).toBe(userLabels.mapping[ref.sourceIndex - 1].label);
      const variants = success(getUniqueVariants(piece.shape)).variants;
      expect(variants).toHaveLength(footprints(ref.rows).length);
      for (const board of [empty, striped, narrow]) {
        const placements = success(
          getValidPlacements(freezeDeep(board), piece),
        ).placements;
        const actual = placements
          .map(({ variant, row, col }) =>
            JSON.stringify(
              variant.flatMap((line, r) =>
                line.flatMap((filled, c) =>
                  filled ? [[row + r, col + c]] : [],
                ),
              ),
            ),
          )
          .sort();
        expect(actual).toEqual(oracle(board, ref.rows));
      }
    },
  );
  it.each([empty, striped, narrow].map((board) => ({ board })))(
    "counts all supplied types against independent oracle %#",
    ({ board }) => {
      const counts = reference.pieces.map(
        ({ rows }) => oracle(board, rows).length,
      );
      const result = calculateMobility(board, getInitialCatalog());
      if (!result.ok) throw new Error(result.error.message);
      expect(result.mobility).toEqual({
        totalPieceTypes: 19,
        playablePieceTypes: counts.filter((count) => count > 0).length,
        totalPlacements: counts.reduce((sum, count) => sum + count, 0),
      });
    },
  );
  it("distinguishes corrected ㅌ from ㅋ while retaining nineteen stable IDs", () => {
    expect(
      footprints(reference.pieces[10].rows)
        .map((cells) => JSON.stringify(cells))
        .sort(),
    ).not.toEqual(
      footprints(reference.pieces[18].rows)
        .map((cells) => JSON.stringify(cells))
        .sort(),
    );
    expect(getInitialCatalog()).toHaveLength(19);
  });
  it("preserves every ID and the other seventeen supplied shapes", () => {
    expect(reference.pieces.map(({ id }) => id)).toEqual(
      previous.pieces.map(({ id }) => id),
    );
    for (const old of previous.pieces.slice(0, 17))
      expect(reference.pieces.find(({ id }) => id === old.id)!.rows).toEqual(
        old.rows,
      );
  });
  it.each(["ZIGZAG_6", "DOUBLE_ARM_LEFT_6"])(
    "%s requires both bottom cells and rejects bottom-row collisions and overflow",
    (id) => {
      const piece = getInitialCatalog().find((p) => p.id === id)!;
      expect(piece.shape).toHaveLength(5);
      expect(piece.shape[4]).toEqual([true, true]);
      expect(piece.shape.flat().filter(Boolean)).toHaveLength(8);
      const rows = piece.shape.map((row) => row.map(Number).join(""));
      const canonical = (board: Board, row: number, col: number) =>
        success(getValidPlacements(board, piece)).placements.some(
          (p) =>
            p.row === row &&
            p.col === col &&
            p.variant.map((line) => line.map(Number).join("")).join("/") ===
              rows.join("/"),
        );
      expect(canonical(empty, 11, 0)).toBe(true);
      expect(canonical(empty, 12, 0)).toBe(false);
      for (const col of [0, 1]) {
        const blocked = createEmptyBoard();
        blocked[4][col] = true;
        expect(canonical(blocked, 0, 0)).toBe(false);
      }
    },
  );
});
