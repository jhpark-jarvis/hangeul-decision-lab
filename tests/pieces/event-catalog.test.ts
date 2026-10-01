import { describe, expect, it } from "vitest";
import reference from "../fixtures/pieces/catalog-v1.json";
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
  it("keeps the supplied reflected pair as separate catalog types", () => {
    expect(
      footprints(reference.pieces[10].rows)
        .map((cells) => JSON.stringify(cells))
        .sort(),
    ).toEqual(
      footprints(reference.pieces[18].rows)
        .map((cells) => JSON.stringify(cells))
        .sort(),
    );
    expect(getInitialCatalog()).toHaveLength(19);
  });
});
