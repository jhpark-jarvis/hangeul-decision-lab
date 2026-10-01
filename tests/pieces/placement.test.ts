import { describe, expect, it } from "vitest";
import { createEmptyBoard, type Board } from "../../src/domain/board/board";
import { getInitialCatalog } from "../../src/domain/pieces/catalog";
import {
  canPlace,
  getValidPlacements,
  placePiece,
} from "../../src/domain/pieces/placement";
import type { Piece, Shape } from "../../src/domain/pieces/types";

function piece(id: string): Piece {
  const found = getInitialCatalog().find((entry) => entry.id === id);
  if (!found) throw new Error("Missing fixture");
  return found;
}
function rows(shape: Shape): string {
  return shape.map((row) => row.map(Number).join("")).join("/");
}

describe("placement boundaries and occupied cells", () => {
  it.each([
    [0, 0],
    [0, 9],
    [15, 0],
    [15, 9],
  ])("places a DOT at corner (%i,%i)", (row, col) => {
    const original = createEmptyBoard();
    expect(canPlace(original, [[true]], row, col)).toBe(true);
    const result = placePiece(original, [[true]], row, col);
    if (!result.ok) throw new Error(result.error.message);
    expect(result.board[row][col]).toBe(true);
    expect(result.board.flat().filter(Boolean)).toHaveLength(1);
    expect(original.flat().some(Boolean)).toBe(false);
  });

  it.each([
    [-1, 0],
    [16, 0],
    [0, -1],
    [0, 10],
  ])("rejects out-of-board (%i,%i)", (row, col) => {
    expect(canPlace(createEmptyBoard(), [[true]], row, col)).toBe(false);
    expect(placePiece(createEmptyBoard(), [[true]], row, col)).toMatchObject({
      ok: false,
      error: { code: "OUT_OF_BOUNDS" },
    });
  });

  it.each([
    [0.5, 0],
    [0, 0.5],
    [NaN, 0],
    [0, NaN],
    [Infinity, 0],
    [0, Infinity],
  ])("rejects noninteger (%s,%s)", (row, col) => {
    expect(canPlace(createEmptyBoard(), [[true]], row, col)).toBe(false);
    expect(placePiece(createEmptyBoard(), [[true]], row, col)).toMatchObject({
      ok: false,
      error: { code: "INVALID_COORDINATE" },
    });
  });

  it("rejects a block that extends past an edge or is larger than the board", () => {
    const board = createEmptyBoard();
    expect(canPlace(board, piece("LINE_3").shape, 15, 7)).toBe(true);
    expect(canPlace(board, piece("LINE_3").shape, 15, 8)).toBe(false);
    const oversized = {
      id: "synthetic",
      name: "wide",
      shape: [Array<boolean>(11).fill(true)],
    };
    expect(placePiece(board, oversized.shape, 0, 0)).toMatchObject({
      ok: false,
      error: { code: "OUT_OF_BOUNDS" },
    });
    // All rotations too large in the 16x10 board.
    expect(
      getValidPlacements(board, {
        ...oversized,
        shape: [Array<boolean>(17).fill(true)],
      }),
    ).toEqual({ ok: true, placements: [] });
  });

  it("preserves an occupied MIEUM hole but rejects an occupied ring cell", () => {
    const board = createEmptyBoard();
    board[5][5] = true;
    const mieum = piece("MIEUM").shape;
    expect(canPlace(board, mieum, 4, 4)).toBe(true);
    const result = placePiece(board, mieum, 4, 4);
    if (!result.ok) throw new Error(result.error.message);
    expect(result.board.flat().filter(Boolean)).toHaveLength(9);
    expect(result.board[5][5]).toBe(true);
    board[4][4] = true;
    const snapshot = structuredClone(board);
    expect(canPlace(board, mieum, 4, 4)).toBe(false);
    expect(placePiece(board, mieum, 4, 4)).toMatchObject({
      ok: false,
      error: { code: "COLLISION" },
    });
    expect(board).toEqual(snapshot);
  });

  it("uses the normalized top-left as origin for padded input", () => {
    const padded = [
      [false, false, false],
      [false, true, false],
      [false, false, false],
    ];
    const result = placePiece(createEmptyBoard(), padded, 15, 9);
    if (!result.ok) throw new Error(result.error.message);
    expect(result.board[15][9]).toBe(true);
    expect(result.board.flat().filter(Boolean)).toHaveLength(1);
  });

  it("returns explicit board/shape/piece errors instead of accepting bad inputs", () => {
    expect(canPlace([], [[true]], 0, 0)).toBe(false);
    expect(canPlace(createEmptyBoard(), [[false]], 0, 0)).toBe(false);
    expect(placePiece([], [[true]], 0, 0)).toMatchObject({
      ok: false,
      error: { code: "INVALID_BOARD" },
    });
    expect(placePiece(createEmptyBoard(), [[false]], 0, 0)).toMatchObject({
      ok: false,
      error: { code: "INVALID_SHAPE" },
    });
    expect(getValidPlacements([], piece("DOT"))).toMatchObject({
      ok: false,
      error: { code: "INVALID_BOARD" },
    });
    expect(getValidPlacements(createEmptyBoard(), null)).toMatchObject({
      ok: false,
      error: { code: "INVALID_PIECE" },
    });
  });

  it("preserves frozen boards/shapes and creates independent output snapshots", () => {
    const board = createEmptyBoard();
    const shape = piece("L_3").shape;
    board.forEach(Object.freeze);
    Object.freeze(board);
    shape.forEach(Object.freeze);
    Object.freeze(shape);
    const first = placePiece(board, shape, 0, 0);
    if (!first.ok) throw new Error(first.error.message);
    const second = placePiece(first.board, [[true]], 15, 9);
    if (!second.ok) throw new Error(second.error.message);
    second.board[0][0] = false;
    expect(first.board[0][0]).toBe(true);
    expect(first.board[15][9]).toBe(false);
    expect(board.flat().some(Boolean)).toBe(false);
    expect(rows(shape)).toBe("10/11");
  });
});

// Hand-written orientations and an occupancy-set oracle, independent of production transforms/canPlace.
const fixtureVariants: Record<string, string[]> = {
  DOT: ["1"],
  LINE_3: ["111", "1/1/1"],
  MIEUM: ["111/101/111"],
  L_3: ["10/11", "11/10", "11/01", "01/11"],
};
function oracle(board: Board, id: string): string[] {
  const occupied = new Set(
    board.flatMap((cells, row) =>
      cells.flatMap((filled, col) => (filled ? [`${row},${col}`] : [])),
    ),
  );
  const expected: string[] = [];
  for (const variant of fixtureVariants[id]) {
    const matrix = variant.split("/");
    const offsets = matrix.flatMap((cells, row) =>
      Array.from(cells).flatMap((cell, col) =>
        cell === "1" ? [[row, col]] : [],
      ),
    );
    for (let row = 0; row < 16; row++) {
      for (let col = 0; col < 10; col++) {
        if (
          offsets.every(
            ([dr, dc]) =>
              row + dr < 16 &&
              col + dc < 10 &&
              !occupied.has(`${row + dr},${col + dc}`),
          )
        ) {
          expected.push(`${variant}@${row},${col}`);
        }
      }
    }
  }
  return expected.sort();
}

describe("all unique valid placements", () => {
  it("enumerates reflected chiral placements with the original variant metadata", () => {
    const shape = [
      [false, true, true],
      [true, true, false],
      [false, true, false],
    ];
    const result = getValidPlacements(createEmptyBoard(), {
      id: "SYNTHETIC_F",
      name: "chiral test fixture",
      shape,
    });
    if (!result.ok) throw new Error(result.error.message);
    expect(result.placements).toHaveLength(896); // 8 orientations * 14 rows * 8 cols.
    const origins = result.placements.filter(
      (placement) => placement.row === 0 && placement.col === 0,
    );
    expect(
      origins.map((placement) => [
        rows(placement.variant),
        placement.rotation,
        placement.flipped,
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
  });
  it.each([
    ["DOT", 160],
    ["LINE_3", 268],
    ["MIEUM", 112],
    ["L_3", 540],
  ])("matches hand-counted empty-board %s=%i", (id, count) => {
    const result = getValidPlacements(createEmptyBoard(), piece(id));
    if (!result.ok) throw new Error(result.error.message);
    expect(result.placements).toHaveLength(count);
    expect(
      result.placements
        .map(
          (placement) =>
            `${rows(placement.variant)}@${placement.row},${placement.col}`,
        )
        .sort(),
    ).toEqual(oracle(createEmptyBoard(), id));
    expect(
      result.placements.every(
        (placement) => placement.pieceId === id && !placement.flipped,
      ),
    ).toBe(true);
  });

  it.each(["DOT", "LINE_3", "MIEUM", "L_3"])(
    "matches independent occupancy oracle on a partially filled board for %s",
    (id) => {
      const board = createEmptyBoard().map((row, r) =>
        row.map((_, c) => (r * 3 + c * 5) % 11 === 0),
      );
      const result = getValidPlacements(board, piece(id));
      if (!result.ok) throw new Error(result.error.message);
      expect(
        result.placements
          .map(
            (placement) =>
              `${rows(placement.variant)}@${placement.row},${placement.col}`,
          )
          .sort(),
      ).toEqual(oracle(board, id));
      for (const placement of result.placements)
        expect(
          placePiece(board, placement.variant, placement.row, placement.col).ok,
        ).toBe(true);
      expect(getValidPlacements(board, piece(id))).toEqual(result);
    },
  );

  it("returns zero placements on a full board", () => {
    const board = createEmptyBoard().map((row) => row.map(() => true));
    for (const entry of getInitialCatalog())
      expect(getValidPlacements(board, entry)).toEqual({
        ok: true,
        placements: [],
      });
  });

  it("detaches each placement matrix from peers, the source piece, and later calls", () => {
    const input = piece("L_3");
    const result = getValidPlacements(createEmptyBoard(), input);
    if (!result.ok) throw new Error(result.error.message);
    const second = structuredClone(result.placements[1]);
    result.placements[0].variant[0][0] = false;
    expect(result.placements[1]).toEqual(second);
    expect(input.shape[0][0]).toBe(true);
    const fresh = getValidPlacements(createEmptyBoard(), input);
    if (!fresh.ok) throw new Error(fresh.error.message);
    expect(fresh.placements[0].variant[0][0]).toBe(true);
  });
});
