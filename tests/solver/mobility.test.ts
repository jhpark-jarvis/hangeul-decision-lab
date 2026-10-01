import { describe, expect, it } from "vitest";
import { createEmptyBoard, type Board } from "../../src/domain/board/board";
import { getInitialCatalog } from "../../src/domain/pieces/catalog";
import {
  calculateMobility,
  countUnfillableGaps,
} from "../../src/domain/solver/mobility";
import { freezeDeep, piece } from "../game/fixtures";

function occupiedExcept(positions: number[][]): Board {
  const board = createEmptyBoard();
  board.forEach((row) => row.fill(true));
  for (const [row, col] of positions) board[row][col] = false;
  return board;
}
const square = [
  [4, 3],
  [4, 4],
  [5, 3],
  [5, 4],
];
const box = Array.from({ length: 9 }, (_, index) => [
  4 + Math.floor(index / 3),
  3 + (index % 3),
]);
const ring = box.filter(([row, col]) => row !== 5 || col !== 4);

// Independent, handwritten occupied footprints. No transforms or placement API.
const footprints: number[][][][] = [
  [[[0, 0]]],
  [
    [
      [0, 0],
      [0, 1],
      [0, 2],
    ],
    [
      [0, 0],
      [1, 0],
      [2, 0],
    ],
  ],
  [
    [
      [0, 0],
      [0, 1],
      [0, 2],
      [1, 0],
      [1, 2],
      [2, 0],
      [2, 1],
      [2, 2],
    ],
  ],
  [
    [
      [0, 0],
      [1, 0],
      [1, 1],
    ],
    [
      [0, 0],
      [0, 1],
      [1, 0],
    ],
    [
      [0, 0],
      [0, 1],
      [1, 1],
    ],
    [
      [0, 1],
      [1, 0],
      [1, 1],
    ],
  ],
];
function oracle(board: Board, typeIndices = [0, 1, 2, 3]) {
  let playablePieceTypes = 0;
  let totalPlacements = 0;
  const covered = new Set<string>();
  for (const orientations of typeIndices.map((index) => footprints[index])) {
    let placements = 0;
    for (const footprint of orientations) {
      for (let row = 0; row < 16; row++)
        for (let col = 0; col < 10; col++) {
          const cells = footprint.map(([dr, dc]) => [row + dr, col + dc]);
          if (cells.some(([r, c]) => r >= 16 || c >= 10 || board[r][c]))
            continue;
          placements++;
          for (const [r, c] of cells) covered.add(`${r},${c}`);
        }
    }
    if (placements > 0) playablePieceTypes++;
    totalPlacements += placements;
  }
  let gaps = 0;
  board.forEach((cells, row) =>
    cells.forEach((occupied, col) => {
      if (!occupied && !covered.has(`${row},${col}`)) gaps++;
    }),
  );
  return {
    mobility: {
      playablePieceTypes,
      totalPieceTypes: typeIndices.length,
      totalPlacements,
    },
    gaps,
  };
}

describe("catalog-relative mobility", () => {
  it.each([
    {
      label: "empty",
      board: createEmptyBoard(),
      playable: 4,
      placements: 1080,
    },
    { label: "full", board: occupiedExcept([]), playable: 0, placements: 0 },
    {
      label: "one isolated hole",
      board: occupiedExcept([[7, 5]]),
      playable: 1,
      placements: 1,
    },
    {
      label: "three-cell corridor",
      board: occupiedExcept([
        [4, 2],
        [4, 3],
        [4, 4],
      ]),
      playable: 2,
      placements: 4,
    },
    {
      label: "two-cell corridor",
      board: occupiedExcept([
        [4, 2],
        [4, 3],
      ]),
      playable: 1,
      placements: 2,
    },
    { label: "2x2", board: occupiedExcept(square), playable: 2, placements: 8 },
    {
      label: "3x3 box",
      board: occupiedExcept(box),
      playable: 4,
      placements: 32,
    },
    {
      label: "3x3 ring with occupied center",
      board: occupiedExcept(ring),
      playable: 4,
      placements: 17,
    },
  ])("matches hand counts on $label", ({ board, playable, placements }) => {
    freezeDeep(board);
    const catalog = freezeDeep(getInitialCatalog());
    expect(calculateMobility(board, catalog)).toEqual({
      ok: true,
      mobility: {
        playablePieceTypes: playable,
        totalPieceTypes: 4,
        totalPlacements: placements,
      },
    });
  });
  it("uses the supplied denominator and keeps equal shapes with different IDs as separate types", () => {
    const board = createEmptyBoard();
    expect(calculateMobility(board, [piece()])).toEqual({
      ok: true,
      mobility: {
        playablePieceTypes: 1,
        totalPieceTypes: 1,
        totalPlacements: 160,
      },
    });
    expect(calculateMobility(board, [piece("LINE_3")])).toEqual({
      ok: true,
      mobility: {
        playablePieceTypes: 1,
        totalPieceTypes: 1,
        totalPlacements: 268,
      },
    });
    expect(
      calculateMobility(board, [piece(), { ...piece(), id: "OTHER_DOT" }]),
    ).toEqual({
      ok: true,
      mobility: {
        playablePieceTypes: 2,
        totalPieceTypes: 2,
        totalPlacements: 320,
      },
    });
    expect(calculateMobility(board, [])).toEqual({
      ok: true,
      mobility: {
        playablePieceTypes: 0,
        totalPieceTypes: 0,
        totalPlacements: 0,
      },
    });
  });
  it("accepts an oversize piece as a valid but unplayable catalog type", () => {
    const big = {
      id: "BIG",
      name: "Synthetic big fixture",
      shape: Array.from({ length: 17 }, () => Array(17).fill(true)),
    };
    expect(calculateMobility(createEmptyBoard(), [big])).toEqual({
      ok: true,
      mobility: {
        playablePieceTypes: 0,
        totalPieceTypes: 1,
        totalPlacements: 0,
      },
    });
  });
  it("normalizes padding and deduplicates symmetric rotations/flip metadata", () => {
    expect(
      calculateMobility(createEmptyBoard(), [
        {
          id: "PADDED_DOT",
          name: "Synthetic padding",
          shape: [
            [false, false, false],
            [false, true, false],
            [false, false, false],
          ],
        },
      ]),
    ).toEqual({
      ok: true,
      mobility: {
        playablePieceTypes: 1,
        totalPieceTypes: 1,
        totalPlacements: 160,
      },
    });
  });
  it.each([
    { catalog: null, code: "INVALID_CATALOG" },
    { catalog: {}, code: "INVALID_CATALOG" },
    { catalog: [piece(), piece()], code: "INVALID_CATALOG" },
    { catalog: Array(1), code: "INVALID_PIECE" },
    { catalog: [null], code: "INVALID_PIECE" },
    { catalog: [{ ...piece(), id: " " }], code: "INVALID_PIECE" },
    { catalog: [{ ...piece(), shape: [[false]] }], code: "INVALID_SHAPE" },
    {
      catalog: [piece(), { ...piece("LINE_3"), shape: [] }],
      code: "INVALID_SHAPE",
    },
  ])(
    "rejects invalid catalog without partial counts %#",
    ({ catalog, code }) => {
      for (const metric of [calculateMobility, countUnfillableGaps]) {
        expect(metric(createEmptyBoard(), catalog)).toMatchObject({
          ok: false,
          error: { code },
        });
        expect(metric(createEmptyBoard(), getInitialCatalog())).toMatchObject({
          ok: true,
        });
      }
    },
  );
  it("rejects malformed board before evaluating a catalog", () => {
    expect(calculateMobility({}, getInitialCatalog())).toMatchObject({
      ok: false,
      error: { code: "INVALID_BOARD" },
    });
    expect(countUnfillableGaps(Array(16), [])).toMatchObject({
      ok: false,
      error: { code: "INVALID_BOARD" },
    });
  });
  it("is deterministic, order-independent and detached from frozen input/previous output", () => {
    const board = freezeDeep(occupiedExcept(box));
    const catalog = freezeDeep(getInitialCatalog());
    const first = calculateMobility(board, catalog);
    const second = calculateMobility(board, [...catalog].reverse());
    expect(second).toEqual(first);
    if (!first.ok || !second.ok) throw new Error("Invalid fixture");
    first.mobility.totalPlacements = -1;
    expect(second.mobility.totalPlacements).toBe(32);
    expect(calculateMobility(board, catalog)).toEqual(second);
    expect(catalog).toEqual(getInitialCatalog());
    expect(board).toEqual(occupiedExcept(box));
  });
});

describe("current-placement gap coverage", () => {
  it.each([
    {
      label: "DOT covers all empty cells",
      board: occupiedExcept([
        [0, 0],
        [15, 9],
      ]),
      catalog: [piece()],
      gaps: 0,
    },
    {
      label: "initial catalog contains DOT",
      board: occupiedExcept([[6, 6]]),
      catalog: getInitialCatalog(),
      gaps: 0,
    },
    {
      label: "LINE cannot cover two-cell corridor",
      board: occupiedExcept([
        [4, 2],
        [4, 3],
      ]),
      catalog: [piece("LINE_3")],
      gaps: 2,
    },
    {
      label: "LINE can cover three-cell corridor",
      board: occupiedExcept([
        [4, 2],
        [4, 3],
        [4, 4],
      ]),
      catalog: [piece("LINE_3")],
      gaps: 0,
    },
    {
      label: "MIEUM false center stays uncovered",
      board: occupiedExcept(box),
      catalog: [piece("MIEUM")],
      gaps: 1,
    },
    {
      label: "MIEUM covers whole empty ring",
      board: occupiedExcept(ring),
      catalog: [piece("MIEUM")],
      gaps: 0,
    },
    {
      label: "empty catalog covers no cells",
      board: createEmptyBoard(),
      catalog: [],
      gaps: 160,
    },
    {
      label: "occupied cells are not gaps",
      board: occupiedExcept([]),
      catalog: [],
      gaps: 0,
    },
  ])("$label", ({ board, catalog, gaps }) => {
    expect(countUnfillableGaps(freezeDeep(board), freezeDeep(catalog))).toEqual(
      { ok: true, count: gaps },
    );
  });
  it("counts uncovered cells rather than connected regions", () => {
    expect(
      countUnfillableGaps(
        occupiedExcept([
          [4, 4],
          [4, 5],
        ]),
        [piece("MIEUM")],
      ),
    ).toEqual({ ok: true, count: 2 });
  });
});

describe("independent occupancy-set oracle", () => {
  const samples = [
    createEmptyBoard(),
    occupiedExcept([]),
    occupiedExcept(square),
    occupiedExcept(box),
    occupiedExcept(ring),
    occupiedExcept([
      [0, 0],
      [15, 9],
    ]),
    Array.from({ length: 16 }, (_, row) =>
      Array.from({ length: 10 }, (_, col) => (row + col) % 2 === 0),
    ),
    Array.from({ length: 16 }, (_, row) =>
      Array.from({ length: 10 }, (_, col) => (row * 10 + col) % 7 < 2),
    ),
  ];
  const catalogs = [[0, 1, 2, 3], [1], [2], [3], [1, 2, 3]];
  const ids = ["DOT", "LINE_3", "MIEUM", "L_3"];
  it.each(
    samples.flatMap((board, index) =>
      catalogs.map((typeIndices, catalogIndex) => ({
        board,
        index,
        typeIndices,
        catalogIndex,
      })),
    ),
  )(
    "matches counts and coverage on sample $index / catalog $catalogIndex",
    ({ board, typeIndices }) => {
      const expected = oracle(board, typeIndices);
      const catalog = freezeDeep(typeIndices.map((index) => piece(ids[index])));
      expect(calculateMobility(freezeDeep(board), catalog)).toEqual({
        ok: true,
        mobility: expected.mobility,
      });
      expect(countUnfillableGaps(board, catalog)).toEqual({
        ok: true,
        count: expected.gaps,
      });
    },
  );
});
