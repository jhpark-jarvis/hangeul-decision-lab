import type { Piece } from "./types";

// Canonical shape reference: tests/fixtures/pieces/catalog-v2.json.
// Names: canonical Hangul label mapping.
// Corrected final rows; legacy piece IDs remain stable.
// Official update813 confirms nineteen types and rotation/reflection, not each footprint.
export const EVENT_CATALOG_VERSION = "2026-10-02-user-labels-v2";
const EVENT_PIECES: Piece[] = [
  { id: "DOT", name: "점", shape: [[true]] },
  { id: "LINE_3", name: "ㅡ", shape: [[true, true, true]] },
  {
    id: "BRANCH_4",
    name: "ㅏ",
    shape: [
      [true, false],
      [true, true],
      [true, false],
    ],
  },
  {
    id: "DOUBLE_BRANCH_7",
    name: "ㅑ",
    shape: [
      [true, false],
      [true, true],
      [true, false],
      [true, true],
      [true, false],
    ],
  },
  {
    id: "DIAGONAL_3",
    name: "ㅅ",
    shape: [
      [false, true, false],
      [true, false, true],
    ],
  },
  {
    id: "L_3",
    name: "ㄴ",
    shape: [
      [true, false],
      [true, true],
    ],
  },
  {
    id: "DIAMOND_4",
    name: "ㅇ",
    shape: [
      [false, true, false],
      [true, false, true],
      [false, true, false],
    ],
  },
  {
    id: "CROWN_6",
    name: "ㅈ",
    shape: [
      [true, true, true],
      [false, true, false],
      [true, false, true],
    ],
  },
  {
    id: "HOOK_4",
    name: "ㄱ",
    shape: [
      [true, true],
      [false, true],
      [false, true],
    ],
  },
  {
    id: "LINE_5",
    name: "ㅣ",
    shape: [[true], [true], [true], [true], [true]],
  },
  {
    id: "DOUBLE_ARM_RIGHT_6",
    name: "ㅋ",
    shape: [
      [true, true],
      [false, true],
      [true, true],
      [false, true],
    ],
  },
  {
    id: "C_5",
    name: "ㄷ",
    shape: [
      [true, true],
      [true, false],
      [true, true],
    ],
  },
  {
    id: "STAR_9",
    name: "ㅎ",
    shape: [
      [false, false, true, false, false],
      [true, true, true, true, true],
      [false, true, false, true, false],
      [false, false, true, false, false],
    ],
  },
  {
    id: "PI_10",
    name: "ㅍ",
    shape: [
      [true, true, true, true],
      [false, true, true, false],
      [true, true, true, true],
    ],
  },
  {
    id: "CROWN_7",
    name: "ㅊ",
    shape: [
      [false, true, false],
      [true, true, true],
      [false, true, false],
      [true, false, true],
    ],
  },
  {
    id: "GRID_10",
    name: "ㅂ",
    shape: [
      [true, false, true],
      [true, true, true],
      [true, false, true],
      [true, true, true],
    ],
  },
  {
    id: "MIEUM",
    name: "ㅁ",
    shape: [
      [true, true, true],
      [true, false, true],
      [true, true, true],
    ],
  },
  {
    id: "ZIGZAG_6",
    name: "ㄹ",
    shape: [
      [true, true],
      [false, true],
      [true, true],
      [true, false],
      [true, true],
    ],
  },
  {
    id: "DOUBLE_ARM_LEFT_6",
    name: "ㅌ",
    shape: [
      [true, true],
      [true, false],
      [true, true],
      [true, false],
      [true, true],
    ],
  },
];

/** Historical API name retained; the current default is the supplied 2026 event catalog. */
export function getInitialCatalog(): Piece[] {
  return EVENT_PIECES.map((piece) => ({
    ...piece,
    shape: piece.shape.map((row) => row.slice()),
  }));
}
