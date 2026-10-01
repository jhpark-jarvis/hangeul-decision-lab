import type { Piece } from "./types";

// Canonical shape reference: tests/fixtures/pieces/catalog-v2.json.
// Official update813 confirms nineteen types and rotation/reflection, not each footprint.
export const EVENT_CATALOG_VERSION = "2026-10-01-user-image-v1";
const EVENT_PIECES: Piece[] = [
  { id: "DOT", name: "1칸", shape: [[true]] },
  { id: "LINE_3", name: "가로 3칸", shape: [[true, true, true]] },
  {
    id: "BRANCH_4",
    name: "4칸 가지",
    shape: [
      [true, false],
      [true, true],
      [true, false],
    ],
  },
  {
    id: "DOUBLE_BRANCH_7",
    name: "7칸 두 가지",
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
    name: "3칸 대각",
    shape: [
      [false, true, false],
      [true, false, true],
    ],
  },
  {
    id: "L_3",
    name: "3칸 L",
    shape: [
      [true, false],
      [true, true],
    ],
  },
  {
    id: "DIAMOND_4",
    name: "4칸 마름모",
    shape: [
      [false, true, false],
      [true, false, true],
      [false, true, false],
    ],
  },
  {
    id: "CROWN_6",
    name: "6칸 지붕",
    shape: [
      [true, true, true],
      [false, true, false],
      [true, false, true],
    ],
  },
  {
    id: "HOOK_4",
    name: "4칸 꺾임",
    shape: [
      [true, true],
      [false, true],
      [false, true],
    ],
  },
  {
    id: "LINE_5",
    name: "세로 5칸",
    shape: [[true], [true], [true], [true], [true]],
  },
  {
    id: "DOUBLE_ARM_RIGHT_6",
    name: "6칸 오른쪽 기둥",
    shape: [
      [true, true],
      [false, true],
      [true, true],
      [false, true],
    ],
  },
  {
    id: "C_5",
    name: "5칸 열린 네모",
    shape: [
      [true, true],
      [true, false],
      [true, true],
    ],
  },
  {
    id: "STAR_9",
    name: "9칸 별",
    shape: [
      [false, false, true, false, false],
      [true, true, true, true, true],
      [false, true, false, true, false],
      [false, false, true, false, false],
    ],
  },
  {
    id: "PI_10",
    name: "10칸 가로 테두리",
    shape: [
      [true, true, true, true],
      [false, true, true, false],
      [true, true, true, true],
    ],
  },
  {
    id: "CROWN_7",
    name: "7칸 지붕",
    shape: [
      [false, true, false],
      [true, true, true],
      [false, true, false],
      [true, false, true],
    ],
  },
  {
    id: "GRID_10",
    name: "10칸 세로 테두리",
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
    name: "6칸 지그재그",
    shape: [
      [true, true],
      [false, true],
      [true, true],
      [true, false],
    ],
  },
  {
    id: "DOUBLE_ARM_LEFT_6",
    name: "6칸 왼쪽 기둥",
    shape: [
      [true, true],
      [true, false],
      [true, true],
      [true, false],
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
