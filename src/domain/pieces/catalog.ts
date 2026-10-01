import type { Piece } from "./types";

// Only the four user-provided development shapes, not the full event catalog.
const INITIAL_PIECES: Piece[] = [
  { id: "DOT", name: "1칸", shape: [[true]] },
  { id: "LINE_3", name: "가로 3칸", shape: [[true, true, true]] },
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
    id: "L_3",
    name: "3칸 L",
    shape: [
      [true, false],
      [true, true],
    ],
  },
];

export function getInitialCatalog(): Piece[] {
  return INITIAL_PIECES.map((piece) => ({
    ...piece,
    shape: piece.shape.map((row) => row.slice()),
  }));
}
