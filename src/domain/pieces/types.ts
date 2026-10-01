export type Shape = boolean[][];
export type Piece = { id: string; name: string; shape: Shape };
export type Rotation = 0 | 90 | 180 | 270;
export type PieceVariant = {
  shape: Shape;
  rotation: Rotation;
  flipped: boolean;
};
export type PieceError = {
  code: "INVALID_SHAPE" | "INVALID_PIECE";
  message: string;
};
export type PieceResult<T> =
  ({ ok: true } & T) | { ok: false; error: PieceError };
export type ShapeResult = PieceResult<{ shape: Shape }>;

export type Placement = {
  pieceId: string;
  variant: Shape;
  rotation: Rotation;
  flipped: boolean;
  row: number;
  col: number;
};
