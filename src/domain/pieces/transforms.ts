import type {
  Piece,
  PieceResult,
  PieceVariant,
  Rotation,
  Shape,
  ShapeResult,
} from "./types";

const ROTATIONS: readonly Rotation[] = [0, 90, 180, 270];

export function validateShape(input: unknown): ShapeResult {
  const invalid: ShapeResult = {
    ok: false,
    error: {
      code: "INVALID_SHAPE",
      message: "점유 칸이 있는 직사각형 boolean 블록이 필요합니다.",
    },
  };
  if (!Array.isArray(input) || input.length === 0) return invalid;
  const first: unknown = input[0];
  if (!Array.isArray(first) || first.length === 0) return invalid;
  const shape: Shape = [];
  let occupied = false;
  for (let row = 0; row < input.length; row++) {
    const source: unknown = input[row];
    if (!Array.isArray(source) || source.length !== first.length)
      return invalid;
    const cells: boolean[] = [];
    for (let col = 0; col < first.length; col++) {
      const cell: unknown = source[col];
      if (typeof cell !== "boolean") return invalid;
      occupied ||= cell;
      cells.push(cell);
    }
    shape.push(cells);
  }
  return occupied ? { ok: true, shape } : invalid;
}

export function validatePiece(input: unknown): PieceResult<{ piece: Piece }> {
  if (
    typeof input !== "object" ||
    input === null ||
    !("id" in input) ||
    typeof input.id !== "string" ||
    input.id.trim() === "" ||
    !("name" in input) ||
    typeof input.name !== "string" ||
    input.name.trim() === "" ||
    !("shape" in input)
  ) {
    return {
      ok: false,
      error: {
        code: "INVALID_PIECE",
        message: "블록 id·이름·shape가 필요합니다.",
      },
    };
  }
  const result = validateShape(input.shape);
  if (!result.ok) return result;
  return {
    ok: true,
    piece: { id: input.id, name: input.name, shape: result.shape },
  };
}

function rotated(shape: Shape): Shape {
  return Array.from({ length: shape[0].length }, (_, row) =>
    Array.from(
      { length: shape.length },
      (_, col) => shape[shape.length - 1 - col][row],
    ),
  );
}

function normalized(shape: Shape): Shape {
  let top = shape.length;
  let bottom = -1;
  let left = shape[0].length;
  let right = -1;
  shape.forEach((cells, row) => {
    cells.forEach((cell, col) => {
      if (!cell) return;
      top = Math.min(top, row);
      bottom = Math.max(bottom, row);
      left = Math.min(left, col);
      right = Math.max(right, col);
    });
  });
  return shape.slice(top, bottom + 1).map((row) => row.slice(left, right + 1));
}

function shapeKey(shape: Shape): string {
  const rows = shape.map((row) =>
    row.map((cell) => (cell ? "1" : "0")).join(""),
  );
  return `${shape.length}x${shape[0].length}:${rows.join("/")}`;
}

export function rotate90(input: unknown): ShapeResult {
  const result = validateShape(input);
  return result.ok ? { ok: true, shape: rotated(result.shape) } : result;
}

export function flipHorizontal(input: unknown): ShapeResult {
  const result = validateShape(input);
  return result.ok
    ? { ok: true, shape: result.shape.map((row) => row.slice().reverse()) }
    : result;
}

export function normalizeShape(input: unknown): ShapeResult {
  const result = validateShape(input);
  return result.ok ? { ok: true, shape: normalized(result.shape) } : result;
}

export function serializeShape(input: unknown): PieceResult<{ key: string }> {
  const result = normalizeShape(input);
  return result.ok ? { ok: true, key: shapeKey(result.shape) } : result;
}

export function getUniqueVariants(
  input: unknown,
): PieceResult<{ variants: PieceVariant[] }> {
  const result = normalizeShape(input);
  if (!result.ok) return result;
  const seen = new Set<string>();
  const variants: PieceVariant[] = [];
  for (const flipped of [false, true]) {
    let shape = result.shape.map((row) =>
      flipped ? row.slice().reverse() : row.slice(),
    );
    for (const rotation of ROTATIONS) {
      const key = shapeKey(shape);
      if (!seen.has(key)) {
        seen.add(key);
        variants.push({ shape, rotation, flipped });
      }
      shape = rotated(shape);
    }
  }
  return { ok: true, variants };
}
