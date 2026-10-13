/** Geometry from an already frozen legal key; no state transition or teacher. */
export function candidateFootprint(key: string): number[][] {
  const value: unknown = JSON.parse(key);
  if (!Array.isArray(value) || typeof value[0] !== "string")
    throw new Error("Footprint action key");
  const bits = Array.from({ length: 16 }, () => Array<number>(10).fill(0));
  const coordinate = (n: unknown, max: number): n is number =>
    typeof n === "number" && Number.isInteger(n) && n >= 0 && n < max;
  const target = () => {
    if (typeof value[1] !== "string" || !value[1] || !coordinate(value[2], 3))
      throw new Error("Footprint target");
  };
  if (value[0] === "reroll") {
    if (value.length !== 3) throw new Error("Footprint reroll key");
    target();
    return bits;
  }
  if (value[0] === "single-cell") {
    if (
      value.length !== 3 ||
      !coordinate(value[1], 16) ||
      !coordinate(value[2], 10)
    )
      throw new Error("Footprint point coordinate");
    bits[value[1]][value[2]] = 1;
    return bits;
  }
  if (value[0] !== "place-piece" || value.length !== 7)
    throw new Error("Footprint kind/length");
  target();
  const [, , , piece, row, col, shape] = value;
  if (
    typeof piece !== "string" ||
    !piece ||
    !coordinate(row, 16) ||
    !coordinate(col, 10) ||
    !Array.isArray(shape) ||
    shape.length < 1 ||
    shape.length > 5 ||
    !Array.isArray(shape[0]) ||
    shape[0].length < 1 ||
    shape[0].length > 5 ||
    shape.some(
      (r: unknown) =>
        !Array.isArray(r) ||
        r.length !== shape[0].length ||
        r.some((cell: unknown) => typeof cell !== "boolean"),
    ) ||
    row + shape.length > 16 ||
    col + shape[0].length > 10 ||
    !shape.some((r: boolean[]) => r.some(Boolean))
  )
    throw new Error("Footprint shape/bounds");
  shape.forEach((r: boolean[], dr: number) =>
    r.forEach((cell, dc) => {
      bits[row + dr][col + dc] = Number(cell);
    }),
  );
  return bits;
}

/** Compact private storage; rows are expanded to N×1×16×10 tensors in Python. */
export function encodeFootprints(keys: string[], budget: number) {
  if (
    !Number.isInteger(budget) ||
    budget < 0 ||
    keys.length < 1 ||
    keys.length > 4096 ||
    keys.length > budget ||
    new Set(keys).size !== keys.length
  )
    throw new Error("Footprint ordered list/budget");
  return {
    footprints: keys.map((key) => candidateFootprint(key).flat().join("")),
    candidates: keys.length,
  };
}
