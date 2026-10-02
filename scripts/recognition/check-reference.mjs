import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { createRequire } from "node:module";
import { performance } from "node:perf_hooks";
import { createHash } from "node:crypto";
import assert from "node:assert/strict";
import ts from "typescript";

// Optional local evidence tool. Uses Next's existing image decoder, never the app runtime.
// Reads the supplied reference in memory; emits geometry/boolean results only.
const sharp = createRequire(import.meta.resolve("next/package.json"))("sharp");
const modules = new Map();
async function source(file) {
  if (modules.has(file)) return modules.get(file);
  let code = ts.transpileModule(readFileSync(file, "utf8"), {
    compilerOptions: {
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2022,
    },
  }).outputText;
  for (const match of [...code.matchAll(/from "(\.[^"]+)"/g)]) {
    const url = await source(resolve(dirname(file), match[1] + ".ts"));
    code = code.replace(match[0], `from "${url}"`);
  }
  const url =
    "data:text/javascript;base64," + Buffer.from(code).toString("base64");
  modules.set(file, url);
  return url;
}
if (!process.argv[2])
  throw new Error(
    "Provide the original reference image locally (V025). No image is bundled.",
  );
const { recognizeAutomaticBoard } = await import(
  await source(resolve("src/features/recognition/automatic.ts"))
);
const { referenceRows } = await import(
  await source(resolve("tests/recognition/automatic-fixture.ts"))
);
const encoded = readFileSync(process.argv[2]);
let data, pixels;
try {
  const imageHash = createHash("sha256").update(encoded).digest("hex");
  assert.equal(
    imageHash,
    "c4b41cc1962aedef6b78dcb07bcd910383c3dddeb3b9dbae4ad5a99cb61b61c7",
    "Reference version does not match the manual oracle",
  );
  const decoded = await sharp(encoded)
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  data = decoded.data;
  pixels = new Uint8ClampedArray(data);
  const frame = {
    width: decoded.info.width,
    height: decoded.info.height,
    timestamp: 1,
    pixels,
  };
  const before = createHash("sha256").update(pixels).digest("hex");
  const start = performance.now(),
    result = recognizeAutomaticBoard(frame),
    elapsedMs = performance.now() - start;
  assert.equal(result.ok, true);
  const compared = result.result.board.flat().map((c) => ({
    row: c.row,
    col: c.col,
    expected: referenceRows[c.row][c.col] === "1",
    actual: c.occupied,
  }));
  const mismatches = compared.filter(
    (c) => c.actual !== null && c.actual !== c.expected,
  );
  const unresolved = compared.filter((c) => c.actual === null);
  const immutable =
    before === createHash("sha256").update(pixels).digest("hex");
  assert.equal(immutable, true);
  assert.deepEqual(mismatches, []);
  assert.equal(compared.filter((c) => c.actual === true).length, 49);
  assert.deepEqual(
    unresolved.map((c) => [c.row, c.col]),
    [
      [1, 0],
      [12, 7],
    ],
  );
  console.log(
    JSON.stringify(
      {
        imageHash,
        dimensions: { width: frame.width, height: frame.height },
        elapsedMs,
        immutable,
        region: result.region,
        recognized: compared.length - unresolved.length,
        occupied: 49,
        expectedOccupied: 49,
        unresolved,
        mismatches,
      },
      null,
      2,
    ),
  );
} finally {
  pixels?.fill(0);
  data?.fill(0);
  encoded.fill(0);
}
