import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { createHash } from "node:crypto";
import { performance } from "node:perf_hooks";
import assert from "node:assert/strict";
import { source } from "./source-loader.mjs";

// Supplied files must remain external to the repository. Only text results leave memory.
const sharp = createRequire(import.meta.resolve("next/package.json"))("sharp");
const { getInitialCatalog } = await import(
  await source("src/domain/pieces/catalog.ts")
);
const { recognizeAutomaticBoard } = await import(
  await source("src/features/recognition/automatic.ts")
);
const { recognizePanelPieces, recognizePieceCards } = await import(
  await source("src/features/recognition/panel-pieces.ts")
);
const templates = [];
const { recognizePanelAbilities, abilityRegions } = await import(
  await source("src/features/recognition/panel-abilities.ts")
);
for (const value of [
  "사용 완료",
  ...Array.from({ length: 10 }, (_, i) => String(i)),
])
  for (const family of [
    "Malgun Gothic",
    "Arial",
    "Segoe UI",
    "Impact",
    "sans-serif",
  ])
    for (const weight of [400, 700]) {
      const rendered = await sharp(
        Buffer.from(
          `<svg width="140" height="32"><rect width="140" height="32" fill="black"/><text x="4" y="24" font-family="${family}" font-size="16" font-weight="${weight}" fill="white">${value}</text></svg>`,
        ),
      )
        .removeAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });
      const { data, info } = rendered;
      templates.push({
        value,
        mask: {
          width: info.width,
          height: info.height,
          data: Uint8Array.from({ length: info.width * info.height }, (_, i) =>
            Number(data[i * info.channels] > 170),
          ),
        },
      });
      data.fill(0);
    }
if (!process.argv[2])
  throw Error(
    "Provide the original full game screenshot externally. No image is bundled.",
  );
const results = [];
for (const [index, path] of process.argv.slice(2).entries()) {
  const encoded = readFileSync(path),
    hash = createHash("sha256").update(encoded).digest("hex");
  const expected =
    index === 0
      ? "c4b41cc1962aedef6b78dcb07bcd910383c3dddeb3b9dbae4ad5a99cb61b61c7"
      : "338277921d717d894a0930eaa55a9a57971eedcc1220998b1c97b7e6a4659e8b";
  if (hash !== expected)
    throw Error("Source does not match the manually observed reference");
  let data, pixels;
  try {
    const decoded = await sharp(encoded)
      .ensureAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
    data = decoded.data;
    pixels = new Uint8ClampedArray(data);
    const frame = {
      width: decoded.info.width,
      height: decoded.info.height,
      pixels,
      timestamp: 1,
    };
    const before = createHash("sha256").update(pixels).digest("hex"),
      start = performance.now();
    const board = index === 0 ? recognizeAutomaticBoard(frame) : null;
    const result =
      index === 0 && board?.ok
        ? recognizePanelPieces(
            frame,
            board.region,
            getInitialCatalog(),
            templates,
          )
        : recognizePieceCards(
            frame,
            [
              { x: 10, y: 30, width: 93, height: 52 },
              { x: 10, y: 86, width: 93, height: 51 },
              { x: 10, y: 142, width: 93, height: 51 },
            ],
            getInitialCatalog(),
            templates,
          );
    const countRegions =
      index === 0 && board?.ok
        ? abilityRegions(board.region)
        : {
            singleCell: { x: 77, y: 285, width: 11, height: 10 },
            reroll: { x: 77, y: 312, width: 11, height: 10 },
            total: { x: 46, y: 261, width: 7, height: 11 },
          };
    const counts = recognizePanelAbilities(frame, countRegions, templates);
    results.push({
      imageHash: hash,
      dimensions: [frame.width, frame.height],
      elapsedMs: performance.now() - start,
      boardRegion: board?.ok ? board.region : null,
      pieces: result.pieces,
      counts,
      expectedCounts: [0, 0, 0],
      reasons: result.reasons,
      immutable: before === createHash("sha256").update(pixels).digest("hex"),
      expected:
        index === 0
          ? ["C_5", "empty", "empty"]
          : ["L_3", "DOUBLE_BRANCH_7", "L_3"],
    });
  } finally {
    data?.fill(0);
    pixels?.fill(0);
    encoded.fill(0);
  }
}
templates.forEach((t) => t.mask.data.fill(0));
for (const result of results) {
  assert.equal(result.immutable, true);
  assert.deepEqual(
    result.pieces.map((p) => (p.empty === true ? "empty" : p.pieceId)),
    result.expected,
  );
  assert(result.pieces.every((p) => p.status === "recognized"));
  assert.deepEqual(
    [
      result.counts.abilities.singleCell.value,
      result.counts.abilities.reroll.value,
      result.counts.total.value,
    ],
    result.expectedCounts,
  );
}
console.log(
  JSON.stringify(
    {
      at: new Date().toISOString(),
      method:
        "External supplied reference read in memory. System-font synthetic usage templates; no original crop/RGB/template storage.",
      results,
      result: "PASS",
    },
    null,
    2,
  ),
);
