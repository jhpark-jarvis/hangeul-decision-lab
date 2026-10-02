import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import assert from "node:assert/strict";
import { chromium } from "@playwright/test";
import { source } from "./source-loader.mjs";

// Original images are decoded only in isolated local browser memory, never saved.
const modules = Object.fromEntries(
  await Promise.all(
    [
      ["fonts", "src/features/recognition/panel-fonts.ts"],
      ["game", "src/features/recognition/automatic-game.ts"],
      ["pieces", "src/features/recognition/panel-pieces.ts"],
      ["counts", "src/features/recognition/panel-abilities.ts"],
      ["catalog", "src/domain/pieces/catalog.ts"],
    ].map(async ([key, path]) => [key, await source(path)]),
  ),
);
const results = [];
if (process.argv.length !== 4)
  throw Error("Supply both original references externally.");
for (const channel of ["chrome", "msedge"]) {
  const browser = await chromium.launch({ channel, headless: true });
  try {
    const page = await browser.newPage();
    await page.route("**/*", (route) =>
      new URL(route.request().url()).origin === "http://127.0.0.1:45001"
        ? route.continue()
        : route.abort(),
    );
    await page.goto("http://127.0.0.1:45001/");
    for (const [index, path] of process.argv.slice(2).entries()) {
      const encoded = readFileSync(path);
      try {
        const imageHash = createHash("sha256").update(encoded).digest("hex");
        assert.equal(
          imageHash,
          index === 0
            ? "c4b41cc1962aedef6b78dcb07bcd910383c3dddeb3b9dbae4ad5a99cb61b61c7"
            : "338277921d717d894a0930eaa55a9a57971eedcc1220998b1c97b7e6a4659e8b",
        );
        const result = await page.evaluate(
          async ({ bytes, modules, index }) => {
            const encoded = new Uint8Array(bytes),
              bitmap = await createImageBitmap(
                new Blob([encoded], { type: "image/png" }),
              ),
              canvas = document.createElement("canvas");
            let pixels,
              templates = [];
            const fonts = await import(modules.fonts);
            try {
              canvas.width = bitmap.width;
              canvas.height = bitmap.height;
              const ctx = canvas.getContext("2d", { willReadFrequently: true });
              ctx.drawImage(bitmap, 0, 0);
              pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
              const frame = {
                width: canvas.width,
                height: canvas.height,
                timestamp: 1,
                pixels,
              };
              const digest = async () =>
                Array.from(
                  new Uint8Array(await crypto.subtle.digest("SHA-256", pixels)),
                )
                  .map((v) => v.toString(16).padStart(2, "0"))
                  .join("");
              const before = await digest(),
                start = performance.now();
              templates = fonts.createPanelTextTemplates();
              let pieces,
                abilities,
                total = null,
                board = null;
              if (index === 0) {
                const game = await import(modules.game),
                  found = game.recognizeAutomaticGame(frame, templates);
                if (!found.ok) throw Error(found.reason);
                pieces = found.result.pieces;
                abilities = found.result.abilities;
                board = {
                  region: found.region,
                  unresolved: found.result.board
                    .flat()
                    .filter((c) => c.occupied === null)
                    .map((c) => [c.row, c.col]),
                  source: found.result.source.kind,
                };
              } else {
                const helper = await import(modules.pieces),
                  catalog = await import(modules.catalog),
                  counts = await import(modules.counts);
                pieces = helper.recognizePieceCards(
                  frame,
                  [
                    { x: 10, y: 30, width: 93, height: 52 },
                    { x: 10, y: 86, width: 93, height: 51 },
                    { x: 10, y: 142, width: 93, height: 51 },
                  ],
                  catalog.getInitialCatalog(),
                  templates,
                ).pieces;
                const read = counts.recognizePanelAbilities(
                  frame,
                  {
                    singleCell: { x: 77, y: 285, width: 11, height: 10 },
                    reroll: { x: 77, y: 312, width: 11, height: 10 },
                    total: { x: 46, y: 261, width: 7, height: 11 },
                  },
                  templates,
                );
                abilities = read.abilities;
                total = read.total;
              }
              return {
                dimensions: [frame.width, frame.height],
                pieces,
                abilities,
                total,
                board,
                elapsedMs: performance.now() - start,
                immutable: before === (await digest()),
                templateCount: templates.length,
              };
            } finally {
              pixels?.fill(0);
              encoded.fill(0);
              fonts.releasePanelTextTemplates(templates);
              canvas.width = 0;
              canvas.height = 0;
              bitmap.close();
            }
          },
          { bytes: Array.from(encoded), modules, index },
        );
        const expected =
          index === 0
            ? ["C_5", "empty", "empty"]
            : ["L_3", "DOUBLE_BRANCH_7", "L_3"];
        assert.deepEqual(
          result.pieces.map((p) => (p.empty ? "empty" : p.pieceId)),
          expected,
        );
        assert(result.pieces.every((p) => p.status === "recognized"));
        assert.deepEqual(
          [result.abilities.singleCell.value, result.abilities.reroll.value],
          [0, 0],
        );
        assert(result.immutable);
        results.push({
          channel,
          version: browser.version(),
          imageHash,
          ...result,
          result: "PASS",
        });
      } finally {
        encoded.fill(0);
      }
    }
  } finally {
    await browser.close();
  }
}
console.log(
  JSON.stringify(
    {
      at: new Date().toISOString(),
      method:
        "System-font Canvas templates and external supplied PNGs, isolated local browser memory only; no native screen chooser or image storage.",
      results,
      result: "PASS",
    },
    null,
    2,
  ),
);
