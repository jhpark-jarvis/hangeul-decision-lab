import { mkdirSync, writeFileSync } from "node:fs";
import { cpus, platform, release } from "node:os";
import { performance } from "node:perf_hooks";
import { describe, expect, it } from "vitest";
import { createEmptyBoard } from "../../src/domain/board/board";
import { getInitialCatalog } from "../../src/domain/pieces/catalog";
import { solveTurn } from "../../src/domain/solver/solver";
import { DEFAULT_SOLVER_CONFIG } from "../../src/domain/solver/types";
import { game, instance } from "../game/fixtures";
import { lineTurn, rescueTurn, success } from "./turn-fixtures";
import { singleRescue, reacquireRescue } from "./ability-fixtures";
import { staggeredBoard } from "./turn-fixtures";

const patterned = createEmptyBoard().map((row, r) =>
  row.map((_, c) => (r * 10 + c) % 7 < 2),
);
const pieces = [
  instance(0, "MIEUM"),
  instance(1, "LINE_3"),
  instance(2, "L_3"),
];
const fixtures = [
  { name: "empty-three-shapes", state: game({ remainingPieces: pieces }) },
  {
    name: "patterned-medium",
    state: game({ board: patterned, remainingPieces: pieces }),
  },
  { name: "dense-line-clear", state: lineTurn() },
  { name: "blocked-piece-rescue", state: rescueTurn() },
  { name: "single-cell-clear-rescue", state: singleRescue() },
  { name: "two-single-cell-rescue", state: singleRescue(2) },
  { name: "reacquisition-between-pieces", state: reacquireRescue() },
  {
    name: "reroll-input-wait",
    state: game({
      board: staggeredBoard(),
      remainingPieces: [instance(2, "MIEUM")],
      abilities: { reroll: 1, singleCell: 0 },
    }),
  },
];

describe("representative solver resource samples", () => {
  it("keeps ordinary and ability cases within the selected node budget", () => {
    const benchmark = process.env.SOLVER_BENCHMARK === "1";
    const catalog = getInitialCatalog();
    const results = fixtures.map(({ name, state }) => {
      if (benchmark) success(solveTurn(state, catalog)); // One warmup per fixture.
      const samples = Array.from({ length: benchmark ? 3 : 1 }, () => {
        const before = process.memoryUsage();
        const start = performance.now();
        const solution = success(solveTurn(state, catalog)).result;
        const elapsedMs = performance.now() - start;
        const after = process.memoryUsage();
        expect(solution.search.visitedNodes).toBeLessThanOrEqual(
          DEFAULT_SOLVER_CONFIG.maxNodes,
        );
        expect(solution.search.memoEntries).toBeLessThanOrEqual(
          solution.search.visitedNodes,
        );
        expect(solution.alternatives.length).toBeLessThanOrEqual(2);
        return {
          elapsedMs,
          heapUsedBefore: before.heapUsed,
          heapUsedAfter: after.heapUsed,
          heapDelta: after.heapUsed - before.heapUsed,
          rssBefore: before.rss,
          rssAfter: after.rss,
          search: solution.search,
          allCurrentPiecesPlaced: solution.evaluation.allCurrentPiecesPlaced,
          usedAbilities: solution.usedAbilities,
        };
      });
      return { name, state, samples };
    });
    if (benchmark) {
      mkdirSync(".next", { recursive: true });
      writeFileSync(
        ".next/solver-benchmark.json",
        JSON.stringify(
          {
            measuredAtUtc: new Date().toISOString(),
            executor: "Codex / Vitest single worker",
            environment: {
              node: process.version,
              platform: platform(),
              osRelease: release(),
              cpuModel: cpus()[0].model,
              logicalCpuCount: cpus().length,
            },
            config: DEFAULT_SOLVER_CONFIG,
            catalog,
            definition:
              "solveTurn ordinary probe plus ability DFS wall time; one warmup then three samples per synthetic fixture. Heap/RSS are before/after observations, not peaks or memory acceptance limits. No browser or future random pieces/reroll outcome.",
            targetMs: 3000,
            verdict: results.every((entry) =>
              entry.samples.every((sample) => sample.elapsedMs <= 3000),
            )
              ? "PASS"
              : "FAIL",
            results,
          },
          null,
          2,
        ) + "\n",
      );
    }
  }, 120_000); // Batch runtime is separate from the per-sample 3s acceptance target.
});
