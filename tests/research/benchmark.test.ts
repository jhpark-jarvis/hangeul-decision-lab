import { describe, expect, it } from "vitest";
import {
  dfsPolicy,
  nearestRank,
  PILOT_SPEC,
  runBenchmark,
  summarize,
  validateBenchmarkSpec,
  type BenchmarkRow,
  type BenchmarkSpec,
} from "../../src/research/benchmark";
import type { ResearchResult } from "../../src/research/types";

function value<T>(r: ResearchResult<T>): T {
  if (!r.ok) throw new Error(r.error);
  return r.value;
}
const spec = (): BenchmarkSpec => ({
  ...structuredClone(PILOT_SPEC),
  seeds: [17],
  families: ["sparse"],
  maxActions: 1,
  repetitions: 2,
});
const tick = () => {
  let time = 0;
  return () => ++time;
};
const first =
  () =>
  ({ legalActions }: Parameters<ReturnType<typeof dfsPolicy>>[0]) => ({
    action: legalActions[0] ?? null,
  });

describe("synthetic benchmark protocol and evidence", () => {
  it("bounds the protocol and rejects duplicates, missing baseline, invalid budgets and oversized matrices", () => {
    expect(validateBenchmarkSpec(PILOT_SPEC).ok).toBe(true);
    for (const bad of [
      null,
      {},
      { ...spec(), split: "test" },
      { ...spec(), seeds: [17, 17] },
      { ...spec(), seeds: [NaN] },
      { ...spec(), families: ["actual-game"] },
      { ...spec(), families: ["sparse", "sparse"] },
      { ...spec(), baselineId: "absent" },
      { ...spec(), repetitions: 0 },
      { ...spec(), maxActions: 0 },
      { ...spec(), configs: [spec().configs[0], spec().configs[0]] },
      {
        ...spec(),
        configs: [
          {
            id: "bad",
            solver: {
              maxNodes: 9000,
              maxAlternatives: 3,
              useMemoization: true,
            },
          },
        ],
      },
      {
        ...PILOT_SPEC,
        seeds: Array.from({ length: 32 }, (_, i) => i),
        maxActions: 100,
        repetitions: 5,
      },
    ])
      expect(validateBenchmarkSpec(bad).ok).toBe(false);
    const detached = value(validateBenchmarkSpec(PILOT_SPEC));
    detached.configs[0].solver.maxNodes = 1;
    expect(PILOT_SPEC.configs[0].solver.maxNodes).toBe(128);
  });
  it("rotates configuration order, pairs identical fixtures and checks deterministic repeats separately from timing", () => {
    const report = value(runBenchmark(spec(), tick(), first));
    expect(report).toMatchObject({
      status: "PASS",
      reproducible: true,
      policyFamily: "injected-test-policy",
    });
    expect(report.rows.map((r) => r.configId)).toEqual([
      "dfs-128",
      "dfs-512",
      "dfs-512",
      "dfs-128",
    ]);
    expect(report.fixtures).toHaveLength(1);
    expect(report.rows.map((r) => r.episode.events[0].before)).toEqual(
      Array(4).fill(report.rows[0].episode.events[0].before),
    );
    expect(report.paired).toEqual([
      {
        fixtureId: report.fixtures[0].id,
        candidateId: "dfs-128",
        actionsDelta: 0,
        clearsDelta: 0,
        setsDelta: 0,
      },
    ]);
    expect(report.summaries[0]).toMatchObject({
      episodes: 2,
      independentFixtures: 1,
      validEpisodes: 2,
      solverDecisions: 0,
    });
  });
  it("records real bounded DFS diagnostics without exposing the fixture tape to the policy", () => {
    const tiny = spec();
    tiny.repetitions = 1;
    tiny.configs = [
      {
        id: "dfs-8",
        solver: { maxNodes: 8, maxAlternatives: 1, useMemoization: true },
      },
    ];
    tiny.baselineId = "dfs-8";
    const report = value(runBenchmark(tiny, tick()));
    expect(report.policyFamily).toBe("current-dfs");
    expect(report.status).toBe("PASS");
    expect(report.rows[0].solverMeasurements[0]).toMatchObject({
      latencyMs: 1,
      complete: false,
    });
    expect(
      report.rows[0].solverMeasurements[0].visitedNodes,
    ).toBeLessThanOrEqual(8);
    expect(report.summaries[0].solverDecisions).toBe(1);
  });
  it("marks diverging repeated traces FAIL while preserving each replay and all raw rows", () => {
    let invocation = 0;
    const report = value(
      runBenchmark(spec(), tick(), () => {
        const choose = invocation++ === 3 ? 1 : 0;
        return ({ legalActions }) => ({ action: legalActions[choose] });
      }),
    );
    expect(report.status).toBe("FAIL");
    expect(report.reproducible).toBe(false);
    expect(report.rows).toHaveLength(4);
    expect(report.rows.every((r) => r.replayValid)).toBe(true);
  });
  it("reports policy exceptions as errors instead of gameover and excludes them from performance denominators", () => {
    const report = value(
      runBenchmark(spec(), tick(), () => () => {
        throw new Error("private error");
      }),
    );
    expect(report.status).toBe("FAIL");
    expect(report.paired).toEqual([]);
    expect(report.summaries[0]).toMatchObject({
      episodes: 2,
      validEpisodes: 0,
      errors: 2,
      meanActions: null,
      meanSolverLatencyMs: null,
      endings: { error: 2, gameover: 0 },
    });
  });
  it("uses nearest-rank p95 and valid episode/decision denominators; preserves horizon versus abstention", () => {
    expect(nearestRank([9, 1, 7, 3], 0.95)).toBe(9);
    expect(nearestRank([9, 1, 7, 3], 0.5)).toBe(3);
    expect(nearestRank([], 0.95)).toBeNull();
    expect(() => nearestRank([NaN], 0.95)).toThrow();
    const report = value(runBenchmark(spec(), tick(), first));
    const rows: BenchmarkRow[] = structuredClone(
      report.rows.filter((r) => r.configId === "dfs-128"),
    );
    rows[0].episode.totals = {
      ...rows[0].episode.totals,
      actions: 4,
      clearedRows: 3,
      completedSets: 1,
    };
    rows[1].episode.ending = { kind: "policy-abstention", detail: "" };
    rows[1].episode.totals = {
      ...rows[1].episode.totals,
      actions: 0,
      clearedRows: 1,
      completedSets: 0,
    };
    rows[0].solverMeasurements = [1, 3, 7, 9].map((latencyMs) => ({
      latencyMs,
      visitedNodes: 8,
      memoPrunedNodes: 2,
      complete: false,
    }));
    const summary = summarize("dfs-128", rows);
    expect(summary).toMatchObject({
      meanActions: 2,
      meanClears: 2,
      meanCompletedSets: 0.5,
      solverDecisions: 4,
      meanSolverLatencyMs: 5,
      p95SolverLatencyMs: 9,
      totalVisitedNodes: 32,
      totalMemoPruned: 8,
      incompleteDecisions: 4,
      endings: { horizon: 1, "policy-abstention": 1 },
    });
    expect(runBenchmark(spec(), () => NaN, first).ok).toBe(false);
  });
});
