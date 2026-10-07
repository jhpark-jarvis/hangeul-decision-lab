import { getInitialCatalog } from "../domain/pieces/catalog";
import { solveTurn } from "../domain/solver/solver";
import {
  DEFAULT_SOLVER_CONFIG,
  type SolverConfig,
} from "../domain/solver/types";
import { runEpisode, verifyReplay } from "./episode";
import { createSyntheticFixture } from "./fixture";
import type {
  EpisodeFixture,
  EpisodePolicy,
  EpisodeResult,
  ResearchResult,
} from "./types";

export type BenchmarkSpec = {
  schemaVersion: 1;
  id: string;
  split: "dev";
  seeds: number[];
  families: ("sparse" | "pressure")[];
  maxActions: number;
  repetitions: number;
  baselineId: string;
  configs: { id: string; solver: SolverConfig }[];
};
export const PILOT_SPEC: BenchmarkSpec = {
  schemaVersion: 1,
  id: "pilot-budget-20261007-v1",
  split: "dev",
  seeds: [17, 42, 2026],
  families: ["sparse", "pressure"],
  maxActions: 12,
  repetitions: 2,
  baselineId: "dfs-512",
  configs: [
    { id: "dfs-128", solver: { ...DEFAULT_SOLVER_CONFIG, maxNodes: 128 } },
    { id: "dfs-512", solver: { ...DEFAULT_SOLVER_CONFIG } },
  ],
};
export type SolverMeasurement = {
  latencyMs: number;
  visitedNodes: number;
  memoPrunedNodes: number;
  complete: boolean;
};
export type BenchmarkRow = {
  fixtureId: string;
  seed: number;
  family: string;
  configId: string;
  repeat: number;
  episodeWallMs: number;
  solverMeasurements: SolverMeasurement[];
  episode: EpisodeResult;
  replayValid: boolean;
};
type PolicyFactory = (
  config: SolverConfig,
  measurements: SolverMeasurement[],
  clock: () => number,
) => EpisodePolicy;
export type BenchmarkReport = {
  schemaVersion: 1;
  status: "PASS" | "FAIL";
  spec: BenchmarkSpec;
  policyFamily: "current-dfs" | "injected-test-policy";
  fixtures: EpisodeFixture[];
  rows: BenchmarkRow[];
  reproducible: boolean;
  summaries: ReturnType<typeof summarize>[];
  paired: {
    fixtureId: string;
    candidateId: string;
    actionsDelta: number;
    clearsDelta: number;
    setsDelta: number;
  }[];
};
export function validateBenchmarkSpec(
  input: unknown,
): ResearchResult<BenchmarkSpec> {
  if (!input || typeof input !== "object")
    return { ok: false, error: "Protocol required" };
  const s = input as BenchmarkSpec;
  const id = (v: unknown) => typeof v === "string" && /^[a-z0-9-]+$/.test(v);
  if (
    s.schemaVersion !== 1 ||
    !id(s.id) ||
    s.split !== "dev" ||
    !Array.isArray(s.seeds) ||
    !s.seeds.length ||
    s.seeds.length > 32 ||
    s.seeds.some((v) => !Number.isInteger(v) || v < 0 || v > 0xffffffff) ||
    new Set(s.seeds).size !== s.seeds.length ||
    !Array.isArray(s.families) ||
    !s.families.length ||
    s.families.some((v) => !["sparse", "pressure"].includes(v)) ||
    new Set(s.families).size !== s.families.length ||
    !Number.isInteger(s.maxActions) ||
    s.maxActions < 1 ||
    s.maxActions > 100 ||
    !Number.isInteger(s.repetitions) ||
    s.repetitions < 1 ||
    s.repetitions > 5 ||
    !Array.isArray(s.configs) ||
    !s.configs.length ||
    s.configs.length > 8 ||
    s.configs.some(
      (c) =>
        !c ||
        !id(c.id) ||
        !c.solver ||
        !Number.isSafeInteger(c.solver.maxNodes) ||
        c.solver.maxNodes < 1 ||
        c.solver.maxNodes > 8192 ||
        !Number.isInteger(c.solver.maxAlternatives) ||
        c.solver.maxAlternatives < 1 ||
        c.solver.maxAlternatives > 3 ||
        typeof c.solver.useMemoization !== "boolean",
    ) ||
    new Set(s.configs.map((c) => c.id)).size !== s.configs.length ||
    !s.configs.some((c) => c.id === s.baselineId) ||
    s.seeds.length *
      s.families.length *
      s.configs.length *
      s.repetitions *
      s.maxActions >
      2000
  ) {
    return {
      ok: false,
      error: "Invalid or oversized development benchmark protocol",
    };
  }
  return {
    ok: true,
    value: {
      schemaVersion: 1,
      id: s.id,
      split: "dev",
      seeds: [...s.seeds],
      families: [...s.families],
      maxActions: s.maxActions,
      repetitions: s.repetitions,
      baselineId: s.baselineId,
      configs: s.configs.map((c) => ({
        id: c.id,
        solver: {
          maxNodes: c.solver.maxNodes,
          maxAlternatives: c.solver.maxAlternatives,
          useMemoization: c.solver.useMemoization,
        },
      })),
    },
  };
}
export function nearestRank(values: number[], quantile: number): number | null {
  if (!values.length) return null;
  if (
    values.some((v) => !Number.isFinite(v)) ||
    quantile <= 0 ||
    quantile > 1 ||
    !Number.isFinite(quantile)
  )
    throw new Error("Invalid quantile data");
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.ceil(quantile * sorted.length) - 1];
}
const mean = (values: number[]) =>
  values.length ? values.reduce((a, b) => a + b, 0) / values.length : null;
export function summarize(configId: string, rows: BenchmarkRow[]) {
  const selected = rows.filter((row) => row.configId === configId);
  const valid = selected.filter(
    (row) => row.episode.ending.kind !== "error" && row.replayValid,
  );
  const decisions = valid.flatMap((row) => row.solverMeasurements);
  return {
    configId,
    episodes: selected.length,
    validEpisodes: valid.length,
    independentFixtures: new Set(selected.map((row) => row.fixtureId)).size,
    errors: selected.length - valid.length,
    endings: Object.fromEntries(
      [
        "gameover",
        "horizon",
        "policy-abstention",
        "tape-exhausted",
        "error",
      ].map((kind) => [
        kind,
        selected.filter((row) => row.episode.ending.kind === kind).length,
      ]),
    ),
    meanActions: mean(valid.map((row) => row.episode.totals.actions)),
    meanClears: mean(valid.map((row) => row.episode.totals.clearedRows)),
    meanCompletedSets: mean(
      valid.map((row) => row.episode.totals.completedSets),
    ),
    meanItems: mean(valid.map((row) => row.episode.totals.acquiredItems)),
    totalSingleCellUses: valid.reduce(
      (sum, row) => sum + row.episode.totals.singleCellUses,
      0,
    ),
    totalRerollUses: valid.reduce(
      (sum, row) => sum + row.episode.totals.rerollUses,
      0,
    ),
    meanEpisodeWallMs: mean(valid.map((row) => row.episodeWallMs)),
    solverDecisions: decisions.length,
    meanSolverLatencyMs: mean(decisions.map((d) => d.latencyMs)),
    p95SolverLatencyMs: nearestRank(
      decisions.map((d) => d.latencyMs),
      0.95,
    ),
    totalVisitedNodes: decisions.reduce((sum, d) => sum + d.visitedNodes, 0),
    totalMemoPruned: decisions.reduce((sum, d) => sum + d.memoPrunedNodes, 0),
    incompleteDecisions: decisions.filter((d) => !d.complete).length,
  };
}
export const dfsPolicy: PolicyFactory = (config, measurements, clock) => {
  const catalog = getInitialCatalog();
  return ({ state }) => {
    const start = clock();
    const solved = solveTurn(state, catalog, config);
    const elapsed = clock() - start;
    if (!Number.isFinite(elapsed) || elapsed < 0)
      throw new Error("Invalid clock");
    if (!solved.ok) throw new Error(solved.error.message);
    measurements.push({
      latencyMs: elapsed,
      visitedNodes: solved.result.search.visitedNodes,
      memoPrunedNodes: solved.result.search.memoPrunedNodes,
      complete: solved.result.search.searchComplete,
    });
    return {
      action: solved.result.actions[0] ?? null,
      search: solved.result.search,
    };
  };
};

export function runBenchmark(
  input: unknown,
  clock: () => number,
  factory: PolicyFactory = dfsPolicy,
): ResearchResult<BenchmarkReport> {
  const validated = validateBenchmarkSpec(input);
  if (!validated.ok) return validated;
  const spec = validated.value;
  const fixtures: EpisodeFixture[] = [];
  for (const family of spec.families)
    for (const seed of spec.seeds) {
      const generated = createSyntheticFixture(seed, family, spec.maxActions);
      if (!generated.ok) return generated;
      fixtures.push(generated.value);
    }
  const rows: BenchmarkRow[] = [];
  for (let repeat = 0; repeat < spec.repetitions; repeat++) {
    for (let index = 0; index < fixtures.length; index++) {
      const fixture = fixtures[index];
      const offset = (repeat + index) % spec.configs.length;
      for (let c = 0; c < spec.configs.length; c++) {
        const config = spec.configs[(c + offset) % spec.configs.length];
        const measurements: SolverMeasurement[] = [];
        const policy = factory(config.solver, measurements, clock);
        const start = clock();
        const played = runEpisode(fixture, policy);
        const elapsed = clock() - start;
        if (!played.ok) return played;
        if (!Number.isFinite(elapsed) || elapsed < 0)
          return { ok: false, error: "Invalid episode clock" };
        rows.push({
          fixtureId: fixture.id,
          seed: fixture.seed,
          family: fixture.id.split("-")[0],
          configId: config.id,
          repeat,
          episodeWallMs: elapsed,
          solverMeasurements: measurements,
          episode: played.value,
          replayValid: verifyReplay(fixture, played.value).ok,
        });
      }
    }
  }
  const traces = new Map<string, string>();
  let reproducible = true;
  for (const row of rows) {
    const key = `${row.fixtureId}:${row.configId}`;
    const trace = JSON.stringify(row.episode);
    if (traces.has(key) && traces.get(key) !== trace) reproducible = false;
    traces.set(key, trace);
  }
  const paired: BenchmarkReport["paired"] = [];
  for (const fixture of fixtures) {
    const baseline = rows.find(
      (row) =>
        row.fixtureId === fixture.id &&
        row.configId === spec.baselineId &&
        row.repeat === 0,
    )!;
    for (const config of spec.configs.filter((c) => c.id !== spec.baselineId)) {
      const candidate = rows.find(
        (row) =>
          row.fixtureId === fixture.id &&
          row.configId === config.id &&
          row.repeat === 0,
      )!;
      if (baseline.replayValid && candidate.replayValid)
        paired.push({
          fixtureId: fixture.id,
          candidateId: config.id,
          actionsDelta:
            candidate.episode.totals.actions - baseline.episode.totals.actions,
          clearsDelta:
            candidate.episode.totals.clearedRows -
            baseline.episode.totals.clearedRows,
          setsDelta:
            candidate.episode.totals.completedSets -
            baseline.episode.totals.completedSets,
        });
    }
  }
  return {
    ok: true,
    value: {
      schemaVersion: 1,
      status:
        reproducible &&
        rows.every(
          (row) => row.replayValid && row.episode.ending.kind !== "error",
        )
          ? "PASS"
          : "FAIL",
      spec,
      policyFamily:
        factory === dfsPolicy ? "current-dfs" : "injected-test-policy",
      fixtures,
      rows,
      reproducible,
      summaries: spec.configs.map((config) => summarize(config.id, rows)),
      paired,
    },
  };
}
