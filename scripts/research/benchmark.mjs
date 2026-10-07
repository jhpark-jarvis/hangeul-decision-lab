import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { performance } from "node:perf_hooks";
import { cpus, platform, release } from "node:os";
import { compileResearch } from "./runtime.mjs";

const root = resolve(fileURLToPath(new URL("../../", import.meta.url)));
const args = process.argv.slice(2);
let protocolPath, outputPath;
for (let i = 0; i < args.length; i += 2) {
  if (!["--protocol", "--output"].includes(args[i]) || !args[i + 1])
    throw new Error("Use --protocol file and/or --output directory");
  if (args[i] === "--protocol") protocolPath = resolve(args[i + 1]);
  else outputPath = resolve(args[i + 1]);
}
outputPath ??= resolve(
  root,
  ".research-output/runs",
  new Date().toISOString().replace(/[:.]/g, "-"),
);
if (existsSync(outputPath))
  throw new Error(
    "Output already exists; prior experiment evidence will not be overwritten",
  );
const loadStart = performance.now();
const runtime = compileResearch(root, "src/research/benchmark.ts");
const { PILOT_SPEC, validateBenchmarkSpec, runBenchmark, dfsPolicy } =
  await import(runtime.url);
const fixtureRuntime = compileResearch(root, "src/research/fixture.ts");
const { createSyntheticFixture } = await import(fixtureRuntime.url);
const spec = validateBenchmarkSpec(
  protocolPath ? JSON.parse(readFileSync(protocolPath, "utf8")) : PILOT_SPEC,
);
if (!spec.ok) throw new Error(spec.error);
const moduleLoadMs = performance.now() - loadStart;
// One solve per configuration on the same separate seed-zero fixture; never counted in summaries.
const warmFixture = createSyntheticFixture(0, "sparse", 1);
if (!warmFixture.ok) throw new Error(warmFixture.error);
const warmStart = performance.now();
for (const config of spec.value.configs)
  dfsPolicy(config.solver, [], () => performance.now())({
    state: warmFixture.value.initialState,
    legalActions: [],
    actionIndex: 0,
  });
const warmupMs = performance.now() - warmStart;
const before = process.memoryUsage();
const startedAtUtc = new Date().toISOString();
console.log(
  `Synthetic dev benchmark: ${spec.value.id}; ${spec.value.seeds.length * spec.value.families.length * spec.value.configs.length * spec.value.repetitions} episodes`,
);
const measured = runBenchmark(spec.value, () => performance.now());
const after = process.memoryUsage();
if (!measured.ok) throw new Error(measured.error);
const hash = (data) => createHash("sha256").update(data).digest("hex");
const git = (args) =>
  execFileSync("git", args, { cwd: root, encoding: "utf8" }).trim();
const report = {
  ...measured.value,
  execution: {
    startedAtUtc,
    finishedAtUtc: new Date().toISOString(),
    executor: "Codex / local Node CLI",
    node: process.version,
    compiler: runtime.compilerVersion,
    platform: platform(),
    osRelease: release(),
    cpu: cpus()[0]?.model,
    cpuCount: cpus().length,
    parallelWorkers: 1,
    revision: git(["rev-parse", "HEAD"]),
    worktree: git(["status", "--porcelain=v1"]),
    moduleLoadMs,
    warmupMs,
    warmupDecisionsPerConfig: 1,
    memoryBefore: before,
    memoryAfter: after,
    definitions: {
      latency:
        "solveTurn entry to return, including its candidates/evaluation/search; excludes simulator legal enumeration/environment/replay/loader",
      episodeWall:
        "episode policy + simulator legality/environment/transition; excludes replay/loader",
      memory:
        "Whole Node process before/after benchmark, including loaded compiler/cache and reports; NOT peak or browser heap acceptance",
      p95: "Nearest-rank ceil(0.95*N), all valid episode solve attempts including policy abstention",
      independentSample:
        "Seed/family fixture; deterministic repeats are not independent samples",
      assumptions:
        "Synthetic equal-weight catalog pieces/current-kind-excluded reroll; initial items only; no actual event probability/counter/new-icon model",
    },
  },
  integrity: {
    sourceHashes: runtime.hashes,
    runnerHashes: Object.fromEntries(
      [
        "scripts/research/benchmark.mjs",
        "scripts/research/runtime.mjs",
        "package.json",
        "pnpm-lock.yaml",
      ].map((name) => [name, hash(readFileSync(resolve(root, name)))]),
    ),
    fixtureHashes: Object.fromEntries(
      measured.value.fixtures.map((f) => [f.id, hash(JSON.stringify(f))]),
    ),
    traceHashes: measured.value.rows.map((row) => ({
      fixtureId: row.fixtureId,
      configId: row.configId,
      repeat: row.repeat,
      sha256: hash(JSON.stringify(row.episode)),
    })),
    protocolSha256: hash(JSON.stringify(spec.value)),
  },
};
mkdirSync(outputPath, { recursive: true });
writeFileSync(
  resolve(outputPath, "protocol.json"),
  JSON.stringify(spec.value, null, 2) + "\n",
);
writeFileSync(
  resolve(outputPath, "report.json"),
  JSON.stringify(report, null, 2) + "\n",
);
const summaries = report.summaries;
const columns = [
  "configId",
  "episodes",
  "validEpisodes",
  "independentFixtures",
  "errors",
  "meanActions",
  "meanClears",
  "meanCompletedSets",
  "meanItems",
  "solverDecisions",
  "meanSolverLatencyMs",
  "p95SolverLatencyMs",
  "totalVisitedNodes",
  "incompleteDecisions",
];
writeFileSync(
  resolve(outputPath, "summary.csv"),
  [
    columns.join(","),
    ...summaries.map((s) => columns.map((key) => s[key] ?? "").join(",")),
  ].join("\n") + "\n",
);
console.log(
  JSON.stringify(
    {
      status: report.status,
      reproducible: report.reproducible,
      output: outputPath,
      summaries,
    },
    null,
    2,
  ),
);
if (report.status !== "PASS") process.exitCode = 1;
