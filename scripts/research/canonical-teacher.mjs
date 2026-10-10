import {
  readFileSync,
  writeFileSync,
  appendFileSync,
  existsSync,
  mkdirSync,
} from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { performance } from "node:perf_hooks";
import { format } from "prettier";
import { compileResearch } from "./runtime.mjs";

const root = resolve(fileURLToPath(new URL("../../", import.meta.url))),
  args = process.argv.slice(2),
  seen = new Set();
let output,
  configPath = resolve(root, "research/experiments/canonical-teacher-v1.json");
for (let i = 0; i < args.length; i += 2) {
  if (
    !args[i + 1] ||
    !["--output", "--protocol"].includes(args[i]) ||
    seen.has(args[i])
  )
    throw new Error("Use --output new-directory [--protocol file]");
  seen.add(args[i]);
  if (args[i] === "--output") output = resolve(args[i + 1]);
  else configPath = resolve(args[i + 1]);
}
if (!output) throw new Error("New output directory required");
if (existsSync(output))
  throw new Error("Existing output will not be overwritten");
const runtime = compileResearch(root, "src/research/canonical-teacher.ts");
const { validateCanonicalProtocol, runCanonicalDiagnostic } = await import(
  runtime.url
);
const protocol = validateCanonicalProtocol(
  JSON.parse(readFileSync(configPath)),
);
const sha = (data) => createHash("sha256").update(data).digest("hex");
const identity = {
  protocolSha256: sha(JSON.stringify(protocol)),
  configSha256: sha(readFileSync(configPath)),
  sourceHashes: runtime.hashes,
  runnerSha256: sha(readFileSync(fileURLToPath(import.meta.url))),
  lockSha256: sha(readFileSync(resolve(root, "pnpm-lock.yaml"))),
};
const metadata = {
  startedAtUtc: new Date().toISOString(),
  executor: "local-cli",
  node: process.version,
  compiler: runtime.compilerVersion,
  revision: execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: root,
    encoding: "utf8",
  }).trim(),
  workingTree: execFileSync("git", ["status", "--short"], {
    cwd: root,
    encoding: "utf8",
  }).trim(),
  memoryBefore: process.memoryUsage(),
};
const write = (name, value) =>
  writeFileSync(resolve(output, name), JSON.stringify(value, null, 2) + "\n", {
    flag: "wx",
  });
mkdirSync(output, { recursive: true });
write("protocol.json", protocol);
write("started.json", { identity, metadata });
writeFileSync(resolve(output, "calls.jsonl"), "", { flag: "wx" });
writeFileSync(resolve(output, "conditions.jsonl"), "", { flag: "wx" });
let completedCalls = 0,
  completedConditions = 0;
const start = performance.now();
try {
  const report = runCanonicalDiagnostic(
    protocol,
    () => performance.now(),
    (fixture, key, expectedHash) => {
      if (sha(key) !== expectedHash)
        throw new Error(`Historical input hash mismatch: ${fixture.id}`);
    },
    (event) => {
      if (event.type === "call") {
        appendFileSync(
          resolve(output, "calls.jsonl"),
          JSON.stringify(event.call) + "\n",
        );
        completedCalls++;
      } else {
        const { calls, ...row } = event.row;
        appendFileSync(
          resolve(output, "conditions.jsonl"),
          JSON.stringify({
            ...row,
            callSequences: calls.map((c) => c.sequence),
          }) + "\n",
        );
        completedConditions++;
        console.log(
          `Verified ${event.row.fixtureId} / ${event.row.condition} / repeat ${event.row.repetition} (${completedCalls} calls)`,
        );
      }
    },
  );
  write("report.json", {
    ...report,
    identity,
    metadata: {
      ...metadata,
      finishedAtUtc: new Date().toISOString(),
      durationMs: performance.now() - start,
      memoryAfter: process.memoryUsage(),
    },
  });
  const summary = {
    schemaVersion: 1,
    id: protocol.id,
    status: report.status,
    protocol,
    summary: report.summary,
    warmup: {
      calls: report.warmup.reduce((n, r) => n + r.calls.length, 0),
      returnedPathsVerified: report.warmup.reduce(
        (n, r) =>
          n +
          r.calls.reduce((m, c) => m + 1 + c.restored.alternatives.length, 0),
        0,
      ),
      excludedFromMeasuredSummary: true,
    },
    identity,
    limits: [
      "Fixed twenty P041 initial states: twelve generated and eight scripted, reported separately; not real event probabilities",
      "Two repetitions check determinism, not additional independent samples",
      "Canonicalization changes only research input slots and array order; original shape orientation and solveTurn best selection are preserved",
      "Identical geometric instances may change physical identity; full semantic path/evaluation/final state invariance is tested separately",
      "All per-call SearchInfo remains in raw evidence; incomplete search, restricted reroll and alternative omissions remain visible",
      "Condition latency includes key/map/legality, solve, inverse restoration and full replay; callback serialization and disk logging are separate",
      "Overall raw runner duration includes condition-summary callbacks; warmup is excluded from measured summary",
      "No new data/teacher labels, held-out test generation/evaluation, training, production integration or external state transmission",
      "Finite pilot gate is not teacher adoption, statistical benefit evidence or app response-time acceptance",
    ],
  };
  writeFileSync(
    resolve(output, "summary.json"),
    await format(JSON.stringify(summary, null, 2) + "\n", { parser: "json" }),
    { flag: "wx" },
  );
  console.log(
    JSON.stringify(
      {
        status: report.status,
        comparisons: report.summary.comparisons,
        invariance: report.summary.invariance.passed,
        repeated: report.summary.repeated.passed,
        candidateForUserReview: report.summary.candidateForUserReview,
      },
      null,
      2,
    ),
  );
  if (report.status !== "PASS") process.exitCode = 1;
} catch (error) {
  write("failure.json", {
    status: "FAIL",
    error: error.message,
    identity,
    metadata: { ...metadata, failedAtUtc: new Date().toISOString() },
    completedCalls,
    completedConditions,
  });
  throw error;
}
