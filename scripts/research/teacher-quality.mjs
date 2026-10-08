import { readFileSync, writeFileSync, existsSync, mkdirSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { performance } from "node:perf_hooks";
import { format } from "prettier";
import { compileResearch } from "./runtime.mjs";
const root = resolve(fileURLToPath(new URL("../../", import.meta.url)));
let protocolPath = resolve(
    root,
    "research/experiments/teacher-quality-v1.json",
  ),
  output;
const args = process.argv.slice(2);
for (let i = 0; i < args.length; i += 2) {
  if (!args[i + 1] || !["--protocol", "--output"].includes(args[i]))
    throw new Error("Use --output new-directory [--protocol file]");
  if (args[i] === "--protocol") protocolPath = resolve(args[i + 1]);
  else output = resolve(args[i + 1]);
}
if (!output) throw new Error("New output directory required");
if (existsSync(output))
  throw new Error("Existing output will not be overwritten");
const protocol = JSON.parse(readFileSync(protocolPath));
const runtime = compileResearch(root, "src/research/teacher-quality.ts");
const { validateQualityProtocol, runQualityPreflight } = await import(
  runtime.url
);
validateQualityProtocol(protocol);
const sha = (d) => createHash("sha256").update(d).digest("hex");
const write = (name, value) =>
  writeFileSync(resolve(output, name), JSON.stringify(value, null, 2) + "\n", {
    flag: "wx",
  });
const startedAtUtc = new Date().toISOString(),
  start = performance.now(),
  memoryBefore = process.memoryUsage();
mkdirSync(output, { recursive: true });
write("protocol.json", protocol);
try {
  const report = runQualityPreflight(protocol);
  const identity = {
    protocolSha256: sha(JSON.stringify(protocol)),
    sourceHashes: runtime.hashes,
    runnerSha256: sha(readFileSync(fileURLToPath(import.meta.url))),
    lockSha256: sha(readFileSync(resolve(root, "pnpm-lock.yaml"))),
  };
  const metadata = {
    startedAtUtc,
    finishedAtUtc: new Date().toISOString(),
    durationMs: performance.now() - start,
    node: process.version,
    compiler: runtime.compilerVersion,
    memoryBefore,
    memoryAfter: process.memoryUsage(),
    revision: execFileSync("git", ["rev-parse", "HEAD"], {
      cwd: root,
      encoding: "utf8",
    }).trim(),
    workingTree: execFileSync("git", ["status", "--short"], {
      cwd: root,
      encoding: "utf8",
    }).trim(),
  };
  write("report.json", { ...report, identity, metadata });
  const summary = {
    schemaVersion: 1,
    id: protocol.id,
    status: report.status,
    experimentStatus: protocol.status,
    protocol,
    summary: report.summary,
    references: report.rows.map((r) => ({
      case: r.fixture.id,
      complete: r.reference.complete,
      visitedNodes: r.reference.visitedNodes,
      leaves: r.reference.leaves,
      optimalFirstActionCount: r.reference.optimalFirstActions.length,
      comparisons: r.comparisons.map((c) => ({
        memo: c.useMemoization,
        complete: c.result.search.searchComplete,
        visitedNodes: c.result.search.visitedNodes,
        scope: c.result.search.scope,
      })),
      transition: r.transition
        ? {
            clearedRows: r.transition.info.clearedRows.length,
            acquiredItems: r.transition.info.acquiredItems.length,
            retainedItems: r.transition.info.retainedItems.length,
            spentAbility: r.transition.info.spentAbility,
            pendingReroll: r.transition.state.pendingReroll !== null,
          }
        : null,
    })),
    identity,
    limits: [
      "Scripted preflight only; proposed train/dev matrix and reserved test are not executed",
      "Ordinary-only complete search shares domain transitions and evaluator; not an independent rules oracle",
      "Equal path evaluation within a finite current-turn tree does not prove equal future survival",
      "Node budgets are not equal wall time; current synchronous DFS has no cooperative deadline",
      "Coverage cases demonstrate mechanics, not representative frequency or policy benefit",
      "No model training, new held-out evaluation or production changes",
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
        experimentStatus: protocol.status,
        summary: report.summary,
        durationMs: metadata.durationMs,
      },
      null,
      2,
    ),
  );
} catch (error) {
  write("failure.json", { status: "FAIL", error: error.message });
  throw error;
}
