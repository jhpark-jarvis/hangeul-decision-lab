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
const root = resolve(fileURLToPath(new URL("../../", import.meta.url)));
const args = process.argv.slice(2);
if (args.length !== 2 || args[0] !== "--output" || !args[1])
  throw new Error("Use --output new-directory");
const output = resolve(args[1]);
if (existsSync(output))
  throw new Error("Existing output will not be overwritten");
const runtime = compileResearch(root, "src/research/item-coverage.ts");
const { ITEM_COVERAGE_PROTOCOL, validateItemProtocol, runItemCoverage } =
  await import(runtime.url);
const protocol = validateItemProtocol(ITEM_COVERAGE_PROTOCOL);
const sha = (d) => createHash("sha256").update(d).digest("hex"),
  write = (name, x) =>
    writeFileSync(resolve(output, name), JSON.stringify(x, null, 2) + "\n", {
      flag: "wx",
    });
const identity = {
  protocolSha256: sha(JSON.stringify(protocol)),
  sourceHashes: runtime.hashes,
  runnerSha256: sha(readFileSync(fileURLToPath(import.meta.url))),
  lockSha256: sha(readFileSync(resolve(root, "pnpm-lock.yaml"))),
};
const metadata = {
  startedAtUtc: new Date().toISOString(),
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
mkdirSync(output, { recursive: true });
write("protocol.json", protocol);
write("started.json", { identity, metadata });
writeFileSync(resolve(output, "cases.jsonl"), "", { flag: "wx" });
const start = performance.now();
try {
  const report = runItemCoverage(
    protocol,
    () => performance.now(),
    (row) => {
      appendFileSync(
        resolve(output, "cases.jsonl"),
        JSON.stringify(row) + "\n",
      );
      console.log(`Verified case: ${row.fixture.id}`);
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
    cases: report.rows.map((r) => ({
      id: r.fixture.id,
      scripted: {
        clearedRows: r.transition.info.clearedRows.length,
        acquiredItems: r.transition.info.acquiredItems.length,
        retainedItems: r.transition.info.retainedItems.length,
        spentAbility: r.transition.info.spentAbility,
        pendingReroll: r.transition.state.pendingReroll !== null,
      },
      ordinary: {
        complete: r.reference.complete,
        visitedNodes: r.reference.visitedNodes,
        optimalFirstActions: r.reference.optimalFirstActions.length,
      },
      selected: r.probes
        .filter((p) => p.repetition === 0)
        .map((p) => ({
          budget: p.budget,
          searchComplete: p.result.search.searchComplete,
          visitedNodes: p.result.search.visitedNodes,
          firstAction: p.result.actions[0]?.type ?? null,
          scriptedFirstMatches: p.scriptedFirstMatches,
          coverage: p.coverage,
        })),
    })),
    identity,
    limits: [
      "Deliberate scripted mechanics coverage, not representative frequency or a training dataset",
      "Scripted transition coverage and solver-selected path coverage are reported separately",
      "Ordinary-only complete reference shares domain/evaluator; it does not prove full ability search or future optimality",
      "Retained events count per transition, repetitions do not create independent states",
      "No new test states, model training, teacher label changes or production integration",
    ],
  };
  writeFileSync(
    resolve(output, "summary.json"),
    await format(JSON.stringify(summary, null, 2) + "\n", { parser: "json" }),
    { flag: "wx" },
  );
  console.log(
    JSON.stringify({ status: report.status, summary: report.summary }, null, 2),
  );
} catch (error) {
  write("failure.json", {
    status: "FAIL",
    error: error.message,
    identity,
    metadata,
  });
  throw error;
}
