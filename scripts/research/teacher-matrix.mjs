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
const proposal = JSON.parse(readFileSync(protocolPath));
const runtime = compileResearch(root, "src/research/teacher-matrix.ts");
const { freezeMatrixProtocol, runTeacherMatrix } = await import(runtime.url);
const protocol = freezeMatrixProtocol(proposal);
const sha = (d) => createHash("sha256").update(d).digest("hex");
const write = (name, value) =>
  writeFileSync(resolve(output, name), JSON.stringify(value, null, 2) + "\n", {
    flag: "wx",
  });
const identity = {
  protocolSha256: sha(JSON.stringify(protocol)),
  sourceHashes: runtime.hashes,
  runnerSha256: sha(readFileSync(fileURLToPath(import.meta.url))),
  lockSha256: sha(readFileSync(resolve(root, "pnpm-lock.yaml"))),
};
const initialMetadata = {
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
write("started.json", { identity, metadata: initialMetadata });
writeFileSync(resolve(output, "probes.jsonl"), "", { flag: "wx" });
const start = performance.now();
try {
  const report = runTeacherMatrix(
    proposal,
    () => performance.now(),
    (row) => {
      appendFileSync(
        resolve(output, "probes.jsonl"),
        JSON.stringify(row) + "\n",
      );
      if ((row.sequence + 1) % 12 === 0)
        console.log(`Verified probes: ${row.sequence + 1}`);
    },
  );
  const metadata = {
    ...initialMetadata,
    finishedAtUtc: new Date().toISOString(),
    durationMs: performance.now() - start,
    memoryAfter: process.memoryUsage(),
  };
  write("report.json", { ...report, identity, metadata });
  const summary = {
    schemaVersion: 1,
    id: "teacher-matrix-20261008-v1",
    status: report.status,
    experimentStatus: "TRAIN_DEV_DIAGNOSTIC_COMPLETE",
    protocol,
    summary: report.summary,
    identity,
    limits: [
      "Initial states only; 12 seed/family states, repetitions are not independent samples",
      "Shared domain path replay is not an independent game rules oracle or a proof of optimal teacher labels",
      "Selected current-turn path coverage is not representative long-run episode frequency",
      "Retained item events count per transition, not distinct icons",
      "Timing includes cold/JIT effects without warmup; node budgets do not equal wall time",
      "Restricted reroll candidates and incomplete DFS remain unchanged; no future tapes are supplied",
      "Reserved test and old test are not generated or evaluated; no training or production integration",
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
        ...report.summary,
        groups: undefined,
        durationMs: metadata.durationMs,
      },
      null,
      2,
    ),
  );
} catch (error) {
  write("failure.json", {
    status: "FAIL",
    error: error.message,
    identity,
    metadata: initialMetadata,
  });
  throw error;
}
