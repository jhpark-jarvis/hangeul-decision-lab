import {
  readFileSync,
  writeFileSync,
  appendFileSync,
  mkdirSync,
  existsSync,
} from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { performance } from "node:perf_hooks";
import { format } from "prettier";
import { compileResearch } from "./runtime.mjs";
const root = resolve(fileURLToPath(new URL("../../", import.meta.url)));
let dataset,
  output,
  protocolPath = resolve(root, "research/cnn/audit-protocol.json");
const args = process.argv.slice(2);
for (let i = 0; i < args.length; i += 2) {
  if (
    !args[i + 1] ||
    !["--dataset", "--output", "--protocol"].includes(args[i])
  )
    throw new Error(
      "Use --dataset directory --output new-directory [--protocol file]",
    );
  if (args[i] === "--dataset") dataset = resolve(args[i + 1]);
  else if (args[i] === "--output") output = resolve(args[i + 1]);
  else protocolPath = resolve(args[i + 1]);
}
if (!dataset || !output) throw new Error("Dataset and new output required");
if (existsSync(output))
  throw new Error("Existing output will not be overwritten");
const hash = (d) => createHash("sha256").update(d).digest("hex");
const dataBody = readFileSync(resolve(dataset, "dataset.json")),
  pack = JSON.parse(dataBody),
  manifest = JSON.parse(readFileSync(resolve(dataset, "manifest.json")));
if (hash(dataBody) !== manifest.dataSha256)
  throw new Error("Dataset hash mismatch");
// Existing complete encoding/replay validator, including every split, before probes.
execFileSync(
  process.execPath,
  [resolve(root, "scripts/research/verify-imitation.mjs"), dataset],
  { cwd: root, stdio: "pipe" },
);
const protocol = JSON.parse(readFileSync(protocolPath));
const runtime = compileResearch(root, "src/research/teacher-audit.ts");
const { runTeacherAudit, validateAuditProtocol } = await import(runtime.url);
validateAuditProtocol(protocol, pack);
const startedAtUtc = new Date().toISOString(),
  start = performance.now(),
  memoryBefore = process.memoryUsage();
mkdirSync(output, { recursive: true });
const write = (name, value) =>
  writeFileSync(resolve(output, name), JSON.stringify(value, null, 2) + "\n", {
    flag: "wx",
  });
try {
  write("protocol.json", protocol);
  write("started.json", { startedAtUtc, dataSha256: manifest.dataSha256 });
  writeFileSync(resolve(output, "probes.jsonl"), "", { flag: "wx" });
  const audit = runTeacherAudit(
    pack,
    protocol,
    () => performance.now(),
    (n, row) => {
      appendFileSync(
        resolve(output, "probes.jsonl"),
        JSON.stringify(row) + "\n",
      );
      console.log(`Audited train/dev states: ${n}`);
    },
  );
  const identity = {
    dataSha256: manifest.dataSha256,
    protocolSha256: hash(JSON.stringify(protocol)),
    sourceHashes: runtime.hashes,
    runnerSha256: hash(readFileSync(fileURLToPath(import.meta.url))),
    lockSha256: hash(readFileSync(resolve(root, "pnpm-lock.yaml"))),
  };
  const metadata = {
    startedAtUtc,
    finishedAtUtc: new Date().toISOString(),
    durationMs: performance.now() - start,
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
    memoryBefore,
    memoryAfter: process.memoryUsage(),
  };
  write("report.json", { ...audit, identity, metadata });
  const summary = {
    schemaVersion: 1,
    id: protocol.id,
    status: audit.status,
    protocol,
    descriptions: audit.descriptions,
    probes: audit.summary,
    identity,
    interpretation:
      "Slot intervention measures this bounded teacher's order sensitivity. Equal evaluations concern selected full paths, not equivalent first-action values or future survival.",
    limits: [
      "16 preselected train/dev initial states; no new test probes or tuning",
      "Top3/memo may omit tied roots; reported ties are a lower bound",
      "Historical test metadata is descriptive; it is not a fresh held-out score",
      "No learning, production solver modification, UI integration or real-game conclusion",
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
        status: audit.status,
        output,
        summary: audit.summary,
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
