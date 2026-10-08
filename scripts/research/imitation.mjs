import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { performance } from "node:perf_hooks";
import { compileResearch } from "./runtime.mjs";
const root = resolve(fileURLToPath(new URL("../../", import.meta.url)));
let protocolPath = resolve(root, "research/cnn/protocol.json"),
  output;
const args = process.argv.slice(2);
for (let i = 0; i < args.length; i += 2) {
  if (!["--protocol", "--output"].includes(args[i]) || !args[i + 1])
    throw new Error("Use --protocol file / --output new-directory");
  if (args[i] === "--protocol") protocolPath = resolve(args[i + 1]);
  else output = resolve(args[i + 1]);
}
output ??= resolve(
  root,
  ".research-output/datasets",
  new Date().toISOString().replace(/[:.]/g, "-"),
);
if (existsSync(output))
  throw new Error("Output already exists; prior data will not be overwritten");
const protocol = JSON.parse(readFileSync(protocolPath, "utf8"));
const runtime = compileResearch(root, "src/research/imitation.ts");
const { buildImitationDataset, validateDatasetSpec, SPLITS } = await import(
  runtime.url
);
const spec = validateDatasetSpec(protocol.dataset);
if (!spec.ok) throw new Error(spec.error);
const startedAtUtc = new Date().toISOString(),
  before = process.memoryUsage(),
  start = performance.now();
const built = buildImitationDataset(
  spec.value,
  () => performance.now(),
  (n) => console.log(`Synthetic teacher episodes: ${n}`),
);
if (!built.ok) throw new Error(built.error);
const after = process.memoryUsage(),
  durationMs = performance.now() - start;
const hash = (data) => createHash("sha256").update(data).digest("hex");
const body = JSON.stringify(built.value);
const manifest = {
  schemaVersion: 1,
  status: built.value.status,
  startedAtUtc,
  finishedAtUtc: new Date().toISOString(),
  durationMs,
  schema: built.value.schema,
  spec: built.value.spec,
  node: process.version,
  compiler: runtime.compilerVersion,
  revision: execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: root,
    encoding: "utf8",
  }).trim(),
  memoryBefore: before,
  memoryAfter: after,
  dataSha256: hash(body),
  protocolSha256: hash(JSON.stringify(protocol)),
  sourceHashes: runtime.hashes,
  runnerSha256: hash(readFileSync(fileURLToPath(import.meta.url))),
  lockSha256: hash(readFileSync(resolve(root, "pnpm-lock.yaml"))),
  workingTree: execFileSync("git", ["status", "--short"], {
    cwd: root,
    encoding: "utf8",
  }).trim(),
  counts: Object.fromEntries(
    SPLITS.map((split) => [
      split,
      {
        episodes: built.value.episodes.filter((e) => e.split === split).length,
        examples: built.value.examples[split].length,
        skips: built.value.skips.filter((s) => s.split === split).length,
        incompleteTeacher: built.value.examples[split].filter(
          (e) => !e.teacher.search.searchComplete,
        ).length,
        maxCandidates: Math.max(
          0,
          ...built.value.examples[split].map((e) => e.candidateKeys.length),
        ),
      },
    ]),
  ),
};
mkdirSync(output, { recursive: true });
writeFileSync(resolve(output, "dataset.json"), body);
writeFileSync(
  resolve(output, "protocol.json"),
  JSON.stringify(protocol, null, 2) + "\n",
);
writeFileSync(
  resolve(output, "manifest.json"),
  JSON.stringify(manifest, null, 2) + "\n",
);
console.log(
  JSON.stringify(
    { status: manifest.status, output, durationMs, counts: manifest.counts },
    null,
    2,
  ),
);
if (manifest.status !== "PASS") process.exitCode = 1;
