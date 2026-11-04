import assert from "node:assert/strict";
import { appendFileSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { compileResearch } from "./runtime.mjs";
import { write, reserveFitOutput } from "./controlled-point-fit-contract.mjs";
import {
  root,
  sha,
  configPath,
  lookaheadSchedule,
  approvedPairs,
  createPairLedger,
  priorFingerprints,
  auditPairBuild,
} from "./lookahead-pairs-contract.mjs";
const git = (args) =>
  execFileSync("git", args, {
    cwd: root,
    encoding: "utf8",
    windowsHide: true,
  }).trim();
export async function runLookaheadPairs(args) {
  assert.deepEqual(args.slice(0, 1), ["--output"]);
  assert.equal(args.length, 2);
  assert.equal(
    git(["status", "--porcelain"]),
    "",
    "Commit frozen implementation before execution",
  );
  const start = 0; // performance.now measures process uptime, including ESM imports.
  const c = approvedPairs(),
    out = reserveFitOutput(args[1]);
  let maxSampledRss = 0;
  const check = () => {
    maxSampledRss = Math.max(maxSampledRss, process.memoryUsage().rss);
    assert(
      (performance.now() - start) / 1000 <= 120 && maxSampledRss <= 2147483648,
      "Time/sampled RSS cap",
    );
  };
  const hashes = () => {
    for (const [p, h] of Object.entries(c.sourceHashes))
      assert.equal(sha(resolve(root, p)), h, p);
    for (const [p, h] of Object.entries(c.inputHashes))
      assert.equal(sha(resolve(root, p)), h, p);
  };
  writeFileSync(resolve(out, "ledger.jsonl"), "", { flag: "wx" });
  const ledger = createPairLedger(
    (row) =>
      appendFileSync(resolve(out, "ledger.jsonl"), JSON.stringify(row) + "\n"),
    check,
  );
  const common = {
    atUtc: new Date().toISOString(),
    executor: "Codex",
    node: process.version,
    pid: process.pid,
    sourceHead: git(["rev-parse", "HEAD"]),
    configSha256: sha(resolve(root, configPath)),
    retry: 0,
  };
  write(resolve(out, "start.json"), common);
  try {
    hashes();
    check();
    const registry = priorFingerprints(root, check);
    write(resolve(out, "prior-fingerprints.json"), registry);
    check();
    const runtime = compileResearch(root, "src/research/lookahead-pairs.ts");
    for (const [p, h] of Object.entries(runtime.hashes))
      assert.equal(c.sourceHashes[p], h, "Unfrozen runtime source " + p);
    const { buildLookaheadPairCase } = await import(runtime.url);
    const summaries = [];
    for (let build = 0; build < 2; build++) {
      const examples = [];
      for (const [index, d] of lookaheadSchedule().entries()) {
        ledger.state(build, index);
        const result = buildLookaheadPairCase(
          d,
          (r) => ledger.reserve({ build, index, ...r }),
          (state) =>
            write(resolve(out, `initial-${build}-${index}.json`), {
              descriptor: d,
              state,
            }),
        );
        examples.push(result);
        check();
      }
      const data = {
        schemaVersion: 1,
        id: "lookahead-pairs-data-v1",
        examples,
      };
      write(resolve(out, `build${build + 1}.json`), data);
      summaries.push(auditPairBuild(data, registry, ledger.rows, build));
      check();
    }
    const counts = ledger.finish();
    assert.equal(
      sha(resolve(out, "build1.json")),
      sha(resolve(out, "build2.json")),
    );
    assert.deepEqual(summaries[0], summaries[1]);
    hashes();
    check();
    const summary = {
      schemaVersion: 1,
      id: "lookahead-pairs-result-v1",
      status: "PASS",
      ...summaries[0],
      actualCompute: counts,
      repeatedBuildsExact: true,
      additionalPythonModelLoadForwardTraining: 0,
      independentGeneralization: false,
      appUsesDfs: true,
    };
    write(resolve(out, "report.json"), {
      status: "PASS",
      ...common,
      elapsedSeconds: (performance.now() - start) / 1000,
      maxSampledRss,
      instantaneousPeakMeasured: false,
      runtime,
      hashes: Object.fromEntries(
        [
          "build1.json",
          "build2.json",
          "ledger.jsonl",
          "prior-fingerprints.json",
        ].map((p) => [p, sha(resolve(out, p))]),
      ),
      summary,
    });
    return summary;
  } catch (error) {
    write(resolve(out, "failure.json"), {
      status: "FAIL",
      ...common,
      error: String(error),
      partial: error.partial ?? null,
      counts: ledger.counts,
      elapsedSeconds: (performance.now() - start) / 1000,
      maxSampledRss,
      automaticRetry: false,
    });
    throw error;
  }
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
)
  console.log(JSON.stringify(await runLookaheadPairs(process.argv.slice(2))));
