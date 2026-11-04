import assert from "node:assert/strict";
import { appendFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { compileResearch } from "./runtime.mjs";
import {
  root,
  hash,
  write,
  reserveFitOutput,
  approvedCoverage,
  auditCoverageData,
  configPath,
} from "./coverage-preflight-contract.mjs";
export async function runCoverage(output) {
  const started = performance.now(),
    c = approvedCoverage(),
    git = (a) =>
      execFileSync("git", a, {
        cwd: root,
        encoding: "utf8",
        windowsHide: true,
      });
  assert.equal(
    git(["status", "--porcelain"]).trim(),
    "",
    "Commit implementation before the single two-build execution",
  );
  const out = reserveFitOutput(output),
    start = {
      atUtc: new Date().toISOString(),
      sourceHead: git(["rev-parse", "HEAD"]).trim(),
      executor: "Codex",
      pid: process.pid,
      node: process.version,
    };
  write(resolve(out, "start.json"), start);
  let maxSampledRss = 0;
  const check = () => {
    maxSampledRss = Math.max(maxSampledRss, process.memoryUsage().rss);
    assert(
      performance.now() - started <= 120000 && maxSampledRss <= 2147483648,
      "Coverage owner time/sampled RSS cap; STOP",
    );
  };
  try {
    const compiled = compileResearch(
        root,
        "src/research/coverage-preflight.ts",
      ),
      files = [
        ...Object.keys(c.contract.sourceHashes),
        ...Object.keys(compiled.hashes),
        configPath,
        "scripts/research/coverage-preflight.mjs",
        "scripts/research/coverage-preflight-contract.mjs",
        "scripts/research/runtime.mjs",
        "pnpm-lock.yaml",
      ],
      sourceHashes = Object.fromEntries(
        [...new Set(files)].map((p) => [p, hash(resolve(root, p))]),
      );
    write(resolve(out, "execution.json"), {
      status: "FROZEN_BEFORE_EXECUTION",
      sourceHead: start.sourceHead,
      configSha256: hash(resolve(root, configPath)),
      sourceHashes,
      compilerVersion: compiled.compilerVersion,
    });
    check();
    const research = await import(compiled.url);
    writeFileSync(resolve(out, "ledger.jsonl"), "", { flag: "wx" });
    let calls = 0,
      reference = 0,
      outcomes = 0;
    for (let build = 1; build <= 2; build++) {
      const reserve = (kind, split, row, index) => {
        check();
        if (kind === "reference") assert(++reference <= 128);
        else assert(kind === "outcome" && ++outcomes <= 5888);
        appendFileSync(
          resolve(out, "ledger.jsonl"),
          JSON.stringify({
            call: ++calls,
            build,
            kind,
            split,
            row,
            index,
            reservedNodes: kind === "reference" ? 256 : 0,
          }) + "\n",
        );
      };
      const data = research.buildCoverage(reserve, check);
      auditCoverageData(data);
      write(resolve(out, `build${build}.json`), data);
      check();
      console.log(
        `Coverage build ${build}/2: ${data.counts.examples} verified boards`,
      );
    }
    assert.equal(reference, 128);
    assert.equal(outcomes, 5888);
    assert.equal(
      hash(resolve(out, "build1.json")),
      hash(resolve(out, "build2.json")),
    );
    for (const [p, h] of Object.entries(sourceHashes))
      assert.equal(hash(resolve(root, p)), h, p);
    check();
    const names = [
      "start.json",
      "execution.json",
      "ledger.jsonl",
      "build1.json",
      "build2.json",
    ];
    const report = {
      status: "PASS",
      atUtc: new Date().toISOString(),
      elapsedSeconds: (performance.now() - started) / 1000,
      maxSampledRss,
      hashes: Object.fromEntries(names.map((p) => [p, hash(resolve(out, p))])),
    };
    write(resolve(out, "report.json"), report);
    return report;
  } catch (error) {
    write(resolve(out, "failure.json"), {
      status: "FAIL",
      atUtc: new Date().toISOString(),
      error: String(error),
      automaticRetry: false,
    });
    throw error;
  }
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  assert(
    process.argv.length === 4 && process.argv[2] === "--output",
    "Fixed --output only",
  );
  console.log(JSON.stringify(await runCoverage(process.argv[3])));
}
