import assert from "node:assert/strict";
import {
  appendFileSync,
  mkdirSync,
  realpathSync,
  writeFileSync,
} from "node:fs";
import { execFileSync } from "node:child_process";
import { isAbsolute, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { runBoundedChild } from "./bounded-child.mjs";
import {
  root,
  sha,
  bytes,
  write,
  reserveFitOutput,
  auditPairsStored,
  boardConfigPath,
  approvedPairedTarget,
  parseBoardArgs,
  auditPairedArm,
  armNames,
} from "./paired-target-contract.mjs";
export async function runPairedTarget(args) {
  const started = 0,
    c = approvedPairedTarget(),
    git = (a) =>
      execFileSync("git", a, {
        cwd: root,
        encoding: "utf8",
        windowsHide: true,
      });
  assert.equal(
    git(["status", "--porcelain"]).trim(),
    "",
    "Commit frozen implementation first",
  );
  const research = realpathSync(resolve(root, ".research-output")),
    inside = (p) => {
      const r = relative(research, realpathSync(p));
      return r && !r.startsWith("..") && !isAbsolute(r);
    };
  const source = resolve(root, args.source),
    python = resolve(root, args.python);
  assert(inside(source) && inside(python));
  const out = reserveFitOutput(args.output, root, source);
  let maxSampledRss = 0;
  const check = () => {
    maxSampledRss = Math.max(maxSampledRss, process.memoryUsage().rss);
    assert(maxSampledRss <= 2147483648, "Owner sampled RSS cap; STOP");
    assert(
      performance.now() - started <= 1260000,
      "Owner1260-second cap; STOP",
    );
  };
  write(resolve(out, "start.json"), {
    atUtc: new Date().toISOString(),
    sourceHead: git(["rev-parse", "HEAD"]).trim(),
    node: process.version,
    executor: "Codex",
    pid: process.pid,
  });
  const children = [];
  try {
    check();
    auditPairsStored(source);
    for (const [p, h] of Object.entries(c.archiveHashes))
      assert.equal(sha(resolve(source, p)), h, p);
    mkdirSync(resolve(out, "source"));
    for (const p of Object.keys(c.archiveHashes))
      writeFileSync(resolve(out, "source", p), bytes(resolve(source, p)), {
        flag: "wx",
      });
    writeFileSync(
      resolve(out, "dataset.json"),
      bytes(resolve(source, "build1.json")),
      { flag: "wx" },
    );
    const prior = resolve(
      root,
      "assets/research/board-context-stop-20261106-v1.json",
    );
    assert.equal(sha(prior), c.priorSummarySha256);
    writeFileSync(resolve(out, "prior-stop-summary.json"), bytes(prior), {
      flag: "wx",
    });
    const paths = git(["ls-files"])
      .trim()
      .split(/\r?\n/)
      .filter((p) => p !== "README.md");
    const sourceHashes = Object.fromEntries(
      [...new Set(paths)].map((p) => [p, sha(resolve(root, p))]),
    );
    write(resolve(out, "execution.json"), {
      status: "FROZEN_BEFORE_EXECUTION",
      sourceHead: git(["rev-parse", "HEAD"]).trim(),
      configSha256: sha(resolve(root, boardConfigPath)),
      sourceHashes,
      inputHashes: c.archiveHashes,
      datasetSha256: sha(resolve(out, "dataset.json")),
    });
    for (const arm of armNames) {
      check();
      if (arm === "candidate") {
        auditPairedArm(out, "control", c);
        write(resolve(out, "candidate-input.json"), {
          executionSha256: sha(resolve(out, "execution.json")),
          controlReportSha256: sha(resolve(out, "control/report.json")),
          controlPairedReportSha256: sha(
            resolve(out, "control/paired-report.json"),
          ),
        });
      }
      const armDir = resolve(out, arm);
      mkdirSync(armDir);
      const result = await runBoundedChild({
        command: python,
        args: [
          "-B",
          resolve(root, "research/cnn/paired_target_fit.py"),
          "--run",
          out,
          "--arm",
          arm,
        ],
        cwd: root,
        timeoutMs: Math.min(600000, 1260000 - (performance.now() - started)),
        onSpawn: (p) =>
          write(resolve(armDir, "child.json"), {
            pid: p.pid,
            ownerPid: process.pid,
            arm,
            purpose: "One approved CPU paired-target occupancy arm",
          }),
        onStdout: (b) => {
          appendFileSync(resolve(armDir, "stdout.log"), b);
          process.stdout.write(b);
        },
        onStderr: (b) => appendFileSync(resolve(armDir, "stderr.log"), b),
      });
      children.push({ arm, result });
      write(resolve(armDir, "child-result.json"), result);
      check();
      assert.equal(result.timedOut, false);
      assert.equal(result.code, 0, "Child failed; STOP/no retry");
      assert(result.elapsedMs <= 600000);
      auditPairedArm(out, arm, c);
    }
    for (const [p, h] of Object.entries(sourceHashes))
      assert.equal(sha(resolve(root, p)), h, p);
    for (const [p, h] of Object.entries(c.archiveHashes)) {
      assert.equal(sha(resolve(source, p)), h);
      assert.equal(sha(resolve(out, "source", p)), h);
    }
    check();
    const names = [
      "start.json",
      "execution.json",
      "dataset.json",
      "prior-stop-summary.json",
      "candidate-input.json",
      "control/report.json",
      "control/paired-report.json",
      "candidate/report.json",
      "candidate/paired-report.json",
    ];
    const report = {
      status: "PASS",
      atUtc: new Date().toISOString(),
      elapsedSeconds: (performance.now() - started) / 1000,
      ownerMaxSampledRss: maxSampledRss,
      children,
      hashes: Object.fromEntries(names.map((p) => [p, sha(resolve(out, p))])),
    };
    write(resolve(out, "report.json"), report);
    return report;
  } catch (error) {
    write(resolve(out, "failure.json"), {
      status: "FAIL",
      atUtc: new Date().toISOString(),
      error: String(error),
      automaticRetry: false,
      elapsedSeconds: (performance.now() - started) / 1000,
      ownerMaxSampledRss: maxSampledRss,
      children,
    });
    throw error;
  }
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
)
  console.log(
    JSON.stringify(
      await runPairedTarget(parseBoardArgs("run", process.argv.slice(2))),
    ),
  );
