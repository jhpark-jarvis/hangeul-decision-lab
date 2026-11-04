import assert from "node:assert/strict";
import {
  appendFileSync,
  mkdirSync,
  readdirSync,
  writeFileSync,
  realpathSync,
} from "node:fs";
import { resolve, relative, isAbsolute } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { runBoundedChild } from "./bounded-child.mjs";
import {
  root,
  json,
  hash,
  bytes,
  write,
  reserveFitOutput,
  configPath,
  approvedCoverageTraining,
  parseCoverageTrainingArgs,
  auditCoverageStored,
  auditMlpStored,
  auditCoverageArm,
  approvedCoverage,
} from "./coverage-training-contract.mjs";
export async function runCoverageTraining(args) {
  const started = performance.now(),
    c = approvedCoverageTraining(),
    git = (a) =>
      execFileSync("git", a, {
        cwd: root,
        encoding: "utf8",
        windowsHide: true,
      });
  assert.equal(
    git(["status", "--porcelain"]).trim(),
    "",
    "Commit implementation first",
  );
  const research = realpathSync(resolve(root, ".research-output")),
    inside = (p) => {
      const r = relative(research, realpathSync(p));
      return r && !r.startsWith("..") && !isAbsolute(r);
    },
    data = resolve(root, args.data),
    original = resolve(root, args.original),
    python = resolve(root, args.python);
  assert(inside(data) && inside(original) && inside(python));
  const out = reserveFitOutput(args.output, root, data),
    check = () =>
      assert(
        performance.now() - started <= 1260000,
        "Owner1260-second cap; STOP",
      );
  write(resolve(out, "start.json"), {
    atUtc: new Date().toISOString(),
    sourceHead: git(["rev-parse", "HEAD"]).trim(),
    executor: "Codex",
    node: process.version,
    pid: process.pid,
  });
  try {
    auditCoverageStored(data);
    auditMlpStored(original);
    assert.equal(hash(resolve(data, "build1.json")), c.dataSha256);
    assert.equal(
      hash(resolve(original, "candidate/initial.json")),
      c.anchor.logitsSha256,
    );
    assert.equal(
      json(resolve(original, "candidate/report.json"))
        .initialLogicalWeightsSha256,
      c.anchor.weightsSha256,
    );
    for (const [from, to] of [
      [resolve(data, "build1.json"), "dataset.json"],
      [resolve(original, "fit-dataset.json"), "anchor-dataset.json"],
    ])
      writeFileSync(resolve(out, to), bytes(from), { flag: "wx" });
    const files = [
        ...Object.keys(approvedCoverage().contract.sourceHashes),
        configPath,
        c.proposalPath,
        "scripts/research/coverage-training.mjs",
        "scripts/research/coverage-training-contract.mjs",
        "scripts/research/replay-coverage-training.mjs",
        "scripts/research/bounded-child.mjs",
        "pnpm-lock.yaml",
        ...readdirSync(resolve(root, "research/cnn"))
          .filter((p) => p.endsWith(".py"))
          .map((p) => "research/cnn/" + p),
      ],
      sourceHashes = Object.fromEntries(
        [...new Set(files)].map((p) => [p, hash(resolve(root, p))]),
      );
    write(resolve(out, "execution.json"), {
      status: "FROZEN_BEFORE_EXECUTION",
      configSha256: hash(resolve(root, configPath)),
      sourceHead: git(["rev-parse", "HEAD"]).trim(),
      sourceHashes,
      inputHashes: {
        "dataset.json": hash(resolve(out, "dataset.json")),
        "anchor-dataset.json": hash(resolve(out, "anchor-dataset.json")),
      },
    });
    const children = [];
    for (const arm of ["control", "diverse"]) {
      check();
      if (arm === "diverse") {
        auditCoverageArm(out, "control", c);
        write(resolve(out, "diverse-input.json"), {
          executionSha256: hash(resolve(out, "execution.json")),
          controlReportSha256: hash(resolve(out, "control/report.json")),
        });
      }
      const armDir = resolve(out, arm);
      mkdirSync(armDir);
      const result = await runBoundedChild({
        command: python,
        args: [
          "-B",
          resolve(root, "research/cnn/coverage_training.py"),
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
            purpose: "one approved coverage arm",
          }),
        onStdout: (b) => {
          appendFileSync(resolve(armDir, "stdout.log"), b);
          process.stdout.write(b);
        },
        onStderr: (b) => appendFileSync(resolve(armDir, "stderr.log"), b),
      });
      check();
      children.push({ arm, result });
      assert.equal(result.timedOut, false);
      assert.equal(result.code, 0, "Child failed; STOP/no retry");
      assert(result.elapsedMs <= 600000);
      auditCoverageArm(out, arm, c);
    }
    for (const [p, h] of Object.entries(sourceHashes))
      assert.equal(hash(resolve(root, p)), h, p);
    check();
    const names = [
        "start.json",
        "execution.json",
        "dataset.json",
        "anchor-dataset.json",
        "diverse-input.json",
        "control/report.json",
        "diverse/report.json",
      ],
      report = {
        status: "PASS",
        atUtc: new Date().toISOString(),
        elapsedSeconds: (performance.now() - started) / 1000,
        children,
        hashes: Object.fromEntries(
          names.map((p) => [p, hash(resolve(out, p))]),
        ),
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
)
  console.log(
    JSON.stringify(
      await runCoverageTraining(
        parseCoverageTrainingArgs("run", process.argv.slice(2)),
      ),
    ),
  );
