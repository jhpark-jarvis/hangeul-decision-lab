import assert from "node:assert/strict";
import {
  appendFileSync,
  mkdirSync,
  readdirSync,
  writeFileSync,
  existsSync,
  realpathSync,
} from "node:fs";
import { resolve, relative, isAbsolute } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { runBoundedChild } from "./bounded-child.mjs";
import { auditStored } from "./controlled-point-fit-contract.mjs";
import {
  root,
  configPath,
  approvedDuration,
  parseDurationArgs,
  hash,
  bytes,
  write,
  reserveFitOutput,
  auditDurationArm,
} from "./fit-duration-contract.mjs";

export async function runDuration(args) {
  const started = performance.now(),
    config = approvedDuration();
  const git = (a) =>
    execFileSync("git", a, { cwd: root, encoding: "utf8", windowsHide: true });
  assert.equal(
    git(["status", "--porcelain"]).trim(),
    "",
    "Commit implementation before the fixed two-run execution",
  );
  const researchRoot = realpathSync(resolve(root, ".research-output"));
  const inside = (p) => {
    const rel = relative(researchRoot, realpathSync(p));
    return rel && !rel.startsWith("..") && !isAbsolute(rel);
  };
  const python = resolve(root, args.python),
    original = resolve(root, args.original);
  assert(
    existsSync(python) && inside(python) && inside(original),
    "Existing local research Python/original archive only",
  );
  const output = reserveFitOutput(args.output, root, original),
    check = () =>
      assert(
        performance.now() - started <= 1260000,
        "1260-second owner budget; STOP",
      );
  write(resolve(output, "start.json"), {
    atUtc: new Date().toISOString(),
    sourceHead: git(["rev-parse", "HEAD"]).trim(),
    executor: "Codex",
    node: process.version,
    pid: process.pid,
    configSha256: hash(resolve(root, configPath)),
  });
  try {
    const prior = auditStored(original);
    assert.equal(prior.summary.fitGate.status, "FAIL");
    assert.equal(
      hash(resolve(original, "dataset.json")),
      config.contract.originalRun.datasetSha256,
    );
    writeFileSync(
      resolve(output, "dataset.json"),
      bytes(resolve(original, "dataset.json")),
      { flag: "wx" },
    );
    const files = [
      ...Object.keys(config.contract.sourceHashes),
      configPath,
      "pnpm-lock.yaml",
      "scripts/research/fit-duration.mjs",
      "scripts/research/fit-duration-contract.mjs",
      "scripts/research/replay-fit-duration.mjs",
      "scripts/research/bounded-child.mjs",
      "scripts/research/validate-fit-duration-proposal.mjs",
      ...readdirSync(resolve(root, "research/cnn"))
        .filter((p) => p.endsWith(".py"))
        .map((p) => "research/cnn/" + p),
    ];
    const sourceHashes = Object.fromEntries(
      [...new Set(files)].map((p) => [p, hash(resolve(root, p))]),
    );
    write(resolve(output, "execution.json"), {
      status: "FROZEN_BEFORE_EXECUTION",
      sourceHead: git(["rev-parse", "HEAD"]).trim(),
      configSha256: hash(resolve(root, configPath)),
      datasetSha256: hash(resolve(output, "dataset.json")),
      originalRunPath: original,
      sourceHashes,
    });
    const children = [];
    for (const arm of ["control", "candidate"]) {
      check();
      if (arm === "candidate") {
        auditDurationArm(output, "control", config);
        write(resolve(output, "candidate-input.json"), {
          executionSha256: hash(resolve(output, "execution.json")),
          controlReportSha256: hash(resolve(output, "control/report.json")),
        });
      }
      const armDir = resolve(output, arm);
      mkdirSync(armDir);
      const result = await runBoundedChild({
        command: python,
        args: [
          "-B",
          resolve(root, "research/cnn/fit_duration.py"),
          "--run",
          output,
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
            purpose: "one bounded duration arm",
          }),
        onStdout: (b) => {
          appendFileSync(resolve(armDir, "stdout.log"), b);
          process.stdout.write(b);
        },
        onStderr: (b) => appendFileSync(resolve(armDir, "stderr.log"), b),
      });
      check();
      children.push({ arm, result });
      assert.equal(result.timedOut, false, "Owned child timeout; STOP");
      assert.equal(result.code, 0, "Child failure; STOP, no retry");
      assert(result.elapsedMs <= 600000, "Child process wall cap");
      auditDurationArm(output, arm, config);
    }
    for (const [p, h] of Object.entries(sourceHashes))
      assert.equal(
        hash(resolve(root, p)),
        h,
        "Source changed during run: " + p,
      );
    const names = [
      "start.json",
      "execution.json",
      "dataset.json",
      "candidate-input.json",
      "control/report.json",
      "candidate/report.json",
    ];
    const report = {
      status: "PASS",
      atUtc: new Date().toISOString(),
      sourceHead: git(["rev-parse", "HEAD"]).trim(),
      elapsedSeconds: (performance.now() - started) / 1000,
      children,
      hashes: Object.fromEntries(
        names.map((p) => [p, hash(resolve(output, p))]),
      ),
    };
    check();
    write(resolve(output, "report.json"), report);
    return report;
  } catch (error) {
    write(resolve(output, "failure.json"), {
      status: "FAIL",
      error: String(error),
      atUtc: new Date().toISOString(),
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
      await runDuration(parseDurationArgs("run", process.argv.slice(2))),
    ),
  );
