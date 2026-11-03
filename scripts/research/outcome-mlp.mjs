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

import {
  root,
  configPath,
  approvedMlp,
  parseTransferArgs,
  hash,
  bytes,
  write,
  reserveFitOutput,
  auditMlpArm,
  auditTransferStored,
} from "./outcome-mlp-contract.mjs";

export async function runMlp(args) {
  const started = performance.now(),
    config = approvedMlp();
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
    auditTransferStored(original);
    for (const [name, key] of [
      ["fit-dataset.json", "fitDatasetSha256"],
      ["transfer-dataset.json", "transferDatasetSha256"],
    ]) {
      assert.equal(
        hash(resolve(original, name)),
        config.contract.original[key],
      );
      writeFileSync(resolve(output, name), bytes(resolve(original, name)), {
        flag: "wx",
      });
    }
    const files = [
      ...Object.keys(config.contract.sourceHashes),
      configPath,
      "pnpm-lock.yaml",
      "scripts/research/outcome-mlp.mjs",
      "scripts/research/outcome-mlp-contract.mjs",
      "scripts/research/replay-outcome-mlp.mjs",
      "scripts/research/bounded-child.mjs",
      "scripts/research/validate-outcome-mlp-proposal.mjs",
      config.proposalPath,
      "scripts/research/test-outcome-mlp-execution.mjs",
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
      inputHashes: Object.fromEntries(
        ["fit-dataset.json", "transfer-dataset.json"].map((p) => [
          p,
          hash(resolve(output, p)),
        ]),
      ),
      originalRunPath: original,
      sourceHashes,
    });
    const children = [];
    for (const arm of ["control", "candidate"]) {
      check();
      if (arm === "candidate") {
        auditMlpArm(output, "control", config);
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
          resolve(root, "research/cnn/outcome_mlp.py"),
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
      auditMlpArm(output, arm, config);
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
      "fit-dataset.json",
      "transfer-dataset.json",
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
      await runMlp(parseTransferArgs("run", process.argv.slice(2))),
    ),
  );
