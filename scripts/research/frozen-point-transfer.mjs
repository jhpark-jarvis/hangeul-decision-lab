import assert from "node:assert/strict";
import {
  appendFileSync,
  existsSync,
  readdirSync,
  realpathSync,
  writeFileSync,
} from "node:fs";
import { resolve, relative, isAbsolute } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { compileResearch } from "./runtime.mjs";
import { runBoundedChild } from "./bounded-child.mjs";
import {
  root,
  configPath,
  approvedTransfer,
  parseTransferArgs,
  hash,
  bytes,
  json,
  write,
  reserveFitOutput,
  auditTransferData,
  auditOriginalTransferInput,
  auditTransferChild,
} from "./frozen-point-transfer-contract.mjs";

export async function runTransfer(args) {
  const started = performance.now(),
    c = approvedTransfer();
  const git = (a) =>
    execFileSync("git", a, { cwd: root, encoding: "utf8", windowsHide: true });
  assert.equal(
    git(["status", "--porcelain"]).trim(),
    "",
    "Commit implementation before the single frozen evaluation",
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
    "Existing local research environment/archive only",
  );
  const output = reserveFitOutput(args.output, root, original);
  const check = () =>
    assert(performance.now() - started <= 660000, "Owner 660-second cap; STOP");
  write(resolve(output, "start.json"), {
    atUtc: new Date().toISOString(),
    sourceHead: git(["rev-parse", "HEAD"]).trim(),
    pid: process.pid,
    executor: "Codex",
    node: process.version,
    configSha256: hash(resolve(root, configPath)),
  });
  try {
    const prior = auditOriginalTransferInput(original);
    assert.equal(prior.arms.candidate.fitGate.status, "PASS");
    assert.equal(
      hash(resolve(original, "candidate/final-weights.pt")),
      c.contract.checkpoint.serializationSha256,
    );
    for (const [from, to] of [
      ["dataset.json", "fit-dataset.json"],
      ["candidate/final1.json", "fit-expected.json"],
    ])
      writeFileSync(resolve(output, to), bytes(resolve(original, from)), {
        flag: "wx",
      });
    check();
    const compiled = compileResearch(
      root,
      "src/research/frozen-point-transfer.ts",
    );
    const research = await import(compiled.url),
      data = research.buildFrozenPointTransfer(check);
    auditTransferData(data, json(resolve(output, "fit-dataset.json")));
    write(resolve(output, "transfer-dataset.json"), data);
    check();
    const files = [
      ...Object.keys(c.contract.sourceHashes),
      ...Object.keys(compiled.hashes),
      configPath,
      "scripts/research/frozen-point-transfer.mjs",
      "scripts/research/frozen-point-transfer-contract.mjs",
      "scripts/research/replay-frozen-point-transfer.mjs",
      "scripts/research/runtime.mjs",
      "scripts/research/bounded-child.mjs",
      "pnpm-lock.yaml",
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
      originalRunPath: original,
      originalCheckpointSha256: c.contract.checkpoint.serializationSha256,
      sourceHashes,
      inputHashes: Object.fromEntries(
        ["fit-dataset.json", "fit-expected.json", "transfer-dataset.json"].map(
          (p) => [p, hash(resolve(output, p))],
        ),
      ),
      compilerVersion: compiled.compilerVersion,
      compilerSha256: hash(
        resolve(root, "node_modules/typescript/lib/typescript.js"),
      ),
    });
    check();
    const child = await runBoundedChild({
      command: python,
      args: [
        "-B",
        resolve(root, "research/cnn/frozen_point_transfer.py"),
        "--output",
        output,
      ],
      cwd: root,
      timeoutMs: Math.min(600000, 660000 - (performance.now() - started)),
      onSpawn: (p) =>
        write(resolve(output, "child.json"), {
          pid: p.pid,
          ownerPid: process.pid,
          purpose: "one frozen 320-call evaluation",
        }),
      onStdout: (b) => appendFileSync(resolve(output, "stdout.log"), b),
      onStderr: (b) => appendFileSync(resolve(output, "stderr.log"), b),
    });
    check();
    assert.equal(child.timedOut, false);
    assert.equal(child.code, 0, "Frozen evaluation failed; STOP/no retry");
    assert(child.elapsedMs <= 600000);
    auditTransferChild(output);
    for (const [p, h] of Object.entries(sourceHashes))
      assert.equal(hash(resolve(root, p)), h, p);
    assert.equal(
      hash(resolve(original, "candidate/final-weights.pt")),
      c.contract.checkpoint.serializationSha256,
    );
    const names = [
      "start.json",
      "execution.json",
      "fit-dataset.json",
      "fit-expected.json",
      "transfer-dataset.json",
      "child-report.json",
    ];
    check();
    const report = {
      status: "PASS",
      atUtc: new Date().toISOString(),
      sourceHead: git(["rev-parse", "HEAD"]).trim(),
      elapsedSeconds: (performance.now() - started) / 1000,
      child,
      hashes: Object.fromEntries(
        names.map((p) => [p, hash(resolve(output, p))]),
      ),
    };
    write(resolve(output, "report.json"), report);
    return report;
  } catch (error) {
    write(resolve(output, "failure.json"), {
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
      await runTransfer(parseTransferArgs("run", process.argv.slice(2))),
    ),
  );
