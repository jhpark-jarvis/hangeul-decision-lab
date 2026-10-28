import assert from "node:assert/strict";
import { appendFileSync, existsSync, readdirSync } from "node:fs";
import { resolve, relative, isAbsolute } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { compileResearch } from "./runtime.mjs";
import { runBoundedChild } from "./bounded-child.mjs";
import {
  root,
  configPath,
  approvedContract,
  json,
  hash,
  write,
  reserveFitOutput,
} from "./controlled-point-fit-contract.mjs";

export function parseFitArgs(args) {
  assert.equal(
    args.length,
    4,
    "Use --python <existing-python> --output <new-directory> only",
  );
  const pairs = [args.slice(0, 2), args.slice(2, 4)];
  assert.deepEqual(pairs.map((p) => p[0]).sort(), ["--output", "--python"]);
  assert(
    pairs.every(
      (p) =>
        typeof p[1] === "string" && p[1].length > 0 && !p[1].startsWith("--"),
    ),
  );
  return Object.fromEntries(pairs.map(([k, v]) => [k.slice(2), v]));
}
export async function runFit(args) {
  const started = performance.now(),
    config = approvedContract();
  const git = (a) =>
    execFileSync("git", a, { cwd: root, encoding: "utf8", windowsHide: true });
  assert.equal(
    git(["status", "--porcelain"]).trim(),
    "",
    "Commit implementation before the single run",
  );
  const python = resolve(root, args.python),
    rel = relative(resolve(root, ".research-output"), python);
  assert(
    rel && !rel.startsWith("..") && !isAbsolute(rel) && existsSync(python),
    "Existing local research Python only",
  );
  const output = reserveFitOutput(args.output, root);
  const check = () =>
    assert(
      performance.now() - started <= 660000,
      "Overall 660-second owner budget; STOP",
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
    const compiled = compileResearch(
      root,
      "src/research/controlled-point-fit.ts",
    );
    const research = await import(compiled.url);
    const data = research.buildControlledPointFit(check);
    write(resolve(output, "dataset.json"), data);
    const paths = [
      "scripts/research/controlled-point-fit.mjs",
      "scripts/research/controlled-point-fit-contract.mjs",
      "scripts/research/replay-controlled-point-fit.mjs",
      "scripts/research/validate-controlled-point-fit-proposal.mjs",
      "scripts/research/runtime.mjs",
      "scripts/research/bounded-child.mjs",
      "scripts/research/point-simulation.mjs",
      configPath,
      "pnpm-lock.yaml",
      ...Object.keys(compiled.hashes),
      ...readdirSync(resolve(root, "research/cnn"))
        .filter((p) => p.endsWith(".py"))
        .map((p) => "research/cnn/" + p),
    ];
    const sourceHashes = Object.fromEntries(
      [...new Set(paths)].map((p) => [p, hash(resolve(root, p))]),
    );
    write(resolve(output, "execution.json"), {
      status: "FROZEN_BEFORE_EXECUTION",
      sourceHead: git(["rev-parse", "HEAD"]).trim(),
      configSha256: hash(resolve(root, configPath)),
      datasetSha256: hash(resolve(output, "dataset.json")),
      sourceHashes,
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
        resolve(root, "research/cnn/controlled_point_fit.py"),
        "--output",
        output,
      ],
      cwd: root,
      timeoutMs: Math.min(600000, 660000 - (performance.now() - started)),
      onSpawn: (p) =>
        write(resolve(output, "child.json"), {
          pid: p.pid,
          ownerPid: process.pid,
          purpose: "one fixed controlled fit",
        }),
      onStdout: (b) => {
        appendFileSync(resolve(output, "child.stdout.log"), b);
        process.stdout.write(b);
      },
      onStderr: (b) => appendFileSync(resolve(output, "child.stderr.log"), b),
    });
    check();
    assert.equal(child.timedOut, false, "Owned child timeout; STOP");
    assert.equal(child.code, 0, "Child failed; STOP, no restart or retry");
    for (const [p, h] of Object.entries(sourceHashes))
      assert.equal(hash(resolve(root, p)), h, "Source changed: " + p);
    const c = json(resolve(output, "child-report.json"));
    assert.equal(c.status, "PASS");
    assert.deepEqual(c.forwardCounts, {
      initial: 16,
      train: 3200,
      final1: 16,
      final2: 16,
    });
    const names = [
      "start.json",
      "dataset.json",
      "execution.json",
      "child-report.json",
      "summary.json",
    ];
    const report = {
      status: "PASS",
      atUtc: new Date().toISOString(),
      sourceHead: git(["rev-parse", "HEAD"]).trim(),
      elapsedSeconds: (performance.now() - started) / 1000,
      child,
      hashes: Object.fromEntries(
        names.map((p) => [p, hash(resolve(output, p))]),
      ),
      budget: config.contract.resources,
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
    JSON.stringify(await runFit(parseFitArgs(process.argv.slice(2)))),
  );
