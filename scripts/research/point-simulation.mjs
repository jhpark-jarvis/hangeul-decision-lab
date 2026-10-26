import assert from "node:assert/strict";
import {
  readFileSync,
  writeFileSync,
  appendFileSync,
  mkdirSync,
  realpathSync,
} from "node:fs";
import { resolve, relative, isAbsolute, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { compileResearch } from "./runtime.mjs";
const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const sha = (p) => createHash("sha256").update(readFileSync(p)).digest("hex");
const inside = (a, b) => {
  const r = relative(a, b);
  return r !== "" && !r.startsWith("..") && !isAbsolute(r);
};
export function parsePointSimulationArgs(args) {
  assert(
    args.length === 2 && args[0] === "--output" && args[1],
    "Use --output .research-output/<new-directory> only",
  );
  return args[1];
}
export function reservePointOutput(output, root = ROOT) {
  const base = resolve(root, ".research-output"),
    out = resolve(root, output);
  assert(inside(base, out), "New owned ignored output only");
  mkdirSync(base, { recursive: true });
  assert(inside(realpathSync(root), realpathSync(base)));
  const parent = realpathSync(dirname(out));
  assert(parent === realpathSync(base) || inside(realpathSync(base), parent));
  mkdirSync(out);
  return out;
}
export async function executePointSimulation(output, root = ROOT) {
  const git = (a) =>
    execFileSync("git", a, {
      cwd: root,
      encoding: "utf8",
      windowsHide: true,
    }).trim();
  assert.equal(
    git(["status", "--porcelain"]),
    "",
    "Commit source/config before actual simulation",
  );
  const out = reservePointOutput(output, root);
  const save = (name, value) =>
    writeFileSync(resolve(out, name), JSON.stringify(value, null, 2) + "\n", {
      flag: "wx",
    });
  const started = performance.now(),
    configPath = resolve(root, "research/experiments/point-simulation-v1.json");
  const metadata = {
    atUtc: new Date().toISOString(),
    sourceHead: git(["rev-parse", "HEAD"]),
    node: process.version,
    programSha256: sha(fileURLToPath(import.meta.url)),
    configSha256: sha(configPath),
    lockSha256: sha(resolve(root, "pnpm-lock.yaml")),
    scope:
      "Controlled synthetic domain/DFS simulation only; no model calls or original dataset changes",
  };
  save("start.json", metadata);
  try {
    const runtime = compileResearch(root, "src/research/point-simulation.ts"),
      research = await import(runtime.url);
    const protocol = research.validatePointProtocol(
      JSON.parse(readFileSync(configPath, "utf8")),
    );
    writeFileSync(resolve(out, "cases.jsonl"), "", { flag: "wx" });
    let emitted = 0;
    const summary = research.runPointSimulation(protocol, (row) => {
      appendFileSync(resolve(out, "cases.jsonl"), JSON.stringify(row) + "\n");
      emitted++;
      assert(
        performance.now() - started <= 300000,
        "Simulation exceeded cooperative 300s limit",
      );
      assert(
        process.memoryUsage().rss <= 2 * 1024 ** 3,
        "Simulation exceeded sampled 2GiB RSS limit",
      );
      if (emitted % 40 === 0)
        process.stdout.write(`Verified ${emitted}/320 controlled cases\n`);
    });
    assert.equal(emitted, 320);
    save("summary.json", summary);
    save("report.json", {
      status: "PASS",
      ...metadata,
      finishedAtUtc: new Date().toISOString(),
      compiler: runtime.compilerVersion,
      sourceHashes: runtime.hashes,
      runtimeSha256: sha(resolve(root, "scripts/research/runtime.mjs")),
      casesSha256: sha(resolve(out, "cases.jsonl")),
      summarySha256: sha(resolve(out, "summary.json")),
      durationMs: performance.now() - started,
      rssAfterBytes: process.memoryUsage().rss,
      limits:
        "Cooperative elapsed/RSS checks at case boundaries, not instantaneous peaks or hard real-time guarantees",
      additionalCompute: summary.additionalCompute,
    });
    return summary;
  } catch (error) {
    save("failure.json", {
      status: "FAIL",
      ...metadata,
      error: String(error),
      automaticRestart: false,
      originalInputsRewritten: false,
    });
    throw error;
  }
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  if (process.argv.length === 3 && process.argv[2] === "--help")
    console.log(
      "node scripts/research/point-simulation.mjs --output .research-output/<new-directory>",
    );
  else {
    const result = await executePointSimulation(
      parsePointSimulationArgs(process.argv.slice(2)),
    );
    console.log(
      JSON.stringify({
        status: result.status,
        summary: result.summary,
        additionalCompute: result.additionalCompute,
      }),
    );
  }
}
