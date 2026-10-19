import {
  readFileSync,
  writeFileSync,
  appendFileSync,
  mkdirSync,
  readdirSync,
} from "node:fs";
import { resolve, relative, isAbsolute } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { runBoundedChild } from "./bounded-child.mjs";
const root = resolve(fileURLToPath(new URL("../../", import.meta.url)));
const json = (p) => JSON.parse(readFileSync(p, "utf8"));
const hash = (p) => createHash("sha256").update(readFileSync(p)).digest("hex");
const save = (p, v) =>
  writeFileSync(p, JSON.stringify(v, null, 2) + "\n", { flag: "wx" });
const check = (ok, why) => {
  if (!ok) throw new Error(why);
};
const git = (args) =>
  execFileSync("git", args, {
    cwd: root,
    encoding: "utf8",
    windowsHide: true,
  }).trim();
const args = process.argv.slice(2),
  opts = {};
if (args.length === 1 && args[0] === "--help") {
  console.log(
    "node scripts/research/point-only-objective.mjs --python <existing-cpu-python> --source <paired> --original <outcome> --experiment <original-spatial> --previous <completed-combined-run> --output <new-ignored-directory>",
  );
  process.exit(0);
}
for (let i = 0; i < args.length; i += 2) {
  check(
    [
      "--python",
      "--source",
      "--original",
      "--experiment",
      "--previous",
      "--output",
    ].includes(args[i]) &&
      args[i + 1] &&
      !opts[args[i]],
    "Unknown/duplicate/missing argument",
  );
  opts[args[i]] = args[i + 1];
}
check(Object.keys(opts).length === 6, "Fixed six arguments; no tuning options");
const started = performance.now();
check(
  git(["status", "--porcelain"]) === "",
  "Commit source/config before execution",
);
const configPath = resolve(
    root,
    "research/experiments/point-only-objective-v1.json",
  ),
  c = json(configPath);
check(
  c.approval === "USER_APPROVED" &&
    c.status === "FROZEN_BEFORE_EXECUTION" &&
    c.training.runs === 2 &&
    c.training.epochs === 12 &&
    c.forwardBudget.totalTopLevelMax === 12840 &&
    c.resources.perChildWallSeconds === 600 &&
    c.resources.totalSupervisorWallSeconds === 1860,
  "Fixed approved contract",
);
const source = resolve(root, opts["--source"]),
  original = resolve(root, opts["--original"]),
  experiment = resolve(root, opts["--experiment"]),
  output = resolve(root, opts["--output"]);
const inside = relative(resolve(root, ".research-output"), output);
check(
  inside && !inside.startsWith("..") && !isAbsolute(inside),
  "New ignored output required",
);
const frozen = json(resolve(experiment, "execution.json"));
const own = [
  "scripts/research/point-only-objective.mjs",
  "scripts/research/bounded-child.mjs",
  "scripts/research/test-bounded-child.mjs",
  "research/experiments/point-only-objective-v1.json",
  "research/experiments/point-only-objective-proposal-v1.json",
  ...readdirSync(resolve(root, "research/cnn"))
    .filter((p) => p.endsWith(".py"))
    .map((p) => "research/cnn/" + p),
];
const sourceHashes = {
  ...frozen.sourceHashes,
  ...Object.fromEntries(own.map((p) => [p, hash(resolve(root, p))])),
};
for (const [p, h] of Object.entries(frozen.sourceHashes))
  check(hash(resolve(root, p)) === h, "Original source changed: " + p);
mkdirSync(output, { recursive: false });
save(resolve(output, "execution.json"), {
  status: "FROZEN_BEFORE_EXECUTION",
  atUtc: new Date().toISOString(),
  sourceHead: git(["rev-parse", "HEAD"]),
  configSha256: hash(configPath),
  sourceHashes,
  previousControlPath: resolve(root, opts["--previous"]),
});
let child = null,
  cancelled = false,
  ownerTimedOut = false;
const progress = (phase, extra = {}) => {
  const row = { atUtc: new Date().toISOString(), phase, ...extra };
  appendFileSync(resolve(output, "progress.jsonl"), JSON.stringify(row) + "\n");
  console.log(JSON.stringify(row));
};
const cancel = () => {
  cancelled = true;
  child?.kill();
};
process.once("SIGINT", cancel);
process.once("SIGTERM", cancel);
const ownerTimer = setTimeout(
  () => {
    ownerTimedOut = true;
    child?.kill();
  },
  Math.max(0, 1860000 - (performance.now() - started)),
);
const observations = [];
try {
  for (const mode of ["preflight", "control", "candidate"]) {
    check(!cancelled && !ownerTimedOut, "Cancelled/owner timeout; STOP");
    if (mode !== "preflight")
      appendFileSync(
        resolve(output, "training-ledger.jsonl"),
        JSON.stringify({
          mode,
          run: mode === "control" ? 1 : 2,
          epochs: 12,
          atUtc: new Date().toISOString(),
        }) + "\n",
      );
    const result = {
      mode,
      ...(await runBoundedChild({
        command: resolve(root, opts["--python"]),
        args: [
          resolve(root, "research/cnn/point_only.py"),
          "--mode",
          mode,
          "--source",
          source,
          "--original",
          original,
          "--experiment",
          experiment,
          "--execution",
          resolve(output, "execution.json"),
          "--output",
          resolve(output, mode),
        ],
        cwd: root,
        timeoutMs: 600000,
        onSpawn: (p) => {
          child = p;
          progress("child-start", { mode, pid: p.pid, timeoutMs: 600000 });
        },
        onStdout: (b) => {
          appendFileSync(resolve(output, mode + ".log"), b);
          process.stdout.write(b);
        },
        onStderr: (b) => {
          appendFileSync(resolve(output, mode + ".log"), b);
          process.stderr.write(b);
        },
      })),
      cancelled,
      ownerTimedOut,
    };
    child = null;
    progress("child-exit", result);
    check(
      result.code === 0 &&
        !result.timedOut &&
        !cancelled &&
        !ownerTimedOut &&
        result.elapsedMs <= 600000,
      mode + " failed/timed out/cancelled; STOP",
    );
    observations.push(result);
    const r = json(resolve(output, mode, "report.json"));
    check(
      r.status === "PASS" &&
        r.modelForwards === (mode === "preflight" ? 856 : 5992),
      "Exact mode count/report",
    );
    if (mode === "control")
      check(r.baselineReproduction, "Control reproduced before candidate");
  }
  check(
    performance.now() - started <= 1860000 && !ownerTimedOut && !cancelled,
    "Total owner budget; STOP",
  );
  const summary = json(resolve(output, "summary.json"));
  const calls = readFileSync(resolve(output, "forward-count.jsonl"), "utf8")
    .trim()
    .split("\n")
    .map(JSON.parse);
  check(
    calls.length === 12840 &&
      calls.every((r, i) => r.call === i + 1 && r.limit === 12840),
    "Global ledger exact",
  );
  for (const [p, h] of Object.entries(sourceHashes))
    check(hash(resolve(root, p)) === h, "Source changed after fit: " + p);
  save(resolve(output, "report.json"), {
    status: "PASS",
    atUtc: new Date().toISOString(),
    executionSha256: hash(resolve(output, "execution.json")),
    rawSha256: hash(resolve(output, "raw.json")),
    summarySha256: hash(resolve(output, "summary.json")),
    ledgerSha256: hash(resolve(output, "forward-count.jsonl")),
    accounting: summary.accounting,
    gate: summary.gate,
    mechanism: summary.mechanism,
    coordinateMechanism: summary.coordinateMechanism,
    processBudgets: observations,
    totalOwnerMs: performance.now() - started,
    preservation: "NOT_VERIFIED; separate original-file audit required",
  });
  save(resolve(output, "freeze.json"), {
    status: "FROZEN_DEV_ONLY",
    executionSha256: hash(resolve(output, "execution.json")),
    reportSha256: hash(resolve(output, "report.json")),
  });
  progress("complete", {
    gate: summary.gate,
    mechanism: summary.mechanism,
    forwards: 12840,
  });
} catch (error) {
  save(resolve(output, "failure.json"), {
    status: "FAIL",
    error: String(error),
    additionalTrainingAllowed: false,
    preservation: "NOT_VERIFIED; separate original-file audit required",
  });
  throw error;
} finally {
  clearTimeout(ownerTimer);
  child?.kill();
  process.removeListener("SIGINT", cancel);
  process.removeListener("SIGTERM", cancel);
}
