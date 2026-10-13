import {
  readFileSync,
  writeFileSync,
  appendFileSync,
  mkdirSync,
} from "node:fs";
import { resolve, relative, isAbsolute } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { spawn, execFileSync } from "node:child_process";
import { compileResearch } from "./runtime.mjs";

const root = resolve(fileURLToPath(new URL("../../", import.meta.url)));
const json = (p) => JSON.parse(readFileSync(p, "utf8"));
const hash = (p) => createHash("sha256").update(readFileSync(p)).digest("hex");
const write = (p, v) =>
  writeFileSync(p, JSON.stringify(v, null, 2) + "\n", { flag: "wx" });
const require = (ok, why) => {
  if (!ok) throw new Error(why);
};
const git = (args) =>
  execFileSync("git", args, {
    cwd: root,
    encoding: "utf8",
    windowsHide: true,
  }).trim();
const args = process.argv.slice(2),
  options = {};
if (args.length === 1 && args[0] === "--help") {
  console.log(
    "node scripts/research/spatial-cnn.mjs --source <paired> --original <outcome> --python <existing-cpu-python> --output <new-ignored-directory>",
  );
  process.exit(0);
}
for (let i = 0; i < args.length; i += 2) {
  require(["--source", "--original", "--python", "--output"].includes(
    args[i],
  ) &&
    args[i + 1] &&
    !options[args[i]], "Unknown, duplicate or missing argument");
  options[args[i]] = args[i + 1];
}
require(Object.keys(options).length ===
  4, "source/original/python/output required; no tuning options");
require(git(["status", "--porcelain"]) ===
  "", "Commit source/config before execution");
const source = resolve(root, options["--source"]),
  original = resolve(root, options["--original"]),
  output = resolve(root, options["--output"]);
const inside = relative(resolve(root, ".research-output"), output);
require(inside &&
  !inside.startsWith("..") &&
  !isAbsolute(inside), "New ignored output required");
const configPath = resolve(
    root,
    "research/experiments/spatial-residual-v1.json",
  ),
  config = json(configPath);
require(config.status === "FROZEN_BEFORE_EXECUTION" &&
  config.approval === "USER_APPROVED" &&
  config.limits.trainingRuns === 2 &&
  config.limits.epochs === 12 &&
  config.limits.footprintCandidates === 149597 &&
  config.limits.childSeconds === 600 &&
  config.limits.workingSetBytes === 2147483648 &&
  config.limits.chunk === 32 &&
  config.limits.teacherDomainTestRollout ===
    0, "Fixed approved spatial contract");
require(hash(resolve(source, "train-dev/dataset.json")) ===
  config.source.datasetSha256, "Original paired dataset hash");
const originalHashes = Object.fromEntries(
  Object.entries(config.source.outcomeHashes).map(([p, h]) => {
    require(hash(resolve(original, p)) === h, "Original outcome hash: " + p);
    return [p, h];
  }),
);
const oldFreeze = json(resolve(source, "freeze.json")),
  oldExecution = json(resolve(original, "execution.json"));
const fixedSources = {
  ...oldFreeze.sourceHashes,
  ...oldExecution.sourceHashes,
};
for (const [p, h] of Object.entries(fixedSources))
  require(hash(resolve(root, p)) === h, "Original frozen source: " + p);
mkdirSync(output, { recursive: false });
let child = null,
  aborting = false;
const progress = (phase, extra = {}) => {
  const event = { at: new Date().toISOString(), phase, ...extra };
  appendFileSync(
    resolve(output, "progress.jsonl"),
    JSON.stringify(event) + "\n",
  );
  console.log(JSON.stringify(event));
};
const stop = () => {
  aborting = true;
  child?.kill();
};
process.once("SIGINT", stop);
process.once("SIGTERM", stop);
const prepareStarted = performance.now();
let preparationMs;
async function python(mode) {
  require(!aborting, "Cancelled; STOP");
  if (mode !== "preflight")
    appendFileSync(
      resolve(output, "training-ledger.jsonl"),
      JSON.stringify({
        mode,
        run: mode === "baseline" ? 1 : 2,
        epochs: 12,
        at: new Date().toISOString(),
      }) + "\n",
    );
  const log = resolve(output, mode + ".log"),
    start = performance.now();
  const result = await new Promise((success, reject) => {
    child = spawn(
      resolve(options["--python"]),
      [
        resolve(root, "research/cnn/spatial.py"),
        "--mode",
        mode,
        "--source",
        source,
        "--original",
        original,
        "--execution",
        resolve(output, "execution.json"),
        "--output",
        resolve(output, mode),
      ],
      { cwd: root, windowsHide: true, stdio: ["ignore", "pipe", "pipe"] },
    );
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child?.kill();
    }, 600000 - preparationMs);
    progress("child-start", {
      mode,
      pid: child.pid,
      timeoutMs: 600000 - preparationMs,
      preparationIncludedMs: preparationMs,
    });
    child.stdout.on("data", (b) => {
      appendFileSync(log, b);
      process.stdout.write(b);
    });
    child.stderr.on("data", (b) => {
      appendFileSync(log, b);
      process.stderr.write(b);
    });
    child.once("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.once("close", (code, signal) => {
      clearTimeout(timer);
      child = null;
      const observation = {
        mode,
        code,
        signal,
        timedOut,
        childMs: performance.now() - start,
        sharedPreparationMs: preparationMs,
        limitMs: 600000,
      };
      progress("child-exit", observation);
      if (code === 0 && !timedOut && !aborting) success(observation);
      else
        reject(
          new Error(
            `${mode} failed (${code ?? signal}), timeout=${timedOut}; STOP`,
          ),
        );
    });
  });
  require(result.childMs + preparationMs <=
    600000, "Whole preparation/child time limit; STOP");
  const report = json(resolve(output, mode, "report.json"));
  require(report.status === "PASS", mode + " report failed");
  return { ...report, processBudget: result };
}
try {
  progress("footprint-encoding");
  const runtime = compileResearch(root, "src/research/spatial-footprint.ts");
  const { encodeFootprints } = await import(runtime.url);
  const pack = json(resolve(source, "train-dev/dataset.json"));
  require(pack.status === "PASS" &&
    pack.stage === "train-dev" &&
    pack.examples.test.length === 0, "Train/dev only");
  let candidates = 0;
  const examples = {};
  for (const split of ["train", "dev"]) {
    require(pack.examples[split].length ===
      (split === "train" ? 339 : 89), "Fixed split size");
    examples[split] = pack.examples[split].map((e) => {
      const result = encodeFootprints(e.candidateKeys, 149597 - candidates);
      candidates += result.candidates;
      require(process.memoryUsage().rss <= 2147483648 &&
        performance.now() - prepareStarted <
          600000, "Encoder time/working-set limit; STOP");
      return { id: e.id, footprints: result.footprints };
    });
  }
  require(candidates === 149597, "Exact footprint count");
  const oldSchema = json(resolve(original, "dataset.json")).schema;
  write(resolve(output, "footprints.json"), {
    schema: {
      ...oldSchema,
      version: "legal-candidate-spatial-residual-v3",
      footprintShape: [1, 16, 10],
    },
    sourceDataSha256: config.source.datasetSha256,
    outcomeDataSha256: originalHashes["dataset.json"],
    candidates,
    examples,
  });
  const paths = [
    "scripts/research/spatial-cnn.mjs",
    "scripts/research/runtime.mjs",
    "research/cnn/spatial.py",
    "research/cnn/test_spatial.py",
    "tests/research/spatial-footprint.test.ts",
    "research/experiments/spatial-residual-v1.json",
  ];
  const sourceHashes = {
    ...fixedSources,
    ...runtime.hashes,
    ...Object.fromEntries(paths.map((p) => [p, hash(resolve(root, p))])),
  };
  preparationMs = performance.now() - prepareStarted;
  require(preparationMs < 600000, "Preparation time limit; STOP");
  write(resolve(output, "execution.json"), {
    status: "FROZEN_BEFORE_EXECUTION",
    at: new Date().toISOString(),
    sourceHead: git(["rev-parse", "HEAD"]),
    configSha256: hash(configPath),
    sourceHashes,
    originalHashes,
    footprintsSha256: hash(resolve(output, "footprints.json")),
    preparationMs,
    compilerVersion: runtime.compilerVersion,
    trainingRuns: 2,
    additionalTeacherDomainTestRollout: 0,
  });
  const preflight = await python("preflight");
  const baseline = await python("baseline");
  require(Object.values(baseline.baselineReproduction).every(
    Boolean,
  ), "Baseline exact before spatial; STOP");
  const spatial = await python("spatial");
  const ledger = readFileSync(resolve(output, "training-ledger.jsonl"), "utf8")
    .trim()
    .split("\n")
    .map((l) => JSON.parse(l));
  require(ledger.length === 2 &&
    ledger[0].mode === "baseline" &&
    ledger[1].mode === "spatial", "Two training runs only");
  for (const [p, h] of Object.entries(originalHashes))
    require(hash(resolve(original, p)) === h, "Original file changed: " + p);
  for (const [p, h] of Object.entries(sourceHashes))
    require(hash(resolve(root, p)) === h, "Source changed: " + p);
  const b = baseline.summary.dev.targets.A512,
    s = spatial.summary.dev.targets.A512;
  require(b.overall.examples === 89 &&
    s.overall.examples === 89 &&
    s.groups.teacherAction["single-cell"].examples === 13 &&
    s.groups.teacherAction.reroll.examples === 8, "Fixed gate denominators");
  const checks = {
    aboveFirstLegal: s.overall.correct >= 48,
    gainAtLeastFive: s.overall.correct >= b.overall.correct + 5,
    singleCellAtLeastThree: s.groups.teacherAction["single-cell"].correct >= 3,
    rerollAtLeastSeven: s.groups.teacherAction.reroll.correct >= 7,
  };
  const report = {
    status: "PASS",
    at: new Date().toISOString(),
    executionSha256: hash(resolve(output, "execution.json")),
    footprintsSha256: hash(resolve(output, "footprints.json")),
    control: preflight.control,
    audit: baseline.audit,
    models: { baseline, spatial },
    gate: {
      status: Object.values(checks).every(Boolean) ? "PASS" : "FAIL",
      checks,
    },
    accounting: {
      observations: 428,
      footprintCandidates: candidates,
      trainingRuns: 2,
      epochsPerRun: 12,
      modelForwards:
        preflight.control.modelForwards +
        baseline.modelForwards +
        spatial.modelForwards,
      teacherCalls: 0,
      domainTransitions: 0,
      newEpisodes: 0,
      testLoaded: false,
      testScored: false,
      policyRollouts: 0,
      additionalTuning: false,
    },
    limits: [
      "Previously examined dev: screening only, not held-out generalization",
      "Gate failure does not permit extra tuning",
      "No app model or policy survival claim",
    ],
  };
  write(resolve(output, "report.json"), report);
  write(resolve(output, "freeze.json"), {
    status: "FROZEN_DEV_ONLY",
    executionSha256: report.executionSha256,
    reportSha256: hash(resolve(output, "report.json")),
    checkpointSha256: {
      baseline: baseline.checkpointSha256,
      spatial: spatial.checkpointSha256,
    },
  });
  progress("complete", {
    status: "PASS",
    gate: report.gate,
    dev: { baseline: b.overall.correct, spatial: s.overall.correct },
  });
} catch (error) {
  write(resolve(output, "failure.json"), {
    status: "FAIL",
    error: String(error),
    additionalTrainingAllowed: false,
    preservation: "NOT_VERIFIED; separate original-file audit required",
  });
  throw error;
} finally {
  child?.kill();
  process.removeListener("SIGINT", stop);
  process.removeListener("SIGTERM", stop);
}
