import {
  readFileSync,
  writeFileSync,
  mkdirSync,
  appendFileSync,
  existsSync,
  copyFileSync,
} from "node:fs";
import { resolve, relative, isAbsolute } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { spawn, execFileSync } from "node:child_process";
import { compileResearch } from "./runtime.mjs";
import { validatePretrainingReuse } from "./outcome-recovery.mjs";

const root = resolve(fileURLToPath(new URL("../../", import.meta.url)));
const json = (path) => JSON.parse(readFileSync(path, "utf8"));
const hash = (path) =>
  createHash("sha256").update(readFileSync(path)).digest("hex");
const write = (path, value) =>
  writeFileSync(path, JSON.stringify(value, null, 2) + "\n", { flag: "wx" });
function require(condition, reason) {
  if (!condition) throw new Error(reason);
}
const argv = process.argv.slice(2),
  options = {};
if (argv.length === 1 && argv[0] === "--help") {
  console.log(
    "node scripts/research/outcome-cnn.mjs --source <original-paired-directory> --python <existing-cpu-python> --output <new-ignored-directory>",
  );
  process.exit(0);
}
for (let i = 0; i < argv.length; i += 2) {
  const key = argv[i];
  require(["--source", "--python", "--output", "--encoded"].includes(key) &&
    argv[i + 1] &&
    !options[key], "Unknown, duplicate or missing argument");
  options[key] = argv[i + 1];
}
require(["--source", "--python", "--output"].every(
  (key) => options[key],
), "source/python/output required; no extra training options");
const output = resolve(root, options["--output"]);
const inside = relative(resolve(root, ".research-output"), output);
require(inside &&
  !inside.startsWith("..") &&
  !isAbsolute(inside), "Output must be a new ignored experiment directory");
const git = (args) =>
  execFileSync("git", args, {
    cwd: root,
    encoding: "utf8",
    windowsHide: true,
  }).trim();
require(git(["status", "--porcelain"]) ===
  "", "Commit source/config before execution");
const source = resolve(root, options["--source"]);
const configPath = resolve(
  root,
  "research/experiments/outcome-features-v1.json",
);
const config = json(configPath);
require(config.status === "FROZEN_BEFORE_EXECUTION" &&
  config.approval === "USER_APPROVED", "Frozen approved config required");
require(config.limits.additionalDomainTransitions === 149597 &&
  config.limits.trainingRuns === 2 &&
  config.training.epochs === 12 &&
  config.source.examples.train === 339 &&
  config.source.examples.dev === 89 &&
  config.features.candidateSize === 74 &&
  config.source.newTeacherCalls === 0 &&
  config.source.testAccess === false, "Fixed experiment limits");
require(hash(resolve(source, "train-dev/dataset.json")) ===
  config.source.dataSha256, "Original train/dev hash");
// Record byte hashes only. Never load, parse, encode or evaluate the original test split.
const originalFiles = [
  "freeze.json",
  "protocol.json",
  "train-dev/dataset.json",
  "train-dev/protocol.json",
  "train-dev/manifest.json",
  "test/dataset.json",
  "test/protocol.json",
  "test/manifest.json",
  "models/A512/checkpoint.pt",
  "models/A512/report.json",
  "models/A1024/checkpoint.pt",
  "models/A1024/report.json",
];
const originalHashes = Object.fromEntries(
  originalFiles.map((p) => [p, hash(resolve(source, p))]),
);
const oldFreeze = json(resolve(source, "freeze.json"));
for (const [p, digest] of Object.entries(oldFreeze.sourceHashes))
  require(hash(resolve(root, p)) ===
    digest, "Original frozen source changed: " + p);
mkdirSync(output, { recursive: false });
const progress = (phase, extra = {}) => {
  const event = { at: new Date().toISOString(), phase, ...extra };
  appendFileSync(
    resolve(output, "progress.jsonl"),
    JSON.stringify(event) + "\n",
  );
  console.log(JSON.stringify(event));
};
let child = null;
const stop = () => {
  child?.kill();
};
process.once("SIGINT", stop);
process.once("SIGTERM", stop);
async function python(mode) {
  const log = resolve(output, `${mode}.log`);
  await new Promise((success, reject) => {
    child = spawn(
      resolve(options["--python"]),
      [
        resolve(root, "research/cnn/outcome.py"),
        "--mode",
        mode,
        "--source",
        source,
        "--augmented",
        resolve(output, "dataset.json"),
        "--execution",
        resolve(output, "execution.json"),
        "--output",
        resolve(output, mode),
      ],
      { cwd: root, windowsHide: true, stdio: ["ignore", "pipe", "pipe"] },
    );
    child.stdout.on("data", (data) => {
      appendFileSync(log, data);
      process.stdout.write(data);
    });
    child.stderr.on("data", (data) => {
      appendFileSync(log, data);
      process.stderr.write(data);
    });
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      child = null;
      if (code === 0) success();
      else reject(new Error(`${mode} failed (${code ?? signal}); STOP`));
    });
  });
  const report = json(resolve(output, mode, "report.json"));
  require(report.status === "PASS", mode + " report failed");
  return report;
}
try {
  progress("encoding");
  const runtime = compileResearch(root, "src/research/action-outcome.ts");
  const { extendOutcome, OUTCOME_SCHEMA } = await import(runtime.url);
  const pack = json(resolve(source, "train-dev/dataset.json"));
  require(pack.status === "PASS" &&
    pack.stage === "train-dev" &&
    pack.examples.test.length === 0, "Train/dev stage only");
  let transitions = 0;
  let recovery = null;
  const examples = {};
  if (options["--encoded"]) {
    const prior = resolve(root, options["--encoded"]);
    const oldExecution = json(resolve(prior, "execution.json"));
    const currentSources = Object.fromEntries(
      Object.keys(oldExecution.sourceHashes).map((p) => [
        p,
        hash(resolve(root, p)),
      ]),
    );
    recovery = validatePretrainingReuse({
      execution: oldExecution,
      configSha256: hash(configPath),
      originalHashes,
      currentSources,
      augmentedSha256: hash(resolve(prior, "dataset.json")),
      progress: readFileSync(resolve(prior, "progress.jsonl"), "utf8")
        .trim()
        .split("\n")
        .map((line) => JSON.parse(line)),
      failure: json(resolve(prior, "failure.json")),
      hasTrainingOutput: ["baseline", "outcome", "report.json"].some((p) =>
        existsSync(resolve(prior, p)),
      ),
    });
    recovery.priorExecutionSha256 = hash(resolve(prior, "execution.json"));
    recovery.priorFailureSha256 = hash(
      resolve(prior, "preflight/failure.json"),
    );
    copyFileSync(
      resolve(prior, "dataset.json"),
      resolve(output, "dataset.json"),
    );
    transitions = recovery.totalDomainTransitions;
    progress("encoded-reuse", recovery);
  } else
    for (const split of ["train", "dev"]) {
      require(pack.examples[split].length ===
        config.source.examples[split], "Fixed observation count");
      examples[split] = pack.examples[split].map((e, i) => {
        const result = extendOutcome(
          e,
          config.limits.additionalDomainTransitions - transitions,
        );
        transitions += result.transitions;
        if ((i + 1) % 50 === 0)
          progress("encoding", { split, observations: i + 1, transitions });
        return { id: e.id, candidateFeatures: result.candidateFeatures };
      });
    }
  require(transitions === 149597, "Exact transition count");
  if (!recovery)
    write(resolve(output, "dataset.json"), {
      schema: OUTCOME_SCHEMA,
      sourceDataSha256: config.source.dataSha256,
      examples,
      transitions,
    });
  const paths = [
    "scripts/research/outcome-cnn.mjs",
    "scripts/research/runtime.mjs",
    "scripts/research/outcome-recovery.mjs",
    "research/cnn/outcome.py",
    "research/cnn/diagnose.py",
    "research/experiments/outcome-features-v1.json",
  ];
  const sourceHashes = {
    ...runtime.hashes,
    ...Object.fromEntries(paths.map((p) => [p, hash(resolve(root, p))])),
  };
  const execution = {
    status: "FROZEN_BEFORE_EXECUTION",
    at: new Date().toISOString(),
    sourceHead: git(["rev-parse", "HEAD"]),
    configSha256: hash(configPath),
    sourceHashes,
    originalHashes,
    augmentedSha256: hash(resolve(output, "dataset.json")),
    transitions,
    teacherCalls: 0,
    testLoaded: false,
    newEpisodes: 0,
    recovery,
  };
  write(resolve(output, "execution.json"), execution);
  progress("preflight", { transitions });
  const control = await python("preflight");
  require(control.control.observations === 428 &&
    control.control.initialLogitsExact, "Initial control");
  progress("baseline");
  const baseline = await python("baseline");
  require(Object.values(baseline.baselineReproduction).every(
    (v) => v === true,
  ), "Baseline mismatch; STOP");
  progress("outcome");
  const outcome = await python("outcome");
  require(JSON.stringify(baseline.audit) ===
    JSON.stringify(outcome.audit), "Initial/common/order audit mismatch");
  const b = baseline.summary.dev.targets.A512,
    o = outcome.summary.dev.targets.A512;
  require(o.overall.examples === 89 &&
    o.groups.teacherAction["single-cell"].examples === 13 &&
    o.groups.teacherAction.reroll.examples === 8, "Gate denominators");
  const checks = {
    aboveFirstLegal: o.overall.correct >= 48,
    gainAtLeastFive: o.overall.correct >= b.overall.correct + 5,
    singleCellAtLeastThree: o.groups.teacherAction["single-cell"].correct >= 3,
    rerollAtLeastSeven: o.groups.teacherAction.reroll.correct >= 7,
  };
  for (const [p, digest] of Object.entries(originalHashes))
    require(hash(resolve(source, p)) === digest, "Original bytes changed: " +
      p);
  for (const [p, digest] of Object.entries({
    ...oldFreeze.sourceHashes,
    ...sourceHashes,
  }))
    require(hash(resolve(root, p)) === digest, "Execution source changed: " +
      p);
  const report = {
    status: "PASS",
    at: new Date().toISOString(),
    executionSha256: hash(resolve(output, "execution.json")),
    augmentedSha256: execution.augmentedSha256,
    control: control.control,
    trainingAudit: baseline.audit,
    models: Object.fromEntries(
      [baseline, outcome].map((r) => [
        r.mode,
        {
          selectedEpoch: r.selectedEpoch,
          checkpointSha256: r.checkpointSha256,
          weightsSha256: r.weightsSha256,
          history: r.history,
          summary: r.summary,
          baselineReproduction: r.baselineReproduction,
          timings: r.timings,
          memory: r.memory,
          environment: r.environment,
        },
      ]),
    ),
    gate: {
      status: Object.values(checks).every(Boolean) ? "PASS" : "FAIL",
      checks,
    },
    accounting: {
      observations: 428,
      transitions,
      teacherCalls: 0,
      newEpisodes: 0,
      trainingRuns: 2,
      epochsPerRun: 12,
      testLoaded: false,
      testScored: false,
      policyRollouts: 0,
      extraTuning: false,
      additionalTransitionsThisExecution: recovery ? 0 : transitions,
    },
    preservation: {
      originalFilesExact: originalFiles.length,
      originalSourcesExact: Object.keys(oldFreeze.sourceHashes).length,
    },
    limits: [
      "Known dev reused for experimental screening; not held-out generalization",
      "Gate failure stops all extra tuning",
      "No application model, RL, new test, policy survival or real-game validation",
    ],
  };
  write(resolve(output, "report.json"), report);
  write(resolve(output, "freeze.json"), {
    status: "FROZEN_DEV_ONLY",
    executionSha256: report.executionSha256,
    reportSha256: hash(resolve(output, "report.json")),
    models: {
      baseline: report.models.baseline,
      outcome: report.models.outcome,
    },
  });
  progress("complete", {
    status: "PASS",
    gate: report.gate,
    dev: { baseline: b.overall, outcome: o.overall },
  });
} catch (error) {
  write(resolve(output, "failure.json"), {
    status: "FAIL",
    error: String(error),
    additionalTrainingAllowed: false,
  });
  throw error;
} finally {
  stop();
  process.removeListener("SIGINT", stop);
  process.removeListener("SIGTERM", stop);
}
