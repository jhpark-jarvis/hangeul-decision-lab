import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, relative, isAbsolute } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { compileResearch } from "./runtime.mjs";

const root = resolve(fileURLToPath(new URL("../../", import.meta.url)));
const json = (p) => JSON.parse(readFileSync(p, "utf8"));
const hash = (p) => createHash("sha256").update(readFileSync(p)).digest("hex");
const write = (p, data) =>
  writeFileSync(p, JSON.stringify(data, null, 2) + "\n", { flag: "wx" });
const check = (ok, reason) => {
  if (!ok) throw new Error(reason);
};
const args = process.argv.slice(2),
  options = {};
if (args.length === 1 && args[0] === "--help") {
  console.log(
    "node scripts/research/action-errors.mjs --source <paired-root> --experiment <frozen-outcome-root> --output <new-ignored-output>",
  );
  process.exit(0);
}
for (let i = 0; i < args.length; i += 2) {
  check(
    ["--source", "--experiment", "--output"].includes(args[i]) &&
      args[i + 1] &&
      !options[args[i]],
    "Unknown/duplicate/missing argument",
  );
  options[args[i]] = args[i + 1];
}
check(Object.keys(options).length === 3, "source/experiment/output required");
const output = resolve(root, options["--output"]),
  inside = relative(resolve(root, ".research-output"), output);
check(
  inside && !inside.startsWith("..") && !isAbsolute(inside),
  "Output must be a new ignored directory",
);
const git = (args) =>
  execFileSync("git", args, {
    cwd: root,
    windowsHide: true,
    encoding: "utf8",
  }).trim();
check(
  git(["status", "--porcelain"]) === "",
  "Commit audit source/config before execution",
);
mkdirSync(output, { recursive: false });
try {
  const source = resolve(options["--source"]),
    experiment = resolve(options["--experiment"]);
  const configPath = resolve(
      root,
      "research/experiments/action-error-audit-v1.json",
    ),
    config = json(configPath);
  check(config.status === "FROZEN_READONLY_AUDIT", "Fixed readonly protocol");
  const input = {
    dataset: resolve(source, "train-dev/dataset.json"),
    execution: resolve(experiment, "execution.json"),
    report: resolve(experiment, "report.json"),
  };
  for (const [key, path] of Object.entries(input))
    check(
      hash(path) === config.source[key + "Sha256"],
      "Frozen input hash: " + key,
    );
  const frozen = json(resolve(source, "freeze.json")),
    execution = json(input.execution),
    report = json(input.report);
  check(
    hash(resolve(source, "freeze.json")) ===
      execution.originalHashes["freeze.json"],
    "Original paired freeze identity",
  );
  for (const [p, digest] of Object.entries({
    ...frozen.sourceHashes,
    ...execution.sourceHashes,
  }))
    check(hash(resolve(root, p)) === digest, "Original frozen source: " + p);
  check(
    report.status === "PASS" &&
      report.executionSha256 === hash(input.execution),
    "Frozen experiment identity",
  );
  const inputs = {
    ...input,
    pairedFreeze: resolve(source, "freeze.json"),
    outcomeFreeze: resolve(experiment, "freeze.json"),
  };
  const sealed = json(inputs.outcomeFreeze);
  check(
    sealed.status === "FROZEN_DEV_ONLY" &&
      sealed.reportSha256 === hash(input.report),
    "Outcome report freeze",
  );
  const predictions = {};
  for (const model of ["baseline", "outcome"]) {
    const raw = resolve(experiment, model, "predictions.json"),
      checkpoint = resolve(experiment, model, "checkpoint.pt");
    check(
      hash(raw) === config.source.predictionSha256[model],
      "Frozen prediction hash",
    );
    check(
      hash(checkpoint) === report.models[model].checkpointSha256,
      "Frozen checkpoint hash",
    );
    inputs[model + "Predictions"] = raw;
    inputs[model + "Checkpoint"] = checkpoint;
    predictions[model] = json(raw);
  }
  const originalHashes = Object.fromEntries(
    Object.entries(inputs).map(([key, p]) => [key, hash(p)]),
  );
  const runtime = compileResearch(root, "src/research/action-errors.ts"),
    api = await import(runtime.url);
  const sourceHashes = {
    ...runtime.hashes,
    ...Object.fromEntries(
      [
        "scripts/research/action-errors.mjs",
        "scripts/research/runtime.mjs",
        "research/experiments/action-error-audit-v1.json",
      ].map((p) => [p, hash(resolve(root, p))]),
    ),
  };
  write(resolve(output, "started.json"), {
    protocolDate: "2026-10-13",
    atUtc: new Date().toISOString(),
    sourceHead: git(["rev-parse", "HEAD"]),
    originalHashes,
    configSha256: hash(configPath),
    sourceHashes,
  });
  const pack = json(input.dataset);
  check(
    pack.status === "PASS" &&
      pack.stage === "train-dev" &&
      pack.examples.test.length === 0,
    "Train/dev only source",
  );
  const raw = {},
    summary = {};
  for (const model of ["baseline", "outcome"]) {
    check(
      Object.keys(predictions[model]).sort().join() === "dev,train",
      "Frozen prediction splits",
    );
    raw[model] = {};
    summary[model] = {};
    for (const split of ["train", "dev"]) {
      check(
        pack.examples[split].length === config.source.examples[split],
        "Fixed observation count",
      );
      const result = api.auditStored(
        pack.examples[split],
        predictions[model][split],
      );
      check(
        result.summary.overall.correct ===
          report.models[model].summary[split].targets.A512.overall.correct,
        "Original hard-label metric drift",
      );
      raw[model][split] = result.rows;
      summary[model][split] = result.summary;
    }
  }
  for (const [key, path] of Object.entries(inputs))
    check(
      hash(path) === originalHashes[key],
      "Original input changed during audit",
    );
  for (const [p, digest] of Object.entries(sourceHashes))
    check(hash(resolve(root, p)) === digest, "Audit source changed");
  write(resolve(output, "raw.json"), raw);
  const result = {
    schemaVersion: 1,
    id: config.id,
    status: "PASS",
    method: config.method,
    identity: {
      inputHashes: originalHashes,
      configSha256: hash(configPath),
      sourceHashes,
    },
    summary,
    accounting: {
      observationsPerModel: 428,
      models: 2,
      trainingRuns: 0,
      inferenceCalls: 0,
      teacherCalls: 0,
      domainTransitions: 0,
      newEpisodes: 0,
      testParsedOrScored: 0,
      labelsOrModelSelectionChanged: false,
    },
    limits: [
      "Stored at-most-three alternative full paths; not a complete equal-action set",
      "Full-path comparator equality does not establish equal long-term value",
      "No relabeling, relaxed accuracy adoption, new training or generalization claim",
      "Model predictions reused; no all-candidate logits or within-kind ranking available",
    ],
  };
  write(resolve(output, "summary.json"), result);
  write(resolve(output, "report.json"), {
    ...result,
    protocolDate: "2026-10-13",
    atUtc: new Date().toISOString(),
    startedSha256: hash(resolve(output, "started.json")),
    rawSha256: hash(resolve(output, "raw.json")),
    environment: { node: process.version, compiler: runtime.compilerVersion },
  });
  console.log(
    JSON.stringify({
      status: "PASS",
      summary: Object.fromEntries(
        Object.entries(summary).map(([name, splits]) => [
          name,
          Object.fromEntries(
            Object.entries(splits).map(([s, v]) => [s, v.overall]),
          ),
        ]),
      ),
    }),
  );
} catch (error) {
  write(resolve(output, "failure.json"), {
    status: "FAIL",
    error: String(error),
    originalInputsPreserved: true,
  });
  throw error;
}
