import assert from "node:assert/strict";
import { resolve } from "node:path";
import {
  root,
  json,
  hash,
  bytes,
  write,
  reserveFitOutput,
  auditCoverageStored,
  auditCoverageData,
  approvedCoverage,
} from "./coverage-preflight-contract.mjs";
import { auditMlpStored, pairedMlp, mlpGate } from "./outcome-mlp-contract.mjs";
export {
  root,
  json,
  hash,
  bytes,
  write,
  reserveFitOutput,
  auditCoverageStored,
  auditMlpStored,
  approvedCoverage,
};
export const configPath = "research/experiments/coverage-training-v1.json";
export const phases = {
  anchor: 16,
  train: 16000,
  fit1: 16,
  dev1: 16,
  final1: 16,
  fit2: 16,
  dev2: 16,
  final2: 16,
};
export function approvedCoverageTraining(base = root) {
  const p = approvedCoverage(base).contract,
    c = json(resolve(base, configPath));
  assert.equal(c.id, "coverage-training-v1");
  assert.equal(c.approval, "USER_APPROVED");
  assert.equal(c.status, "FROZEN_BEFORE_EXECUTION");
  assert.equal(c.proposalSha256, hash(resolve(base, c.proposalPath)));
  assert.deepEqual(c.training, p.futureTraining);
  const data = json(
    resolve(base, "assets/research/coverage-preflight-20261104-v1.json"),
  );
  assert.equal(data.status, "PASS");
  assert.equal(c.dataSha256, data.dataSha256);
  assert.equal(
    c.preflightAssetSha256,
    hash(resolve(base, "assets/research/coverage-preflight-20261104-v1.json")),
  );
  assert.equal(
    c.anchor.datasetSha256,
    json(resolve(base, "research/experiments/outcome-mlp-v1.json")).contract
      .original.fitDatasetSha256,
  );
  assert.deepEqual(
    c.modelConfig,
    json(resolve(base, "research/experiments/outcome-mlp-v1.json")).contract
      .shared.modelConfig,
  );
  assert.deepEqual(c.resources, {
    childSeconds: 600,
    ownerSeconds: 1260,
    sampledWorkingSetBytes: 2147483648,
    instantaneousPeakMeasured: false,
  });
  assert.deepEqual(c.boundaries, {
    model2: true,
    checkpointLoads: 0,
    additionalTuningRetry: 0,
    additionalDomainDataset: 0,
    appUsesDfs: true,
  });
  return c;
}
export function parseCoverageTrainingArgs(kind, args) {
  assert(["run", "replay"].includes(kind));
  const names =
    kind === "run"
      ? ["--data", "--original", "--output", "--python"]
      : ["--output", "--recorded"];
  assert.equal(args.length, 2 * names.length, "Fixed paths only");
  const pairs = [];
  for (let i = 0; i < args.length; i += 2) pairs.push(args.slice(i, i + 2));
  assert.deepEqual(pairs.map((p) => p[0]).sort(), names);
  assert(pairs.every((p) => p[1] && !p[1].startsWith("--")));
  return Object.fromEntries(pairs.map(([k, v]) => [k.slice(2), v]));
}
export function auditCoverageTrainingLedger(rows, arm) {
  assert(["control", "diverse"].includes(arm));
  assert.equal(rows.length, 16112);
  let offset = 0;
  const start = arm === "control" ? 0 : 16112;
  for (const [phase, n] of Object.entries(phases)) {
    for (let i = 0; i < n; i++)
      assert.deepEqual(rows[offset + i], {
        call: start + offset + i + 1,
        arm,
        phase,
        phaseCall: i + 1,
        step: phase === "train" ? Math.floor(i / 16) + 1 : null,
        example: i % 16,
      });
    offset += n;
  }
}
export function auditCoveragePredictions(rows, examples) {
  assert.equal(rows.length, 16);
  assert.equal(examples.length, 16);
  rows.forEach((r, i) => {
    const e = examples[i],
      s = r.logits;
    assert.equal(r.id, e.id);
    assert.equal(r.label, e.label);
    assert.equal(s.length, 46);
    assert(s.every((v) => Number.isFinite(v) && Math.fround(v) === v));
    const max = Math.max(...s),
      index = s.indexOf(max),
      nll =
        Math.log(s.reduce((a, v) => a + Math.exp(v - max), 0)) +
        max -
        s[e.label],
      margin = s[e.label] - Math.max(...s.filter((_, j) => j !== e.label)),
      rank =
        1 +
        s.filter(
          (v, j) =>
            j !== e.label &&
            (v > s[e.label] || (v === s[e.label] && j < e.label)),
        ).length;
    assert.equal(r.index, index);
    assert.equal(r.correct, index === e.label);
    assert.equal(r.rank, rank);
    for (const [a, b] of [
      [r.nll, nll],
      [r.margin, margin],
    ])
      assert(
        Number.isFinite(a) &&
          Math.abs(a - b) <= 2e-6 * Math.max(1, Math.abs(a), Math.abs(b)),
      );
  });
  return {
    examples: 16,
    correct: rows.filter((r) => r.correct).length,
    meanNll: rows.reduce((a, r) => a + r.nll, 0) / 16,
    minimumTargetMargin: Math.min(...rows.map((r) => r.margin)),
    maximumTargetRank: Math.max(...rows.map((r) => r.rank)),
  };
}
export function auditCoverageArm(
  recorded,
  arm,
  c = approvedCoverageTraining(),
) {
  assert(["control", "diverse"].includes(arm));
  const path = (p) => resolve(recorded, arm, p),
    r = json(path("report.json")),
    data = json(resolve(recorded, "dataset.json"));
  assert.equal(r.status, "PASS");
  assert.equal(r.arm, arm);
  assert.equal(r.executionSha256, hash(resolve(recorded, "execution.json")));
  const names = [
    "anchor.json",
    "history.json",
    "fit1.json",
    "dev1.json",
    "final1.json",
    "fit2.json",
    "dev2.json",
    "final2.json",
    "forward-count.jsonl",
    "final-weights.pt",
    "model-intent.json",
    "summary.json",
  ];
  assert.deepEqual(Object.keys(r.hashes).sort(), names.sort());
  for (const p of names) assert.equal(hash(path(p)), r.hashes[p]);
  assert.equal(hash(path("anchor.json")), c.anchor.logitsSha256);
  assert.equal(r.initialLogicalWeightsSha256, c.anchor.weightsSha256);
  assert.equal(r.anchorExact, true);
  assert.equal(r.parameters, 769);
  assert.equal(r.topLevelModels, 1);
  assert.equal(r.checkpointLoads, 0);
  assert.deepEqual(json(path("model-intent.json")), {
    topLevelModels: 1,
    checkpointLoads: 0,
    arm,
  });
  assert.deepEqual(r.forwardCounts, phases);
  auditCoverageTrainingLedger(
    bytes(path("forward-count.jsonl"))
      .toString()
      .trim()
      .split("\n")
      .map(JSON.parse),
    arm,
  );
  assert(
    r.budget.wallTimeSeconds <= 600 &&
      r.budget.maxSampledWorkingSetBytes <= 2147483648 &&
      r.budget.instantaneousPeakMeasured === false,
  );
  assert.equal(r.environment.cuda, null);
  assert.equal(r.environment.threads, 2);
  assert.equal(r.environment.interopThreads, 1);
  const h = json(path("history.json"));
  assert.equal(h.length, 1000);
  h.forEach((s, i) => {
    assert.equal(s.step, i + 1);
    assert(Number.isFinite(s.meanTrainingNll) && s.meanTrainingNll >= 0);
  });
  const splits = {};
  for (const split of ["fit", "dev", "final"]) {
    assert.equal(hash(path(split + "1.json")), hash(path(split + "2.json")));
    const metrics = auditCoveragePredictions(
      json(path(split + "1.json")),
      data.splits[split === "fit" ? arm : split],
    );
    splits[split] = { metrics, gate: mlpGate(metrics) };
  }
  const summary = { arm, updates: 1000, splits };
  assert.deepEqual(r.summary, summary);
  assert.deepEqual(json(path("summary.json")), summary);
  if (arm === "diverse")
    assert.deepEqual(json(resolve(recorded, "diverse-input.json")), {
      executionSha256: hash(resolve(recorded, "execution.json")),
      controlReportSha256: hash(resolve(recorded, "control/report.json")),
    });
  return { summary, report: r };
}
export function auditCoverageTrainingStored(recorded, base = root) {
  const c = approvedCoverageTraining(base),
    path = (p) => resolve(recorded, p),
    r = json(path("report.json")),
    f = json(path("execution.json"));
  assert.equal(r.status, "PASS");
  assert(r.elapsedSeconds <= 1260);
  assert.equal(r.children.length, 2);
  r.children.forEach((c, i) => {
    assert.equal(c.arm, ["control", "diverse"][i]);
    assert.equal(c.result.code, 0);
    assert.equal(c.result.timedOut, false);
    assert(c.result.elapsedMs <= 600000);
  });
  assert.equal(f.configSha256, hash(resolve(base, configPath)));
  assert.deepEqual(Object.keys(r.hashes).sort(), [
    "anchor-dataset.json",
    "control/report.json",
    "dataset.json",
    "diverse-input.json",
    "diverse/report.json",
    "execution.json",
    "start.json",
  ]);
  for (const [p, h] of Object.entries(r.hashes)) assert.equal(hash(path(p)), h);
  assert.deepEqual(f.inputHashes, {
    "dataset.json": c.dataSha256,
    "anchor-dataset.json": c.anchor.datasetSha256,
  });
  for (const [p, h] of Object.entries(f.inputHashes))
    assert.equal(hash(path(p)), h);
  const data = json(path("dataset.json"));
  auditCoverageData(data);
  const control = auditCoverageArm(recorded, "control", c),
    diverse = auditCoverageArm(recorded, "diverse", c),
    paired = {};
  for (const split of ["dev", "final"])
    paired[split] = pairedMlp(
      json(path(`control/${split}1.json`)),
      json(path(`diverse/${split}1.json`)),
    );
  const baselines = {};
  for (const split of ["dev", "final"]) {
    const rows = data.splits[split];
    baselines[split] = {
      examples: 16,
      firstLegalCorrect: rows.filter((e) => e.label === 0).length,
      immediateClearCorrect: rows.filter(
        (e) =>
          e.candidateFeatures
            .map((f) => f[52])
            .indexOf(Math.max(...e.candidateFeatures.map((f) => f[52]))) ===
          e.label,
      ).length,
    };
  }
  return {
    schemaVersion: 1,
    id: "coverage-training-v1",
    technicalStatus: "PASS",
    dataSha256: c.dataSha256,
    control: control.summary,
    diverse: diverse.summary,
    paired,
    baselines,
    reproduction: {
      originalOld16InitialAnchorExact: true,
      final48EachRepeatedExact: true,
    },
    actualCompute: {
      topLevelModels: 2,
      checkpointLoads: 0,
      topLevelForwards: 32224,
      trainingRuns: 2,
      optimizerUpdates: 2000,
      trainingObservationsEach: 16000,
      trainingCandidatesEach: 736000,
    },
    timing: {
      ownerWallSeconds: r.elapsedSeconds,
      children: r.children.map(({ arm, result }) => ({
        arm,
        processWallSeconds: result.elapsedMs / 1000,
        pythonBody: (arm === "control" ? control : diverse).report.budget,
      })),
    },
    additionalPythonModelLoadForwardTrainingDomainDataset: 0,
    interpretation:
      "New FP32 inputs within a prospective synthetic immediate-clear family; not natural-game generalization, future planning or app adoption. Data coverage varies, model and exposures match.",
    appUsesDfs: true,
  };
}
