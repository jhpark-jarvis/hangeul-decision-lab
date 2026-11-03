import assert from "node:assert/strict";
import { resolve } from "node:path";
import {
  root,
  json,
  hash,
  bytes,
  write,
  reserveFitOutput,
  auditTransferStored,
  auditTransferData,
  auditEvaluationRows,
  parseTransferArgs,
} from "./frozen-point-transfer-contract.mjs";
import {
  proposalPath,
  validateMlpProposal,
} from "./validate-outcome-mlp-proposal.mjs";
export {
  root,
  json,
  hash,
  bytes,
  write,
  reserveFitOutput,
  parseTransferArgs,
  auditTransferStored,
};
export const configPath = "research/experiments/outcome-mlp-v1.json";
export const phases = {
  initial: 16,
  train: 16000,
  fit1: 16,
  transfer1: 144,
  fit2: 16,
  transfer2: 144,
};
export function approvedMlp(base = root) {
  const p = json(resolve(base, proposalPath));
  validateMlpProposal(p, base);
  const c = json(resolve(base, configPath));
  assert.equal(c.schemaVersion, 1);
  assert.equal(c.id, "outcome-mlp-v1");
  assert.equal(c.status, "FROZEN_BEFORE_EXECUTION");
  assert.equal(c.approval, "USER_APPROVED");
  assert.equal(c.proposalPath, proposalPath);
  assert.equal(c.proposalSha256, hash(resolve(base, proposalPath)));
  assert.deepEqual(
    c.contract,
    Object.fromEntries(
      [
        "original",
        "shared",
        "contrast",
        "controls",
        "forwardBudget",
        "resources",
        "screening",
        "sourceHashes",
      ].map((k) => [k, p[k]]),
    ),
  );
  assert.deepEqual(c.boundaries, {
    existingSourcesImmutable: true,
    noNewDataDomainTeacher: true,
    checkpointLoads: 0,
    retryOrTuning: false,
    publicReplayAdditionalModelLoadForwardTraining: 0,
    inputEquivalenceNotGeneralization: true,
    appUsesDfs: true,
  });
  return c;
}
export function auditMlpLedger(rows, arm) {
  assert(["control", "candidate"].includes(arm));
  assert.equal(rows.length, 16336);
  let offset = 0;
  const global = arm === "control" ? 0 : 16336;
  for (const [phase, n] of Object.entries(phases)) {
    for (let i = 0; i < n; i++)
      assert.deepEqual(rows[offset + i], {
        call: global + offset + i + 1,
        arm,
        phase,
        phaseCall: i + 1,
        step: phase === "train" ? Math.floor(i / 16) + 1 : null,
        example: phase === "train" ? i % 16 : i,
      });
    offset += n;
  }
}
export function mlpGate(s) {
  assert([16, 144].includes(s.examples));
  assert(Number.isFinite(s.meanNll) && Number.isFinite(s.minimumTargetMargin));
  const checks = {
    allCorrect: s.correct === s.examples,
    meanNllAtMost005: s.meanNll <= 0.05,
    everyTargetMarginPositive: s.minimumTargetMargin > 0,
  };
  return {
    status: Object.values(checks).every(Boolean) ? "PASS" : "FAIL",
    checks,
  };
}
export function auditMlpArm(recorded, arm, config = approvedMlp()) {
  assert(["control", "candidate"].includes(arm));
  const path = (p) => resolve(recorded, arm, p),
    r = json(path("report.json")),
    f = json(resolve(recorded, "execution.json")),
    c = config.contract;
  assert.equal(r.status, "PASS");
  assert.equal(r.arm, arm);
  assert.equal(r.executionSha256, hash(resolve(recorded, "execution.json")));
  const names = [
    "initial.json",
    "history.json",
    "fit1.json",
    "transfer1.json",
    "fit2.json",
    "transfer2.json",
    "forward-count.jsonl",
    "summary.json",
    "final-weights.pt",
    "model-intent.json",
  ];
  assert.deepEqual(Object.keys(r.hashes).sort(), names.sort());
  for (const n of names) assert.equal(hash(path(n)), r.hashes[n]);
  assert.deepEqual(Object.keys(f.inputHashes).sort(), [
    "fit-dataset.json",
    "transfer-dataset.json",
  ]);
  for (const [n, h] of Object.entries(f.inputHashes))
    assert.equal(hash(resolve(recorded, n)), h);
  assert.equal(f.inputHashes["fit-dataset.json"], c.original.fitDatasetSha256);
  assert.equal(
    f.inputHashes["transfer-dataset.json"],
    c.original.transferDatasetSha256,
  );
  const fit = json(resolve(recorded, "fit-dataset.json")),
    transfer = json(resolve(recorded, "transfer-dataset.json"));
  auditTransferData(transfer, fit);
  assert.equal(hash(path("fit1.json")), hash(path("fit2.json")));
  assert.equal(hash(path("transfer1.json")), hash(path("transfer2.json")));
  const initial = auditEvaluationRows(json(path("initial.json")), fit.examples),
    fitMetrics = auditEvaluationRows(json(path("fit1.json")), fit.examples),
    transferMetrics = auditEvaluationRows(
      json(path("transfer1.json")),
      transfer.examples,
    );
  const summary = {
    arm,
    initial,
    fit: fitMetrics,
    transfer: transferMetrics,
    fitGate: mlpGate(fitMetrics),
    transferGate: mlpGate(transferMetrics),
    updates: 1000,
  };
  assert.deepEqual(r.summary, summary);
  assert.deepEqual(json(path("summary.json")), summary);
  const history = json(path("history.json"));
  assert.equal(history.length, 1000);
  history.forEach((s, i) => {
    assert.equal(s.step, i + 1);
    assert(Number.isFinite(s.meanTrainingNll) && s.meanTrainingNll >= 0);
  });
  auditMlpLedger(
    bytes(path("forward-count.jsonl"))
      .toString("utf8")
      .trim()
      .split("\n")
      .map(JSON.parse),
    arm,
  );
  assert.deepEqual(r.forwardCounts, phases);
  assert.equal(r.globalOffset, arm === "control" ? 0 : 16336);
  assert.deepEqual(json(path("model-intent.json")), {
    topLevelModels: 1,
    checkpointLoads: 0,
    arm,
  });
  assert.equal(r.topLevelModels, 1);
  assert.equal(r.checkpointLoads, 0);
  assert.equal(r.trainingRuns, 1);
  assert.equal(r.parameters, arm === "control" ? 13292 : 769);
  assert.equal(r.environment.cuda, null);
  assert.equal(r.environment.threads, 2);
  assert.equal(r.environment.interopThreads, 1);
  assert(
    r.budget.wallTimeSeconds <= 600 &&
      r.budget.maxSampledWorkingSetBytes <= 2147483648 &&
      r.budget.instantaneousPeakMeasured === false,
  );
  if (arm === "control") {
    for (const [n, k] of [
      ["initial.json", "initialFitLogitsSha256"],
      ["history.json", "history1000Sha256"],
      ["fit1.json", "finalFitLogitsSha256"],
      ["transfer1.json", "finalTransferLogitsSha256"],
    ])
      assert.equal(hash(path(n)), c.original[k]);
    assert.equal(
      r.initialLogicalWeightsSha256,
      c.original.initialLogicalWeightsSha256,
    );
    assert.equal(
      r.finalLogicalWeightsSha256,
      c.original.finalLogicalWeightsSha256,
    );
    assert.equal(r.controlExact, true);
  } else {
    assert.equal(r.controlExact, null);
    assert.deepEqual(json(resolve(recorded, "candidate-input.json")), {
      executionSha256: hash(resolve(recorded, "execution.json")),
      controlReportSha256: hash(resolve(recorded, "control/report.json")),
    });
  }
  return { summary, report: r };
}
export function pairedMlp(control, candidate) {
  assert.equal(control.length, candidate.length);
  const s = {
    examples: control.length,
    bothCorrect: 0,
    recovered: 0,
    regressed: 0,
    bothWrong: 0,
    changedChoices: 0,
  };
  control.forEach((a, i) => {
    const b = candidate[i];
    assert.equal(a.id, b.id);
    assert.equal(a.label, b.label);
    s[
      a.correct
        ? b.correct
          ? "bothCorrect"
          : "regressed"
        : b.correct
          ? "recovered"
          : "bothWrong"
    ]++;
    s.changedChoices += Number(a.index !== b.index);
  });
  return s;
}
export function auditMlpStored(recorded, base = root) {
  const config = approvedMlp(base),
    f = json(resolve(recorded, "execution.json")),
    owner = json(resolve(recorded, "report.json"));
  assert.equal(f.status, "FROZEN_BEFORE_EXECUTION");
  assert.equal(f.configSha256, hash(resolve(base, configPath)));
  assert.equal(owner.status, "PASS");
  assert(owner.elapsedSeconds <= 1260);
  assert.equal(owner.children.length, 2);
  const names = [
    "start.json",
    "execution.json",
    "fit-dataset.json",
    "transfer-dataset.json",
    "candidate-input.json",
    "control/report.json",
    "candidate/report.json",
  ];
  assert.deepEqual(Object.keys(owner.hashes).sort(), names.sort());
  for (const n of names)
    assert.equal(hash(resolve(recorded, n)), owner.hashes[n]);
  owner.children.forEach((c, i) => {
    assert.equal(c.arm, ["control", "candidate"][i]);
    assert.equal(c.result.code, 0);
    assert.equal(c.result.timedOut, false);
    assert(c.result.elapsedMs <= 600000);
  });
  for (const [p, h] of Object.entries(config.contract.sourceHashes))
    assert.equal(f.sourceHashes[p], h);
  const control = auditMlpArm(recorded, "control", config),
    candidate = auditMlpArm(recorded, "candidate", config);
  const pairs = {},
    byRow = [],
    byColumn = [],
    transfer = json(resolve(recorded, "transfer-dataset.json")).examples,
    rows = json(resolve(recorded, "candidate/transfer1.json"));
  for (const split of ["fit", "transfer"])
    pairs[split] = pairedMlp(
      json(resolve(recorded, `control/${split}1.json`)),
      json(resolve(recorded, `candidate/${split}1.json`)),
    );
  const group = (indices) => {
    const s = indices.map((i) => rows[i]);
    return {
      examples: s.length,
      correct: s.filter((r) => r.correct).length,
      meanNll: s.reduce((a, r) => a + r.nll, 0) / s.length,
      minimumTargetMargin: Math.min(...s.map((r) => r.margin)),
      maximumTargetRank: Math.max(...s.map((r) => r.rank)),
    };
  };
  for (let r = 0; r < 16; r++)
    byRow.push(
      group(
        transfer.flatMap((e, i) =>
          Math.floor(e.fixtureIndex / 10) === r ? [i] : [],
        ),
      ),
    );
  for (let c = 0; c < 10; c++)
    byColumn.push(
      group(transfer.flatMap((e, i) => (e.fixtureIndex % 10 === c ? [i] : []))),
    );
  return {
    schemaVersion: 1,
    id: "outcome-mlp-v1",
    technicalStatus: "PASS",
    control: control.summary,
    candidate: candidate.summary,
    paired: pairs,
    candidateTransferByRow: byRow,
    candidateTransferByColumn: byColumn,
    reproduction: {
      originalControlExact: true,
      all160FinalPredictionsRepeatedExact: true,
    },
    actualCompute: {
      topLevelModels: 2,
      checkpointLoads: 0,
      topLevelForwards: 32672,
      trainingRuns: 2,
      optimizerUpdates: 2000,
    },
    timing: {
      ownerWallSeconds: owner.elapsedSeconds,
      children: owner.children.map(({ arm, result }) => ({
        arm,
        processWallSeconds: result.elapsedMs / 1000,
        ...{
          pythonBody: (arm === "control" ? control : candidate).report.budget,
        },
      })),
    },
    interpretation:
      "Outcome22 transfer is a label-preserving permutation of known fit inputs; bypass diagnostic, not independent generalization. Architecture, capacity and board context change together. Existing wider tasks and app adoption are untested.",
    additionalPythonModelLoadForwardTraining: 0,
    appUsesDfs: true,
  };
}
