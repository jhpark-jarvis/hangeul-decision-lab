import assert from "node:assert/strict";
import { resolve } from "node:path";
import {
  root,
  json,
  hash,
  bytes,
  auditRows,
  reserveFitOutput,
  write,
} from "./controlled-point-fit-contract.mjs";
import {
  validateProposal,
  proposalPath,
  phaseBudget,
  durationScreen,
} from "./validate-fit-duration-proposal.mjs";
export { root, json, hash, bytes, reserveFitOutput, write };
export const configPath =
  "research/experiments/controlled-fit-duration-v1.json";
export function approvedDuration(base = root) {
  const proposal = json(resolve(base, proposalPath));
  validateProposal(proposal, base);
  const c = json(resolve(base, configPath));
  assert.equal(c.id, "controlled-fit-duration-v1");
  assert.equal(c.status, "FROZEN_BEFORE_EXECUTION");
  assert.equal(c.approval, "USER_APPROVED");
  assert.equal(c.proposalPath, proposalPath);
  assert.equal(c.proposalSha256, hash(resolve(base, proposalPath)));
  const keys = [
    "originalRun",
    "sourceHashes",
    "shared",
    "contrast",
    "controls",
    "forwardBudget",
    "resources",
    "screening",
  ];
  assert.deepEqual(
    c.contract,
    Object.fromEntries(keys.map((k) => [k, proposal[k]])),
  );
  assert.deepEqual(c.boundaries, {
    originalFilesImmutable: true,
    domainReferenceOrNewDataset: false,
    checkpointLoads: 0,
    retryOrExtraTuning: false,
    appUsesDfs: true,
    publicReplayAdditionalModelLoadForwardTraining: 0,
    rawWeightsLogitsOptimizerStatePrivate: true,
  });
  return c;
}
export function parseDurationArgs(kind, args) {
  const names =
    kind === "run"
      ? ["--original", "--output", "--python"]
      : ["--output", "--recorded"];
  assert(["run", "replay"].includes(kind));
  assert.equal(
    args.length,
    names.length * 2,
    "Fixed named paths only; no training overrides",
  );
  const pairs = [];
  for (let i = 0; i < args.length; i += 2) pairs.push(args.slice(i, i + 2));
  assert.deepEqual(pairs.map((p) => p[0]).sort(), names);
  assert(
    pairs.every(
      (p) => typeof p[1] === "string" && p[1] && !p[1].startsWith("--"),
    ),
  );
  return Object.fromEntries(pairs.map(([k, v]) => [k.slice(2), v]));
}
export function auditDurationLedger(rows, arm) {
  assert(["control", "candidate"].includes(arm));
  const updates = arm === "control" ? 200 : 1000,
    budget = phaseBudget(updates),
    offset = arm === "control" ? 0 : 3248;
  assert.equal(rows.length, budget.total);
  let local = 0;
  for (const [phase, n] of [
    ["initial", 16],
    ["train", updates * 16],
    ["final1", 16],
    ["final2", 16],
  ]) {
    for (let i = 0; i < n; i++)
      assert.deepEqual(rows[local + i], {
        call: offset + local + i + 1,
        arm,
        phase,
        phaseCall: i + 1,
        step: phase === "train" ? Math.floor(i / 16) + 1 : null,
        example: i % 16,
      });
    local += n;
  }
}
export function auditDurationArm(recorded, arm, config = approvedDuration()) {
  const path = (p) => resolve(recorded, arm, p),
    r = json(path("report.json")),
    c = config.contract;
  assert.equal(r.status, "PASS");
  assert.equal(r.arm, arm);
  assert.equal(r.topLevelModels, 1);
  assert.equal(r.checkpointLoads, 0);
  assert.equal(r.trainingRuns, 1);
  assert.equal(r.executionSha256, hash(resolve(recorded, "execution.json")));
  assert.equal(r.datasetSha256, c.originalRun.datasetSha256);
  const names = [
    "initial.json",
    "final1.json",
    "final2.json",
    "history.json",
    "prefix-history.json",
    "prefix.json",
    "optimizer-step200.json",
    "forward-count.jsonl",
    "final-weights.pt",
    "summary.json",
  ];
  assert.deepEqual(Object.keys(r.hashes).sort(), [...names].sort());
  for (const p of names) assert.equal(hash(path(p)), r.hashes[p], p);
  const data = json(resolve(recorded, "dataset.json"));
  assert.equal(
    hash(resolve(recorded, "dataset.json")),
    c.originalRun.datasetSha256,
  );
  const initial = auditRows(json(path("initial.json")), data.examples),
    final = auditRows(json(path("final1.json")), data.examples);
  assert.deepEqual(r.summary, json(path("summary.json")));
  assert.deepEqual(r.summary.initial, initial);
  assert.deepEqual(r.summary.final, final);
  const checks = {
    allCorrect: final.correct === 16,
    meanNllAtMost005: final.meanNll <= 0.05,
    everyTargetMarginPositive: final.minimumTargetMargin > 0,
  };
  assert.deepEqual(r.summary.fitGate, {
    status: Object.values(checks).every(Boolean) ? "PASS" : "FAIL",
    checks,
  });
  const steps = arm === "control" ? 200 : 1000,
    budget = phaseBudget(steps);
  assert.equal(r.summary.updates, steps);
  assert.equal(r.modelForwards, budget.total);
  assert.equal(r.globalOffset, arm === "control" ? 0 : 3248);
  assert.deepEqual(r.forwardCounts, {
    initial: 16,
    train: steps * 16,
    final1: 16,
    final2: 16,
  });
  assert.equal(
    r.hashes["initial.json"],
    c.originalRun.initialPredictionsSha256,
  );
  assert.equal(
    r.initialWeightsSha256,
    c.originalRun.initialLogicalWeightsSha256,
  );
  assert.equal(r.hashes["final1.json"], r.hashes["final2.json"]);
  const prefix = json(path("prefix.json"));
  assert.deepEqual(r.prefix, prefix);
  assert.equal(prefix.step, 200);
  assert.equal(prefix.weightsSha256, c.originalRun.finalLogicalWeightsSha256);
  assert.equal(prefix.historySha256, c.originalRun.historySha256);
  assert.equal(r.hashes["prefix-history.json"], c.originalRun.historySha256);
  assert.match(prefix.optimizerSha256, /^[0-9a-f]{64}$/);
  const history = json(path("history.json")),
    prefixHistory = json(path("prefix-history.json"));
  assert.equal(history.length, steps);
  history.forEach((v, i) => {
    assert.equal(v.step, i + 1);
    assert(Number.isFinite(v.meanTrainingNll) && v.meanTrainingNll >= 0);
  });
  assert.deepEqual(history.slice(0, 200), prefixHistory);
  auditDurationLedger(
    bytes(path("forward-count.jsonl"))
      .toString("utf8")
      .trim()
      .split("\n")
      .map(JSON.parse),
    arm,
  );
  assert(
    r.budget.wallTimeSeconds <= 600 &&
      r.budget.maxSampledWorkingSetBytes <= 2147483648,
  );
  if (arm === "control") {
    assert.deepEqual(r.originalReproduction, {
      initial: true,
      history: true,
      finalRows: true,
      logicalWeights: true,
      metrics: true,
    });
    assert.equal(r.hashes["history.json"], c.originalRun.historySha256);
    assert.equal(r.hashes["final1.json"], c.originalRun.finalPredictionsSha256);
    assert.equal(r.finalWeightsSha256, c.originalRun.finalLogicalWeightsSha256);
    const old = json(
      resolve(root, "assets/research/controlled-point-fit-20261028-v1.json"),
    ).summary;
    for (const k of ["initial", "final", "fitGate"])
      assert.deepEqual(r.summary[k], old[k]);
  } else assert.equal(r.originalReproduction, null);
  return { report: r, history, finalRows: json(path("final1.json")) };
}
export function auditDurationStored(recorded, base = root) {
  const config = approvedDuration(base),
    frozen = json(resolve(recorded, "execution.json")),
    owner = json(resolve(recorded, "report.json"));
  assert.equal(owner.status, "PASS");
  assert.equal(owner.children.length, 2);
  assert(owner.elapsedSeconds <= 1260);
  assert.equal(frozen.status, "FROZEN_BEFORE_EXECUTION");
  assert.equal(frozen.configSha256, hash(resolve(base, configPath)));
  assert.equal(frozen.datasetSha256, config.contract.originalRun.datasetSha256);
  for (const [p, h] of Object.entries(owner.hashes)) {
    assert(
      [
        "start.json",
        "execution.json",
        "dataset.json",
        "candidate-input.json",
        "control/report.json",
        "candidate/report.json",
      ].includes(p),
    );
    assert.equal(hash(resolve(recorded, p)), h, p);
  }
  assert.equal(Object.keys(owner.hashes).length, 6);
  owner.children.forEach((r, i) => {
    assert.equal(r.arm, i === 0 ? "control" : "candidate");
    assert.equal(r.result.code, 0);
    assert.equal(r.result.timedOut, false);
    assert(r.result.elapsedMs <= 600000);
  });
  const control = auditDurationArm(recorded, "control", config),
    candidate = auditDurationArm(recorded, "candidate", config);
  assert.deepEqual(control.report.prefix, candidate.report.prefix);
  assert.equal(
    control.report.hashes["optimizer-step200.json"],
    candidate.report.hashes["optimizer-step200.json"],
  );
  const phase = json(resolve(recorded, "candidate-input.json"));
  assert.deepEqual(phase, {
    executionSha256: hash(resolve(recorded, "execution.json")),
    controlReportSha256: hash(resolve(recorded, "control/report.json")),
  });
  const transitions = {
    bothCorrect: 0,
    recovered: 0,
    regressed: 0,
    bothWrong: 0,
  };
  for (let i = 0; i < 16; i++) {
    const a = control.finalRows[i].correct,
      b = candidate.finalRows[i].correct;
    transitions[
      a ? (b ? "bothCorrect" : "regressed") : b ? "recovered" : "bothWrong"
    ]++;
  }
  const screen = durationScreen(
    control.report.summary.final,
    candidate.report.summary.final,
    true,
  );
  const arms = Object.fromEntries(
    [
      ["control", control],
      ["candidate", candidate],
    ].map(([arm, v], i) => [
      arm,
      {
        ...v.report.summary,
        topLevelForwards: v.report.modelForwards,
        processWallSeconds: owner.children[i].result.elapsedMs / 1000,
        pythonBodyWallSeconds: v.report.budget.wallTimeSeconds,
        maxSampledWorkingSetBytes: v.report.budget.maxSampledWorkingSetBytes,
        targetMassGeometricMean: Math.exp(-v.report.summary.final.meanNll),
        savedPreUpdateLoss: [1, 100, 200, 400, 600, 800, 1000]
          .filter((step) => step <= v.history.length)
          .map((step) => v.history[step - 1]),
      },
    ]),
  );
  return {
    schemaVersion: 1,
    id: "controlled-fit-duration-v1",
    technicalStatus: "PASS",
    durationScreen: screen,
    arms,
    pairedTransitions: transitions,
    nllDelta:
      candidate.report.summary.final.meanNll -
      control.report.summary.final.meanNll,
    controls: {
      original200Exact: true,
      initialsExact: true,
      modelAndHistoryPrefix200Exact: true,
      concurrentOptimizerPrefix200Exact: true,
      originalOptimizerStateAvailable: false,
    },
    actualCompute: {
      topLevelModels: 2,
      checkpointLoads: 0,
      trainingRuns: 2,
      updates: 1200,
      topLevelForwards: 19296,
      domainReferenceOrNewDataset: 0,
    },
    ownerWallSeconds: owner.elapsedSeconds,
    instantaneousMemoryPeakMeasured: false,
    interpretation:
      "Single-seed fit-only update-count contrast with five times candidate training exposure, outcome shortcut retained; no equal-compute, calibration, spatial mechanism, kind-selection, old train/dev repair, generalization, survival or app adoption claim.",
    replayValidation:
      "Stored artifact hashes, approved contract and original proposal source identity; full execution source freeze was checked by owner/child. Weights and optimizer state are read as bytes/JSON only, with no Python/model/optimizer execution.",
    additionalModelCreationLoadForwardTraining: 0,
    appUsesDfs: true,
  };
}
