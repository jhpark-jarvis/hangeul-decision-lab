import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import {
  root,
  json,
  sha,
  auditPairsStored,
} from "./lookahead-pairs-contract.mjs";
import {
  bytes,
  write,
  reserveFitOutput,
} from "./controlled-point-fit-contract.mjs";
import { inputRows } from "./lookahead-information.mjs";
import {
  boardProposalPath,
  boardContract,
  validateBoardProposal,
  checkBoardOutcomes,
} from "./validate-board-context-proposal.mjs";
export { root, json, sha, bytes, write, reserveFitOutput, auditPairsStored };
export const boardConfigPath = "research/experiments/board-context-v1.json";
export const phases = { initial: 32, train: 32000, final1: 32, final2: 32 };
export const armNames = ["control", "candidate"];
export const armFiles = [
  "initial.json",
  "final1.json",
  "final2.json",
  "history.json",
  "input-identities.json",
  "forward-count.jsonl",
  "model-intent.json",
  "summary.json",
  "final-weights.pt",
];
export function approvedBoardContext(base = root) {
  const p = json(resolve(base, boardProposalPath));
  validateBoardProposal(p, base);
  const c = json(resolve(base, boardConfigPath));
  assert.equal(c.id, "board-context-v1");
  assert.equal(c.approval, "USER_APPROVED");
  assert.equal(c.status, "FROZEN_BEFORE_EXECUTION");
  assert.equal(c.proposalPath, boardProposalPath);
  assert.equal(c.proposalSha256, sha(resolve(base, boardProposalPath)));
  assert.deepEqual(c.contract, boardContract);
  assert.deepEqual(c.archiveHashes, p.archiveHashes);
  assert.deepEqual(c.boundaries, {
    domainReferenceMaterialization: 0,
    checkpointLoad: 0,
    retryTuning: 0,
    rawDataWeightsLogitsPrivate: true,
    appUsesDfs: true,
    publicReplayAdditionalModelLoadForwardTraining: 0,
  });
  return c;
}
export function parseBoardArgs(kind, args) {
  assert(["run", "replay"].includes(kind));
  const names =
    kind === "run"
      ? ["--output", "--python", "--source"]
      : ["--output", "--recorded"];
  assert.equal(args.length, names.length * 2, "Fixed paths only");
  const pairs = [];
  for (let i = 0; i < args.length; i += 2) pairs.push(args.slice(i, i + 2));
  assert.deepEqual(pairs.map((p) => p[0]).sort(), names);
  assert(pairs.every((p) => p[1] && !p[1].startsWith("--")));
  return Object.fromEntries(pairs.map(([k, v]) => [k.slice(2), v]));
}
export function auditBoardLedger(rows, arm) {
  assert(armNames.includes(arm));
  assert.equal(rows.length, 32096);
  let offset = 0;
  const start = arm === "control" ? 0 : 32096;
  for (const [phase, n] of Object.entries(phases)) {
    for (let i = 0; i < n; i++)
      assert.deepEqual(rows[offset + i], {
        call: start + offset + i + 1,
        arm,
        phase,
        phaseCall: i + 1,
        step: phase === "train" ? Math.floor(i / 32) + 1 : null,
        example: i % 32,
      });
    offset += n;
  }
  return rows.length;
}
export function fp32InputKey(values) {
  const b = Buffer.alloc(values.length * 4);
  values.forEach((v, i) => {
    assert(Number.isFinite(v));
    b.writeFloatLE(v, i * 4);
  });
  return createHash("sha256").update(b).digest("hex");
}
export function auditInputIdentity(rows, keys) {
  const seen = new Map();
  let total = 0;
  assert.equal(rows.length, keys.length);
  rows.forEach((r, i) => {
    assert.equal(r.logits.length, keys[i].length);
    r.logits.forEach((v, j) => {
      const k = keys[i][j];
      if (seen.has(k))
        assert(Object.is(seen.get(k), v), "Same input/logit identity drift");
      else seen.set(k, v);
      total++;
    });
  });
  return {
    uniqueInputs: seen.size,
    candidateRows: total,
    allIdenticalInputLogitsExact: true,
  };
}
const near = (a, b) =>
  assert(
    Number.isFinite(a) &&
      Number.isFinite(b) &&
      Math.abs(a - b) <= 2e-6 * Math.max(1, Math.abs(a), Math.abs(b)),
    "Stored FP32 metric drift",
  );
export function auditBoardRows(rows, examples) {
  assert.equal(rows.length, 32);
  assert.equal(examples.length, 32);
  rows.forEach((r, i) => {
    assert.equal(r.example, i);
    assert.equal(r.label, examples[i].label);
    const s = r.logits;
    assert.equal(s.length, 32);
    assert(s.every((v) => Number.isFinite(v) && Math.fround(v) === v));
    const max = Math.max(...s),
      index = s.indexOf(max),
      label = r.label;
    assert(Number.isInteger(label) && label >= 0 && label < 32);
    const nll =
      Math.log(s.reduce((n, v) => n + Math.exp(v - max), 0)) + max - s[label];
    assert.equal(r.index, index);
    assert.equal(r.correct, index === label);
    assert.equal(
      r.rank,
      1 +
        s.filter(
          (v, j) =>
            j !== label && (v > s[label] || (v === s[label] && j < label)),
        ).length,
    );
    near(r.nll, nll);
    assert(r.nll >= 0);
    assert.equal(
      r.margin,
      s[label] - Math.max(...s.filter((_, j) => j !== label)),
    );
  });
  return {
    examples: 32,
    correct: rows.filter((r) => r.correct).length,
    meanNll: rows.reduce((a, r) => a + r.nll, 0) / 32,
    minMargin: Math.min(...rows.map((r) => r.margin)),
    maxRank: Math.max(...rows.map((r) => r.rank)),
    cohorts: Object.fromEntries(
      [
        ["orderLucky", 0],
        ["hard", 1],
      ].map(([name, parity]) => [
        name,
        {
          examples: 16,
          correct: rows.filter((r, i) => i % 2 === parity && r.correct).length,
        },
      ]),
    ),
  };
}
export function boardFitGate(s) {
  const checks = {
    correct32: s.correct === 32,
    meanNllAtMost005: s.meanNll <= 0.05,
    allMarginsPositive: s.minMargin > 0,
  };
  return {
    status: Object.values(checks).every(Boolean) ? "PASS" : "FAIL",
    checks,
  };
}
export function auditBoardArm(recorded, arm, c = approvedBoardContext()) {
  assert(armNames.includes(arm));
  const path = (p) => resolve(recorded, arm, p),
    r = json(path("report.json"));
  assert.equal(r.status, "PASS");
  assert.equal(r.arm, arm);
  assert.equal(r.executionSha256, sha(resolve(recorded, "execution.json")));
  assert.deepEqual(Object.keys(r.hashes).sort(), [...armFiles].sort());
  for (const p of armFiles) assert.equal(sha(path(p)), r.hashes[p]);
  assert.equal(r.parameters, 13292);
  assert.equal(r.topLevelModels, 1);
  assert.equal(r.checkpointLoads, 0);
  assert.deepEqual(json(path("model-intent.json")), {
    topLevelModels: 1,
    checkpointLoads: 0,
    arm,
  });
  assert.deepEqual(r.forwardCounts, phases);
  auditBoardLedger(
    bytes(path("forward-count.jsonl"))
      .toString()
      .trim()
      .split("\n")
      .map(JSON.parse),
    arm,
  );
  const b = r.budget;
  assert(b.wallTimeSeconds >= 0 && b.wallTimeSeconds <= 600);
  assert(
    b.maxSampledWorkingSetBytes > 0 &&
      b.maxSampledWorkingSetBytes <= 2147483648,
  );
  assert.equal(b.instantaneousPeakMeasured, false);
  assert.equal(r.environment.cuda, null);
  assert.equal(r.environment.threads, 2);
  assert.equal(r.environment.interopThreads, 1);
  for (const k of ["initialLogicalWeightsSha256", "finalLogicalWeightsSha256"])
    assert.match(r[k], /^[a-f0-9]{64}$/);
  const history = json(path("history.json"));
  assert.equal(history.length, 1000);
  history.forEach((r, i) => {
    assert.equal(r.step, i + 1);
    assert(Number.isFinite(r.meanTrainingNll) && r.meanTrainingNll >= 0);
  });
  const data = json(resolve(recorded, "dataset.json"));
  assert.equal(
    sha(resolve(recorded, "dataset.json")),
    c.archiveHashes["build1.json"],
  );
  const expectedKeys = data.examples.map((e) =>
    inputRows(e, arm === "control" ? "maskedSpatial" : "fullSpatial").map(
      fp32InputKey,
    ),
  );
  assert.deepEqual(json(path("input-identities.json")), expectedKeys);
  assert.equal(sha(path("final1.json")), sha(path("final2.json")));
  const metrics = {};
  for (const phase of ["initial", "final1", "final2"]) {
    const rows = json(path(phase + ".json"));
    metrics[phase] = auditBoardRows(rows, data.examples);
    if (arm === "control") {
      const identity = auditInputIdentity(rows, expectedKeys);
      assert.equal(identity.uniqueInputs, 512);
      assert.equal(identity.candidateRows, 1024);
      assert(metrics[phase].correct <= 16);
      if (phase === "final1") assert.deepEqual(r.nativeIdentity, identity);
    }
  }
  if (arm === "candidate") assert.equal(r.nativeIdentity, null);
  const summary = {
    arm,
    updates: 1000,
    initial: metrics.initial,
    final: metrics.final1,
    fitGate: boardFitGate(metrics.final1),
  };
  assert.deepEqual(r.summary, summary);
  assert.deepEqual(json(path("summary.json")), summary);
  return { report: r, summary };
}
export function compareBoardArms(control, candidate) {
  assert.equal(control.length, 32);
  assert.equal(candidate.length, 32);
  let recovered = 0,
    regressed = 0,
    bothCorrect = 0,
    bothWrong = 0;
  control.forEach((a, i) => {
    const b = candidate[i];
    assert.equal(a.example, b.example);
    assert.equal(a.label, b.label);
    if (a.correct && b.correct) bothCorrect++;
    else if (a.correct) regressed++;
    else if (b.correct) recovered++;
    else bothWrong++;
  });
  return {
    examples: 32,
    recovered,
    regressed,
    bothCorrect,
    bothWrong,
    netCorrect: recovered - regressed,
  };
}
export function auditBoardStored(recorded, base = root) {
  const c = approvedBoardContext(base),
    path = (p) => resolve(recorded, p),
    r = json(path("report.json")),
    f = json(path("execution.json"));
  assert.equal(r.status, "PASS");
  assert(r.elapsedSeconds >= 0 && r.elapsedSeconds <= 1260);
  assert.equal(f.status, "FROZEN_BEFORE_EXECUTION");
  assert.equal(f.configSha256, sha(resolve(base, boardConfigPath)));
  assert.equal(f.datasetSha256, c.archiveHashes["build1.json"]);
  assert.equal(sha(path("dataset.json")), f.datasetSha256);
  assert.deepEqual(f.inputHashes, c.archiveHashes);
  for (const [p, h] of Object.entries(f.inputHashes))
    assert.equal(sha(path("source/" + p)), h);
  auditPairsStored(path("source"), base);
  const names = [
    "start.json",
    "execution.json",
    "dataset.json",
    "candidate-input.json",
    "control/report.json",
    "candidate/report.json",
  ];
  assert.deepEqual(Object.keys(r.hashes).sort(), names.sort());
  for (const [p, h] of Object.entries(r.hashes)) assert.equal(sha(path(p)), h);
  assert.equal(r.children.length, 2);
  r.children.forEach((child, i) => {
    assert.equal(child.arm, armNames[i]);
    assert.equal(child.result.code, 0);
    assert.equal(child.result.timedOut, false);
    assert(child.result.elapsedMs <= 600000);
  });
  assert.deepEqual(json(path("candidate-input.json")), {
    executionSha256: sha(path("execution.json")),
    controlReportSha256: sha(path("control/report.json")),
  });
  const control = auditBoardArm(recorded, "control", c),
    candidate = auditBoardArm(recorded, "candidate", c);
  assert.equal(
    control.report.initialLogicalWeightsSha256,
    candidate.report.initialLogicalWeightsSha256,
  );
  const paired = compareBoardArms(
    json(path("control/final1.json")),
    json(path("candidate/final1.json")),
  );
  const outcome = checkBoardOutcomes(
    { ...control.summary.final, technicalPass: true },
    { ...candidate.summary.final, technicalPass: true },
  );
  return {
    schemaVersion: 1,
    id: "board-context-v1",
    technicalStatus: "PASS",
    contrastOutcome: outcome,
    control: control.summary,
    candidate: candidate.summary,
    paired,
    identity: {
      sameInitialLogicalWeights: true,
      controlIdenticalInputLogitsExact: true,
      finalRepeatsExact: true,
    },
    actualCompute: {
      topLevelModels: 2,
      checkpointLoads: 0,
      trainingRuns: 2,
      optimizerUpdates: 2000,
      topLevelForwards: 64192,
      domainReferenceMaterializations: 0,
    },
    timing: {
      ownerWallSeconds: r.elapsedSeconds,
      children: r.children.map((child, i) => ({
        arm: child.arm,
        processWallSeconds: child.result.elapsedMs / 1000,
        pythonBudget: [control, candidate][i].report.budget,
      })),
    },
    additionalReplayPythonModelLoadForwardTraining: 0,
    interpretation:
      "Fit-only occupancy information contrast on stored32, no independent generalization, residual-mechanism claim, old CNN failure resolution or app adoption",
    appUsesDfs: true,
  };
}
