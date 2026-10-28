import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import test from "node:test";
import {
  validateProposal,
  proposalPath,
  phaseBudget,
  durationScreen,
  checkSyntheticPrefix,
} from "./validate-fit-duration-proposal.mjs";
const proposal = JSON.parse(
  readFileSync(new URL("../../" + proposalPath, import.meta.url), "utf8"),
);
test("static fixed duration proposal passes with no model execution", () => {
  const r = validateProposal(proposal);
  assert.equal(r.actualModelCreationLoadForwardTraining, 0);
  assert.equal(r.actualDomainOrDatasetMaterialization, 0);
});
test("all proposed phase calls including initial and two final repeats are counted", () => {
  assert.equal(phaseBudget(200).total, 3248);
  assert.equal(phaseBudget(1000).total, 16048);
  assert.equal(phaseBudget(200).total + phaseBudget(1000).total, 19296);
  for (const n of [0, -1, 1.5, 1001, NaN]) assert.throws(() => phaseBudget(n));
});
const old = {
  correct: 15,
  meanNll: 2.5084412693977356,
  minimumTargetMargin: -0.30619263648986816,
};
const good = { correct: 16, meanNll: 0.05, minimumTargetMargin: 0.1 };
test("top1 or lower loss alone does not pass the duration screen", () => {
  assert.equal(durationScreen(old, good, true).status, "PASS");
  for (const candidate of [
    { ...good, meanNll: 0.051 },
    { ...good, correct: 15 },
    { ...good, minimumTargetMargin: 0 },
  ])
    assert.equal(durationScreen(old, candidate, true).status, "FAIL");
});
test("technical mismatch or an already successful control cannot claim the proposed result", () => {
  assert.equal(
    durationScreen(old, good, false).status,
    "BLOCKED_TECHNICAL_FAILURE",
  );
  assert.equal(
    durationScreen(good, { ...good, meanNll: 0.01 }, true).status,
    "FAIL",
  );
  assert.throws(() => durationScreen(old, { ...good, meanNll: NaN }, true));
});
const original = {
  history: Array.from({ length: 200 }, (_, i) => 3 - i / 1000),
  initialWeights: "1".repeat(64),
  initialLogits: "2".repeat(64),
  finalWeights200: "3".repeat(64),
};
const control = { ...original, optimizerState200: "4".repeat(64) };
const candidate = {
  ...control,
  history: [...original.history, ...Array(800).fill(2.8)],
};
test("synthetic prefix checks compare the new optimizer state only between concurrent arms", () => {
  assert.equal(checkSyntheticPrefix(original, control, candidate), true);
  for (const mutation of [
    (c) => (c.history[199] += 0.001),
    (c) => (c.initialWeights = "0".repeat(64)),
    (c) => (c.finalWeights200 = "0".repeat(64)),
    (c) => (c.optimizerState200 = "0".repeat(64)),
  ]) {
    const c = structuredClone(candidate);
    mutation(c);
    assert.throws(() => checkSyntheticPrefix(original, control, c));
  }
});
for (const [name, change] of [
  ["approval escalation", (p) => (p.approval = "USER_APPROVED")],
  ["completed status", (p) => (p.status = "PASS")],
  ["duration sweep", (p) => (p.contrast.candidateUpdates = 2000)],
  ["extra arm", (p) => (p.contrast.runs = 3)],
  ["learning-rate change", (p) => (p.shared.model.config.learningRate = 0.01)],
  ["changed loss", (p) => (p.shared.loss = "auxiliary")],
  ["new dataset", (p) => (p.originalRun.datasetSha256 = "0".repeat(64))],
  ["easier gate", (p) => (p.shared.fitAcceptance.maximumMeanNll = 3)],
  ["extra forward", (p) => p.forwardBudget.totalTopLevelMax++],
  [
    "prefix inference",
    (p) => (p.forwardBudget.extraPrefixOrTelemetryForwards = 16),
  ],
  ["checkpoint load", (p) => (p.forwardBudget.checkpointLoads = 1)],
  [
    "warm start",
    (p) => (p.contrast.initialization = "load old final checkpoint"),
  ],
  [
    "optimizer archive claim",
    (p) => (p.originalRun.optimizerStateArchived = true),
  ],
  [
    "missing prefix control",
    (p) => (p.controls.candidatePrefix200 = "not required"),
  ],
  [
    "source drift",
    (p) =>
      (p.sourceHashes["research/cnn/controlled_point_fit.py"] = "0".repeat(64)),
  ],
  ["source traversal", (p) => (p.sourceHashes["../secret"] = "0".repeat(64))],
  ["evidence drift", (p) => (p.evidence.sha256 = "0".repeat(64))],
  ["looser memory", (p) => (p.resources.sampledWorkingSetBytes *= 2)],
  ["test access", (p) => (p.boundaries.reservedTestOrDevSelection = true)],
  [
    "runner implementation",
    (p) => (p.boundaries.trainingRunnerImplemented = true),
  ],
])
  test(`reject ${name}`, () => {
    const p = structuredClone(proposal);
    change(p);
    assert.throws(() => validateProposal(p));
  });
test("CLI rejects execution arguments", () => {
  const r = spawnSync(
    process.execPath,
    [
      fileURLToPath(
        new URL("./validate-fit-duration-proposal.mjs", import.meta.url),
      ),
      "--train",
    ],
    { encoding: "utf8", windowsHide: true },
  );
  assert.notEqual(r.status, 0);
  assert.match(r.stderr, /Static validation only/);
});
