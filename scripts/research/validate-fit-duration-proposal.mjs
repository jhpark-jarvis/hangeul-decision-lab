import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../", import.meta.url));
export const proposalPath =
  "research/experiments/controlled-fit-duration-proposal-v1.json";
const read = (base, p) => readFileSync(resolve(base, p));
const json = (base, p) => JSON.parse(read(base, p).toString("utf8"));
const hash = (base, p) =>
  createHash("sha256").update(read(base, p)).digest("hex");
const originalIdentity = {
  datasetSha256:
    "862353fc79a5bc12a97da008b9ce0467ac026d28a4dc86fb211767ca45d12666",
  executionSha256:
    "0a4a49182506a00074644b5a0823a0f62037a00fbda6331c7c62cc3872701dea",
  initialPredictionsSha256:
    "ce4f971b347e118dc51ed7213fd408962a2ce7e1f38fda169f473876f36a4a52",
  historySha256:
    "48c5b88490f9aaf6ee8cfaa5e089623775365f4d2d63ca629e2d5147b6a6cf43",
  finalPredictionsSha256:
    "b4998858d1745ab171b331f08500dbc1a996bf78c3fed5c1655189b374054811",
  initialLogicalWeightsSha256:
    "8c36dfe1cd14e50da02db90bbc0e7e1d7dc6f5025711b4aa4818ee02f91d2c6b",
  finalLogicalWeightsSha256:
    "2714943db1f96b810ad99adeae0945c47b0bc665cf637f8ebbbe9d77e871b53a",
  optimizerStateArchived: false,
};

// Static arithmetic and synthetic recorded-value checks only; no runtime training interface.
export function phaseBudget(updates) {
  assert(Number.isInteger(updates) && updates >= 1 && updates <= 1000);
  return {
    initial: 16,
    training: updates * 16,
    finalTwoRepeats: 32,
    total: 16 + updates * 16 + 32,
  };
}
function fit(summary) {
  assert(
    Number.isInteger(summary.correct) &&
      summary.correct >= 0 &&
      summary.correct <= 16,
  );
  assert(Number.isFinite(summary.meanNll) && summary.meanNll >= 0);
  assert(Number.isFinite(summary.minimumTargetMargin));
  return (
    summary.correct === 16 &&
    summary.meanNll <= 0.05 &&
    summary.minimumTargetMargin > 0
  );
}
export function durationScreen(control, candidate, technicalExact) {
  const controlFit = fit(control),
    candidateFit = fit(candidate);
  assert.equal(typeof technicalExact, "boolean");
  return {
    controlFit,
    candidateFit,
    status: !technicalExact
      ? "BLOCKED_TECHNICAL_FAILURE"
      : candidateFit && !controlFit && candidate.meanNll < control.meanNll
        ? "PASS"
        : "FAIL",
  };
}
export function checkSyntheticPrefix(original, control, candidate) {
  assert.equal(original.history.length, 200);
  assert.equal(control.history.length, 200);
  assert.equal(candidate.history.length, 1000);
  for (const rows of [original.history, control.history, candidate.history])
    assert(rows.every((v) => Number.isFinite(v) && v >= 0));
  assert.deepEqual(control.history, original.history);
  assert.deepEqual(candidate.history.slice(0, 200), original.history);
  for (const key of ["initialWeights", "initialLogits", "finalWeights200"]) {
    assert.match(original[key], /^[0-9a-f]{64}$/);
    assert.equal(control[key], original[key]);
    assert.equal(candidate[key], original[key]);
  }
  assert.match(control.optimizerState200, /^[0-9a-f]{64}$/);
  assert.equal(candidate.optimizerState200, control.optimizerState200);
  return true;
}
export function validateProposal(p, base = root) {
  assert.equal(p.schemaVersion, 1);
  assert.equal(p.id, "controlled-fit-duration-proposal-v1");
  assert.equal(p.status, "PROPOSED_NOT_RUN");
  assert.equal(p.approval, "DESIGN_AND_PUBLICATION_ONLY");
  assert.equal(p.executionRequiresSeparateUserDecision, true);
  assert.equal(
    p.evidence.path,
    "assets/research/controlled-point-fit-20261028-v1.json",
  );
  assert.equal(hash(base, p.evidence.path), p.evidence.sha256);
  const evidence = json(base, p.evidence.path),
    s = evidence.summary,
    d = evidence.diagnostics;
  assert.equal(evidence.status, "PASS_STORED_REPLAY");
  assert.equal(evidence.researchVerdict, "FAIL");
  assert.deepEqual(p.evidence.observed, {
    fitStatus: s.fitGate.status,
    examples: 16,
    initialCorrect: s.initial.correct,
    finalCorrect: s.final.correct,
    meanFinalNll: s.final.meanNll,
    minimumFinalMargin: s.final.minimumTargetMargin,
    lastPreUpdateNll: d.savedTrainingLossMilestones.at(-1).meanTrainingNll,
    previousMilestoneStep: 175,
    previousPreUpdateNll: d.savedTrainingLossMilestones.find(
      (r) => r.step === 175,
    ).meanTrainingNll,
  });
  assert.equal(s.final.correct, 15);
  assert.equal(s.fitGate.status, "FAIL");
  assert(
    p.evidence.observed.lastPreUpdateNll <
      p.evidence.observed.previousPreUpdateNll,
  );
  assert.deepEqual(p.originalRun, originalIdentity);
  const old = json(
    base,
    "research/experiments/controlled-point-fit-proposal-v1.json",
  );
  const current = json(
    base,
    "research/experiments/controlled-point-fit-v1.json",
  ).contract;
  const sources = [
    ...Object.keys(old.sourceHashes),
    "research/cnn/controlled_point_fit.py",
    "scripts/research/controlled-point-fit-contract.mjs",
    "scripts/research/controlled-fit-diagnostics.mjs",
    "research/experiments/controlled-point-fit-v1.json",
  ];
  assert.deepEqual(Object.keys(p.sourceHashes).sort(), [...sources].sort());
  for (const path of sources)
    assert.equal(hash(base, path), p.sourceHashes[path], path);
  assert.deepEqual(p.shared.input, current.input);
  assert.deepEqual(p.shared.model, current.model);
  assert.equal(p.shared.loss, current.fit.loss);
  assert.equal(p.shared.order, current.fit.order);
  assert.deepEqual(p.shared.fitAcceptance, current.acceptance.fit);
  assert.equal(p.shared.marginRule, current.acceptance.margin);
  assert.deepEqual(p.shared.dataset, {
    examples: 16,
    candidatesPerExample: 31,
    totalCandidates: 496,
    split: "fit-only",
    reuse:
      "Exact stored original dataset bytes, label indices, features, footprints and order; no regeneration, domain, reference or new examples",
  });
  assert.match(p.shared.selection, /no score scaling, temperature or routing/);
  assert.deepEqual(p.contrast, {
    factor: "optimizer-update-count",
    controlUpdates: 200,
    candidateUpdates: 1000,
    updateRatio: 5,
    runs: 2,
    optimizer: "Adam",
    batchSize: 16,
    initialization:
      "Two fresh same-seed cold spatial models in sequential child processes; no warm start or checkpoint load",
    fixed:
      "Same data, architecture, 74 features, footprint, seed, native flat float32 CE, mean loss, optimizer, learning rate, per-step weight decay and fixed observation order",
    selection:
      "Control final step 200 and candidate final step 1000 only; no early stopping, best checkpoint, extra seed, duration sweep or continuation",
  });
  assert.equal(
    p.controls.order,
    "Complete exact 200-update control before creating the 1000-update candidate; mismatch stops the entire run",
  );
  assert.match(
    p.controls.control,
    /all 200 mean training losses, final logical weights/,
  );
  assert.match(p.controls.control, /two repeats exact/);
  assert.match(
    p.controls.candidateInitial,
    /no extra preflight model or forwards/,
  );
  assert.match(
    p.controls.candidatePrefix200,
    /step-200 optimizer state fingerprint exact to concurrent control/,
  );
  assert.match(
    p.controls.optimizerFingerprint,
    /original optimizer state was not saved/,
  );
  assert.match(p.controls.prefixObservations, /no forward, checkpoint load/);
  assert.deepEqual(p.forwardBudget, {
    control: phaseBudget(200),
    candidate: phaseBudget(1000),
    totalTopLevelMax: 19296,
    topLevelModels: 2,
    checkpointLoads: 0,
    extraPrefixOrTelemetryForwards: 0,
    scope:
      "Reserve before each top-level spatial call; nested common/residual modules are not extra top-level calls. No warmup or other model calls",
  });
  assert.equal(
    p.forwardBudget.control.total + p.forwardBudget.candidate.total,
    p.forwardBudget.totalTopLevelMax,
  );
  for (const [k, v] of Object.entries({
    children: 2,
    sequential: true,
    perChildProcessWallSeconds: 600,
    totalOwnerWallSeconds: 1260,
    sampledWorkingSetBytes: 2147483648,
  }))
    assert.equal(p.resources[k], v, k);
  assert.match(p.resources.timing, /caps include interpreter\/import startup/);
  assert.match(p.resources.failure, /no retry, restart, sweep or extra update/);
  assert.match(p.screening.fit, /unchanged 16\/16, mean NLL <= 0.05/);
  assert.match(p.screening.durationScreen, /all required/);
  assert.match(p.screening.durationScreen, /16\/16 alone cannot pass/);
  assert.match(
    p.screening.interpretation,
    /cumulative regularization and Adam state evolve/,
  );
  assert.match(p.screening.compute, /not an equal-compute or equal-time/);
  for (const [k, v] of Object.entries({
    designModelCreationLoadForwardTraining: 0,
    designDomainReferenceDatasetMaterialization: 0,
    trainingRunnerImplemented: false,
    originalFilesImmutable: true,
    newDataLabelsFeaturesArchitecture: false,
    reservedTestOrDevSelection: false,
    rolloutOrRl: false,
    appUsesDfs: true,
    rawWeightsLogitsOptimizerStatePrivate: true,
  }))
    assert.equal(p.boundaries[k], v, k);
  assert.match(p.boundaries.originalBudget, /new and unapproved/);
  assert.match(
    p.boundaries.future,
    /Separate explicit execution and results-publication decision/,
  );
  return {
    status: "PASS_STATIC_PROPOSAL",
    experimentStatus: p.status,
    controlUpdates: 200,
    candidateUpdates: 1000,
    proposedTopLevelForwards: 19296,
    actualModelCreationLoadForwardTraining: 0,
    actualDomainOrDatasetMaterialization: 0,
    limitation:
      "Static Node checks do not verify native FP32 training, prefix reproducibility, fitting or a causal mechanism.",
  };
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  assert.equal(
    process.argv.length,
    2,
    "Static validation only; no training or execution arguments",
  );
  console.log(JSON.stringify(validateProposal(json(root, proposalPath))));
}
