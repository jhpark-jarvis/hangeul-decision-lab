import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { approvedDuration } from "./fit-duration-contract.mjs";

const root = fileURLToPath(new URL("../../", import.meta.url));
export const proposalPath =
  "research/experiments/frozen-point-transfer-proposal-v1.json";
const json = (base, path) =>
  JSON.parse(readFileSync(resolve(base, path), "utf8"));
const hash = (base, path) =>
  createHash("sha256")
    .update(readFileSync(resolve(base, path)))
    .digest("hex");

/** Index arithmetic only. Does not construct a board, label, tensor or model. */
export function transferPartition() {
  const fit = Array.from({ length: 16 }, (_, r) => 10 * r + ((3 * r) % 10));
  const seen = new Set(fit);
  const transfer = Array.from({ length: 160 }, (_, i) => i).filter(
    (i) => !seen.has(i),
  );
  return { fit, transfer };
}

/** Synthetic or saved scalar gate; never an inference or checkpoint selector. */
export function transferScreen(summary, technicalExact) {
  assert.equal(typeof technicalExact, "boolean");
  assert.equal(summary.examples, 144);
  assert(
    Number.isInteger(summary.correct) &&
      summary.correct >= 0 &&
      summary.correct <= 144,
  );
  assert(Number.isFinite(summary.meanNll) && summary.meanNll >= 0);
  assert(Number.isFinite(summary.minimumTargetMargin));
  const checks = {
    allCorrect: summary.correct === 144,
    meanNllAtMost005: summary.meanNll <= 0.05,
    everyTargetMarginPositive: summary.minimumTargetMargin > 0,
  };
  return {
    status: !technicalExact
      ? "BLOCKED_TECHNICAL_FAILURE"
      : Object.values(checks).every(Boolean)
        ? "PASS"
        : "FAIL",
    checks,
  };
}

export function validateTransferProposal(p, base = root) {
  const original = approvedDuration(base).contract;
  assert.equal(p.schemaVersion, 1);
  assert.equal(p.id, "frozen-point-transfer-proposal-v1");
  assert.equal(p.status, "PROPOSED_NOT_RUN");
  assert.equal(p.approval, "LOCAL_DESIGN_ONLY");
  assert.equal(p.executionRequiresSeparateUserDecision, true);
  assert.equal(
    p.evidence.path,
    "assets/research/frozen-point-transfer-design-20261029-v1.json",
  );
  assert.equal(hash(base, p.evidence.path), p.evidence.sha256);
  const evidence = json(base, p.evidence.path);
  assert.equal(evidence.status, "PASS_SAVED_ANALYSIS_ONLY");
  assert.equal(evidence.observedTransfer, "NOT_RUN");
  assert.deepEqual(evidence.additionalCompute, {
    python: 0,
    modelCreation: 0,
    checkpointLoad: 0,
    forward: 0,
    training: 0,
    domainReference: 0,
    datasetMaterialization: 0,
  });
  assert.equal(
    hash(base, evidence.sourceAggregate.path),
    evidence.sourceAggregate.sha256,
  );
  const aggregate = json(base, evidence.sourceAggregate.path);
  assert.equal(aggregate.durationScreen.status, "PASS");
  assert.deepEqual(evidence.fit, aggregate.arms.candidate.final);
  assert.deepEqual(
    evidence.lossMilestones,
    aggregate.arms.candidate.savedPreUpdateLoss,
  );
  const paths = [
    "src/research/point-simulation.ts",
    "src/research/controlled-point-fit.ts",
    "src/research/imitation.ts",
    "src/research/action-outcome.ts",
    "src/research/spatial-footprint.ts",
    "research/cnn/model.py",
    "research/cnn/outcome.py",
    "research/cnn/spatial.py",
    "research/cnn/controlled_point_fit.py",
    "research/cnn/fit_duration.py",
    "research/experiments/controlled-fit-duration-v1.json",
  ];
  assert.deepEqual(Object.keys(p.sourceHashes).sort(), paths.sort());
  for (const path of paths)
    assert.equal(hash(base, path), p.sourceHashes[path], path);
  assert.equal(
    p.checkpoint.source,
    "controlled-fit-duration-v1/candidate/final-1000",
  );
  assert.equal(
    p.checkpoint.serializationSha256,
    "abb5d67b6afbb16a275e2bfde7d33821a2fb5d44497007203a034d7dae090cec",
  );
  assert.equal(
    p.checkpoint.logicalWeightsSha256,
    "424704f9ed88d76a3f77dcd2d3ff4493451ea66fdc20ac3f2cc5df4cb03a1981",
  );
  assert.equal(
    p.checkpoint.fitPredictionsSha256,
    "bdc0560426ead312bf171490617f3b3e05743c2c8d453be262cc7b3da755ae30",
  );
  assert.equal(
    p.checkpoint.fitDatasetSha256,
    original.originalRun.datasetSha256,
  );
  assert.match(
    p.checkpoint.selection,
    /Already fixed final step 1000; no new checkpoint selection, retraining or optimizer construction/,
  );
  assert.match(
    p.checkpoint.loading,
    /weights_only=True.*strict keys\/shapes\/finite values.*eval and no_grad/,
  );
  const { fit, transfer } = transferPartition();
  const columnCounts = Array.from(
    { length: 10 },
    (_, c) => transfer.filter((i) => i % 10 === c).length,
  );
  for (let r = 0; r < 16; r++)
    assert.equal(transfer.filter((i) => Math.floor(i / 10) === r).length, 9);
  assert.deepEqual(evidence.split, {
    population: 160,
    fit: 16,
    transfer: 144,
    fitCandidates: 496,
    transferCandidates: 4464,
    transferPerRow: 9,
    transferColumnCounts: columnCounts,
  });
  const split = p.split;
  for (const [k, v] of Object.entries({
    profile: "controlled-single-point-line-unlock",
    population: 160,
    rows: 16,
    cols: 10,
    fitFormula: "10*r + (3*r)%10 for r=0..15",
    fitExamples: fit.length,
    transferExamples: transfer.length,
    order:
      "Fit in increasing row order; complement in increasing original fixture index",
    candidatesPerExample: 31,
    transferPerRow: 9,
    transferColumnCounts: columnCounts,
    overlap: 0,
  }))
    assert.deepEqual(split[k], v, k);
  assert.match(
    split.definition,
    /no new rows\/columns, random seeds or reserved test split/,
  );
  assert.match(split.priorExposure, /not a pristine held-out benchmark/);
  assert.deepEqual(p.input, {
    planes: original.shared.input.planes,
    metadata: 140,
    candidateFeatures: 74,
    footprint: [1, 16, 10],
    modelParameters: 13292,
    chunk: 32,
    residualScale: 1,
    selection:
      "Unchanged full legal-list raw argmax; first legal index on ties",
    shortcut:
      "Outcome index 52 clearedRows/16 remains; even complete transfer cannot prove spatial reasoning or residual necessity",
  });
  const material = p.materialization;
  for (const [k, v] of Object.entries({
    legalCandidates: transfer.length * 31,
    outcomeTransitions: transfer.length * 31,
    referenceMaxNodesPerExample: 256,
    totalReferenceNodeCap: transfer.length * 256,
    dfsCalls: 0,
    oldTrainDevTestOrLabelChanges: false,
  }))
    assert.equal(material[k], v, k);
  assert.match(
    material.fit,
    /Read original 16 stored examples byte-for-byte; do not regenerate/,
  );
  assert.match(
    material.transfer,
    /Future separate isolated 144-example artifact only/,
  );
  assert.match(
    material.preflight,
    /before creating the model; STOP on mismatch/,
  );
  assert.deepEqual(p.forwardBudget, {
    fitControl: 16,
    transfer: 144,
    repeatFit: 16,
    repeatTransfer: 144,
    totalTopLevelMax: 320,
    topLevelModels: 1,
    checkpointLoads: 1,
    trainingRuns: 0,
    optimizerUpdates: 0,
    warmupOrExtraCalls: 0,
    order:
      "fit-control16 exact to stored candidate final logits/metrics -> transfer144 -> fit-repeat16 -> transfer-repeat144; STOP before transfer on control mismatch",
  });
  assert.deepEqual(p.resources, {
    runtime:
      "Existing CPU PyTorch only; no install, GPU, new runtime, service or external transmission",
    children: 1,
    childProcessWallSeconds: 600,
    ownerWallSeconds: 660,
    sampledWorkingSetBytes: 2147483648,
    processTimerIncludesImports: true,
    instantaneousPeakMeasured: false,
    failure:
      "Pre-call global/phase/example ledger, exclusive new output, STOP on hash/load/control/finite/schema/reference/repeat/budget failure; retain partial data, no retry or tuning",
  });
  assert.deepEqual(p.screening.transfer, {
    examples: 144,
    correct: 144,
    maximumMeanNll: 0.05,
    minimumEveryTargetMargin: 0,
  });
  assert.match(
    p.screening.technical,
    /original fit data\/predictions replay.*320-call pre-ledger/,
  );
  assert.match(
    p.screening.rule,
    /No threshold tuning.*merging fit examples into the transfer denominator/,
  );
  assert.match(p.screening.descriptive, /not excluded observations/);
  assert.match(
    p.screening.interpretation,
    /No out-of-distribution.*spatial-mechanism or app-adoption claim/,
  );
  assert.deepEqual(p.boundaries, {
    designPythonModelLoadForwardTraining: 0,
    designDomainReferenceMaterialization: 0,
    runnerImplemented: false,
    originalSourceDataModelImmutable: true,
    additionalTraining: false,
    featureArchitectureScoreChanges: false,
    reservedTestDevOrRollout: false,
    appUsesDfs: true,
    rawDataWeightsLogitsPrivate: true,
    publication:
      "Local README/proposal only in this task; a new explicit publication decision is required before public preparation or push",
    future:
      "Separate user decision required for isolated 144-example materialization, model creation/load and up to 320 evaluation forwards; previous 19296-call training budget is spent",
  });
  return {
    status: "PASS_STATIC_DESIGN",
    proposal: p.id,
    execution: "NOT_RUN",
    partition: { fit: 16, transfer: 144, overlap: 0 },
    proposedEvaluationForwards: 320,
    proposedTrainingRuns: 0,
    actualPythonModelLoadForwardTraining: 0,
    actualDomainReferenceMaterialization: 0,
    publication: "LOCAL_ONLY",
  };
}

if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  assert.equal(
    process.argv.length,
    2,
    "Static validator takes no run, data, checkpoint or tuning arguments",
  );
  console.log(
    JSON.stringify(validateTransferProposal(json(root, proposalPath))),
  );
}
