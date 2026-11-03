import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { approvedDuration } from "./fit-duration-contract.mjs";
const root = fileURLToPath(new URL("../../", import.meta.url));
export const proposalPath = "research/experiments/outcome-mlp-proposal-v1.json";
const json = (base, p) => JSON.parse(readFileSync(resolve(base, p), "utf8"));
const hash = (base, p) =>
  createHash("sha256")
    .update(readFileSync(resolve(base, p)))
    .digest("hex");
export function mlpBudget() {
  return {
    initialFit: 16,
    training: 1000 * 16,
    finalFitFirst: 16,
    finalTransferFirst: 144,
    finalFitRepeat: 16,
    finalTransferRepeat: 144,
    total: 16 + 1000 * 16 + 2 * (16 + 144),
  };
}
export function validateMlpProposal(p, base = root) {
  const prior = approvedDuration(base).contract;
  assert.equal(p.schemaVersion, 1);
  assert.equal(p.id, "outcome-mlp-proposal-v1");
  assert.equal(p.status, "PROPOSED_NOT_RUN");
  assert.equal(p.approval, "DESIGN_AND_PUBLICATION_ONLY");
  assert.equal(p.executionRequiresSeparateUserDecision, true);
  assert.equal(
    p.evidence.path,
    "assets/research/outcome-input-equivalence-20261103-v1.json",
  );
  assert.equal(p.evidence.sha256, hash(base, p.evidence.path));
  const e = json(base, p.evidence.path);
  assert.equal(e.status, "PASS_STORED_INPUT_AUDIT");
  assert.equal(e.additionalPythonModelLoadForwardTrainingDomainDataset, 0);
  assert.equal(e.previousTransferGate, "FAIL");
  assert.equal(hash(base, e.sourceAggregate.path), e.sourceAggregate.sha256);
  assert.deepEqual(
    e.previousTransfer,
    json(base, e.sourceAggregate.path).transfer,
  );
  assert.deepEqual(e.inputAudit, {
    fitExamples: 16,
    transferExamples: 144,
    fitCandidateRows: 496,
    transferCandidateRows: 4464,
    fitUniqueOutcomeVectors: 136,
    transferUniqueOutcomeVectors: 136,
    transferUnseenOutcomeVectors: 0,
    fitOutcomeLabelConflicts: 0,
    uniqueFitBags: 16,
    labelPreservingBagMatches: 144,
    fitUniqueFull74Vectors: 496,
    transferUniqueFull74Vectors: 624,
    transferUnseenFull74Vectors: 144,
    transferTargetRowsUnseenFull74: 144,
    transferNonTargetRowsUnseenFull74: 0,
  });
  const paths = [
    ...Object.keys(prior.sourceHashes),
    "research/cnn/fit_duration.py",
    "research/cnn/frozen_point_transfer.py",
    "research/experiments/controlled-fit-duration-v1.json",
    "research/experiments/frozen-point-transfer-v1.json",
    "scripts/research/analyze-outcome-equivalence.mjs",
  ];
  assert.deepEqual(
    Object.keys(p.sourceHashes).sort(),
    [...new Set(paths)].sort(),
  );
  for (const path of paths)
    assert.equal(hash(base, path), p.sourceHashes[path], path);
  assert.equal(p.original.fitDatasetSha256, prior.originalRun.datasetSha256);
  assert.equal(
    p.original.transferDatasetSha256,
    "142c26aabd0dfbb863be9f0c50037849f22e7cffbec3f2c0a65753a849f01588",
  );
  assert.equal(
    p.original.initialFitLogitsSha256,
    prior.originalRun.initialPredictionsSha256,
  );
  assert.equal(
    p.original.initialLogicalWeightsSha256,
    prior.originalRun.initialLogicalWeightsSha256,
  );
  assert.equal(
    p.original.history1000Sha256,
    "2413bce01298f376bbd1a584f7633a64c1b0f3dec0585582cdb54332546b9590",
  );
  assert.equal(
    p.original.finalFitLogitsSha256,
    "bdc0560426ead312bf171490617f3b3e05743c2c8d453be262cc7b3da755ae30",
  );
  assert.equal(
    p.original.finalTransferLogitsSha256,
    "593c36101f71ca93f2ba52f9a7e743883711b218446ea10a129ac516cab5ec3c",
  );
  assert.equal(
    p.original.finalLogicalWeightsSha256,
    "424704f9ed88d76a3f77dcd2d3ff4493451ea66fdc20ac3f2cc5df4cb03a1981",
  );
  assert.deepEqual(p.shared.modelConfig, prior.shared.model.config);
  assert.equal(p.shared.interopThreads, 1);
  assert.deepEqual(
    { ...p.shared.data, materialization: null },
    {
      fitExamples: 16,
      transferExamples: 144,
      candidatesEach: 31,
      fitCandidateRows: 496,
      transferCandidateRows: 4464,
      materialization: null,
    },
  );
  assert.match(
    p.shared.data.materialization,
    /Exact existing stored bytes.*no new domain\/teacher\/labels\/data/,
  );
  assert.deepEqual(p.shared.training, {
    optimizer: "Adam",
    learningRate: 0.001,
    weightDecay: 0.0001,
    seed: 20261008,
    updates: 1000,
    batchExamples: 16,
    loss: "Native CPU float32 flat full-list cross_entropy; sum of 16 per-example losses divided by 16; one optimizer update after all 16 backwards",
    order:
      "Fixed original increasing row order; no shuffle or vectorization of the reference control",
    selection:
      "Final update1000 only; initial fit16 and final fit16/transfer144 twice, no early stop, best checkpoint, extra seed or tuning",
  });
  assert.equal(
    p.shared.prediction,
    "Original full-list raw argmax and first legal index on ties; no score scaling or routing",
  );
  assert.deepEqual(p.contrast.control, {
    architecture: "Unmodified SpatialCandidateCNN v3",
    parameters: 13292,
    planes: [3, 16, 10],
    metadata: 140,
    candidateFeatures: 74,
    footprint: [1, 16, 10],
    chunk: 32,
    residualScale: 1,
  });
  assert.deepEqual(p.contrast.candidate, {
    architecture: "Linear(22,32,bias=True) -> ReLU -> Linear(32,1,bias=True)",
    input:
      "Only stored candidateFeatures columns52..73, in original31-candidate order",
    layers: [22, 32, 1],
    activation: "ReLU",
    biases: true,
    parameters: 22 * 32 + 32 + 32 + 1,
    initialization:
      "Cold torch.nn.Linear default initialization after existing configure(seed20261008); no checkpoint load or original-CNN weight mapping",
    excludedInputs:
      "No board/metadata/footprint, absolute candidate coordinates, candidate key, instance/id, label or reference score enters forward",
  });
  assert.match(
    p.contrast.interpretation,
    /not a single-component causal ablation or equal-FLOPs comparison/,
  );
  assert.equal(
    p.controls.order,
    "Run complete exact spatial control before creating candidate model",
  );
  assert.match(
    p.controls.controlExact,
    /all1000 pre-update losses\/history.*STOP on mismatch/,
  );
  assert.match(p.controls.candidateInitial, /not expected equal/);
  assert.match(p.controls.candidateRepeat, /byteexact.*before-call ledger/);
  assert.match(p.controls.noReferenceModel, /no extra reference network/);
  assert.deepEqual(p.forwardBudget, {
    perArm: mlpBudget(),
    arms: 2,
    totalTopLevelMax: 2 * mlpBudget().total,
    topLevelModels: 2,
    trainingRuns: 2,
    checkpointLoads: 0,
    extraTelemetryOrWarmup: 0,
  });
  assert.deepEqual(p.resources, {
    runtime:
      "Existing CPU PyTorch only; no install/GPU/service/external transmission",
    sequentialChildren: 2,
    childProcessWallSeconds: 600,
    ownerWallSeconds: 1260,
    sampledWorkingSetBytes: 2147483648,
    processTimerIncludesImports: true,
    instantaneousPeakMeasured: false,
    failure:
      "Reserve global/arm/phase/example before each call; STOP on source/data/control/finite/ledger/time/RSS/cancel failure, retain partial artifacts, no retry/restart/tuning",
  });
  assert.deepEqual(p.screening.fit, {
    examples: 16,
    correct: 16,
    maximumMeanNll: 0.05,
    minimumEveryTargetMargin: 0,
  });
  assert.deepEqual(p.screening.transfer, {
    examples: 144,
    correct: 144,
    maximumMeanNll: 0.05,
    minimumEveryTargetMargin: 0,
  });
  assert.match(p.screening.comparison, /retain FAIL unless all gates pass/);
  assert.match(
    p.screening.inputEquivalence,
    /not an independent generalization result/,
  );
  assert.match(p.screening.limits, /post-hoc/);
  assert.deepEqual(p.boundaries, {
    modelRunnerImplemented: false,
    executionConfigImplemented: false,
    currentPythonModelLoadForwardTraining: 0,
    currentDomainTeacherMaterialization: 0,
    originalSourceDataModelImmutable: true,
    newScoreFeaturePolicyAdopted: false,
    appUsesDfs: true,
    rawWeightsLogitsAndPermutationWitnessesPrivate: true,
    publication:
      "Proposal/Node tools/tests/non-sensitive input counts/README only, no raw vectors or per-example permutation witnesses",
    future:
      "Explicit user model/input/two-CPU32672 execution decision required; past320/19296 budgets are spent",
  });
  return {
    status: "PASS_STATIC_PROPOSAL",
    proposal: p.id,
    execution: "NOT_RUN",
    candidateParameters: 769,
    proposedTotalForwards: 32672,
    actualPythonModelLoadForwardTrainingDomainDataset: 0,
    inputEquivalenceIsNotGeneralization: true,
  };
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  assert.equal(
    process.argv.length,
    2,
    "Static proposal validator accepts no execution/tuning arguments",
  );
  console.log(JSON.stringify(validateMlpProposal(json(root, proposalPath))));
}
