import assert from "node:assert/strict";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { root, json, sha } from "./lookahead-pairs-contract.mjs";
export const boardProposalPath =
  "research/experiments/board-context-proposal-v1.json";
export const boardContract = {
  dataset: {
    states: 32,
    candidatesEach: 32,
    totalCandidates: 1024,
    split: "FIT_ONLY_NO_GENERALIZATION",
    source:
      "Exact P110 build1 bytes and ordered unique quality roots; no generation or reference",
    labels:
      "Single complete quality root per example; no relabeling or target-dependent features",
  },
  model: {
    architecture: "Unmodified SpatialCandidateCNN v3",
    parameters: 13292,
    chunk: 32,
    residualScale: 1,
    initialization:
      "Two sequential cold same-seed models; exact initial logical weights across arms; checkpoint loads zero",
    config: {
      seed: 20261008,
      epochs: 12,
      batchSize: 8,
      threads: 2,
      convChannels: 8,
      embedding: 32,
      poolRows: 4,
      poolCols: 5,
      learningRate: 0.001,
      weightDecay: 0.0001,
    },
    interopThreads: 1,
  },
  input: {
    planes: [3, 16, 10],
    metadata: 140,
    candidateFeatures: 74,
    footprint: [1, 16, 10],
    factor: "OCCUPIED_PLANE_ONLY",
    control:
      "Set occupied plane channel 0 to literal FP32 zero in both common and residual branches; preserve item planes",
    candidate: "Use original three planes",
    unchanged:
      "All ordered features, metadata, footprints, targets, architecture, seed, loss, optimizer, order, update count and selection",
    excluded:
      "No descriptor, family, arm, coordinate key, reference evaluation or paths enter forward",
  },
  fit: {
    runs: 2,
    updatesEach: 1000,
    observationsPerUpdate: 32,
    optimizer: "Adam",
    loss: "Mean native FP32 flat cross_entropy over original 32 candidates; unique quality root target",
    order:
      "Original P110 descriptor order; one optimizer step after all 32 per-example backward calls",
    selection:
      "Final step 1000 only; original first-index argmax; no early stopping, extra seed, sweep or tuning",
  },
  budget: {
    initialEach: 32,
    trainingEach: 32000,
    finalEach: 64,
    totalEach: 32096,
    totalTopLevelForwardMax: 64192,
    modelCreations: 2,
    checkpointLoads: 0,
    domainReferenceMaterializations: 0,
  },
  resources: {
    sequentialChildren: 2,
    childWallSeconds: 600,
    ownerWallSeconds: 1260,
    sampledRssBytes: 2147483648,
    retries: 0,
    runtime: "Existing local CPU PyTorch only; no install/GPU/service",
  },
  validation: {
    beforeExecution:
      "Separate explicit user decision, new execution config/source hashes and clean commit; original proposal preserved",
    beforeCandidate:
      "Control technical/hash/budget/repeat checks PASS; its expected quality failure is not a technical STOP",
    sharedInitial:
      "Initial logical weight hashes exact across arms; initial logits are NOT expected equal with different planes",
    nativePointwise:
      "Control identical per-candidate FP32 inputs must yield exactly identical saved logits across all occurrences; otherwise technical FAIL, no comparison claim",
    counting:
      "Durable pre-call global/arm/phase ledger; top-level calls include initial/training/final repeats, no extra warmup or telemetry",
    stop: "Nonfinite/input/hash/schema/initial weights/ledger/repeat/resource mismatch STOP; no retry or tuning",
  },
  acceptance: {
    candidate: {
      correct: 32,
      maximumMeanNll: 0.05,
      minimumEveryTargetMarginExclusive: 0,
    },
    pairedMinimumNetCorrect: 1,
    controlPointwiseUpperCorrect: 16,
    technical:
      "Budget and exact repeated final logits checked separately from fit gates",
    interpretation:
      "Full fit PASS with control ceiling validates occupancy availability for this fixed fit-only task, not residual mechanism, independent generalization, original CNN error resolution or app adoption",
  },
  exclusions: [
    "new data",
    "reference",
    "teacher",
    "new architecture",
    "new loss",
    "checkpoint continuation",
    "dev/test selection",
    "held-out access",
    "RL",
    "app integration",
    "installation",
    "service",
    "GPU",
    "deployment",
    "blanket publication",
  ],
};
export const boardSourcePaths = [
  "research/cnn/model.py",
  "research/cnn/outcome.py",
  "research/cnn/spatial.py",
  "research/cnn/controlled_point_fit.py",
  "research/experiments/spatial-residual-v1.json",
  "scripts/research/lookahead-information.mjs",
  "scripts/research/validate-board-context-proposal.mjs",
];
export const boardEvidencePath =
  "assets/research/lookahead-information-20261106-v1.json";
export function checkBoardOutcomes(control, candidate) {
  for (const s of [control, candidate]) {
    assert.equal(s.examples, 32);
    assert(Number.isInteger(s.correct) && s.correct >= 0 && s.correct <= 32);
    assert(Number.isFinite(s.meanNll) && s.meanNll >= 0);
    assert(Number.isFinite(s.minMargin));
    assert.equal(typeof s.technicalPass, "boolean");
  }
  if (!control.technicalPass || !candidate.technicalPass)
    return "TECHNICAL_FAIL";
  if (control.correct > 16) return "CONTROL_IDENTITY_VIOLATION";
  if (
    candidate.correct !== 32 ||
    candidate.meanNll > 0.05 ||
    candidate.minMargin <= 0
  )
    return "FIT_FAIL";
  return candidate.correct - control.correct >= 1
    ? "FIT_ONLY_CONTRAST_PASS"
    : "NO_GAIN";
}
export function validateBoardProposal(p, base = root) {
  assert.deepEqual(
    Object.keys(p).sort(),
    [
      "schemaVersion",
      "id",
      "status",
      "approval",
      "executionRequiresSeparateUserDecision",
      "contract",
      "sourceHashes",
      "evidenceSha256",
      "archiveHashes",
    ].sort(),
  );
  assert.equal(p.schemaVersion, 1);
  assert.equal(p.id, "board-context-proposal-v1");
  assert.equal(p.status, "PROPOSED_NOT_RUN");
  assert.equal(p.approval, "DESIGN_ONLY");
  assert.equal(p.executionRequiresSeparateUserDecision, true);
  assert.deepEqual(p.contract, boardContract);
  assert.deepEqual(
    Object.keys(p.sourceHashes).sort(),
    [...boardSourcePaths].sort(),
  );
  for (const [path, h] of Object.entries(p.sourceHashes))
    assert.equal(sha(resolve(base, path)), h, path);
  assert.equal(sha(resolve(base, boardEvidencePath)), p.evidenceSha256);
  const parent = json(
    resolve(base, "research/experiments/lookahead-information-v1.json"),
  );
  assert.deepEqual(p.archiveHashes, parent.archiveHashes);
  const s = json(resolve(base, boardEvidencePath));
  assert.equal(s.status, "PASS");
  assert.equal(s.results.maskedSpatial.pairedPointwiseUpperBoundCorrect, 16);
  assert.equal(s.results.fullSpatial.pointwise.allPositiveMarginFeasible, true);
  const b = boardContract.budget;
  assert.equal(b.initialEach + b.trainingEach + b.finalEach, b.totalEach);
  assert.equal(2 * b.totalEach, b.totalTopLevelForwardMax);
  assert.equal(boardContract.fit.updatesEach * 32, b.trainingEach);
  return {
    status: "PASS_STATIC_CONTRACT",
    execution: "PROPOSED_NOT_RUN",
    modelRunnerImplemented: false,
    proposedForwardMax: b.totalTopLevelForwardMax,
    proposedCpuTrainingRuns: 2,
    proposedUpdatesEach: 1000,
    actualDomainReferenceDataPythonModelLoadForwardTraining: 0,
  };
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  assert.equal(
    process.argv.length,
    2,
    "Static validator accepts no training or override flags",
  );
  console.log(
    JSON.stringify(
      validateBoardProposal(json(resolve(root, boardProposalPath))),
    ),
  );
}
