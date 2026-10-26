import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../", import.meta.url));
export const proposalPath =
  "research/experiments/controlled-point-fit-proposal-v1.json";
const bytes = (base, path) => readFileSync(resolve(base, path));
const json = (base, path) => JSON.parse(bytes(base, path).toString("utf8"));
const sha = (body) => createHash("sha256").update(body).digest("hex");

// Geometry selection arithmetic only: no fixture construction, domain or NN calls.
export function subsetIndices() {
  return Array.from({ length: 16 }, (_, row) => 10 * row + ((3 * row) % 10));
}
export function validateProposal(p, base = root) {
  assert.equal(p.schemaVersion, 1);
  assert.equal(p.id, "controlled-point-fit-proposal-v1");
  assert.equal(p.status, "PROPOSED_NOT_RUN");
  assert.equal(p.approval, "DESIGN_AND_PUBLICATION_ONLY");
  assert.equal(p.executionRequiresSeparateUserDecision, true);
  assert.equal(
    p.evidence.path,
    "assets/research/point-simulation-20261026-v1.json",
  );
  assert.equal(sha(bytes(base, p.evidence.path)), p.evidence.sha256);
  const evidence = json(base, p.evidence.path);
  assert.equal(evidence.status, "PASS");
  assert.equal(evidence.summary.fixtures, 160);
  assert.equal(evidence.summary.reference.complete, 160);
  assert.equal(evidence.summary.reference.uniqueOptimalFirstActions, 160);
  assert.equal(evidence.summary.policies["immediate-clear"].correctFirst, 160);
  const sources = [
    "src/research/point-simulation.ts",
    "src/research/imitation.ts",
    "src/research/action-outcome.ts",
    "src/research/spatial-footprint.ts",
    "research/cnn/model.py",
    "research/cnn/outcome.py",
    "research/cnn/spatial.py",
    "research/experiments/spatial-residual-v1.json",
  ];
  assert.deepEqual(Object.keys(p.sourceHashes).sort(), [...sources].sort());
  for (const path of sources)
    assert.equal(sha(bytes(base, path)), p.sourceHashes[path], path);
  const original = json(base, sources.at(-1));
  const configKeys = [
    "seed",
    "epochs",
    "batchSize",
    "threads",
    "convChannels",
    "embedding",
    "poolRows",
    "poolCols",
    "learningRate",
    "weightDecay",
  ];
  assert.deepEqual(
    p.model.config,
    Object.fromEntries(configKeys.map((key) => [key, original.training[key]])),
  );
  assert.equal(p.model.architecture, "Unmodified SpatialCandidateCNN v3");
  assert.equal(p.model.parameters, original.training.parameterCounts.spatial);
  assert.equal(p.model.chunk, 32);
  assert.equal(p.model.residualScale, 1);
  assert.equal(p.model.interopThreads, 1);
  assert.match(p.model.initialization, /no old checkpoint load/);
  assert.deepEqual(p.input.planes, [3, 16, 10]);
  assert.equal(p.input.metadata, 140);
  assert.equal(p.input.candidateFeatures, 74);
  assert.deepEqual(p.input.footprint, [1, 16, 10]);
  assert.match(
    p.input.construction,
    /no target, label, key, fixture id or reference score/,
  );
  assert.match(p.input.shortcut, /index 52/);
  assert.match(p.input.shortcut, /cannot establish spatial reasoning/);
  const indices = subsetIndices();
  assert.equal(new Set(indices).size, 16);
  assert.equal(new Set(indices.map((i) => i % 10)).size, 10);
  assert(indices.every((i, r) => Math.floor(i / 10) === r));
  assert.equal(p.dataset.profile, evidence.protocol.profile);
  assert.equal(
    p.dataset.selection,
    "For each row r=0..15, choose pointFixtures()[10*r + (3*r)%10]; preserve increasing row order",
  );
  for (const [k, v] of Object.entries({
    examples: 16,
    legalCandidatesPerExample: 31,
    totalCandidates: 496,
    distinctRows: 16,
    distinctColumns: 10,
    split: "fit-only",
    oldDatasetOrLabelChanges: false,
    referenceMaxNodesPerExample: 256,
    totalReferenceNodeCap: 4096,
    outcomeTransitions: 496,
    dfsCalls: 0,
  }))
    assert.equal(p.dataset[k], v, k);
  assert.match(p.dataset.label, /STOP on incomplete or nonunique reference/);
  assert.match(p.dataset.materialization, /Future approved execution only/);
  assert.equal(p.fit.runs, 1);
  assert.equal(p.fit.optimizer, "Adam");
  assert.equal(p.fit.steps, 200);
  assert.equal(p.fit.observationsPerStep, 16);
  assert.match(p.fit.order, /one optimizer update after all 16/);
  assert.match(p.fit.loss, /Mean native float32 cross_entropy/);
  assert.match(p.fit.loss, /flat loss only/);
  assert.match(p.fit.selection, /Final step 200 only/);
  assert.match(p.fit.selection, /no early stop, best-step reselection/);
  assert.match(p.fit.evaluation, /same in-memory final model/);
  assert.match(p.fit.oldFitRunner, /Do not call or modify fit_fixed/);
  assert.deepEqual(p.forwardBudget, {
    initial: 16,
    training: 3200,
    finalTwoRepeats: 32,
    totalTopLevelMax: 3248,
    modelCreations: 1,
    checkpointLoads: 0,
    scope:
      "Count before each top-level SpatialCandidateCNN call; internal common/residual modules are not separate top-level calls; no extra telemetry or warmup forward",
  });
  assert.equal(p.forwardBudget.training, p.fit.steps * p.dataset.examples);
  assert.equal(
    p.forwardBudget.totalTopLevelMax,
    p.forwardBudget.initial +
      p.forwardBudget.training +
      p.forwardBudget.finalTwoRepeats,
  );
  assert.equal(p.resources.children, 1);
  assert.equal(p.resources.childWallSeconds, 600);
  assert.equal(p.resources.supervisorWallSeconds, 660);
  assert.equal(p.resources.sampledWorkingSetBytes, 2147483648);
  assert.match(p.resources.limits, /Hard supervisor timeout/);
  assert.match(p.resources.limits, /instantaneous peak not measured/);
  assert.deepEqual(p.acceptance.fit, {
    correct: 16,
    examples: 16,
    maximumMeanNll: 0.05,
    minimumEveryTargetMargin: 0,
  });
  assert.match(p.acceptance.margin, /strictly greater than zero/);
  assert.match(
    p.acceptance.decision,
    /Technical PASS and fitting PASS are separate/,
  );
  for (const [k, v] of Object.entries({
    designModelCreationLoadForwardTraining: 0,
    designDomainTeacherDatasetMaterialization: 0,
    trainingRunnerImplemented: false,
    originalFilesImmutable: true,
    reservedTestAccess: false,
    devSelection: false,
    kindSelectionTested: false,
    rolloutOrRl: false,
    appUsesDfs: true,
    rawDatasetWeightsLogitsPrivate: true,
  }))
    assert.equal(p.boundaries[k], v, k);
  assert.match(p.boundaries.future, /Separate explicit decision required/);
  assert.match(p.boundaries.failure, /no retry or automatic tuning/);
  return {
    status: "PASS_STATIC_PROPOSAL",
    experimentStatus: p.status,
    subsetExamples: 16,
    coveredColumns: 10,
    proposedForwardLimit: 3248,
    actualModelCreationLoadForwardTraining: 0,
    actualDatasetMaterialization: 0,
  };
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  assert.equal(
    process.argv.length,
    2,
    "No execution options; static validation only",
  );
  console.log(JSON.stringify(validateProposal(json(root, proposalPath))));
}
