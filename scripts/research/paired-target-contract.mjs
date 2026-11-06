import assert from "node:assert/strict";
import { resolve } from "node:path";
import {
  root,
  json,
  sha,
  bytes,
  write,
  reserveFitOutput,
  auditPairsStored,
  parseBoardArgs,
  phases,
  armNames,
  armFiles,
  auditBoardLedger,
  fp32InputKey,
  auditBoardRows,
  boardFitGate,
  compareBoardArms,
} from "./board-context-contract.mjs";
import { inputRows } from "./lookahead-information.mjs";
import { pairedAnchorProof, inputScoreDrift } from "./board-context-stop.mjs";
import {
  pairProposalPath,
  pairContract,
  validatePairProposal,
  stopEvidencePath,
} from "./validate-paired-target-proposal.mjs";
import { checkBoardOutcomes } from "./validate-board-context-proposal.mjs";
export {
  root,
  json,
  sha,
  bytes,
  write,
  reserveFitOutput,
  auditPairsStored,
  parseBoardArgs,
  armNames,
};
export const boardConfigPath =
  "research/experiments/paired-target-identity-v1.json";
export function approvedPairedTarget(base = root) {
  const p = json(resolve(base, pairProposalPath));
  validatePairProposal(p, base);
  const c = json(resolve(base, boardConfigPath));
  assert.equal(c.id, "paired-target-identity-v1");
  assert.equal(c.status, "FROZEN_BEFORE_EXECUTION");
  assert.equal(c.approval, "USER_APPROVED");
  assert.equal(c.proposalPath, pairProposalPath);
  assert.equal(c.proposalSha256, sha(resolve(base, pairProposalPath)));
  assert.deepEqual(c.contract, pairContract.inheritedContrast);
  assert.deepEqual(c.revision, pairContract.revision);
  assert.deepEqual(
    c.archiveHashes,
    json(resolve(base, "research/experiments/board-context-v1.json"))
      .archiveHashes,
  );
  assert.equal(c.priorSummarySha256, sha(resolve(base, stopEvidencePath)));
  assert.deepEqual(c.boundaries, {
    newForwardMax: 64192,
    priorForwards: 32,
    cumulativeForwardMax: 64224,
    modelCreations: 2,
    checkpointLoads: 0,
    retryTuning: 0,
    newDataDomainReference: 0,
    rawPrivate: true,
    publicReplayAdditionalModelForwardTraining: 0,
  });
  return c;
}
export function phaseIdentity(rows, keys) {
  const drift = inputScoreDrift(rows, keys);
  return {
    anchors: pairedAnchorProof(rows, keys),
    globalDrift: drift.summary,
    mismatchTargetOccurrences: drift.cases
      .flatMap((c) => c.occurrences)
      .filter((x) => rows[x.example].label === x.index).length,
  };
}
export function auditPairedArm(recorded, arm, c = approvedPairedTarget()) {
  assert(armNames.includes(arm));
  const path = (p) => resolve(recorded, arm, p),
    r = json(path("report.json")),
    paired = json(path("paired-report.json"));
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
  const h = json(path("history.json"));
  assert.equal(h.length, 1000);
  h.forEach((v, i) => {
    assert.equal(v.step, i + 1);
    assert(Number.isFinite(v.meanTrainingNll) && v.meanTrainingNll >= 0);
  });
  const data = json(resolve(recorded, "dataset.json"));
  assert.equal(
    sha(resolve(recorded, "dataset.json")),
    c.archiveHashes["build1.json"],
  );
  const keys = data.examples.map((e) =>
    inputRows(e, arm === "control" ? "maskedSpatial" : "fullSpatial").map(
      fp32InputKey,
    ),
  );
  assert.deepEqual(json(path("input-identities.json")), keys);
  assert.equal(sha(path("final1.json")), sha(path("final2.json")));
  assert.equal(paired.status, "PASS");
  assert.equal(paired.arm, arm);
  assert.equal(paired.baseReportSha256, sha(path("report.json")));
  assert.equal(paired.identityRule, "PAIRED_TARGET_EXACT_TOLERANCE_ZERO");
  assert.equal(paired.additionalModelLoadForwardTraining, 0);
  assert.equal(paired.identityCalls, arm === "control" ? 4 : 0);
  const phaseNames = ["initial", "final1", "final2"],
    metrics = {},
    identities = {};
  assert.deepEqual(
    Object.keys(paired.phaseHashes).sort(),
    arm === "control" ? phaseNames.map((p) => p + "-identity.json").sort() : [],
  );
  for (const phase of phaseNames) {
    const rows = json(path(phase + ".json"));
    metrics[phase] = auditBoardRows(rows, data.examples);
    if (arm === "control") {
      identities[phase] = phaseIdentity(rows, keys);
      assert.equal(identities[phase].anchors.pairs, 16);
      assert.equal(identities[phase].globalDrift.uniqueInputs, 512);
      assert(metrics[phase].correct <= 16);
      assert.equal(
        sha(path(phase + "-identity.json")),
        paired.phaseHashes[phase + "-identity.json"],
      );
      assert.deepEqual(json(path(phase + "-identity.json")), identities[phase]);
      if (phase === "final1")
        assert.deepEqual(r.nativeIdentity, identities[phase]);
    } else
      identities[phase] = { globalDrift: inputScoreDrift(rows, keys).summary };
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
  return { report: r, summary, identities };
}
export function auditPairedStored(recorded, base = root) {
  const c = approvedPairedTarget(base),
    path = (p) => resolve(recorded, p),
    r = json(path("report.json")),
    f = json(path("execution.json"));
  assert.equal(r.status, "PASS");
  assert(r.elapsedSeconds >= 0 && r.elapsedSeconds <= 1260);
  assert(r.ownerMaxSampledRss > 0 && r.ownerMaxSampledRss <= 2147483648);
  assert.equal(f.status, "FROZEN_BEFORE_EXECUTION");
  assert.equal(f.configSha256, sha(resolve(base, boardConfigPath)));
  assert.equal(f.datasetSha256, c.archiveHashes["build1.json"]);
  assert.equal(sha(path("dataset.json")), f.datasetSha256);
  assert.deepEqual(f.inputHashes, c.archiveHashes);
  for (const [p, h] of Object.entries(f.inputHashes))
    assert.equal(sha(path("source/" + p)), h);
  auditPairsStored(path("source"), base);
  assert.equal(sha(path("prior-stop-summary.json")), c.priorSummarySha256);
  const names = [
    "start.json",
    "execution.json",
    "dataset.json",
    "prior-stop-summary.json",
    "candidate-input.json",
    "control/report.json",
    "control/paired-report.json",
    "candidate/report.json",
    "candidate/paired-report.json",
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
    controlPairedReportSha256: sha(path("control/paired-report.json")),
  });
  const control = auditPairedArm(recorded, "control", c),
    candidate = auditPairedArm(recorded, "candidate", c);
  assert.equal(
    control.report.initialLogicalWeightsSha256,
    candidate.report.initialLogicalWeightsSha256,
  );
  const paired = compareBoardArms(
    json(path("control/final1.json")),
    json(path("candidate/final1.json")),
  );
  const contrastOutcome = checkBoardOutcomes(
    { ...control.summary.final, technicalPass: true },
    { ...candidate.summary.final, technicalPass: true },
  );
  return {
    schemaVersion: 1,
    id: "paired-target-identity-v1",
    technicalStatus: "PASS",
    contrastOutcome,
    control: control.summary,
    candidate: candidate.summary,
    paired,
    identity: {
      sameInitialLogicalWeights: true,
      pairedTargetAnchorsExactAllPhases: true,
      finalRepeatsExact: true,
      tolerance: 0,
    },
    identityDiagnostics: {
      control: control.identities,
      candidate: candidate.identities,
    },
    actualCompute: {
      topLevelModels: 2,
      checkpointLoads: 0,
      trainingRuns: 2,
      optimizerUpdates: 2000,
      newTopLevelForwards: 64192,
      priorStoppedForwards: 32,
      cumulativeTopLevelForwards: 64224,
      domainReferenceMaterializations: 0,
    },
    timing: {
      ownerWallSeconds: r.elapsedSeconds,
      ownerMaxSampledRss: r.ownerMaxSampledRss,
      children: r.children.map((child, i) => ({
        arm: child.arm,
        processWallSeconds: child.result.elapsedMs / 1000,
        pythonBudget: [control, candidate][i].report.budget,
      })),
      instantaneousPeakMeasured: false,
    },
    additionalReplayModelLoadForwardTraining: 0,
    interpretation:
      "Same-CNN occupancy fit-only contrast, exact target anchors with zero tolerance. No independent generalization, residual necessity, original CNN failure resolution or app adoption.",
    appUsesDfs: true,
  };
}
