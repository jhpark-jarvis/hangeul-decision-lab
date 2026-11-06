import assert from "node:assert/strict";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { root, json, sha } from "./board-context-contract.mjs";
import { boardContract } from "./validate-board-context-proposal.mjs";
import { pairedAnchorProof } from "./board-context-stop.mjs";
export const pairProposalPath =
  "research/experiments/paired-target-identity-proposal-v1.json";
export const revisedNativeRule =
  "For each counterfactual pair, require exact FP32 identity of both opposing target inputs, bitwise-equal native scores for each anchor across the pair, and unchanged relative first-index tie priority. Require control top1 <=16/32. Global non-anchor input-class drift is reported without changing scores or applying tolerances.";
export const pairContract = {
  inheritedContrast: {
    ...structuredClone(boardContract),
    validation: {
      ...boardContract.validation,
      nativePointwise: revisedNativeRule,
    },
  },
  revision: {
    onlyScientificContractChange:
      "Required native identity scope: opposing target anchors within each pair",
    requiredAt: ["initial", "final1", "final2"],
    globalDrift:
      "Measure every identical-input class, signed-zero/ULP gaps and target occurrence counts; diagnostic only outside required anchors",
    tolerance: 0,
    targetPairScoreEquality:
      "Bitwise exact; one ULP target difference still STOP",
    proof:
      "Both opposite target inputs occur in both original ordered lists; each anchor's score is equal across that pair; tie priority has same relative order. Both opposite targets cannot win first-index argmax. Sixteen disjoint pairs imply at most16 correct.",
    candidate:
      "Unchanged original planes, inputs and gates; initial logical weights exact to fresh control; initial logits not forced equal",
  },
  execution: {
    status: "PROPOSED_NOT_RUN",
    runnerImplemented: false,
    newExecutionConfigRequired: true,
    freshAttempt:
      "New explicit approval required after P115 STOP; no reuse of unused V1 budget",
    originalAttempt: {
      modelCreations: 1,
      initialForwards: 32,
      optimizerUpdates: 0,
      retries: 0,
    },
    newProposed: {
      modelCreations: 2,
      trainingRuns: 2,
      updatesEach: 1000,
      topLevelForwardMax: 64192,
      checkpointLoads: 0,
    },
    maximumAcrossOldAndNewIfApproved: 64224,
    oldSourceConfigDataFailureImmutable: true,
  },
  claims: {
    notANumericalToleranceIncrease: true,
    noNewModelInputLossArchitectureOrData: true,
    fitOnly: true,
    noIndependentGeneralization: true,
    noResidualMechanismClaim: true,
    oldFailedContractNotWaived: true,
  },
};
export const pairSourcePaths = [
  "research/cnn/board_context_fit.py",
  "research/experiments/board-context-v1.json",
  "scripts/research/board-context-contract.mjs",
  "scripts/research/board-context-stop.mjs",
  "scripts/research/validate-board-context-proposal.mjs",
  "scripts/research/validate-paired-target-proposal.mjs",
];
export const stopEvidencePath =
  "assets/research/board-context-stop-20261106-v1.json";
export function checkedPairProof(rows, keys) {
  for (const r of rows)
    assert(r.logits.every((v) => Number.isFinite(v) && Math.fround(v) === v));
  return pairedAnchorProof(rows, keys);
}
export function validatePairProposal(p, base = root) {
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
    ].sort(),
  );
  assert.equal(p.schemaVersion, 1);
  assert.equal(p.id, "paired-target-identity-proposal-v1");
  assert.equal(p.status, "PROPOSED_NOT_RUN");
  assert.equal(p.approval, "DESIGN_ONLY");
  assert.equal(p.executionRequiresSeparateUserDecision, true);
  assert.deepEqual(p.contract, pairContract);
  assert.deepEqual(
    Object.keys(p.sourceHashes).sort(),
    [...pairSourcePaths].sort(),
  );
  for (const [path, h] of Object.entries(p.sourceHashes))
    assert.equal(sha(resolve(base, path)), h, path);
  assert.equal(p.evidenceSha256, sha(resolve(base, stopEvidencePath)));
  const s = json(resolve(base, stopEvidencePath));
  assert.equal(s.experimentTechnicalStatus, "FAIL");
  assert.equal(s.actualCompute.totalTopLevelForwards, 32);
  assert.equal(s.actualCompute.optimizerUpdates, 0);
  assert.equal(s.drift.maximumUlpGap, 1);
  assert.equal(s.pairedTargetProofAtInitialOnly.pairs, 16);
  assert.equal(s.pairedTargetProofAtInitialOnly.exactTargetAnchorScores, true);
  assert.equal(
    pairContract.execution.maximumAcrossOldAndNewIfApproved,
    32 + 64192,
  );
  return {
    status: "PASS_STATIC_CONTRACT",
    execution: "PROPOSED_NOT_RUN",
    newLearningDecisionRequired: true,
    proposedNewCpuRuns: 2,
    proposedNewUpdatesEach: 1000,
    proposedNewForwards: 64192,
    actualAdditionalDomainReferenceDataPythonModelLoadForwardTraining: 0,
  };
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  assert.equal(process.argv.length, 2, "No execution/tuning flags");
  console.log(
    JSON.stringify(validatePairProposal(json(resolve(root, pairProposalPath)))),
  );
}
