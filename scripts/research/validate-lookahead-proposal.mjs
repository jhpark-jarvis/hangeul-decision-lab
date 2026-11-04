import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const root = fileURLToPath(new URL("../../", import.meta.url));
export const proposalPath =
  "research/experiments/lookahead-pairs-proposal-v1.json";
export const sha = (path) =>
  createHash("sha256").update(readFileSync(path)).digest("hex");
export const readJson = (path) => JSON.parse(readFileSync(path, "utf8"));

// Integer descriptors only. This module never constructs a board or calls domain.
export const contract = {
  generator: {
    version: "two-well-mieum-counterfactual-v1",
    rows: 16,
    cols: 10,
    firstRows: [0, 2, 4, 6],
    columns: [0, 2, 5, 7],
    secondRowDelta: 7,
    arms: ["early-aligned", "late-aligned"],
    familyCount: 16,
    diagnosticStates: 32,
    repetitions: 2,
    background:
      "Two holes at (3*row+phase) mod10 and the following column mod10; phase=(3*firstRow+column) mod10.",
    wells:
      "Override background in each three-row well. Top has one hole: column+1 in early well, column+2 in late well. Bottom has holes column,column+1,column+2. Middle has column,column+2 when aligned, otherwise column+1,column+2. Exactly one aligned well per arm.",
    pieces: ["MIEUM", "MIEUM"],
    slots: [0, 1],
    initialPoint: 1,
    initialReroll: 0,
    hiddenItems: 0,
    pendingReroll: null,
    initialCompletedRows: 0,
    expectedPointOnlyCandidates: 32,
    seed: null,
    augmentation: false,
    trainingDevTestSplit: "NONE_DIAGNOSTIC_ONLY",
    encoding:
      "Unchanged encodeDecision52 + actionOutcome22 and encodeFootprints. Constructor/arm/family/quality labels never enter features. 24/74 are diagnostic projections only, not new model inputs.",
  },
  reference: {
    implementation:
      "NEW_NOT_IMPLEMENTED; existing pointReference rejects this two-MIEUM domain",
    traversal:
      "Full ordered legal tree once per state; retain the best terminal evaluation conditional on EACH root, without memoization or production pruning.",
    evaluator:
      "Existing evaluateState/compareEvaluations lexicographic order and current 19-piece catalog, unchanged.",
    terminal:
      "Stop only at no remaining pieces or gameover. No next supply, reroll, items, reacquisition, random future or early stopping at next-playing.",
    maxActions: 3,
    maxVisitedNodesPerState: 64,
    rootCountsAsNode: true,
    reserveBeforeEveryEdgeApply: true,
    maxApplyPerState: 63,
    rewards:
      "Accumulate actual clearedRows and acquiredItems along each path before terminal evaluateState; no new score or partial-depth terminal fallback.",
    incomplete:
      "Retain partial report; STOP, no reference labels or baseline-success verdict; no retry, quota refill, tuning or extra samples.",
    qualityTargets:
      "All root indices whose complete conditional best terminal evaluation equals the complete global best by compareEvaluations. Single-label representative is first ordered quality member, auxiliary only; existing labels unchanged.",
  },
  gates: {
    structure:
      "All 32 planned states: initial point-only legal list of32, no items or full rows; exactly two immediate1-row clears and all others0; legal order and inputs immutable.",
    primary:
      "16 late-aligned states: complete reference, all-pieces-placed terminal exists, immediate-clear-tied candidates have different best quality and first-legal immediate baseline is outside the quality set.",
    control:
      "16 early-aligned states: same complete and salvageability checks, fixed immediate baseline is inside the quality set. Report the worse tied immediate action too.",
    pairing:
      "Each family has two different boards, matching metadata and exact ordered FP32 outcome22 matrices, equal root count, disjoint nonempty quality-target index sets. Expected two unique point-clear roots; reference must establish them, never overwrite labels from geometry.",
    representation:
      "Audit ordered22 identity and quality-set information bound; report22/24/74 and board-plane distinctions separately. Board-aware CNN is not restricted to22 and no model is executed.",
    novelty:
      "Require32 distinct board/D2 groups and no board/D2 overlap with prior train-dev, fit16, transfer144 and coverage64. Report FP32 target/row/tagged-multiset overlap and cross-family duplicates, not independent generalization or a new training split.",
    holdoutBoundary:
      "Original held-out archive access0; no model fitting or evaluation split created. Counterfactual arm pairs stay in one family in any separately approved future split.",
    quotas:
      "Any incomplete/structure/pair/primary/control/novelty/repeat mismatch fails this fixed campaign; retain every attempted descriptor and rejection. No resampling, split reassignment or replacement.",
  },
  proposedCompute: {
    uniqueStates: 32,
    builds: 2,
    stateMaterializations: 64,
    referenceCalls: 64,
    referenceNodeReservations: 4096,
    referenceApplyCallsMax: 4032,
    outcomeApplyCalls: 2048,
    totalApplyCallsMax: 6080,
    extraWitnessReplayCalls: 0,
    productionDfsCalls: 0,
    pythonModelLoadForwardTraining: 0,
  },
  resources: {
    nodeWallSeconds: 120,
    sampledRssBytes: 2147483648,
    instantaneousPeakMeasured: false,
    retries: 0,
    newPackagesRuntimesServicesGpu: 0,
  },
  stage: {
    current: "DESIGN_AND_STATIC_VALIDATION_ONLY",
    next: "Separate user decision required for new diagnostic states, reference and quality targets. Generator/runner/execution config not implemented.",
    following:
      "Any model, feature, loss, training, test, RL, app adoption or publication is a separate decision; no learning budget here.",
  },
};

export const sourcePaths = [
  "src/domain/pieces/catalog.ts",
  "src/domain/game/actions.ts",
  "src/domain/game/hidden-items.ts",
  "src/domain/game/game-state.ts",
  "src/domain/solver/evaluator.ts",
  "src/research/point-simulation.ts",
  "src/research/imitation.ts",
  "src/research/action-outcome.ts",
  "src/research/spatial-footprint.ts",
  "scripts/research/audit-coverage-inputs.mjs",
  "scripts/research/representation-alias.mjs",
  "assets/research/coverage-training-20261104-v1.json",
  "assets/research/representation-alias-20261104-v1.json",
  "assets/research/stored-alias-replay-20261104-v2.json",
  "scripts/research/validate-lookahead-proposal.mjs",
];
export const inputPaths = [
  ".research-output/paired-teacher/paired-teacher-cnn-20261011-v1/train-dev/dataset.json",
  ".research-output/action-outcome-train-dev-20261012-recovered/dataset.json",
  ".research-output/controlled-point-fit-20261028-v1/dataset.json",
  ".research-output/frozen-point-transfer-20261101-v1/transfer-dataset.json",
  ".research-output/coverage-preflight-20261104-v1/build1.json",
];

export function lookaheadSchedule() {
  return contract.generator.firstRows.flatMap((firstRow, ri) =>
    contract.generator.columns.flatMap((column, ci) =>
      contract.generator.arms.map((arm) => ({
        familyIndex: ri * 4 + ci,
        firstRow,
        secondRow: firstRow + 7,
        column,
        phase: (3 * firstRow + column) % 10,
        arm,
      })),
    ),
  );
}

function checkedIndices(indices, n, nonempty = true) {
  assert(Array.isArray(indices) && (!nonempty || indices.length > 0));
  assert(indices.every((i) => Number.isSafeInteger(i) && i >= 0 && i < n));
  assert.deepEqual(
    indices,
    [...new Set(indices)].sort((a, b) => a - b),
  );
}
/** Scalar gate records only, not a reference or an action-selection policy. */
export function classifyLookaheadRecord(r) {
  assert(Number.isSafeInteger(r.rootCount) && r.rootCount > 0);
  assert.equal(typeof r.complete, "boolean");
  if (!r.complete) return "INCOMPLETE";
  checkedIndices(r.salvageableIndices, r.rootCount, false);
  checkedIndices(r.qualityIndices, r.rootCount);
  checkedIndices(r.maxImmediateIndices, r.rootCount);
  assert.equal(
    r.baselineIndex,
    r.maxImmediateIndices[0],
    "First legal immediate tie",
  );
  if (!r.salvageableIndices.length) return "DEAD_END";
  assert(r.qualityIndices.every((i) => r.salvageableIndices.includes(i)));
  const winners = r.maxImmediateIndices.filter((i) =>
    r.qualityIndices.includes(i),
  );
  if (!winners.length) return "IMMEDIATE_COUNT_TRAP";
  if (winners.length === r.maxImmediateIndices.length) return "EASY";
  return r.qualityIndices.includes(r.baselineIndex)
    ? "ORDER_LUCKY_CONTROL"
    : "HARD";
}

/** Ideal deterministic selector of the ENTIRE identical ordered22 input; no NN. */
export function qualityInputBound(rows) {
  assert(Array.isArray(rows) && rows.length > 0);
  const groups = new Map();
  for (const r of rows) {
    assert.equal(r.complete, true);
    assert.match(r.ordered22Sha256, /^[a-f0-9]{64}$/);
    assert(Number.isSafeInteger(r.rootCount) && r.rootCount > 0);
    checkedIndices(r.qualityIndices, r.rootCount);
    if (!groups.has(r.ordered22Sha256))
      groups.set(r.ordered22Sha256, {
        rootCount: r.rootCount,
        support: new Map(),
        rows: 0,
      });
    const g = groups.get(r.ordered22Sha256);
    assert.equal(g.rootCount, r.rootCount);
    g.rows++;
    for (const i of r.qualityIndices)
      g.support.set(i, (g.support.get(i) || 0) + 1);
  }
  return {
    examples: rows.length,
    orderedInputGroups: groups.size,
    idealQualityMembershipUpperBoundCorrect: [...groups.values()].reduce(
      (n, g) => n + Math.max(...g.support.values()),
      0,
    ),
  };
}

export function validateLookaheadProposal(p, base = root) {
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
      "inputHashes",
    ].sort(),
  );
  assert.equal(p.schemaVersion, 1);
  assert.equal(p.id, "lookahead-pairs-proposal-v1");
  assert.equal(p.status, "PROPOSED_NOT_RUN");
  assert.equal(p.approval, "DESIGN_ONLY");
  assert.equal(p.executionRequiresSeparateUserDecision, true);
  assert.deepEqual(p.contract, contract);
  for (const [key, paths] of [
    ["sourceHashes", sourcePaths],
    ["inputHashes", inputPaths],
  ]) {
    assert.deepEqual(Object.keys(p[key]).sort(), [...paths].sort());
    for (const path of paths)
      assert.equal(p[key][path], sha(resolve(base, path)), path);
  }
  const s = lookaheadSchedule();
  assert.equal(s.length, 32);
  assert.equal(new Set(s.map((d) => d.familyIndex)).size, 16);
  for (const d of s) {
    const rows = [
      d.firstRow,
      d.firstRow + 1,
      d.firstRow + 2,
      d.secondRow,
      d.secondRow + 1,
      d.secondRow + 2,
    ];
    assert.equal(new Set(rows).size, 6);
    assert(rows.every((r) => r >= 0 && r < 16));
    assert(d.column >= 0 && d.column + 2 < 10);
    assert.equal(2 * 1 + 2 * 3 + 12 * 2, 32); // Hole-count arithmetic, no board materialization.
  }
  const b = contract.proposedCompute;
  assert.equal(b.stateMaterializations, b.uniqueStates * b.builds);
  assert.equal(
    b.referenceNodeReservations,
    b.referenceCalls * contract.reference.maxVisitedNodesPerState,
  );
  assert.equal(
    b.referenceApplyCallsMax,
    b.referenceNodeReservations - b.referenceCalls,
  );
  assert.equal(b.outcomeApplyCalls, b.stateMaterializations * 32);
  assert.equal(
    b.totalApplyCallsMax,
    b.referenceApplyCallsMax + b.outcomeApplyCalls,
  );
  const prior = readJson(
    resolve(base, "assets/research/stored-alias-replay-20261104-v2.json"),
  );
  assert.equal(prior.status, "PASS");
  assert.equal(prior.wholeTurnSalvageability, "NOT_EVALUATED");
  return {
    status: "PASS_STATIC_CONTRACT",
    execution: "PROPOSED_NOT_RUN",
    descriptors: 32,
    counterfactualFamilies: 16,
    proposedCompute: b,
    actualDomainReferenceDatasetModelLoadForwardTraining: 0,
    publication: "NOT_AUTHORIZED_THIS_TASK",
  };
}

if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  assert.equal(
    process.argv.length,
    2,
    "Static validation accepts no execution/tuning options",
  );
  console.log(
    JSON.stringify(
      validateLookaheadProposal(readJson(resolve(root, proposalPath))),
    ),
  );
}
