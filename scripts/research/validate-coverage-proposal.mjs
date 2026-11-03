import assert from "node:assert/strict";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { root, json, hash } from "./outcome-mlp-contract.mjs";
export const proposalPath =
  "research/experiments/coverage-preflight-proposal-v1.json";
/** Descriptor arithmetic only; no board, domain, labels, model or dataset materialization. */
export function coverageSchedule() {
  const splits = { control: [], diverse: [], dev: [], final: [] };
  for (let r = 0; r < 16; r++)
    for (const name of Object.keys(splits))
      splits[name].push({
        row: r,
        a:
          name === "control"
            ? 0
            : name === "diverse"
              ? 1 + (r % 3)
              : name === "dev"
                ? 5
                : 7,
        phase: name === "dev" ? 1 : name === "final" ? 2 : 0,
      });
  return splits;
}
export function validateCoverageProposal(p, base = root) {
  assert.equal(p.schemaVersion, 1);
  assert.equal(p.id, "coverage-preflight-proposal-v1");
  assert.equal(p.status, "PROPOSED_NOT_RUN");
  assert.equal(p.approval, "DESIGN_ONLY");
  assert.equal(p.executionRequiresSeparateUserDecision, true);
  assert.equal(p.evidence.path, "assets/research/outcome-mlp-20261103-v1.json");
  assert.equal(p.evidence.sha256, hash(resolve(base, p.evidence.path)));
  const prior = json(resolve(base, p.evidence.path));
  assert.equal(prior.candidate.transfer.correct, 144);
  assert.equal(prior.control.transfer.correct, 109);
  assert.equal(prior.actualCompute.topLevelForwards, 32672);
  assert.deepEqual(
    Object.keys(p.sourceHashes).sort(),
    [
      "src/research/point-simulation.ts",
      "src/research/action-outcome.ts",
      "src/research/imitation.ts",
      "src/research/spatial-footprint.ts",
      "src/domain/game/actions.ts",
      "research/cnn/outcome_mlp.py",
      "research/experiments/outcome-mlp-v1.json",
      p.evidence.path,
      "scripts/research/audit-coverage-inputs.mjs",
    ].sort(),
  );
  for (const [path, h] of Object.entries(p.sourceHashes))
    assert.equal(hash(resolve(base, path)), h, path);
  const d = p.data;
  assert.equal(d.rows, 16);
  assert.equal(d.cols, 10);
  assert.equal(d.candidatesEach, 46);
  assert.equal(d.totalExamples, 64);
  assert.equal(d.newTrainingExamples, 32);
  assert.equal(d.devExamples, 16);
  assert.equal(d.finalExamples, 16);
  assert.equal(d.targetRow, "r=0..15 once per split");
  assert.equal(d.targetColumn, "(3*r+phase) mod10");
  assert.deepEqual([d.trainingPhase, d.devPhase, d.finalPhase], [0, 1, 2]);
  assert.deepEqual(d.allowedOffsets, {
    two: [0, 1],
    three: [0, 1, 4],
    four: [0, 1, 4, 5],
  });
  assert.equal(
    d.nonTargetHoleStart,
    "(3*boardRow+phase) mod10; offset addition mod10 affects column selection only, never a board translation",
  );
  assert.equal(
    d.rowOrder,
    "The fifteen non-target row indices (r+k) mod16 for k=1..15; enumeration only, no row permutation",
  );
  assert.equal(
    d.allocation,
    "First a non-target rows have two holes, last a have four holes, the rest have three; target row has exactly one hole",
  );
  assert.deepEqual(d.splits, {
    control: { examples: 16, a: "0" },
    diverse: { examples: 16, a: "1+(r mod3)" },
    dev: { examples: 16, a: "5" },
    final: { examples: 16, a: "7" },
  });
  const schedule = coverageSchedule();
  for (const rows of Object.values(schedule)) {
    assert.equal(rows.length, 16);
    for (const { a } of rows)
      assert.equal(1 + 2 * a + 3 * (15 - 2 * a) + 4 * a, 46);
  }
  assert.deepEqual(
    schedule.control.map((e) => [e.row, e.phase]),
    schedule.diverse.map((e) => [e.row, e.phase]),
  );
  // No run of three empty cells in the upper mask, including any row-count subset.
  const mask = new Set(d.allowedOffsets.four),
    shift = (k) => new Set([...mask].map((x) => (x + k + 20) % 10));
  assert(
    [...mask].every((c) => !(mask.has((c + 1) % 10) && mask.has((c + 2) % 10))),
  );
  assert([...mask].every((c) => !(shift(3).has(c) && shift(6).has(c))));
  for (const [a, b] of [
    [-6, -3],
    [-3, 3],
    [3, 6],
  ])
    assert(!(shift(a).has(0) && shift(b).has(0)));
  const f = p.preflight;
  assert.equal(f.referenceRuns, 2 * 64);
  assert.equal(f.referenceNodeCap, 2 * 64 * 256);
  assert.equal(f.outcomeTransitionCap, 2 * 64 * 46);
  assert.equal(f.dfsCalls, 0);
  assert.equal(f.pythonModelLoadForwardTraining, 0);
  assert.equal(f.all64BoardGroupsUnique, true);
  assert.deepEqual(f.novelty, {
    dev: {
      examples: 16,
      candidateRows: 736,
      unseenTargets: 16,
      unseenRows: 736,
    },
    final: {
      examples: 16,
      candidateRows: 736,
      unseenTargets: 16,
      unseenRows: 736,
    },
    devFinalGroupAndVectorOverlap: 0,
    trainingLabelConflicts: 0,
  });
  assert.match(f.leakage, /UNION/);
  assert.match(f.float, /Math.fround/);
  assert.match(f.failure, /STOP/);
  assert.match(f.failure, /retry\/tuning\/fallback0/);
  assert.deepEqual(p.resources, {
    nodeWallSeconds: 120,
    sampledWorkingSetBytes: 2147483648,
    instantaneousPeakMeasured: false,
    newPackagesRuntimesServices: 0,
  });
  const t = p.futureTraining;
  assert.equal(t.status, "NOT_AUTHORIZED_NOT_IMPLEMENTED");
  assert.equal(t.runs, 2);
  assert.equal(t.initialOldFitAnchorEach, 16);
  assert.equal(t.updatesEach, 1000);
  assert.equal(t.fullBatchEach, 16);
  assert.equal(t.trainingCandidatesEach, 46);
  assert.equal(t.trainingObservationExposuresEach, 16000);
  assert.equal(t.trainingCandidateExposuresEach, 736000);
  assert.equal(t.finalEvaluationEach, 2 * 3 * 16);
  assert.equal(t.forwardsEach, 16 + 16000 + 96);
  assert.equal(t.totalTopLevelForwardMax, 2 * t.forwardsEach);
  assert.equal(t.checkpointLoads, 0);
  assert.equal(t.seed, 20261008);
  assert.equal(t.executionRequiresNewDecision, true);
  assert.deepEqual(p.boundaries, {
    currentPythonModelLoadForwardTraining: 0,
    currentDomainTeacherMaterialization: 0,
    generatorImplemented: false,
    trainingRunnerImplemented: false,
    executionConfigImplemented: false,
    originalSourceDataModelImmutable: true,
    appUsesDfs: true,
    actualScreenAcceptance: "BLOCKED_NO_GAME_SCREEN",
    publication:
      "Local README/proposal/tools only in this task; no prepare or push without new explicit authorization",
    limits:
      "Prospective synthetic diagnostic after previous post-hoc evidence, not natural game generalization. All labels may remain separable by immediate-clear feature52; no claim of future planning, old339/89 repair, action-kind accuracy, survival or app adoption.",
  });
  return {
    status: "PASS_STATIC_PROPOSAL",
    proposal: p.id,
    execution: "NOT_RUN",
    proposedNewBoards: 64,
    proposedTwoBuildReferenceCap: 32768,
    proposedTwoBuildOutcomeTransitions: 5888,
    nextStageModelForwardTraining: 0,
    futureUnapprovedForwards: 32224,
    actualPythonModelLoadForwardTrainingDomainDataset: 0,
  };
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  assert.equal(
    process.argv.length,
    2,
    "Static validator accepts no run/tuning options",
  );
  console.log(
    JSON.stringify(validateCoverageProposal(json(resolve(root, proposalPath)))),
  );
}
