import assert from "node:assert/strict";
import test from "node:test";
import { spawnSync } from "node:child_process";
import { json, root } from "./outcome-mlp-contract.mjs";
import { resolve } from "node:path";
import {
  boardGroup,
  auditCoverageSplits,
  coveragePreflightGate,
} from "./audit-coverage-inputs.mjs";
import {
  coverageSchedule,
  validateCoverageProposal,
  proposalPath,
} from "./validate-coverage-proposal.mjs";
const vector = (x) => Array(22).fill(x),
  group = (i) => i.toString(16).padStart(64, "0"),
  row = (i, values, label = 0) => ({
    boardGroup: group(i),
    candidates: values.map(vector),
    label,
  });
test("board group conservatively includes both reflections and rejects malformed stored planes", () => {
  const b = Array.from({ length: 16 }, (_, r) =>
    Array.from({ length: 10 }, (_, c) => Number((r * 7 + c * 3) % 11 < 4)),
  );
  assert.equal(boardGroup(b), boardGroup(b.map((r) => r.toReversed())));
  assert.equal(boardGroup(b), boardGroup(b.toReversed()));
  assert.throws(() => boardGroup(b.slice(1)));
  const bad = structuredClone(b);
  bad[0][0] = 0.5;
  assert.throws(() => boardGroup(bad));
});
test("FP32 collisions are counted as known inputs even on distinct boards", () => {
  const a = auditCoverageSplits([row(1, [0.5, 0.25])], {
    dev: [row(2, [0.25, 0.5 + 1e-9], 1)],
  });
  assert.equal(a.evaluations.dev.boardGroupOverlap, 0);
  assert.equal(a.evaluations.dev.unseenCandidateRows, 0);
  assert.equal(a.evaluations.dev.labelPreservingBagMatches, 1);
});
test("multiset multiplicity and label tags cannot be replaced by set membership", () => {
  const training = [row(1, [0.1, 0.2, 0.2])];
  const a = auditCoverageSplits(training, {
    dev: [row(2, [0.1, 0.1, 0.2]), row(3, [0.1, 0.2, 0.2], 1)],
  });
  assert.equal(a.evaluations.dev.labelPreservingBagMatches, 0);
  assert.equal(a.evaluations.dev.unseenCandidateRows, 0);
  const conflicts = auditCoverageSplits(
    [row(1, [0.1, 0.2]), row(2, [0.1, 0.2], 1)],
    { dev: [row(3, [0.3, 0.4])] },
  );
  assert.equal(conflicts.training.conflictingFp32Labels, 2);
});
test("final novelty is measured against both training arms and dev", () => {
  const a = auditCoverageSplits([row(1, [0.1, 0.2]), row(2, [0.3, 0.4])], {
    dev: [row(3, [0.5, 0.6])],
    final: [row(3, [0.3, 0.6])],
  });
  assert.equal(a.evaluations.final.unseenTargetRows, 0);
  assert.equal(a.evaluationPairs[0].boardGroupOverlap, 1);
  assert.equal(a.evaluationPairs[0].candidateRowsSeenInOther, 1);
  assert.equal(coveragePreflightGate(a).status, "FAIL");
});
function scalarAudit() {
  const rows = (start, n) =>
    Array.from({ length: n }, (_, i) =>
      row(
        start + i,
        Array.from(
          { length: 46 },
          (_, j) => (start * 46 + i * 46 + j) / 100000,
        ),
      ),
    );
  return auditCoverageSplits(rows(1, 32), {
    dev: rows(100, 16),
    final: rows(200, 16),
  });
}
test("all fixed gate conditions are mandatory, including denominator and singleton evaluation names", () => {
  const a = scalarAudit();
  assert.equal(coveragePreflightGate(a).status, "PASS");
  for (const change of [
    (x) => x.training.examples--,
    (x) => x.training.candidateRows--,
    (x) => x.training.uniqueBoardGroups--,
    (x) => x.training.conflictingFp32Labels++,
    (x) => x.evaluations.dev.examples--,
    (x) => x.evaluations.dev.uniqueBoardGroups--,
    (x) => x.evaluations.final.unseenTargetRows--,
    (x) => x.evaluations.final.unseenCandidateRows--,
    (x) => x.evaluations.dev.boardGroupOverlap++,
    (x) => x.evaluations.dev.labelPreservingBagMatches++,
    (x) => x.evaluationPairs[0].candidateRowsSeenInOther++,
    (x) => delete x.evaluations.final,
  ]) {
    const bad = structuredClone(a);
    change(bad);
    assert.equal(coveragePreflightGate(bad).status, "FAIL");
  }
});
test("malformed features, labels and nonfinite values fail before novelty accounting", () => {
  for (const change of [
    (x) => (x.label = -1),
    (x) => x.candidates[0].pop(),
    (x) => (x.candidates[0][0] = NaN),
    (x) => (x.candidates[0][0] = 1.01),
    (x) => (x.boardGroup = "unchecked-id"),
  ]) {
    const bad = row(1, [0.1, 0.2]);
    change(bad);
    assert.throws(() =>
      auditCoverageSplits([bad], { dev: [row(2, [0.3, 0.4])] }),
    );
  }
});
test("frozen coverage proposal preserves sources, matched exposures and separate execution decisions", () => {
  const p = json(resolve(root, proposalPath)),
    result = validateCoverageProposal(p);
  assert.equal(result.proposedNewBoards, 64);
  assert.equal(result.nextStageModelForwardTraining, 0);
  const s = coverageSchedule();
  assert.equal(s.control.length, 16);
  assert.equal(
    s.diverse.reduce((n, e) => n + Number(e.a > 0), 0),
    16,
  );
  for (const change of [
    (x) => (x.approval = "USER_APPROVED"),
    (x) => (x.data.candidatesEach = 31),
    (x) => (x.data.splits.final.a = "3"),
    (x) => x.preflight.outcomeTransitionCap++,
    (x) => x.preflight.novelty.final.unseenRows--,
    (x) => x.futureTraining.totalTopLevelForwardMax++,
    (x) => (x.boundaries.generatorImplemented = true),
    (x) => (x.sourceHashes["research/cnn/outcome_mlp.py"] = "0".repeat(64)),
  ]) {
    const bad = structuredClone(p);
    change(bad);
    assert.throws(() => validateCoverageProposal(bad));
  }
});
test("static CLI rejects execution and tuning arguments", () => {
  const r = spawnSync(
    process.execPath,
    ["scripts/research/validate-coverage-proposal.mjs", "--run"],
    { cwd: root, encoding: "utf8", windowsHide: true },
  );
  assert.notEqual(r.status, 0);
  assert.match(r.stderr, /Static validator accepts no run/);
});
