import assert from "node:assert/strict";
import test from "node:test";
import {
  approvedCoverageTraining,
  parseCoverageTrainingArgs,
  auditCoverageTrainingLedger,
  auditCoveragePredictions,
  phases,
} from "./coverage-training-contract.mjs";
test("approved training freezes accepted data and equal exposures", () => {
  const c = approvedCoverageTraining();
  assert.equal(c.training.totalTopLevelForwardMax, 32224);
  assert.equal(c.training.trainingCandidateExposuresEach, 736000);
  assert.equal(c.boundaries.checkpointLoads, 0);
});
test("only fixed input/output paths, never data or schedule overrides", () => {
  assert.throws(() =>
    parseCoverageTrainingArgs("run", [
      "--data",
      "a",
      "--original",
      "b",
      "--output",
      "c",
      "--steps",
      "1000",
    ]),
  );
  assert.throws(() =>
    parseCoverageTrainingArgs("replay", ["--output", "a", "--output", "b"]),
  );
  assert.deepEqual(
    parseCoverageTrainingArgs("replay", ["--recorded", "a", "--output", "b"]),
    { recorded: "a", output: "b" },
  );
});
function ledger(arm) {
  const rows = [];
  for (const [phase, n] of Object.entries(phases))
    for (let i = 0; i < n; i++)
      rows.push({
        call: (arm === "control" ? 0 : 16112) + rows.length + 1,
        arm,
        phase,
        phaseCall: i + 1,
        step: phase === "train" ? Math.floor(i / 16) + 1 : null,
        example: i % 16,
      });
  return rows;
}
test("control and diverse ledgers have one32224 sequence; corrupt reservations reject", () => {
  for (const arm of ["control", "diverse"]) {
    const rows = ledger(arm);
    auditCoverageTrainingLedger(rows, arm);
    for (const change of [
      (r) => r.pop(),
      (r) => r.push(r[0]),
      (r) => r[16].step++,
      (r) => (r[0].arm = "wrong"),
      (r) => (r[16016].phase = "dev1"),
    ]) {
      const bad = structuredClone(rows);
      change(bad);
      assert.throws(() => auditCoverageTrainingLedger(bad, arm));
    }
  }
});
test("stored46-candidate ranks/NLL/ties are independent and reject invalid rows", () => {
  const examples = Array.from({ length: 16 }, (_, i) => ({
      id: String(i),
      label: 1,
    })),
    rows = examples.map((e) => ({
      id: e.id,
      label: 1,
      index: 0,
      correct: false,
      rank: 2,
      margin: 0,
      nll: Math.log(46),
      logits: Array(46).fill(0),
    }));
  const s = auditCoveragePredictions(rows, examples);
  assert.equal(s.correct, 0);
  assert.equal(s.maximumTargetRank, 2);
  for (const mutate of [
    (r) => (r[0].index = 1),
    (r) => (r[0].rank = 1),
    (r) => (r[0].nll = 0),
    (r) => r[0].logits.pop(),
    (r) => (r[0].id = "bad"),
    (r) => (r[0].logits[0] = NaN),
  ]) {
    const bad = structuredClone(rows);
    mutate(bad);
    assert.throws(() => auditCoveragePredictions(bad, examples));
  }
});
