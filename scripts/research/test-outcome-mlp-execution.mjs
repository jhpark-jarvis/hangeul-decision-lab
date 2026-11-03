import assert from "node:assert/strict";
import test from "node:test";
import {
  approvedMlp,
  phases,
  auditMlpLedger,
  mlpGate,
  pairedMlp,
  parseTransferArgs,
} from "./outcome-mlp-contract.mjs";
test("approved MLP execution preserves historical proposal and fixed costs", () => {
  const c = approvedMlp();
  assert.equal(c.contract.forwardBudget.totalTopLevelMax, 32672);
  assert.equal(c.contract.contrast.candidate.parameters, 769);
  assert.equal(c.boundaries.checkpointLoads, 0);
  assert.throws(() =>
    parseTransferArgs("run", [
      "--python",
      "p",
      "--original",
      "o",
      "--steps",
      "1000",
    ]),
  );
});
function ledger(arm) {
  const rows = [],
    offset = arm === "control" ? 0 : 16336;
  for (const [phase, n] of Object.entries(phases))
    for (let i = 0; i < n; i++)
      rows.push({
        call: offset + rows.length + 1,
        arm,
        phase,
        phaseCall: i + 1,
        step: phase === "train" ? Math.floor(i / 16) + 1 : null,
        example: phase === "train" ? i % 16 : i,
      });
  return rows;
}
test("two ledgers exactly cover 32672 calls and reject corrupt reservations", () => {
  for (const arm of ["control", "candidate"]) {
    const rows = ledger(arm);
    auditMlpLedger(rows, arm);
    for (const mutate of [
      (r) => r.pop(),
      (r) => r.push(r[0]),
      (r) => r[16].step++,
      (r) => r[16032].example++,
      (r) => (r[0].arm = "wrong"),
      (r) => r[0].call++,
    ]) {
      const bad = structuredClone(rows);
      mutate(bad);
      assert.throws(() => auditMlpLedger(bad, arm));
    }
  }
});
test("fit and transfer gates retain all three requirements and finite metrics", () => {
  for (const n of [16, 144]) {
    const s = {
      examples: n,
      correct: n,
      meanNll: 0.05,
      minimumTargetMargin: 1,
    };
    assert.equal(mlpGate(s).status, "PASS");
    for (const [k, v] of [
      ["correct", n - 1],
      ["meanNll", 0.05001],
      ["minimumTargetMargin", 0],
    ])
      assert.equal(mlpGate({ ...s, [k]: v }).status, "FAIL");
    assert.throws(() => mlpGate({ ...s, meanNll: NaN }));
  }
});
test("paired recovery and regression use aligned full denominators", () => {
  const a = [true, false, true, false].map((correct, i) => ({
      id: String(i),
      label: 1,
      index: correct ? 1 : 0,
      correct,
    })),
    b = [true, true, false, false].map((correct, i) => ({
      id: String(i),
      label: 1,
      index: correct ? 1 : 0,
      correct,
    }));
  assert.deepEqual(pairedMlp(a, b), {
    examples: 4,
    bothCorrect: 1,
    recovered: 1,
    regressed: 1,
    bothWrong: 1,
    changedChoices: 2,
  });
  assert.throws(() => pairedMlp(a, b.slice(1)));
  assert.throws(() => pairedMlp(a, b.toReversed()));
});
