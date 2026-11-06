import assert from "node:assert/strict";
import test from "node:test";
import {
  approvedBoardContext,
  parseBoardArgs,
  auditBoardLedger,
  auditInputIdentity,
  fp32InputKey,
  auditBoardRows,
  boardFitGate,
  compareBoardArms,
  phases,
} from "./board-context-contract.mjs";
test("Approved execution keeps immutable proposal and single-factor budget", () => {
  const c = approvedBoardContext();
  assert.equal(c.contract.budget.totalTopLevelForwardMax, 64192);
  assert.equal(c.contract.input.factor, "OCCUPIED_PLANE_ONLY");
  assert.equal(c.boundaries.domainReferenceMaterialization, 0);
});
test("CLI accepts fixed path options and rejects tuning, duplicates and missing paths", () => {
  assert.deepEqual(
    parseBoardArgs("run", ["--source", "a", "--output", "b", "--python", "c"]),
    { source: "a", output: "b", python: "c" },
  );
  for (const args of [
    ["--recorded", "a", "--output", "b", "--epochs", "5"],
    ["--output", "a", "--output", "b"],
    ["--recorded", "--output", "--output", "a"],
  ])
    assert.throws(() => parseBoardArgs("replay", args));
});
const ledger = (arm) => {
  const rows = [];
  for (const [phase, n] of Object.entries(phases))
    for (let i = 0; i < n; i++)
      rows.push({
        call: (arm === "control" ? 0 : 32096) + rows.length + 1,
        arm,
        phase,
        phaseCall: i + 1,
        step: phase === "train" ? Math.floor(i / 32) + 1 : null,
        example: i % 32,
      });
  return rows;
};
test("All64192 pre-call positions are exact without performing forwards", () => {
  for (const arm of ["control", "candidate"])
    assert.equal(auditBoardLedger(ledger(arm), arm), 32096);
});
test("Ledger rejects count, arm offset, step and order drift", () => {
  for (const change of [
    (r) => r.pop(),
    (r) => r[0].call++,
    (r) => (r[32].step = 2),
    (r) => (r[1].example = 0),
  ]) {
    const rows = ledger("control");
    change(rows);
    assert.throws(() => auditBoardLedger(rows, "control"));
  }
});
test("FP32 byte identity is little endian and signed zeros remain distinct", () => {
  assert.equal(fp32InputKey([Math.fround(1 / 3)]), fp32InputKey([1 / 3]));
  assert.notEqual(fp32InputKey([0]), fp32InputKey([-0]));
  assert.throws(() => fp32InputKey([NaN]));
});
test("Input equality requires bitwise identical saved logits", () => {
  const rows = [{ logits: [1, 2] }, { logits: [2, 1] }],
    keys = [
      ["a", "b"],
      ["b", "a"],
    ];
  assert.equal(auditInputIdentity(rows, keys).uniqueInputs, 2);
  rows[1].logits[0] = 3;
  assert.throws(() => auditInputIdentity(rows, keys));
});
const synthetic = () => {
  const examples = Array.from({ length: 32 }, () => ({ label: 0 }));
  const rows = examples.map((_, i) => ({
    example: i,
    label: 0,
    index: 0,
    correct: true,
    nll: Math.fround(Math.log(32)),
    rank: 1,
    margin: 0,
    logits: Array(32).fill(0),
  }));
  return { examples, rows };
};
test("Stored full-list metrics preserve first-index ties and cohort denominators", () => {
  const { examples, rows } = synthetic();
  const m = auditBoardRows(rows, examples);
  assert.equal(m.correct, 32);
  assert.equal(m.cohorts.hard.examples, 16);
  assert.equal(m.minMargin, 0);
  assert.equal(boardFitGate(m).status, "FAIL");
});
test("Stored metrics reject wrong labels, non-FP32 logits, NLL and rank drift", () => {
  for (const change of [
    (r) => (r[0].label = 1),
    (r) => (r[0].logits[0] = 1 / 3),
    (r) => r[0].nll++,
    (r) => r[0].rank++,
  ]) {
    const { examples, rows } = synthetic();
    change(rows);
    assert.throws(() => auditBoardRows(rows, examples));
  }
});
test("Recovery and regression use paired32 without pooling repeats", () => {
  const { rows } = synthetic(),
    other = structuredClone(rows);
  rows[0].correct = false;
  other[1].correct = false;
  assert.deepEqual(compareBoardArms(rows, other), {
    examples: 32,
    recovered: 1,
    regressed: 1,
    bothCorrect: 30,
    bothWrong: 0,
    netCorrect: 0,
  });
  other[0].label = 1;
  assert.throws(() => compareBoardArms(rows, other));
});
