import assert from "node:assert/strict";
import test from "node:test";
import {
  approvedTransfer,
  parseTransferArgs,
  auditTransferLedger,
  auditEvaluationRows,
  phases,
} from "./frozen-point-transfer-contract.mjs";
test("approved frozen execution inherits the original design and one model/load, zero training", () => {
  const c = approvedTransfer();
  assert.equal(c.contract.forwardBudget.totalTopLevelMax, 320);
  assert.equal(c.boundaries.optimizerUpdates, 0);
  assert.equal(
    c.contract.checkpoint.source,
    "controlled-fit-duration-v1/candidate/final-1000",
  );
});
test("path-only run and stored replay reject tuning, duplicate and missing arguments", () => {
  assert.deepEqual(
    parseTransferArgs("run", [
      "--python",
      "p",
      "--original",
      "a",
      "--output",
      "b",
    ]),
    { python: "p", original: "a", output: "b" },
  );
  assert.deepEqual(
    parseTransferArgs("replay", ["--recorded", "a", "--output", "b"]),
    { recorded: "a", output: "b" },
  );
  for (const args of [
    [],
    ["--recorded", "a", "--recorded", "b"],
    ["--recorded", "a", "--output", "b", "--extra", "1"],
  ])
    assert.throws(() => parseTransferArgs("replay", args));
});
function ledger() {
  let offset = 0;
  return Object.entries(phases).flatMap(([phase, n]) => {
    const r = Array.from({ length: n }, (_, i) => ({
      call: offset + i + 1,
      phase,
      phaseCall: i + 1,
      step: null,
      example: i,
    }));
    offset += n;
    return r;
  });
}
test("320-call ledger requires fit control before any transfer and exact final repeats", () => {
  const rows = ledger();
  auditTransferLedger(rows);
  assert.equal(rows[16].call, 17);
  assert.equal(rows[160].phase, "fit-repeat");
  for (const mutate of [
    (r) => r.pop(),
    (r) => r.push(r[0]),
    (r) => (r[0].phase = "transfer"),
    (r) => (r[16].call = 1),
    (r) => (r[17].example = 0),
  ]) {
    const r = structuredClone(rows);
    mutate(r);
    assert.throws(() => auditTransferLedger(r));
  }
});
test("saved row audit enforces full logits and native first-index tie without new inference", () => {
  const examples = Array.from({ length: 144 }, (_, i) => ({
    id: `synthetic-${i}`,
    label: 1,
  }));
  const rows = examples.map((e) => ({
    id: e.id,
    label: 1,
    index: 0,
    correct: false,
    nll: Math.fround(Math.log(31)),
    rank: 2,
    margin: 0,
    logits: Array(31).fill(0),
  }));
  const s = auditEvaluationRows(rows, examples);
  assert.equal(s.examples, 144);
  assert.equal(s.correct, 0);
  assert.equal(s.maximumTargetRank, 2);
  for (const mutate of [
    (r) => (r[0].index = 1),
    (r) => (r[0].nll = 0),
    (r) => (r[0].rank = 1),
    (r) => (r[0].logits[0] = Infinity),
    (r) => (r[0].label = 0),
    (r) => r.pop(),
  ]) {
    const r = structuredClone(rows);
    mutate(r);
    assert.throws(() => auditEvaluationRows(r, examples));
  }
});
