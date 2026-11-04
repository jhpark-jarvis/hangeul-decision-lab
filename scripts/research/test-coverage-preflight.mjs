import assert from "node:assert/strict";
import test from "node:test";
import {
  approvedCoverage,
  auditCoverageLedger,
} from "./coverage-preflight-contract.mjs";
function ledger() {
  const rows = [];
  for (let build = 1; build <= 2; build++)
    for (const split of ["control", "diverse", "dev", "final"])
      for (let row = 0; row < 16; row++)
        for (let i = -1; i < 46; i++)
          rows.push({
            call: rows.length + 1,
            build,
            kind: i < 0 ? "reference" : "outcome",
            split,
            row,
            index: i < 0 ? 0 : i,
            reservedNodes: i < 0 ? 256 : 0,
          });
  return rows;
}
test("approved preflight retains original design and zero NN budget", () => {
  const c = approvedCoverage();
  assert.equal(c.contract.preflight.pythonModelLoadForwardTraining, 0);
  assert.equal(c.contract.preflight.referenceNodeCap, 32768);
});
test("two-build ledger reserves every reference cap and every outcome before calls", () => {
  const rows = ledger();
  auditCoverageLedger(rows);
  assert.equal(rows.filter((r) => r.kind === "reference").length, 128);
  assert.equal(rows.filter((r) => r.kind === "outcome").length, 5888);
});
test("incomplete, over-budget or wrong-order preflight ledger is rejected", () => {
  for (const mutate of [
    (r) => r.pop(),
    (r) => r.push(r[0]),
    (r) => (r[0].reservedNodes = 255),
    (r) => r[1].index++,
    (r) => (r[0].split = "dev"),
    (r) => (r[3008].build = 1),
  ]) {
    const bad = ledger();
    mutate(bad);
    assert.throws(() => auditCoverageLedger(bad));
  }
});
