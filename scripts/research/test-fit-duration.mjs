import assert from "node:assert/strict";
import test from "node:test";
import {
  approvedDuration,
  parseDurationArgs,
  auditDurationLedger,
} from "./fit-duration-contract.mjs";
test("approved execution preserves the proposal, original data and fixed two-arm budget", () => {
  const c = approvedDuration();
  assert.equal(c.contract.forwardBudget.totalTopLevelMax, 19296);
  assert.equal(c.contract.contrast.runs, 2);
  assert.equal(c.boundaries.checkpointLoads, 0);
});
test("only explicit run/replay paths are accepted, never tuning or duplicate overrides", () => {
  assert.deepEqual(
    parseDurationArgs("run", [
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
    parseDurationArgs("replay", ["--recorded", "a", "--output", "b"]),
    { recorded: "a", output: "b" },
  );
  for (const args of [
    [],
    ["--recorded", "a", "--output", "b", "--forward"],
    ["--recorded", "a", "--recorded", "b"],
    ["--train", "1", "--output", "b"],
  ])
    assert.throws(() => parseDurationArgs("replay", args));
  assert.throws(() =>
    parseDurationArgs("run", [
      "--python",
      "p",
      "--original",
      "a",
      "--steps",
      "1000",
    ]),
  );
});
function ledger(arm) {
  const rows = [],
    updates = arm === "control" ? 200 : 1000,
    offset = arm === "control" ? 0 : 3248;
  for (const [phase, n] of [
    ["initial", 16],
    ["train", updates * 16],
    ["final1", 16],
    ["final2", 16],
  ])
    for (let i = 0; i < n; i++)
      rows.push({
        call: offset + rows.length + 1,
        arm,
        phase,
        phaseCall: i + 1,
        step: phase === "train" ? Math.floor(i / 16) + 1 : null,
        example: i % 16,
      });
  return rows;
}
test("both local ledgers form one fixed 19296-call global sequence", () => {
  const a = ledger("control"),
    b = ledger("candidate");
  auditDurationLedger(a, "control");
  auditDurationLedger(b, "candidate");
  assert.equal(a.length + b.length, 19296);
  assert.equal(b[0].call, 3249);
  assert.equal(b.at(-1).call, 19296);
});
test("partial, extra, reordered or wrong-arm reservations are rejected", () => {
  const rows = ledger("candidate");
  for (const mutate of [
    (r) => r.pop(),
    (r) => r.push(r[0]),
    (r) => (r[0].call = 1),
    (r) => (r[16].step = 2),
    (r) => (r[0].arm = "control"),
    (r) => (r[17].example = 0),
  ]) {
    const c = structuredClone(rows);
    mutate(c);
    assert.throws(() => auditDurationLedger(c, "candidate"));
  }
});
