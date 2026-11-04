import assert from "node:assert/strict";
import test from "node:test";
import {
  schedule,
  validateSelection,
  createLedger,
  auditReplayRows,
  parseStoredAliasArgs,
} from "./stored-alias-replay-contract.mjs";

// Scalar verification records only; no synthetic GameState, transition or NN.
function selection() {
  return {
    schemaVersion: 1,
    cases: [4, 5].map((n, i) => ({
      split: "train",
      sourceIndex: i,
      fixtureId: String(i),
      stateSha256: "a".repeat(64),
      members22: Array.from({ length: n }, (_, j) => j),
      label: 1,
      storedTeacherTerminalSurvivable: false,
      storedTeacherAllPiecesPlaced: false,
      teacherSearchComplete: true,
      eligibleLiveAlias: true,
      ordinaryPlacements: 1,
      storedMaskSurvivors: i === 0 ? [0, 0, 1, 1] : [1, 0, 0, 1, 1],
    })),
  };
}
function records(s) {
  return s.cases.map((c) =>
    c.members22.map((member, i) => ({
      member,
      target: member === c.label,
      earliest: i === 0,
      nextPhase: c.storedMaskSurvivors[i] ? "playing" : "gameover",
      ordinaryPlacements: c.storedMaskSurvivors[i],
      outcomeExact: true,
      inputUnchanged: true,
    })),
  );
}
test("exact replay schedule reserves 18 direct and 18 outcome calls", () => {
  const s = selection(),
    calls = schedule(s.cases);
  assert.equal(calls.length, 36);
  assert.equal(calls.filter((c) => c.kind === "direct").length, 18);
  assert.deepEqual(calls[0], {
    pass: 0,
    caseIndex: 0,
    member: 0,
    kind: "direct",
  });
  assert.equal(calls[35].pass, 1);
});
test("ledger persists before granting a transition and rejects exhausted cap", () => {
  const s = selection(),
    persisted = [],
    l = createLedger(s, (r) => persisted.push(r));
  for (const call of schedule(s.cases)) {
    l.reserve(call);
    assert.deepEqual(l.rows, persisted);
  }
  assert.equal(l.finish().length, 36);
  assert.throws(() => l.reserve(schedule(s.cases)[0]), /cap/);
});
test("failed durable reservation does not grant a call", () => {
  const s = selection(),
    l = createLedger(s, () => {
      throw new Error("disk failure");
    });
  assert.throws(() => l.reserve(schedule(s.cases)[0]), /disk failure/);
  assert.equal(l.rows.length, 0);
});
test("out of order, incomplete, and over-budget schedules fail closed", () => {
  const s = selection(),
    l = createLedger(s, () => {});
  assert.throws(() => l.reserve(schedule(s.cases)[1]), /order/);
  assert.throws(() => l.finish());
  assert.equal(l.rows.length, 0);
});
test("selection rejects changed split, duplicate fixture and altered class", () => {
  for (const change of [
    (s) => (s.cases[0].split = "test"),
    (s) => (s.cases[1].fixtureId = "0"),
    (s) => (s.cases[0].members22[1] = 0),
    (s) => s.cases[0].storedMaskSurvivors.fill(0),
  ]) {
    const s = selection();
    change(s);
    assert.throws(() => validateSelection(s));
  }
});
test("saved-only replay checks phases, repetitions, exact features and ledger", () => {
  const s = selection(),
    r = records(s),
    ledger = schedule(s.cases).map((e, i) => ({ call: i + 1, ...e }));
  const a = auditReplayRows(s, [r, structuredClone(r)], ledger);
  assert.equal(a.nextGameoverPerPass, 4);
  assert.equal(a.nextPlayingPerPass, 5);
  assert.equal(a.mixedPhaseStates, 2);
  const altered = structuredClone(r);
  altered[0][0].outcomeExact = false;
  assert.throws(() => auditReplayRows(s, [r, altered], ledger));
  assert.throws(() => auditReplayRows(s, [r, r], ledger.slice(1)));
  r[0][0].nextPhase = "playing";
  assert.throws(() => auditReplayRows(s, [r, r], ledger));
});
test("sampled resource failure is checked before a reservation", () => {
  const s = selection(),
    l = createLedger(
      s,
      () => assert.fail("must not persist"),
      () => {
        throw new Error("resource cap");
      },
    );
  assert.throws(() => l.reserve(schedule(s.cases)[0]), /resource cap/);
  assert.equal(l.rows.length, 0);
});
test("CLI only permits one new ignored output, no tuning or test args", () => {
  assert.deepEqual(
    parseStoredAliasArgs(["--output", ".research-output/check"]),
    { output: ".research-output/check" },
  );
  for (const a of [
    [],
    ["--source", "x"],
    ["--output", "--train"],
    ["--output", "x", "--repeat", "3"],
  ])
    assert.throws(() => parseStoredAliasArgs(a));
});
