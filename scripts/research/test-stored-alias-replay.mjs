import assert from "node:assert/strict";
import test from "node:test";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { compileResearch } from "./runtime.mjs";
import { root, hash, json } from "./controlled-point-fit-contract.mjs";
import { auditRecordedAlias } from "./stored-alias-replay.mjs";
import { replayStoredAliasEvidence } from "./replay-stored-alias.mjs";
import {
  schedule,
  validateSelection,
  createLedger,
  auditReplayRows,
  parseStoredAliasArgs,
  configPath,
} from "./stored-alias-replay-contract.mjs";

const { aliasItemIdentity } = await import(
  compileResearch(root, "src/research/alias-item-identity.ts").url
);
test("item normalization order is not a semantic mutation", () => {
  const items = [
    { type: "reroll", col: 3, row: 8 },
    { row: 2, col: 1, type: "single-cell" },
  ];
  const original = JSON.stringify(items);
  const normalized = [
    { row: 2, col: 1, type: "single-cell" },
    { row: 8, col: 3, type: "reroll" },
  ];
  assert.notEqual(JSON.stringify(items), JSON.stringify(normalized));
  assert.equal(aliasItemIdentity(items), aliasItemIdentity(normalized));
  assert.equal(JSON.stringify(items), original);
});
test("item identity detects missing, duplicated, moved or retyped icons", () => {
  const items = [{ row: 1, col: 3, type: "reroll" }];
  for (const changed of [
    [],
    [...items, ...items],
    [{ row: 1, col: 4, type: "reroll" }],
    [{ row: 1, col: 3, type: "single-cell" }],
  ])
    assert.notEqual(aliasItemIdentity(items), aliasItemIdentity(changed));
});

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

function savedFixture() {
  const base = mkdtempSync(
    resolve(root, ".research-output/stored-alias-scalar-test-"),
  );
  const recorded = resolve(base, "recorded");
  mkdirSync(recorded);
  const put = (p, v) => {
    mkdirSync(dirname(p), { recursive: true });
    writeFileSync(p, JSON.stringify(v, null, 2) + "\n");
  };
  const s = selection(),
    r = records(s),
    ledger = schedule(s.cases).map((e, i) => ({ call: i + 1, ...e }));
  put(resolve(base, "selection.json"), s);
  const c = {
    id: "stored-alias-replay-v2",
    selection: "selection.json",
    hashes: { "selection.json": hash(resolve(base, "selection.json")) },
    resources: { seconds: 60, sampledRssBytes: 2147483648 },
  };
  put(resolve(base, configPath), c);
  put(resolve(recorded, "private-rows.json"), [r, r]);
  writeFileSync(
    resolve(recorded, "ledger.jsonl"),
    ledger.map((l) => JSON.stringify(l)).join("\n") + "\n",
  );
  const summary = {
    schemaVersion: 1,
    status: "PASS",
    ...auditReplayRows(s, [r, r], ledger),
    newTeacherReferenceModelLoadForwardTrainingDataset: 0,
    wholeTurnSalvageability: "NOT_EVALUATED",
    teacherLabelError: "NOT_ESTABLISHED",
  };
  put(resolve(recorded, "summary.json"), summary);
  const report = {
    status: "PASS",
    contractId: c.id,
    configSha256: hash(resolve(base, configPath)),
    elapsedSeconds: 1,
    maxSampledRss: 1024,
    instantaneousPeakMeasured: false,
    originalInputsAndSourcesUnchanged: true,
    hashes: Object.fromEntries(
      ["private-rows.json", "summary.json", "ledger.jsonl"].map((p) => [
        p,
        hash(resolve(recorded, p)),
      ]),
    ),
  };
  put(resolve(recorded, "report.json"), report);
  return { base, recorded, summary, report, put };
}
test("saved replay reconstructs scalar evidence and rejects inflated utility claims", () => {
  const f = savedFixture();
  assert.deepEqual(auditRecordedAlias(f.recorded, f.base), f.summary);
  const changed = { ...f.summary, wholeTurnSalvageability: "PASS" };
  f.put(resolve(f.recorded, "summary.json"), changed);
  f.report.hashes["summary.json"] = hash(resolve(f.recorded, "summary.json"));
  f.put(resolve(f.recorded, "report.json"), f.report);
  assert.throws(() => auditRecordedAlias(f.recorded, f.base));
});
test("saved replay checks config hash, exact evidence set and resource bounds", () => {
  const f = savedFixture();
  for (const change of [
    (r) => (r.configSha256 = "0".repeat(64)),
    (r) => (r.elapsedSeconds = 61),
    (r) => (r.hashes = { "summary.json": r.hashes["summary.json"] }),
    (r) => (r.originalInputsAndSourcesUnchanged = false),
  ]) {
    const r = structuredClone(f.report);
    change(r);
    f.put(resolve(f.recorded, "report.json"), r);
    assert.throws(() => auditRecordedAlias(f.recorded, f.base));
  }
  f.put(resolve(f.recorded, "report.json"), f.report);
  const s = json(resolve(f.base, "selection.json"));
  s.cases[0].label = 0;
  f.put(resolve(f.base, "selection.json"), s);
  assert.throws(() => auditRecordedAlias(f.recorded, f.base));
});
test("saved-replay CLI rejects execution and tuning options before I/O", () => {
  for (const a of [
    [],
    ["--output", "x"],
    ["--run", "x", "--output", "y"],
    ["--recorded", "x", "--output", "y", "--repeat", "2"],
  ])
    assert.throws(() => replayStoredAliasEvidence(a));
});
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
