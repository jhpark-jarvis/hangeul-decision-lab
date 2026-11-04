import assert from "node:assert/strict";
import test from "node:test";
import { spawnSync } from "node:child_process";
import { resolve } from "node:path";
import {
  root,
  proposalPath,
  readJson,
  lookaheadSchedule,
  validateLookaheadProposal,
  classifyLookaheadRecord,
  qualityInputBound,
} from "./validate-lookahead-proposal.mjs";
const p = () => readJson(resolve(root, proposalPath));
const record = (extra = {}) => ({
  rootCount: 4,
  complete: true,
  salvageableIndices: [2],
  qualityIndices: [2],
  maxImmediateIndices: [0, 2],
  baselineIndex: 0,
  ...extra,
});
const boundRow = (qualityIndices, hash = "a".repeat(64)) => ({
  rootCount: 4,
  complete: true,
  ordered22Sha256: hash,
  qualityIndices,
});

test("frozen design validates source/input bytes without materialization", () => {
  const r = validateLookaheadProposal(p());
  assert.equal(r.descriptors, 32);
  assert.equal(r.actualDomainReferenceDatasetModelLoadForwardTraining, 0);
  assert.equal(r.execution, "PROPOSED_NOT_RUN");
});
test("integer schedule preserves all16 families and both ordered arms", () => {
  const s = lookaheadSchedule();
  assert.equal(s.length, 32);
  assert.equal(new Set(s.map((r) => JSON.stringify(r))).size, 32);
  for (let i = 0; i < 16; i++) {
    const [a, b] = s.filter((d) => d.familyIndex === i);
    assert.equal(a.arm, "early-aligned");
    assert.equal(b.arm, "late-aligned");
    assert.deepEqual({ ...a, arm: null }, { ...b, arm: null });
    assert.equal(a.secondRow - a.firstRow, 7);
    assert(a.secondRow + 2 < 16 && a.column + 2 < 10);
  }
});
test("proposal rejects implicit adoption, extra fields and execution flags", () => {
  for (const change of [
    (v) => (v.approval = "USER_APPROVED"),
    (v) => (v.executionRequiresSeparateUserDecision = false),
    (v) => (v.status = "ACCEPTED"),
    (v) => (v.extra = true),
  ]) {
    const v = p();
    change(v);
    assert.throws(() => validateLookaheadProposal(v));
  }
});
test("proposal rejects changed family geometry or root candidate arithmetic", () => {
  for (const change of [
    (v) => (v.contract.generator.secondRowDelta = 6),
    (v) => (v.contract.generator.columns[0] = 8),
    (v) => (v.contract.generator.expectedPointOnlyCandidates = 31),
  ]) {
    const v = p();
    change(v);
    assert.throws(() => validateLookaheadProposal(v));
  }
});
test("proposal rejects increased reference, feature or retry compute", () => {
  for (const change of [
    (v) => (v.contract.reference.maxVisitedNodesPerState = 65),
    (v) => (v.contract.proposedCompute.totalApplyCallsMax = 6081),
    (v) => (v.contract.resources.retries = 1),
    (v) => (v.contract.proposedCompute.extraWitnessReplayCalls = 1),
  ]) {
    const v = p();
    change(v);
    assert.throws(() => validateLookaheadProposal(v));
  }
});
test("proposal rejects missing, altered and extra source/input hashes", () => {
  for (const key of ["sourceHashes", "inputHashes"]) {
    const changed = p();
    changed[key][Object.keys(changed[key])[0]] = "0".repeat(64);
    assert.throws(() => validateLookaheadProposal(changed));
    const missing = p();
    delete missing[key][Object.keys(missing[key])[0]];
    assert.throws(() => validateLookaheadProposal(missing));
    const extra = p();
    extra[key]["unapproved-path"] = "0".repeat(64);
    assert.throws(() => validateLookaheadProposal(extra));
  }
});
test("hard cohort requires a salvageable winner among tied immediate actions", () => {
  assert.equal(classifyLookaheadRecord(record()), "HARD");
  assert.equal(
    classifyLookaheadRecord(
      record({ salvageableIndices: [0], qualityIndices: [0] }),
    ),
    "ORDER_LUCKY_CONTROL",
  );
});
test("incomplete and all-dead-end examples cannot enter a primary cohort", () => {
  assert.equal(
    classifyLookaheadRecord({ rootCount: 4, complete: false }),
    "INCOMPLETE",
  );
  assert.equal(
    classifyLookaheadRecord(record({ salvageableIndices: [] })),
    "DEAD_END",
  );
});
test("easy and immediate-count traps have separate denominators", () => {
  assert.equal(
    classifyLookaheadRecord(
      record({ salvageableIndices: [0, 2], qualityIndices: [0, 2] }),
    ),
    "EASY",
  );
  assert.equal(
    classifyLookaheadRecord(
      record({ salvageableIndices: [3], qualityIndices: [3] }),
    ),
    "IMMEDIATE_COUNT_TRAP",
  );
});
test("classification refuses a different tie rule or impossible quality witness", () => {
  assert.throws(() => classifyLookaheadRecord(record({ baselineIndex: 2 })));
  assert.throws(() =>
    classifyLookaheadRecord(record({ salvageableIndices: [0] })),
  );
  assert.throws(() =>
    classifyLookaheadRecord(record({ qualityIndices: [2, 2] })),
  );
});
test("opposite quality targets on identical ordered input imply at most half correct", () => {
  const r = qualityInputBound([boundRow([0]), boundRow([2])]);
  assert.deepEqual(r, {
    examples: 2,
    orderedInputGroups: 1,
    idealQualityMembershipUpperBoundCorrect: 1,
  });
});
test("quality-set overlap can remove the hard-label bound", () => {
  assert.equal(
    qualityInputBound([boundRow([0, 2]), boundRow([1, 2])])
      .idealQualityMembershipUpperBoundCorrect,
    2,
  );
});
test("bound counts actual target support across larger alias classes", () => {
  const rows = [
    boundRow([0]),
    boundRow([0, 1]),
    boundRow([1]),
    boundRow([2]),
    boundRow([3], "b".repeat(64)),
  ];
  assert.equal(
    qualityInputBound(rows).idealQualityMembershipUpperBoundCorrect,
    3,
  );
});
test("bound refuses incomplete, malformed or inconsistent input groups", () => {
  for (const rows of [
    [],
    [boundRow([])],
    [{ ...boundRow([0]), complete: false }],
    [{ ...boundRow([0]), ordered22Sha256: "bad" }],
    [boundRow([0]), { ...boundRow([1]), rootCount: 3 }],
  ])
    assert.throws(() => qualityInputBound(rows));
});
test("static CLI rejects execution flags without invoking an owner", () => {
  const result = spawnSync(
    process.execPath,
    [
      resolve(root, "scripts/research/validate-lookahead-proposal.mjs"),
      "--execute",
    ],
    { encoding: "utf8", windowsHide: true },
  );
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /no execution/);
});
