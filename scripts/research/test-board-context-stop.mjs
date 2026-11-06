import assert from "node:assert/strict";
import test from "node:test";
import {
  ulpDistance,
  pairedAnchorProof,
  inputScoreDrift,
  runBoardStop,
} from "./board-context-stop.mjs";
test("ULP distance covers signed zeros and adjacent finite floats", () => {
  assert.equal(ulpDistance(1, 1 + 2 ** -23), 1);
  assert.equal(ulpDistance(-1, -1 - 2 ** -23), 1);
  assert.equal(ulpDistance(0, -0), 1);
  assert.throws(() => ulpDistance(NaN, 0));
});
const pair = () => ({
  rows: [
    { label: 0, logits: [2, 1, 0] },
    { label: 1, logits: [2, 1, 0] },
  ],
  keys: [
    ["a", "b", "x"],
    ["a", "b", "x"],
  ],
});
test("Exact opposing anchors and tie order prove one correct at most", () => {
  const p = pair();
  assert.equal(pairedAnchorProof(p.rows, p.keys).nativeCorrectUpperBound, 1);
});
test("Irrelevant non-target ULP drift does not change anchor proof", () => {
  const p = pair();
  p.rows[1].logits[2] = 2 ** -149;
  assert.equal(inputScoreDrift(p.rows, p.keys).summary.mismatchGroups, 1);
  assert.equal(pairedAnchorProof(p.rows, p.keys).exactTargetAnchorScores, true);
});
test("Anchor proof rejects even one ULP target drift without tolerance", () => {
  const p = pair();
  p.rows[1].logits[0] = 2 + 2 ** -22;
  assert.throws(() => pairedAnchorProof(p.rows, p.keys));
});
test("Missing anchor, duplicate key, equal target and changed tie priority reject", () => {
  for (const change of [
    (p) => (p.keys[1][0] = "z"),
    (p) => (p.keys[0][2] = "a"),
    (p) => (p.rows[1].label = 0),
    (p) => {
      p.keys[1] = ["b", "a", "x"];
      p.rows[1] = { label: 0, logits: [1, 2, 0] };
    },
  ]) {
    const p = pair();
    change(p);
    assert.throws(() => pairedAnchorProof(p.rows, p.keys));
  }
});
test("Drift report separates same-index and shifted-index associations", () => {
  const r = [{ logits: [1, 2] }, { logits: [2 + 2 ** -22, 1] }],
    k = [
      ["a", "b"],
      ["b", "a"],
    ];
  const d = inputScoreDrift(r, k).summary;
  assert.equal(d.mismatchGroups, 1);
  assert.equal(d.maximumUlpGap, 1);
  assert.equal(d.differentIndexMismatchGroups, 1);
  assert.equal(d.sameIndexMismatchGroups, 0);
});
test("Public drift aggregate contains no coordinate, values, hash or case rows", () => {
  const p = pair();
  const s = inputScoreDrift(p.rows, p.keys).summary;
  assert.deepEqual(Object.keys(s), [
    "candidateRows",
    "uniqueInputs",
    "repeatedInputGroups",
    "mismatchGroups",
    "maximumAbsoluteGap",
    "maximumUlpGap",
    "sameIndexMismatchGroups",
    "differentIndexMismatchGroups",
  ]);
});
test("Stored-only CLI rejects any execution, extra or missing arguments", () => {
  for (const args of [
    [],
    ["--train"],
    ["--recorded", "a", "--output", "b", "--retry", "1"],
  ])
    assert.throws(() => runBoardStop(args));
});
