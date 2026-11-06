import assert from "node:assert/strict";
import test from "node:test";
import {
  approvedPairedTarget,
  phaseIdentity,
} from "./paired-target-contract.mjs";
import { pairContract } from "./validate-paired-target-proposal.mjs";
test("Approved new execution exactly inherits revised contract and separate budget", () => {
  const c = approvedPairedTarget();
  assert.deepEqual(c.contract, pairContract.inheritedContrast);
  assert.equal(c.boundaries.newForwardMax, 64192);
  assert.equal(c.boundaries.cumulativeForwardMax, 64224);
  assert.equal(c.boundaries.retryTuning, 0);
});
test("Phase diagnostic records non-target drift with exact opposing anchors", () => {
  const rows = [
      { label: 0, logits: [2, 1, 0] },
      { label: 1, logits: [2, 1, 2 ** -149] },
    ],
    keys = [
      ["a", "b", "x"],
      ["a", "b", "x"],
    ];
  const r = phaseIdentity(rows, keys);
  assert.equal(r.anchors.nativeCorrectUpperBound, 1);
  assert.equal(r.globalDrift.maximumUlpGap, 1);
  assert.equal(r.mismatchTargetOccurrences, 0);
});
test("Cross-family target participation in global drift is counted but pair anchors remain required", () => {
  const rows = [
      { label: 0, logits: [2, 1] },
      { label: 1, logits: [2, 1] },
      { label: 0, logits: [2 + 2 ** -22, 1] },
      { label: 1, logits: [2 + 2 ** -22, 1] },
    ],
    keys = rows.map(() => ["a", "b"]);
  const r = phaseIdentity(rows, keys);
  assert.equal(r.mismatchTargetOccurrences, 2);
  assert.equal(r.anchors.pairs, 2);
  rows[3].logits[0] = 2;
  assert.throws(() => phaseIdentity(rows, keys));
});
test("Exact target diagnostics retain signed-zero identity and original score values", () => {
  const rows = [
      { label: 0, logits: [0, -1] },
      { label: 1, logits: [0, -1] },
    ],
    keys = [
      ["a", "b"],
      ["a", "b"],
    ],
    before = structuredClone(rows);
  phaseIdentity(rows, keys);
  assert.deepEqual(rows, before);
  rows[1].logits[0] = -0;
  assert.throws(() => phaseIdentity(rows, keys));
});
