import assert from "node:assert/strict";
import test from "node:test";
import { resolve } from "node:path";
import { root, json } from "./board-context-contract.mjs";
import {
  pairProposalPath,
  validatePairProposal,
  checkedPairProof,
} from "./validate-paired-target-proposal.mjs";
const proposal = () => json(resolve(root, pairProposalPath));
test("Revised design is an unexecuted new budget with immutable original failure", () => {
  const s = validatePairProposal(proposal());
  assert.equal(s.proposedNewForwards, 64192);
  assert.equal(
    s.actualAdditionalDomainReferenceDataPythonModelLoadForwardTraining,
    0,
  );
});
for (const [name, change] of [
  ["authorization", (p) => (p.approval = "USER_APPROVED")],
  ["tolerance", (p) => (p.contract.revision.tolerance = 1e-6)],
  ["extra updates", (p) => p.contract.inheritedContrast.fit.updatesEach++],
  [
    "thread/model config",
    (p) => (p.contract.inheritedContrast.model.config.threads = 1),
  ],
  [
    "budget reuse",
    (p) => (p.contract.execution.maximumAcrossOldAndNewIfApproved = 64192),
  ],
  ["anchor requirement", (p) => p.contract.revision.requiredAt.pop()],
  [
    "old contract waiver",
    (p) => (p.contract.claims.oldFailedContractNotWaived = false),
  ],
  [
    "source identity",
    (p) => (p.sourceHashes[Object.keys(p.sourceHashes)[0]] = "0".repeat(64)),
  ],
  ["evidence identity", (p) => (p.evidenceSha256 = "0".repeat(64))],
])
  test("Reject changed " + name, () => {
    const p = proposal();
    change(p);
    assert.throws(() => validatePairProposal(p));
  });
test("Exact relevant anchors allow non-target drift without modifying any score", () => {
  const rows = [
      { label: 0, logits: [2, 1, 0] },
      { label: 1, logits: [2, 1, 2 ** -149] },
    ],
    keys = [
      ["a", "b", "c"],
      ["a", "b", "c"],
    ],
    before = JSON.stringify(rows);
  assert.equal(checkedPairProof(rows, keys).nativeCorrectUpperBound, 1);
  assert.equal(JSON.stringify(rows), before);
  rows[1].logits[0] = 2 + 2 ** -22;
  assert.throws(() => checkedPairProof(rows, keys));
});
test("Nonfinite or non-FP32 anchors are never admitted", () => {
  for (const value of [NaN, Infinity, 1 / 3])
    assert.throws(() =>
      checkedPairProof(
        [
          { label: 0, logits: [value, 0] },
          { label: 1, logits: [value, 0] },
        ],
        [
          ["a", "b"],
          ["a", "b"],
        ],
      ),
    );
});
