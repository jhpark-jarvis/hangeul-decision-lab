import assert from "node:assert/strict";
import test from "node:test";
import { root, json } from "./lookahead-pairs-contract.mjs";
import { resolve } from "node:path";
import {
  boardProposalPath,
  validateBoardProposal,
  checkBoardOutcomes,
} from "./validate-board-context-proposal.mjs";
const proposal = () => json(resolve(root, boardProposalPath));
test("Board context contract remains static, explicit and unexecuted", () => {
  const s = validateBoardProposal(proposal());
  assert.equal(s.actualDomainReferenceDataPythonModelLoadForwardTraining, 0);
  assert.equal(s.proposedForwardMax, 64192);
  assert.equal(s.modelRunnerImplemented, false);
});
for (const [name, change] of [
  ["authorization", (p) => (p.approval = "USER_APPROVED")],
  ["single input factor", (p) => (p.contract.input.factor = "FOOTPRINT_TOO")],
  ["architecture", (p) => p.contract.model.parameters++],
  ["training extension", (p) => p.contract.fit.updatesEach++],
  ["extra inference", (p) => p.contract.budget.totalTopLevelForwardMax++],
  ["test split", (p) => (p.contract.dataset.split = "TEST")],
  [
    "native identity check",
    (p) => (p.contract.validation.nativePointwise = "optional"),
  ],
  [
    "source identity",
    (p) => (p.sourceHashes[Object.keys(p.sourceHashes)[0]] = "0".repeat(64)),
  ],
  ["evidence identity", (p) => (p.evidenceSha256 = "0".repeat(64))],
  [
    "parent archive identity",
    (p) => (p.archiveHashes["build1.json"] = "0".repeat(64)),
  ],
])
  test("Reject changed " + name, () => {
    const p = proposal();
    change(p);
    assert.throws(() => validateBoardProposal(p));
  });
const control = {
    examples: 32,
    correct: 16,
    meanNll: 1,
    minMargin: -1,
    technicalPass: true,
  },
  candidate = {
    examples: 32,
    correct: 32,
    meanNll: 0.01,
    minMargin: 1,
    technicalPass: true,
  };
test("Expected control quality failure does not block a valid fit-only contrast", () =>
  assert.equal(
    checkBoardOutcomes(control, candidate),
    "FIT_ONLY_CONTRAST_PASS",
  ));
test("Fit failure, technical stop and identity violation are distinguished", () => {
  assert.equal(
    checkBoardOutcomes({ ...control, technicalPass: false }, candidate),
    "TECHNICAL_FAIL",
  );
  assert.equal(
    checkBoardOutcomes({ ...control, correct: 17 }, candidate),
    "CONTROL_IDENTITY_VIOLATION",
  );
  for (const c of [
    { ...candidate, correct: 31 },
    { ...candidate, meanNll: 0.06 },
    { ...candidate, minMargin: 0 },
  ])
    assert.equal(checkBoardOutcomes(control, c), "FIT_FAIL");
  assert.throws(() =>
    checkBoardOutcomes(control, { ...candidate, meanNll: NaN }),
  );
});
