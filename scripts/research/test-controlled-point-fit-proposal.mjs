import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";
import {
  subsetIndices,
  validateProposal,
  proposalPath,
} from "./validate-controlled-point-fit-proposal.mjs";

const root = new URL("../../", import.meta.url);
const proposal = JSON.parse(readFileSync(new URL(proposalPath, root), "utf8"));
test("frozen proposal is statically verifiable without imports or materialization", () => {
  const result = validateProposal(proposal);
  assert.equal(result.actualModelCreationLoadForwardTraining, 0);
  assert.equal(result.actualDatasetMaterialization, 0);
});
test("fixed subset covers rows and all columns without selecting measured predictions", () => {
  assert.deepEqual(
    subsetIndices(),
    [0, 13, 26, 39, 42, 55, 68, 71, 84, 97, 100, 113, 126, 139, 142, 155],
  );
});
for (const [name, change] of [
  ["execution authorization", (p) => (p.approval = "USER_APPROVED")],
  ["training status", (p) => (p.status = "COMPLETE")],
  [
    "different subset",
    (p) => (p.dataset.selection = "best sixteen predictions"),
  ],
  ["loss tuning", (p) => (p.fit.loss = "flat + kind auxiliary")],
  ["step extension", (p) => p.fit.steps++],
  ["extra forward", (p) => p.forwardBudget.totalTopLevelMax++],
  ["checkpoint selection", (p) => (p.fit.selection = "highest fit accuracy")],
  ["held-out access", (p) => (p.boundaries.reservedTestAccess = true)],
  [
    "source drift",
    (p) => (p.sourceHashes["research/cnn/spatial.py"] = "0".repeat(64)),
  ],
  ["source traversal", (p) => (p.sourceHashes["../secret"] = "0".repeat(64))],
  ["evidence drift", (p) => (p.evidence.sha256 = "0".repeat(64))],
  ["looser fit criterion", (p) => (p.acceptance.fit.correct = 15)],
  ["new parameters", (p) => p.model.parameters++],
  ["changed learning rate", (p) => (p.model.config.learningRate = 0.01)],
  ["missing old hyperparameter", (p) => delete p.model.config.learningRate],
  ["new runtime budget", (p) => p.resources.children++],
])
  test(`reject ${name}`, () => {
    const changed = structuredClone(proposal);
    change(changed);
    assert.throws(() => validateProposal(changed));
  });
test("CLI rejects execution arguments rather than offering a training entry", () => {
  const result = spawnSync(
    process.execPath,
    [
      fileURLToPath(
        new URL(
          "./validate-controlled-point-fit-proposal.mjs",
          import.meta.url,
        ),
      ),
      "--train",
    ],
    { cwd: root, encoding: "utf8", windowsHide: true },
  );
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /static validation only/);
});
