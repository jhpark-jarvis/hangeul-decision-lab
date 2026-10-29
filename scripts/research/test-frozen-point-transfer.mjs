import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import {
  transferPartition,
  transferScreen,
  validateTransferProposal,
} from "./validate-frozen-point-transfer.mjs";

const proposal = () =>
  JSON.parse(
    readFileSync(
      new URL(
        "../../research/experiments/frozen-point-transfer-proposal-v1.json",
        import.meta.url,
      ),
      "utf8",
    ),
  );
test("static CLI rejects execution and data arguments with a failing exit and no result", () => {
  const cli = fileURLToPath(
    new URL("./validate-frozen-point-transfer.mjs", import.meta.url),
  );
  const result = spawnSync(process.execPath, [cli, "--run"], {
    encoding: "utf8",
    windowsHide: true,
    timeout: 10000,
  });
  assert.equal(result.status, 1);
  assert.equal(result.signal, null);
  assert.equal(result.stdout, "");
  assert.match(
    result.stderr,
    /Static validator takes no run, data, checkpoint or tuning arguments/,
  );
});
test("complement partitions every original index once, excludes optimized targets, and covers each row and column", () => {
  const { fit, transfer } = transferPartition();
  assert.equal(new Set([...fit, ...transfer]).size, 160);
  assert.deepEqual(
    [...fit, ...transfer].sort((a, b) => a - b),
    Array.from({ length: 160 }, (_, i) => i),
  );
  assert.equal(fit.filter((i) => transfer.includes(i)).length, 0);
  assert(fit.every((i, r) => i === 10 * r + ((3 * r) % 10)));
  assert.equal(transfer.length, 144);
  assert(transfer.every((v, i) => i === 0 || v > transfer[i - 1]));
  assert.deepEqual(
    Array.from(
      { length: 16 },
      (_, r) => transfer.filter((i) => Math.floor(i / 10) === r).length,
    ),
    Array(16).fill(9),
  );
  assert.deepEqual(
    Array.from(
      { length: 10 },
      (_, c) => transfer.filter((i) => i % 10 === c).length,
    ),
    [14, 15, 14, 14, 15, 14, 14, 15, 15, 14],
  );
});
test("transfer uses only its 144 examples and all three gates, strict positive margin and technical precondition", () => {
  const good = {
    examples: 144,
    correct: 144,
    meanNll: 0.05,
    minimumTargetMargin: 0.00001,
  };
  assert.equal(transferScreen(good, true).status, "PASS");
  assert.equal(transferScreen(good, false).status, "BLOCKED_TECHNICAL_FAILURE");
  for (const delta of [
    { correct: 143 },
    { meanNll: 0.050000001 },
    { minimumTargetMargin: 0 },
    { minimumTargetMargin: -1 },
  ])
    assert.equal(transferScreen({ ...good, ...delta }, true).status, "FAIL");
  for (const delta of [
    { examples: 160 },
    { correct: 145 },
    { correct: 143.5 },
    { meanNll: NaN },
    { meanNll: -1 },
    { minimumTargetMargin: Infinity },
  ])
    assert.throws(() => transferScreen({ ...good, ...delta }, true));
});
test("static contract anchors the selected checkpoint and exact stored positive result without loading either model", () => {
  const result = validateTransferProposal(proposal());
  assert.equal(result.execution, "NOT_RUN");
  assert.equal(result.proposedEvaluationForwards, 320);
  assert.equal(result.actualPythonModelLoadForwardTraining, 0);
  assert.equal(result.actualDomainReferenceMaterialization, 0);
});
test("drift in checkpoint, source, saved evidence, denominator or fit-complement membership is rejected", () => {
  for (const mutate of [
    (p) => (p.checkpoint.serializationSha256 = "0".repeat(64)),
    (p) => (p.checkpoint.logicalWeightsSha256 = "0".repeat(64)),
    (p) => (p.sourceHashes["research/cnn/spatial.py"] = "0".repeat(64)),
    (p) => (p.evidence.sha256 = "0".repeat(64)),
    (p) => (p.split.overlap = 1),
    (p) => (p.split.fitFormula = "10*r"),
    (p) => (p.split.transferExamples = 160),
    (p) => p.split.transferColumnCounts[0]++,
    (p) => (p.screening.transfer.examples = 160),
    (p) => (p.screening.transfer.maximumMeanNll = 0.1),
  ]) {
    const changed = proposal();
    mutate(changed);
    assert.throws(() => validateTransferProposal(changed));
  }
});
test("no silent expansion to execution, training, extra load/forward, reference budget, score changes or publication", () => {
  for (const mutate of [
    (p) => (p.status = "USER_APPROVED"),
    (p) => (p.approval = "EXECUTION_APPROVED"),
    (p) => p.forwardBudget.totalTopLevelMax++,
    (p) => p.forwardBudget.checkpointLoads++,
    (p) => (p.forwardBudget.trainingRuns = 1),
    (p) => (p.forwardBudget.warmupOrExtraCalls = 1),
    (p) => p.materialization.totalReferenceNodeCap++,
    (p) => p.resources.childProcessWallSeconds++,
    (p) => (p.input.residualScale = 0),
    (p) => (p.boundaries.runnerImplemented = true),
    (p) => (p.boundaries.publication = "PUSH"),
    (p) => (p.boundaries.featureArchitectureScoreChanges = true),
  ]) {
    const changed = proposal();
    mutate(changed);
    assert.throws(() => validateTransferProposal(changed));
  }
});
