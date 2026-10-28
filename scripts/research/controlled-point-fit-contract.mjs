import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync, mkdirSync, realpathSync } from "node:fs";
import { resolve, relative, isAbsolute, dirname, basename } from "node:path";
import { fileURLToPath } from "node:url";
import {
  validateProposal,
  proposalPath,
} from "./validate-controlled-point-fit-proposal.mjs";

export const root = fileURLToPath(new URL("../../", import.meta.url));
export const configPath = "research/experiments/controlled-point-fit-v1.json";
export const bytes = (p) => readFileSync(p);
export const json = (p) => JSON.parse(bytes(p).toString("utf8"));
export const hash = (p) => createHash("sha256").update(bytes(p)).digest("hex");
export const write = (p, value) =>
  writeFileSync(p, JSON.stringify(value, null, 2) + "\n", { flag: "wx" });
const inside = (base, p) => {
  const r = relative(base, p);
  return r !== "" && !r.startsWith("..") && !isAbsolute(r);
};
export function reserveFitOutput(output, base = root, recorded = null) {
  const research = resolve(base, ".research-output"),
    out = resolve(base, output);
  assert(inside(research, out), "New owned ignored output only");
  mkdirSync(research, { recursive: true });
  assert(
    inside(realpathSync(base), realpathSync(research)),
    "Redirected research root",
  );
  const parent = realpathSync(dirname(out));
  assert(
    parent === realpathSync(research) || inside(realpathSync(research), parent),
    "Redirected output parent",
  );
  if (recorded) {
    const source = realpathSync(recorded),
      actual = resolve(parent, basename(out));
    assert(
      source !== actual && !inside(source, actual) && !inside(actual, source),
      "Replay output overlaps input",
    );
  }
  mkdirSync(out);
  return out;
}
export function approvedContract(base = root) {
  const proposal = json(resolve(base, proposalPath));
  validateProposal(proposal, base);
  const config = json(resolve(base, configPath));
  assert.equal(config.id, "controlled-point-fit-v1");
  assert.equal(config.status, "FROZEN_BEFORE_EXECUTION");
  assert.equal(config.approval, "USER_APPROVED");
  assert.equal(config.proposalPath, proposalPath);
  assert.equal(config.proposalSha256, hash(resolve(base, proposalPath)));
  const keys = [
    "dataset",
    "input",
    "model",
    "fit",
    "forwardBudget",
    "resources",
    "acceptance",
  ];
  assert.deepEqual(
    config.contract,
    Object.fromEntries(keys.map((k) => [k, proposal[k]])),
  );
  assert.deepEqual(config.boundaries, {
    originalFilesImmutable: true,
    oldDatasetOrLabelChanges: false,
    reservedTestOrRollout: false,
    appUsesDfs: true,
    publicReplayAdditionalModelLoadForwardTraining: 0,
    rawDataWeightsLogitsPrivate: true,
    retryOrAdditionalTuning: false,
  });
  return config;
}
const near = (a, b) =>
  assert(
    Number.isFinite(a) &&
      Number.isFinite(b) &&
      Math.abs(a - b) <= 2e-6 * Math.max(1, Math.abs(a), Math.abs(b)),
    "Stored float32 metric drift",
  );
export function auditRows(rows, examples) {
  assert.equal(rows.length, 16);
  assert.equal(examples.length, 16);
  const metrics = rows.map((r, i) => {
    const e = examples[i];
    assert.equal(r.id, `controlled-fit-${i}`);
    assert.equal(e.id, r.id);
    assert.equal(r.label, e.label);
    assert(Number.isInteger(r.label) && r.label >= 0 && r.label < 31);
    const s = r.logits;
    assert.equal(s.length, 31);
    assert(s.every((v) => Number.isFinite(v) && Math.fround(v) === v));
    const max = Math.max(...s),
      index = s.indexOf(max);
    const nll =
      Math.log(s.reduce((a, v) => a + Math.exp(v - max), 0)) + max - s[r.label];
    const margin = s[r.label] - Math.max(...s.filter((_, j) => j !== r.label));
    const rank =
      1 +
      s.filter(
        (v, j) =>
          j !== r.label &&
          (v > s[r.label] || (v === s[r.label] && j < r.label)),
      ).length;
    assert.equal(r.index, index);
    assert.equal(r.correct, index === r.label);
    assert.equal(r.rank, rank);
    near(r.nll, nll);
    near(r.margin, margin);
    return { correct: r.correct, nll: r.nll, margin: r.margin, rank };
  });
  return {
    examples: 16,
    correct: metrics.filter((r) => r.correct).length,
    meanNll: metrics.reduce((a, r) => a + r.nll, 0) / 16,
    minimumTargetMargin: Math.min(...metrics.map((r) => r.margin)),
    maximumTargetRank: Math.max(...metrics.map((r) => r.rank)),
  };
}
export function auditLedger(rows) {
  assert.equal(rows.length, 3248);
  const phases = [
    ["initial", 16],
    ["train", 3200],
    ["final1", 16],
    ["final2", 16],
  ];
  let offset = 0;
  for (const [phase, count] of phases) {
    for (let i = 0; i < count; i++)
      assert.deepEqual(rows[offset + i], {
        call: offset + i + 1,
        phase,
        phaseCall: i + 1,
        step: phase === "train" ? Math.floor(i / 16) + 1 : null,
        example: i % 16,
      });
    offset += count;
  }
}
export function auditStored(recorded, base = root) {
  approvedContract(base);
  const report = json(resolve(recorded, "report.json"));
  assert.equal(report.status, "PASS");
  assert.equal(report.child.code, 0);
  assert.equal(report.child.timedOut, false);
  assert(report.elapsedSeconds <= 660);
  for (const [p, h] of Object.entries(report.hashes)) {
    assert(
      [
        "start.json",
        "dataset.json",
        "execution.json",
        "child-report.json",
        "summary.json",
      ].includes(p),
    );
    assert.equal(hash(resolve(recorded, p)), h, p);
  }
  assert.equal(Object.keys(report.hashes).length, 5);
  const frozen = json(resolve(recorded, "execution.json"));
  assert.equal(frozen.configSha256, hash(resolve(base, configPath)));
  assert.equal(frozen.status, "FROZEN_BEFORE_EXECUTION");
  assert.equal(frozen.datasetSha256, hash(resolve(recorded, "dataset.json")));
  const child = json(resolve(recorded, "child-report.json"));
  assert.equal(child.status, "PASS");
  assert.equal(child.topLevelModels, 1);
  assert.equal(child.checkpointLoads, 0);
  assert.equal(child.trainingRuns, 1);
  assert(
    child.budget.wallTimeSeconds <= 600 &&
      child.budget.maxSampledWorkingSetBytes <= 2147483648,
  );
  const names = [
    "dataset.json",
    "execution.json",
    "initial.json",
    "final1.json",
    "final2.json",
    "history.json",
    "forward-count.jsonl",
    "summary.json",
    "final-weights.pt",
  ];
  assert.deepEqual(Object.keys(child.hashes).sort(), [...names].sort());
  for (const p of names)
    assert.equal(hash(resolve(recorded, p)), child.hashes[p], p);
  const data = json(resolve(recorded, "dataset.json"));
  assert.equal(data.split, "fit-only");
  assert.equal(data.counts.examples, 16);
  assert.equal(data.counts.candidates, 496);
  assert.equal(data.counts.outcomeTransitions, 496);
  assert(data.counts.referenceNodes > 0 && data.counts.referenceNodes <= 4096);
  assert.equal(data.counts.dfsCalls, 0);
  assert.equal(
    hash(resolve(recorded, "final1.json")),
    hash(resolve(recorded, "final2.json")),
  );
  auditLedger(
    bytes(resolve(recorded, "forward-count.jsonl"))
      .toString("utf8")
      .trim()
      .split("\n")
      .map(JSON.parse),
  );
  const initial = auditRows(
    json(resolve(recorded, "initial.json")),
    data.examples,
  );
  const final = auditRows(
    json(resolve(recorded, "final1.json")),
    data.examples,
  );
  const checks = {
    allCorrect: final.correct === 16,
    meanNllAtMost005: final.meanNll <= 0.05,
    everyTargetMarginPositive: final.minimumTargetMargin > 0,
  };
  const summary = json(resolve(recorded, "summary.json"));
  assert.deepEqual(summary.initial, initial);
  assert.deepEqual(summary.final, final);
  assert.deepEqual(summary.fitGate, {
    status: Object.values(checks).every(Boolean) ? "PASS" : "FAIL",
    checks,
  });
  assert.deepEqual(child.summary, summary);
  assert.equal(summary.technicalStatus, "PASS");
  assert.equal(summary.updates, 200);
  assert.equal(summary.trainingRuns, 1);
  assert.equal(summary.topLevelForwardCalls, 3248);
  assert.equal(summary.finalRepeatsExact, true);
  assert.equal(summary.heldOutOrKindSelectionTested, false);
  assert.equal(summary.spatialMechanismTested, false);
  assert.equal(summary.appUsesDfs, true);
  const history = json(resolve(recorded, "history.json"));
  assert.equal(history.length, 200);
  history.forEach((r, i) => {
    assert.equal(r.step, i + 1);
    assert(Number.isFinite(r.meanTrainingNll) && r.meanTrainingNll >= 0);
  });
  return {
    status: "PASS_STORED_REPLAY",
    summary,
    datasetCounts: data.counts,
    budget: {
      ownerWallSeconds: report.elapsedSeconds,
      childWallSeconds: child.budget.wallTimeSeconds,
      maxSampledWorkingSetBytes: child.budget.maxSampledWorkingSetBytes,
      instantaneousPeakMeasured: false,
    },
    sourceValidation:
      "Original graph freeze verified by owner/child; this data-only replay verifies stored artifact hashes, approved contract and the eight original proposal source hashes, without re-running the domain or model.",
    additionalModelCreationLoadForwardTraining: 0,
  };
}
