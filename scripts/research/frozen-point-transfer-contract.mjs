import assert from "node:assert/strict";
import { resolve } from "node:path";
import {
  root,
  json,
  hash,
  bytes,
  write,
  reserveFitOutput,
  approvedDuration,
  auditDurationStored,
} from "./fit-duration-contract.mjs";
import {
  validateTransferProposal,
  proposalPath,
  transferPartition,
  transferScreen,
} from "./validate-frozen-point-transfer.mjs";
export { root, json, hash, bytes, write, reserveFitOutput };
export const configPath = "research/experiments/frozen-point-transfer-v1.json";
export const phases = {
  "fit-control": 16,
  transfer: 144,
  "fit-repeat": 16,
  "transfer-repeat": 144,
};
const contractKeys = [
  "checkpoint",
  "split",
  "input",
  "materialization",
  "forwardBudget",
  "resources",
  "screening",
  "sourceHashes",
];
export function approvedTransfer(base = root) {
  const p = json(resolve(base, proposalPath));
  validateTransferProposal(p, base);
  const c = json(resolve(base, configPath));
  assert.equal(c.id, "frozen-point-transfer-v1");
  assert.equal(c.status, "FROZEN_BEFORE_EXECUTION");
  assert.equal(c.approval, "USER_APPROVED");
  assert.equal(c.proposalPath, proposalPath);
  assert.equal(c.proposalSha256, hash(resolve(base, proposalPath)));
  assert.deepEqual(
    c.contract,
    Object.fromEntries(contractKeys.map((k) => [k, p[k]])),
  );
  assert.deepEqual(
    c.modelConfig,
    approvedDuration(base).contract.shared.model.config,
  );
  assert.deepEqual(c.boundaries, {
    trainingRuns: 0,
    optimizerUpdates: 0,
    retryOrExtraTuning: false,
    existingSourceDataModelImmutable: true,
    rawDataWeightsLogitsPrivate: true,
    publicReplayAdditionalModelLoadForwardTraining: 0,
    appUsesDfs: true,
  });
  return c;
}
export function parseTransferArgs(kind, args) {
  assert(["run", "replay"].includes(kind));
  const names =
    kind === "run"
      ? ["--original", "--output", "--python"]
      : ["--output", "--recorded"];
  assert.equal(
    args.length,
    names.length * 2,
    "Fixed paths only; no execution overrides",
  );
  const pairs = [];
  for (let i = 0; i < args.length; i += 2) pairs.push(args.slice(i, i + 2));
  assert.deepEqual(pairs.map((p) => p[0]).sort(), names);
  assert(
    pairs.every(
      (p) => typeof p[1] === "string" && p[1] && !p[1].startsWith("--"),
    ),
  );
  return Object.fromEntries(pairs.map(([k, v]) => [k.slice(2), v]));
}
export function auditTransferData(data, fit) {
  assert.equal(data.schemaVersion, 1);
  assert.equal(data.id, "frozen-point-transfer-data-v1");
  assert.equal(data.split, "controlled-transfer-only");
  assert.equal(fit.examples.length, 16);
  assert.deepEqual(
    { ...data.counts, referenceNodes: 0 },
    {
      examples: 144,
      candidates: 4464,
      referenceNodes: 0,
      referenceNodeCap: 36864,
      outcomeTransitions: 4464,
      dfsCalls: 0,
    },
  );
  assert(
    Number.isInteger(data.counts.referenceNodes) &&
      data.counts.referenceNodes > 0 &&
      data.counts.referenceNodes <= 36864,
  );
  const indices = transferPartition().transfer;
  assert.equal(data.examples.length, 144);
  const vector = (v, n) =>
    assert(
      Array.isArray(v) &&
        v.length === n &&
        v.every(
          (x) =>
            typeof x === "number" && Number.isFinite(x) && x >= 0 && x <= 1,
        ),
    );
  let nodes = 0;
  data.examples.forEach((e, i) => {
    const index = indices[i],
      row = Math.floor(index / 10),
      col = index % 10;
    assert.equal(e.fixtureIndex, index);
    assert.equal(e.id, `controlled-transfer-${index}`);
    assert.equal(e.reference.complete, true);
    assert.equal(e.reference.unique, true);
    assert(
      Number.isInteger(e.reference.visitedNodes) &&
        e.reference.visitedNodes > 0 &&
        e.reference.visitedNodes <= 256,
    );
    nodes += e.reference.visitedNodes;
    const o = e.observation;
    assert.deepEqual(Object.keys(o).sort(), ["metadata", "planes"]);
    vector(o.metadata, 140);
    assert.deepEqual(o.metadata, fit.examples[0].observation.metadata);
    assert.equal(o.planes.length, 3);
    o.planes.forEach((p) => {
      assert.equal(p.length, 16);
      p.forEach((r) => {
        vector(r, 10);
        assert(r.every((v) => v === 0 || v === 1));
      });
    });
    const keys = [];
    for (let r = 0; r < 16; r++)
      for (let c = 0; c < 10; c++) {
        assert.equal(
          o.planes[0][r][c],
          Number(
            r === row
              ? c !== col
              : c !== (3 * r) % 10 && c !== (3 * r + 1) % 10,
          ),
        );
        assert.equal(o.planes[1][r][c], 0);
        assert.equal(o.planes[2][r][c], 0);
        if (o.planes[0][r][c] === 0) keys.push(["single-cell", r, c]);
      }
    assert.deepEqual(e.candidateKeys.map(JSON.parse), keys);
    assert.equal(keys.length, 31);
    assert(Number.isInteger(e.label) && e.label >= 0 && e.label < 31);
    assert.deepEqual(keys[e.label], ["single-cell", row, col]);
    assert.equal(e.candidateFeatures.length, 31);
    assert.equal(e.footprints.length, 31);
    keys.forEach(([, r, c], j) => {
      const f = e.candidateFeatures[j];
      vector(f, 74);
      assert.deepEqual(f.slice(0, 3), [0, 1, 0]);
      assert.deepEqual(f.slice(25, 27), [r / 15, c / 9]);
      assert.equal(f[52], j === e.label ? 1 / 16 : 0);
      assert.equal(
        e.footprints[j],
        Array.from({ length: 160 }, (_, k) =>
          k === 10 * r + c ? "1" : "0",
        ).join(""),
      );
    });
  });
  assert.equal(nodes, data.counts.referenceNodes);
  return data.examples;
}
export function auditTransferLedger(rows) {
  assert.equal(rows.length, 320);
  let offset = 0;
  for (const [phase, n] of Object.entries(phases)) {
    for (let i = 0; i < n; i++)
      assert.deepEqual(rows[offset + i], {
        call: offset + i + 1,
        phase,
        phaseCall: i + 1,
        step: null,
        example: i,
      });
    offset += n;
  }
}
const near = (a, b) =>
  assert(
    Number.isFinite(a) &&
      Number.isFinite(b) &&
      Math.abs(a - b) <= 2e-6 * Math.max(1, Math.abs(a), Math.abs(b)),
    "Stored native FP32 metric drift",
  );
export function auditEvaluationRows(rows, examples) {
  assert.equal(rows.length, examples.length);
  assert([16, 144].includes(rows.length));
  assert.equal(new Set(rows.map((r) => r.id)).size, rows.length);
  rows.forEach((r, i) => {
    const e = examples[i],
      s = r.logits;
    assert.equal(r.id, e.id);
    assert.equal(r.label, e.label);
    assert.equal(s.length, 31);
    assert(s.every((v) => Number.isFinite(v) && Math.fround(v) === v));
    const max = Math.max(...s),
      index = s.indexOf(max),
      nll =
        Math.log(s.reduce((a, v) => a + Math.exp(v - max), 0)) +
        max -
        s[e.label],
      margin = s[e.label] - Math.max(...s.filter((_, j) => j !== e.label));
    const rank =
      1 +
      s.filter(
        (v, j) =>
          j !== e.label &&
          (v > s[e.label] || (v === s[e.label] && j < e.label)),
      ).length;
    assert.equal(r.index, index);
    assert.equal(r.correct, index === e.label);
    assert.equal(r.rank, rank);
    near(r.nll, nll);
    near(r.margin, margin);
  });
  return {
    examples: rows.length,
    correct: rows.filter((r) => r.correct).length,
    meanNll: rows.reduce((a, r) => a + r.nll, 0) / rows.length,
    minimumTargetMargin: Math.min(...rows.map((r) => r.margin)),
    maximumTargetRank: Math.max(...rows.map((r) => r.rank)),
  };
}
export function auditTransferChild(recorded, base = root) {
  const c = approvedTransfer(base),
    path = (p) => resolve(recorded, p),
    frozen = json(path("execution.json")),
    r = json(path("child-report.json"));
  assert.equal(frozen.status, "FROZEN_BEFORE_EXECUTION");
  assert.equal(frozen.configSha256, hash(resolve(base, configPath)));
  assert.equal(r.status, "PASS");
  assert.equal(r.executionSha256, hash(path("execution.json")));
  assert.deepEqual(Object.keys(frozen.inputHashes).sort(), [
    "fit-dataset.json",
    "fit-expected.json",
    "transfer-dataset.json",
  ]);
  for (const [p, h] of Object.entries(frozen.inputHashes))
    assert.equal(hash(path(p)), h);
  assert.equal(
    frozen.inputHashes["fit-dataset.json"],
    c.contract.checkpoint.fitDatasetSha256,
  );
  assert.equal(
    frozen.inputHashes["fit-expected.json"],
    c.contract.checkpoint.fitPredictionsSha256,
  );
  const fit = json(path("fit-dataset.json")),
    transfer = json(path("transfer-dataset.json"));
  auditTransferData(transfer, fit);
  const names = [
    "fit-control.json",
    "transfer.json",
    "fit-repeat.json",
    "transfer-repeat.json",
    "forward-count.jsonl",
    "summary.json",
    "model-intent.json",
    "load-intent.json",
  ];
  assert.deepEqual(Object.keys(r.hashes).sort(), names.sort());
  for (const p of names) assert.equal(hash(path(p)), r.hashes[p]);
  assert.equal(
    hash(path("fit-control.json")),
    c.contract.checkpoint.fitPredictionsSha256,
  );
  assert.equal(hash(path("fit-repeat.json")), hash(path("fit-control.json")));
  assert.equal(hash(path("transfer.json")), hash(path("transfer-repeat.json")));
  auditTransferLedger(
    bytes(path("forward-count.jsonl"))
      .toString("utf8")
      .trim()
      .split("\n")
      .map(JSON.parse),
  );
  assert.deepEqual(r.forwardCounts, phases);
  assert.equal(
    r.loadedLogicalWeightsSha256,
    c.contract.checkpoint.logicalWeightsSha256,
  );
  assert.equal(r.finalLogicalWeightsSha256, r.loadedLogicalWeightsSha256);
  assert.deepEqual(json(path("model-intent.json")), {
    topLevelModels: 1,
    trainingRuns: 0,
  });
  assert.deepEqual(json(path("load-intent.json")), {
    checkpointLoads: 1,
    weightsOnly: true,
    serializationSha256: c.contract.checkpoint.serializationSha256,
  });
  const fitMetrics = auditEvaluationRows(
      json(path("fit-control.json")),
      fit.examples,
    ),
    rows = json(path("transfer.json")),
    transferMetrics = auditEvaluationRows(rows, transfer.examples),
    screen = transferScreen(transferMetrics, true);
  const summary = {
    technicalStatus: "PASS",
    fit: fitMetrics,
    transfer: transferMetrics,
    transferGate: { status: screen.status, checks: screen.checks },
    topLevelModels: 1,
    checkpointLoads: 1,
    topLevelForwards: 320,
    trainingRuns: 0,
    optimizerUpdates: 0,
  };
  assert.deepEqual(r.summary, summary);
  assert.deepEqual(json(path("summary.json")), summary);
  assert(
    r.budget.wallTimeSeconds <= 600 &&
      r.budget.maxSampledWorkingSetBytes <= 2147483648 &&
      r.budget.instantaneousPeakMeasured === false,
  );
  const group = (indices) => {
    const selected = indices.map((i) => rows[i]);
    return {
      examples: selected.length,
      correct: selected.filter((r) => r.correct).length,
      meanNll: selected.reduce((a, r) => a + r.nll, 0) / selected.length,
      minimumTargetMargin: Math.min(...selected.map((r) => r.margin)),
      maximumTargetRank: Math.max(...selected.map((r) => r.rank)),
    };
  };
  const byRow = Array.from({ length: 16 }, (_, k) =>
    group(
      transfer.examples.flatMap((e, i) =>
        Math.floor(e.fixtureIndex / 10) === k ? [i] : [],
      ),
    ),
  );
  const byColumn = Array.from({ length: 10 }, (_, k) =>
    group(
      transfer.examples.flatMap((e, i) =>
        e.fixtureIndex % 10 === k ? [i] : [],
      ),
    ),
  );
  return { summary, byRow, byColumn, report: r, construction: transfer.counts };
}
export function auditTransferStored(recorded, base = root) {
  const c = approvedTransfer(base),
    owner = json(resolve(recorded, "report.json")),
    frozen = json(resolve(recorded, "execution.json"));
  assert.equal(owner.status, "PASS");
  assert(owner.elapsedSeconds <= 660);
  assert.equal(owner.child.code, 0);
  assert.equal(owner.child.timedOut, false);
  assert(owner.child.elapsedMs <= 600000);
  const names = [
    "start.json",
    "execution.json",
    "fit-dataset.json",
    "fit-expected.json",
    "transfer-dataset.json",
    "child-report.json",
  ];
  assert.deepEqual(Object.keys(owner.hashes).sort(), names.sort());
  for (const p of names)
    assert.equal(hash(resolve(recorded, p)), owner.hashes[p]);
  const audited = auditTransferChild(recorded, base);
  assert.equal(
    frozen.originalCheckpointSha256,
    c.contract.checkpoint.serializationSha256,
  );
  return {
    schemaVersion: 1,
    id: "frozen-point-transfer-v1",
    technicalStatus: "PASS",
    ...audited.summary,
    byRow: audited.byRow,
    byColumn: audited.byColumn,
    construction: audited.construction,
    reproduction: {
      originalFitExact: true,
      all160RepeatedExact: true,
      frozenWeightsExact: true,
    },
    actualCompute: {
      topLevelModels: 1,
      checkpointLoads: 1,
      topLevelForwards: 320,
      trainingRuns: 0,
      optimizerUpdates: 0,
    },
    timing: {
      ownerWallSeconds: owner.elapsedSeconds,
      childProcessWallSeconds: owner.child.elapsedMs / 1000,
      pythonBodyWallSeconds: audited.report.budget.wallTimeSeconds,
      maxSampledWorkingSetBytes:
        audited.report.budget.maxSampledWorkingSetBytes,
      rssSamples: audited.report.budget.samples,
      instantaneousPeakMeasured: false,
    },
    interpretation: c.contract.screening.interpretation,
    additionalPythonModelLoadForwardTraining: 0,
    appUsesDfs: true,
    replayScope:
      "Node saved hashes/rows/metrics/ledger only; no source re-encoding, domain calls, checkpoint deserialization or new inference. Full source freeze was enforced by the owner/child.",
  };
}
export function auditOriginalTransferInput(original) {
  return auditDurationStored(original);
}
