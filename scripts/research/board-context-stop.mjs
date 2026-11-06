import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  root,
  json,
  sha,
  bytes,
  write,
  reserveFitOutput,
  approvedBoardContext,
  auditBoardRows,
  fp32InputKey,
  auditPairsStored,
} from "./board-context-contract.mjs";
import { inputRows } from "./lookahead-information.mjs";
export const stopConfigPath =
  "research/experiments/board-context-stop-audit-v1.json";
export function ulpDistance(a, b) {
  assert(
    Number.isFinite(a) &&
      Number.isFinite(b) &&
      Math.fround(a) === a &&
      Math.fround(b) === b,
  );
  const order = (v) => {
    const buffer = Buffer.alloc(4);
    buffer.writeFloatLE(v);
    const n = buffer.readUInt32LE();
    return n >= 0x80000000 ? 0xffffffff - n : n + 0x80000000;
  };
  return Math.abs(order(a) - order(b));
}
/** Only two opposing target scores per pair are needed for the first-index bound. */
export function pairedAnchorProof(rows, keys) {
  assert(
    rows.length > 0 && rows.length % 2 === 0 && rows.length === keys.length,
  );
  let exactPairs = 0;
  for (let i = 0; i < rows.length; i += 2) {
    const a = rows[i],
      b = rows[i + 1],
      ka = keys[i],
      kb = keys[i + 1];
    assert.equal(ka.length, a.logits.length);
    assert.equal(kb.length, b.logits.length);
    assert.equal(new Set(ka).size, ka.length);
    assert.equal(new Set(kb).size, kb.length);
    for (const [r, k] of [
      [a, ka],
      [b, kb],
    ])
      assert(Number.isInteger(r.label) && r.label >= 0 && r.label < k.length);
    const targetA = ka[a.label],
      targetB = kb[b.label],
      aInB = kb.indexOf(targetA),
      bInA = ka.indexOf(targetB);
    assert.notEqual(
      targetA,
      targetB,
      "Opposing target inputs must be distinct",
    );
    assert(
      aInB >= 0 && bInA >= 0,
      "Both anchor inputs must occur in both lists",
    );
    assert.equal(
      a.label < bInA,
      aInB < b.label,
      "First-index tie priority must agree",
    );
    assert(
      Object.is(a.logits[a.label], b.logits[aInB]),
      "Target A native score differs",
    );
    assert(
      Object.is(a.logits[bInA], b.logits[b.label]),
      "Target B native score differs",
    );
    exactPairs++;
  }
  return {
    pairs: exactPairs,
    anchorComparisons: exactPairs * 2,
    scoreOccurrences: exactPairs * 4,
    exactTargetAnchorScores: true,
    sameTiePriority: true,
    nativeCorrectUpperBound: rows.length - exactPairs,
  };
}
export function inputScoreDrift(rows, keys) {
  assert.equal(rows.length, keys.length);
  const groups = new Map();
  rows.forEach((r, i) => {
    assert.equal(r.logits.length, keys[i].length);
    r.logits.forEach((v, j) => {
      assert(Number.isFinite(v) && Math.fround(v) === v);
      const k = keys[i][j];
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k).push({ example: i, index: j, value: v });
    });
  });
  const cases = [];
  let repeated = 0,
    maxAbs = 0,
    maxUlp = 0,
    sameIndex = 0,
    differentIndex = 0;
  for (const g of groups.values()) {
    if (g.length > 1) repeated++;
    if (g.every((x) => Object.is(x.value, g[0].value))) continue;
    const gap =
      Math.max(...g.map((x) => x.value)) - Math.min(...g.map((x) => x.value));
    let ulp = 0;
    for (const a of g)
      for (const b of g) ulp = Math.max(ulp, ulpDistance(a.value, b.value));
    maxAbs = Math.max(maxAbs, gap);
    maxUlp = Math.max(maxUlp, ulp);
    if (new Set(g.map((x) => x.index)).size === 1) sameIndex++;
    else differentIndex++;
    cases.push({ occurrences: g, absoluteGap: gap, ulpGap: ulp });
  }
  return {
    summary: {
      candidateRows: rows.reduce((n, r) => n + r.logits.length, 0),
      uniqueInputs: groups.size,
      repeatedInputGroups: repeated,
      mismatchGroups: cases.length,
      maximumAbsoluteGap: maxAbs,
      maximumUlpGap: maxUlp,
      sameIndexMismatchGroups: sameIndex,
      differentIndexMismatchGroups: differentIndex,
    },
    cases,
  };
}
export function auditBoardStop(recorded, base = root) {
  const c = json(resolve(base, stopConfigPath)),
    path = (p) => resolve(recorded, p),
    original = approvedBoardContext(base);
  assert.equal(c.id, "board-context-stop-audit-v1");
  assert.equal(c.mode, "STORED_ONLY_NO_RETRY");
  for (const [p, h] of Object.entries(c.sourceHashes))
    assert.equal(sha(resolve(base, p)), h, p);
  for (const [p, h] of Object.entries(c.artifactHashes))
    assert.equal(sha(path(p)), h, p);
  const f = json(path("execution.json"));
  assert.equal(
    f.configSha256,
    sha(resolve(base, "research/experiments/board-context-v1.json")),
  );
  assert.deepEqual(f.inputHashes, original.archiveHashes);
  assert.equal(
    sha(path("dataset.json")),
    original.archiveHashes["build1.json"],
  );
  for (const [p, h] of Object.entries(f.inputHashes))
    assert.equal(sha(path("source/" + p)), h, p);
  auditPairsStored(path("source"), base);
  const sourceProof = json(path("source-at-stop.json"));
  assert.equal(sourceProof.sourceHead, f.sourceHead);
  assert.deepEqual(sourceProof.originalSourceHashes, f.sourceHashes);
  assert.equal(sourceProof.sourceGraphExactAtStop, true);
  assert.equal(json(path("failure.json")).automaticRetry, false);
  assert.equal(json(path("control/failure.json")).retry, false);
  assert.equal(
    json(path("control/failure.json")).error,
    "Identical native input produced different logits; STOP",
  );
  assert.deepEqual(json(path("control/model-intent.json")), {
    topLevelModels: 1,
    checkpointLoads: 0,
    arm: "control",
  });
  const ledger = bytes(path("control/forward-count.jsonl"))
    .toString()
    .trim()
    .split("\n")
    .map(JSON.parse);
  assert.equal(ledger.length, 32);
  ledger.forEach((r, i) =>
    assert.deepEqual(r, {
      call: i + 1,
      arm: "control",
      phase: "initial",
      phaseCall: i + 1,
      step: null,
      example: i,
    }),
  );
  for (const p of [
    "candidate",
    "report.json",
    "control/history.json",
    "control/final1.json",
    "control/final2.json",
    "control/final-weights.pt",
  ])
    assert(!existsSync(path(p)), p);
  const examples = json(path("dataset.json")).examples,
    rows = json(path("control/initial.json")),
    keys = json(path("control/input-identities.json"));
  assert.deepEqual(
    keys,
    examples.map((e) => inputRows(e, "maskedSpatial").map(fp32InputKey)),
  );
  const metrics = auditBoardRows(rows, examples),
    drift = inputScoreDrift(rows, keys),
    anchors = pairedAnchorProof(rows, keys);
  const memory = bytes(path("control/memory.jsonl"))
    .toString()
    .trim()
    .split("\n")
    .map(JSON.parse);
  assert(
    memory.length > 0 &&
      memory.every(
        (r) => r.elapsedSeconds <= 600 && r.workingSetBytes <= 2147483648,
      ),
  );
  const summary = {
    schemaVersion: 1,
    id: "board-context-stop-audit-v1",
    auditStatus: "PASS",
    experimentTechnicalStatus: "FAIL",
    fitStatus: "NOT_RUN",
    stopReason: "Initial native identical-input bitwise mismatch",
    initial: metrics,
    drift: drift.summary,
    pairedTargetProofAtInitialOnly: anchors,
    actualCompute: {
      topLevelModels: 1,
      checkpointLoads: 0,
      initialForwards: 32,
      trainingForwards: 0,
      finalForwards: 0,
      totalTopLevelForwards: 32,
      optimizerUpdates: 0,
      candidateModels: 0,
      retries: 0,
      domainReferenceMaterializations: 0,
    },
    observations: {
      ownerStartToFailureTimestampSeconds:
        (Date.parse(json(path("failure.json")).atUtc) -
          Date.parse(json(path("start.json")).atUtc)) /
        1000,
      childLastSampleSeconds: memory.at(-1).elapsedSeconds,
      childMaxSampledRssBytes: Math.max(
        ...memory.map((r) => r.workingSetBytes),
      ),
      instantaneousPeakMeasured: false,
    },
    conclusions: {
      allInputExactGuardFailed: true,
      pairAnchorsStillExactAtInitial: true,
      observedDifferentIndexAssociationIsNotCausalLocalization: true,
      noTrainingOrModelQualityConclusion: true,
      noAutomaticToleranceOrRetry: true,
    },
    additionalReplayPythonModelLoadForwardTraining: 0,
  };
  return { summary, cases: drift.cases };
}
export function runBoardStop(args) {
  assert.equal(args.length, 4);
  assert.equal(args[0], "--recorded");
  assert.equal(args[2], "--output");
  assert(args[1] && args[3]);
  const recorded = resolve(root, args[1]),
    out = reserveFitOutput(args[3], root, recorded),
    result = auditBoardStop(recorded);
  write(resolve(out, "summary.json"), result.summary);
  write(resolve(out, "private-cases.json"), result.cases);
  write(resolve(out, "report.json"), {
    status: "PASS",
    atUtc: new Date().toISOString(),
    configSha256: sha(resolve(root, stopConfigPath)),
    summarySha256: sha(resolve(out, "summary.json")),
    privateCasesSha256: sha(resolve(out, "private-cases.json")),
    additionalModelLoadForwardTraining: 0,
  });
  return result.summary;
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
)
  console.log(JSON.stringify(runBoardStop(process.argv.slice(2))));
