import assert from "node:assert/strict";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import {
  root,
  json,
  hash,
  write,
  reserveFitOutput,
} from "./coverage-training-contract.mjs";
export const configPath = "research/experiments/representation-alias-v1.json";
const digest = (v) =>
  createHash("sha256").update(JSON.stringify(v)).digest("hex");
export function fp32Key(values) {
  assert(
    Array.isArray(values) &&
      values.length > 0 &&
      values.every(
        (v) => typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 1,
      ),
  );
  return JSON.stringify(values.map(Math.fround));
}
/** Structural bounds under an ideal deterministic candidate-pointwise score, not a neural forward. */
export function aliasBounds(features, label) {
  assert(
    Array.isArray(features) &&
      features.length >= 1 &&
      Number.isInteger(label) &&
      label >= 0 &&
      label < features.length,
  );
  const keys = features.map(fp32Key);
  assert(keys.every((k) => JSON.parse(k).length === features[0].length));
  const members = keys.flatMap((k, i) => (k === keys[label] ? [i] : []));
  return {
    members,
    multiplicity: members.length,
    earlierDuplicate: members[0] < label,
    positiveMarginPossible: members.length === 1,
    idealNllLowerBound: Math.log(members.length),
  };
}
export function summarizeAliases(rows) {
  const summary = {
    examples: rows.length,
    aliasTargets: 0,
    earlierDuplicateTargets: 0,
    positiveMarginPossibleExamples: 0,
    hardLabelTop1UpperBoundCorrect: 0,
    meanIdealNllLowerBound: 0,
    multiplicityHistogram: {},
  };
  for (const r of rows) {
    summary.aliasTargets += Number(r.multiplicity > 1);
    summary.earlierDuplicateTargets += Number(r.earlierDuplicate);
    summary.positiveMarginPossibleExamples += Number(r.positiveMarginPossible);
    summary.hardLabelTop1UpperBoundCorrect += Number(!r.earlierDuplicate);
    summary.meanIdealNllLowerBound += r.idealNllLowerBound;
    summary.multiplicityHistogram[r.multiplicity] =
      (summary.multiplicityHistogram[r.multiplicity] || 0) + 1;
  }
  summary.meanIdealNllLowerBound = rows.length
    ? summary.meanIdealNllLowerBound / rows.length
    : null;
  return summary;
}
export function storedMaskSurvivors(keys, members) {
  const ordinary = keys.filter((k) => k[0] === "place-piece");
  for (const k of ordinary) {
    assert.equal(k.length, 7);
    assert(Number.isInteger(k[4]) && Number.isInteger(k[5]));
    assert(Array.isArray(k[6]) && k[6].length > 0);
    k[6].forEach((r) =>
      assert(Array.isArray(r) && r.every((x) => typeof x === "boolean")),
    );
  }
  return members.map((i) => {
    assert.equal(keys[i][0], "single-cell");
    const [, r, c] = keys[i];
    assert(
      Number.isInteger(r) &&
        r >= 0 &&
        r < 16 &&
        Number.isInteger(c) &&
        c >= 0 &&
        c < 10,
    );
    return ordinary.filter((k) => k[6][r - k[4]]?.[c - k[5]] !== true).length;
  });
}
export function analyzeRepresentations(examples, augmented, check = () => {}) {
  assert.equal(examples.length, augmented.length);
  const privateCases = [],
    all22 = [],
    all74 = [],
    all24 = [],
    point22 = [],
    point74 = [],
    point24 = [];
  let candidates = 0;
  examples.forEach((e, index) => {
    check();
    const a = augmented[index];
    assert.equal(e.id, a.id);
    assert.equal(e.candidateKeys.length, e.candidateFeatures.length);
    assert.equal(a.candidateFeatures.length, e.candidateFeatures.length);
    const label = e.labels.A512;
    assert(
      Number.isInteger(label) && label >= 0 && label < e.candidateKeys.length,
    );
    assert(e.state && e.teachers.A512.result.search);
    a.candidateFeatures.forEach((f, j) => {
      assert.equal(f.length, 74);
      assert.equal(e.candidateFeatures[j].length, 52);
      assert.deepEqual(f.slice(0, 52), e.candidateFeatures[j]);
      fp32Key(f);
      candidates++;
    });
    const b22 = aliasBounds(
        a.candidateFeatures.map((f) => f.slice(52)),
        label,
      ),
      b74 = aliasBounds(a.candidateFeatures, label),
      b24 = aliasBounds(
        a.candidateFeatures.map((f) => [...f.slice(52), ...f.slice(25, 27)]),
        label,
      );
    all22.push(b22);
    all74.push(b74);
    all24.push(b24);
    const keys = e.candidateKeys.map(JSON.parse);
    if (keys[label][0] !== "single-cell") return;
    point22.push(b22);
    point74.push(b74);
    point24.push(b24);
    const ordinary = keys.filter((k) => k[0] === "place-piece").length,
      f = a.candidateFeatures[label];
    const deadEnd =
      b22.multiplicity > 1 &&
      ordinary === 0 &&
      e.state.abilities.singleCell === 1 &&
      e.state.abilities.reroll === 0 &&
      b22.members.every((i) => keys[i][0] === "single-cell") &&
      f[52] === 0 &&
      f[53] === 0 &&
      f[54] === 0 &&
      f[57] === 0;
    privateCases.push({
      sourceIndex: index,
      id: e.id,
      fixtureId: e.fixtureId,
      stateSha256: digest(e.state),
      label,
      members22: b22.members,
      multiplicity: b22.multiplicity,
      earlierDuplicate: b22.earlierDuplicate,
      members74: b74.members,
      teacherSearchComplete: e.teachers.A512.result.search.searchComplete,
      teacherAgreesA1024: label === e.labels.A1024,
      storedTeacherTerminalSurvivable:
        e.teachers.A512.result.evaluation.survivable,
      storedTeacherAllPiecesPlaced:
        e.teachers.A512.result.evaluation.allCurrentPiecesPlaced,
      ordinaryPlacements: ordinary,
      pointOnly: keys.every((k) => k[0] === "single-cell"),
      targetClearsRows: f[52] > 0,
      monotoneDeadEndClass: deadEnd,
      eligibleLiveAlias:
        b22.multiplicity > 1 &&
        ordinary > 0 &&
        e.state.abilities.singleCell === 1 &&
        e.state.abilities.reroll === 0 &&
        e.state.pendingReroll === null &&
        e.state.remainingPieces.length <= 3 &&
        e.state.hiddenItems.length <= 2 &&
        f[52] === 0 &&
        f[53] === 0 &&
        f[54] === 0 &&
        b22.members.every((i) => keys[i][0] === "single-cell") &&
        e.teachers.A512.result.search.searchComplete === true,
    });
  });
  const cohort = (predicate) => {
    const selected = privateCases.filter(predicate),
      indices = new Set(selected.map((c) => c.sourceIndex));
    const selectedBounds = point22.filter((_, i) =>
      indices.has(privateCases[i].sourceIndex),
    );
    return {
      examples: selected.length,
      distinctFixtures: new Set(selected.map((r) => r.fixtureId)).size,
      bounds22: summarizeAliases(selectedBounds),
    };
  };
  for (const row of privateCases)
    if (row.eligibleLiveAlias)
      row.storedMaskSurvivors = storedMaskSurvivors(
        examples[row.sourceIndex].candidateKeys.map(JSON.parse),
        row.members22,
      );
  const live = privateCases.filter((r) => r.eligibleLiveAlias),
    geometry = {
      cases: live.length,
      distinctFixtures: new Set(live.map((r) => r.fixtureId)).size,
      mixedOrdinaryAvailability: live.filter(
        (r) =>
          Math.min(...r.storedMaskSurvivors) === 0 &&
          Math.max(...r.storedMaskSurvivors) > 0,
      ).length,
      earliestAliasPreservesOrdinary: live.filter(
        (r) => r.storedMaskSurvivors[0] > 0,
      ).length,
      teacherHardLabelPreservesOrdinary: live.filter(
        (r) => r.storedMaskSurvivors[r.members22.indexOf(r.label)] > 0,
      ).length,
      storedTeacherTerminalSurvivable: live.filter(
        (r) => r.storedTeacherTerminalSurvivable,
      ).length,
    };
  return {
    summary: {
      all: {
        outcome22: summarizeAliases(all22),
        outcomePlusCoordinates24: summarizeAliases(all24),
        full74: summarizeAliases(all74),
      },
      point: {
        outcome22: summarizeAliases(point22),
        outcomePlusCoordinates24: summarizeAliases(point24),
        full74: summarizeAliases(point74),
      },
      cohorts: {
        searchComplete: cohort((r) => r.teacherSearchComplete),
        searchIncomplete: cohort((r) => !r.teacherSearchComplete),
        pointOnly: cohort((r) => r.pointOnly),
        targetClears: cohort((r) => r.targetClearsRows),
        targetDoesNotClear: cohort((r) => !r.targetClearsRows),
        monotoneDeadEnd: cohort((r) => r.monotoneDeadEndClass),
        liveAlias: cohort((r) => r.eligibleLiveAlias),
      },
      candidates,
      storedGeometry: geometry,
    },
    privateCases,
  };
}
export function auditStoredRepresentations(
  source,
  outcome,
  base = root,
  check = () => {},
) {
  const c = json(resolve(base, configPath));
  assert.equal(c.id, "representation-alias-v1");
  assert.equal(c.status, "READ_ONLY_STORED_AUDIT");
  assert.deepEqual(c.splits, { train: 339, dev: 89 });
  assert.deepEqual(c.pointExamples, { train: 52, dev: 13 });
  assert.deepEqual(c.boundaries, {
    newModelLoadForwardTrainingDomainDataset: 0,
    testSplitAccess: false,
    storedA512LabelsUnchanged: true,
    caseIdentifiersAndCoordinatesPrivate: true,
    appUsesDfs: true,
  });
  for (const [p, h] of Object.entries(c.sourceHashes))
    assert.equal(hash(resolve(base, p)), h, p);
  const inputs = {
    sourceDataset: resolve(source, "train-dev/dataset.json"),
    sourceManifest: resolve(source, "train-dev/manifest.json"),
    sourceProtocol: resolve(source, "train-dev/protocol.json"),
    augmentedDataset: resolve(outcome, "dataset.json"),
    augmentedExecution: resolve(outcome, "execution.json"),
  };
  for (const [k, p] of Object.entries(inputs))
    assert.equal(hash(p), c.inputHashes[k], k);
  const manifest = json(inputs.sourceManifest),
    execution = json(inputs.augmentedExecution),
    pack = json(inputs.sourceDataset),
    aug = json(inputs.augmentedDataset);
  assert.equal(manifest.status, "PASS");
  assert.equal(manifest.stage, "train-dev");
  assert.equal(manifest.dataSha256, c.inputHashes.sourceDataset);
  assert.equal(manifest.protocolSha256, c.inputHashes.sourceProtocol);
  assert.equal(execution.augmentedSha256, c.inputHashes.augmentedDataset);
  assert.equal(aug.sourceDataSha256, manifest.dataSha256);
  assert.deepEqual(Object.keys(aug.examples).sort(), ["dev", "train"]);
  assert.equal(pack.examples.test.length, 0);
  assert.equal(pack.stage, "train-dev");
  assert.equal(pack.status, "PASS");
  assert.equal(aug.transitions, 149597);
  check();
  const summary = {
      schemaVersion: 1,
      id: "representation-alias-v1",
      status: "PASS_STORED_AUDIT",
      splits: {},
      totalCandidateRows: 0,
      assumption:
        "Ideal deterministic candidate-pointwise f(FP32 outcome22), raw full-list softmax and first-legal tie order; no board, coordinates, index or set context. Not measured native NN output equality or a bound for the existing74-feature CNN.",
      limits:
        "Hard-label mimicry only, not unavoidable gameplay error. Stored A512 complete/incomplete labels are preserved; alias-class action quality needs separate conditional reference evidence.",
      additionalPythonModelLoadForwardTrainingDomainDataset: 0,
      appUsesDfs: true,
    },
    cases = {};
  for (const split of ["train", "dev"]) {
    assert.equal(pack.examples[split].length, c.splits[split]);
    const result = analyzeRepresentations(
      pack.examples[split],
      aug.examples[split],
      check,
    );
    assert.equal(
      result.summary.point.outcome22.examples,
      c.pointExamples[split],
    );
    summary.splits[split] = result.summary;
    summary.totalCandidateRows += result.summary.candidates;
    cases[split] = result.privateCases;
  }
  assert.equal(summary.totalCandidateRows, 149597);
  const selected = [],
    seen = new Set();
  for (const split of ["train", "dev"])
    for (const row of cases[split])
      if (row.eligibleLiveAlias && !seen.has(row.stateSha256)) {
        seen.add(row.stateSha256);
        selected.push({ split, ...row });
      }
  return {
    summary,
    cases,
    selection: {
      schemaVersion: 1,
      sourceDataSha256: c.inputHashes.sourceDataset,
      augmentedSha256: c.inputHashes.augmentedDataset,
      cases: selected,
    },
  };
}
export function runRepresentationAudit(args) {
  assert.equal(
    execFileSync("git", ["status", "--porcelain"], {
      cwd: root,
      encoding: "utf8",
      windowsHide: true,
    }).trim(),
    "",
    "Commit implementation before stored audit",
  );
  const source = resolve(root, args.source),
    outcome = resolve(root, args.outcome),
    out = reserveFitOutput(args.output, root, source),
    started = performance.now();
  let maxSampledRss = 0;
  const check = () => {
    maxSampledRss = Math.max(maxSampledRss, process.memoryUsage().rss);
    assert(
      (performance.now() - started) / 1000 <= 120 &&
        maxSampledRss <= 2147483648,
      "Stored audit sampled time/RSS cap",
    );
  };
  try {
    const result = auditStoredRepresentations(source, outcome, root, check);
    write(resolve(out, "summary.json"), result.summary);
    write(resolve(out, "private-cases.json"), result.cases);
    write(resolve(out, "private-reference-selection.json"), result.selection);
    const elapsed = (performance.now() - started) / 1000,
      rss = process.memoryUsage().rss;
    assert(
      elapsed <= 120 && rss <= 2147483648,
      "Stored audit sampled time/RSS cap",
    );
    write(resolve(out, "report.json"), {
      status: "PASS",
      atUtc: new Date().toISOString(),
      executor: "Codex",
      node: process.version,
      elapsedSeconds: elapsed,
      rssAfterBytes: rss,
      maxSampledRss,
      sourceHead: execFileSync("git", ["rev-parse", "HEAD"], {
        cwd: root,
        encoding: "utf8",
        windowsHide: true,
      }).trim(),
      configSha256: hash(resolve(root, configPath)),
      instantaneousPeakMeasured: false,
      hashes: Object.fromEntries(
        [
          "summary.json",
          "private-cases.json",
          "private-reference-selection.json",
        ].map((p) => [p, hash(resolve(out, p))]),
      ),
      additionalPythonModelLoadForwardTrainingDomainDataset: 0,
    });
    return result.summary;
  } catch (error) {
    write(resolve(out, "failure.json"), {
      status: "FAIL",
      error: String(error),
      automaticRetry: false,
    });
    throw error;
  }
}
export function parseAliasArgs(args) {
  assert.equal(args.length, 6, "Use --source --outcome --output only");
  const pairs = [];
  for (let i = 0; i < args.length; i += 2) pairs.push(args.slice(i, i + 2));
  assert.deepEqual(pairs.map((p) => p[0]).sort(), [
    "--outcome",
    "--output",
    "--source",
  ]);
  assert(pairs.every((p) => p[1] && !p[1].startsWith("--")));
  return Object.fromEntries(pairs.map(([k, v]) => [k.slice(2), v]));
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
)
  console.log(
    JSON.stringify(
      runRepresentationAudit(parseAliasArgs(process.argv.slice(2))),
    ),
  );
