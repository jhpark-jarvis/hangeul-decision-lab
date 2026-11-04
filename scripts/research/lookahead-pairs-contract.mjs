import assert from "node:assert/strict";
import { resolve } from "node:path";
import {
  root,
  proposalPath,
  readJson as json,
  sha,
  contract,
  lookaheadSchedule,
  validateLookaheadProposal,
  classifyLookaheadRecord,
  qualityInputBound,
} from "./validate-lookahead-proposal.mjs";
import { boardGroup } from "./audit-coverage-inputs.mjs";
import { createHash } from "node:crypto";
export { root, json, sha, contract, lookaheadSchedule };
export const configPath = "research/experiments/lookahead-pairs-v1.json";
export const digest = (v) =>
  createHash("sha256").update(JSON.stringify(v)).digest("hex");
const fp = (v) => digest(v.map(Math.fround));
export const projection = (e, n) =>
  e.candidateFeatures.map((f) =>
    n === 22
      ? f.slice(52)
      : n === 24
        ? [...f.slice(52), ...f.slice(25, 27)]
        : f,
  );

export function approvedPairs(base = root) {
  const p = json(resolve(base, proposalPath));
  validateLookaheadProposal(p, base);
  const c = json(resolve(base, configPath));
  assert.equal(c.id, "lookahead-pairs-v1");
  assert.equal(c.approval, "USER_APPROVED_FIXED_EXECUTION");
  assert.equal(c.proposalSha256, sha(resolve(base, proposalPath)));
  assert.deepEqual(c.budget, contract.proposedCompute);
  assert.deepEqual(c.resources, contract.resources);
  return c;
}

/** Durable pre-call ledger. Root counts as a node but never as a transition. */
export function createPairLedger(persist, check = () => {}) {
  const rows = [],
    counts = {
      states: 0,
      roots: 0,
      nodes: 0,
      referenceApply: 0,
      outcomeApply: 0,
      totalApply: 0,
    };
  let state = -1,
    localNodes = 0,
    lastRoot = -1,
    outcomes = 0,
    rootSeen = false;
  function accept(r) {
    check();
    const candidate = { event: rows.length + 1, ...r };
    persist(candidate);
    rows.push(candidate);
  }
  return {
    rows,
    counts,
    state(build, index) {
      assert.equal(build * 32 + index, counts.states);
      assert(counts.states < 64);
      if (state >= 0) assert.equal(outcomes, 32);
      accept({
        build,
        index,
        kind: "state",
        rootIndex: -1,
        depth: 0,
        key: null,
      });
      state++;
      counts.states++;
      localNodes = 0;
      lastRoot = -1;
      outcomes = 0;
      rootSeen = false;
    },
    reserve(r) {
      assert(state >= 0);
      const { build, index, kind, rootIndex, depth, key } = r;
      assert.equal(build * 32 + index, state);
      assert(["root", "edge", "outcome"].includes(kind));
      if (kind === "root") {
        assert(!rootSeen);
        assert.deepEqual(
          { rootIndex, depth, key },
          { rootIndex: -1, depth: 0, key: null },
        );
      } else {
        assert(rootSeen);
        assert(Number.isInteger(rootIndex) && rootIndex >= 0 && rootIndex < 32);
        assert(typeof key === "string" && key.length > 0);
        assert(Number.isInteger(depth) && depth >= 1 && depth <= 3);
      }
      if (kind === "edge") {
        assert.equal(outcomes, 0);
        assert(localNodes < 64);
        assert(rootIndex === lastRoot || rootIndex === lastRoot + 1);
        if (rootIndex !== lastRoot) assert.equal(depth, 1);
        else assert(depth > 1, "Root edge already covered");
        assert(counts.referenceApply < 4032 && counts.totalApply < 6080);
      }
      if (kind === "outcome") {
        assert.equal(lastRoot, 31);
        assert.equal(rootIndex, outcomes);
        assert.equal(depth, 1);
        assert(counts.outcomeApply < 2048 && counts.totalApply < 6080);
      }
      accept(r);
      if (kind === "root") {
        rootSeen = true;
        counts.roots++;
        counts.nodes++;
        localNodes++;
      }
      if (kind === "edge") {
        lastRoot = rootIndex;
        localNodes++;
        counts.nodes++;
        counts.referenceApply++;
        counts.totalApply++;
      }
      if (kind === "outcome") {
        outcomes++;
        counts.outcomeApply++;
        counts.totalApply++;
      }
      assert(counts.nodes <= 4096);
    },
    finish() {
      assert.deepEqual(
        [counts.states, counts.roots, counts.outcomeApply, outcomes],
        [64, 64, 2048, 32],
      );
      return { ...counts };
    },
  };
}
export function auditPairLedger(rows) {
  const l = createPairLedger(() => {});
  for (const row of rows) {
    const { event, ...r } = row;
    assert.equal(event, l.rows.length + 1);
    if (r.kind === "state") l.state(r.build, r.index);
    else l.reserve(r);
    assert.deepEqual(l.rows.at(-1), row);
  }
  return l.finish();
}

export function evaluationAxes(e) {
  assert.equal(typeof e.survivable, "boolean");
  assert.equal(typeof e.allCurrentPiecesPlaced, "boolean");
  const values = [
    Number(e.survivable),
    Number(e.allCurrentPiecesPlaced),
    e.mobility.playablePieceTypes,
    e.mobility.totalPlacements,
    e.clearedRows,
    e.acquiredItems,
    e.remainingAbilities.reroll + e.remainingAbilities.singleCell,
    -e.isolatedEmptyCells,
    -e.unfillableGaps,
    e.emptyCells,
  ];
  assert(values.every(Number.isSafeInteger));
  return values;
}
export function compareStoredEvaluations(a, b) {
  const x = evaluationAxes(a),
    y = evaluationAxes(b);
  for (let i = 0; i < x.length; i++)
    if (x[i] !== y[i]) return x[i] > y[i] ? 1 : -1;
  return 0;
}

/** Fingerprints of prior stored data; no domain call and no new game state. */
export function priorFingerprints(base = root, check = () => {}) {
  const p = json(resolve(base, proposalPath)),
    paths = Object.keys(p.inputHashes);
  for (const path of paths)
    assert.equal(sha(resolve(base, path)), p.inputHashes[path]);
  const groups = new Set(),
    vectors = { 22: new Set(), 24: new Set(), 74: new Set() },
    targets = { 22: new Set(), 24: new Set(), 74: new Set() },
    bags = new Set();
  let examples = 0,
    candidates = 0;
  const add = (source, aug = source) => {
    check();
    const board =
      source.state?.board.map((r) => r.map(Number)) ??
      source.observation.planes[0];
    groups.add(boardGroup(board));
    const e = { candidateFeatures: aug.candidateFeatures },
      label = source.label ?? source.labels.A512;
    assert(
      e.candidateFeatures.length > 0 &&
        label >= 0 &&
        label < e.candidateFeatures.length,
    );
    examples++;
    candidates += e.candidateFeatures.length;
    for (const n of [22, 24, 74]) {
      const rows = projection(e, n);
      for (const f of rows) vectors[n].add(fp(f));
      targets[n].add(fp(rows[label]));
    }
    bags.add(
      digest(
        projection(e, 22)
          .map((f, i) => [fp(f), i === label])
          .sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b))),
      ),
    );
  };
  const pack = json(resolve(base, paths[0])),
    aug = json(resolve(base, paths[1]));
  assert.deepEqual(pack.examples.test, []);
  for (const split of ["train", "dev"]) {
    assert.equal(pack.examples[split].length, aug.examples[split].length);
    pack.examples[split].forEach((e, i) => {
      assert.equal(e.id, aug.examples[split][i].id);
      add(e, aug.examples[split][i]);
    });
  }
  for (const path of paths.slice(2)) {
    const d = json(resolve(base, path));
    const rows = d.examples ?? Object.values(d.splits).flat();
    for (const e of rows) add(e);
  }
  return {
    schemaVersion: 1,
    sourceInputHashes: p.inputHashes,
    examples,
    candidates,
    groups: [...groups].sort(),
    vectors: Object.fromEntries(
      Object.entries(vectors).map(([k, v]) => [k, [...v].sort()]),
    ),
    targets: Object.fromEntries(
      Object.entries(targets).map(([k, v]) => [k, [...v].sort()]),
    ),
    bags: [...bags].sort(),
  };
}

function auditCase(e, expected) {
  assert.deepEqual(e.descriptor, expected);
  assert.equal(e.state.board.length, 16);
  assert(
    e.state.board.every(
      (r) => r.length === 10 && r.every((v) => typeof v === "boolean"),
    ),
  );
  const keys = e.candidateKeys;
  assert.equal(keys.length, 32);
  assert.equal(new Set(keys).size, 32);
  assert.equal(e.candidateFeatures.length, 32);
  assert.deepEqual(
    e.observation.planes[0],
    e.state.board.map((r) => r.map(Number)),
  );
  const expectedKeys = [];
  e.state.board.forEach((row, r) =>
    row.forEach((v, c) => {
      assert.equal(typeof v, "boolean");
      if (!v) expectedKeys.push(JSON.stringify(["single-cell", r, c]));
    }),
  );
  assert.deepEqual(keys, expectedKeys);
  assert.deepEqual(e.state.abilities, { singleCell: 1, reroll: 0 });
  assert.deepEqual(e.state.hiddenItems, []);
  assert.equal(e.state.pendingReroll, null);
  assert.equal(e.state.remainingPieces.length, 2);
  for (const [i, p] of e.state.remainingPieces.entries()) {
    assert.equal(p.pieceIndex, i);
    assert.equal(p.piece.id, "MIEUM");
    assert.deepEqual(p.piece.shape, [
      [true, true, true],
      [true, false, true],
      [true, true, true],
    ]);
  }
  for (const f of e.candidateFeatures)
    assert(
      f.length === 74 && f.every((x) => Number.isFinite(x) && x >= 0 && x <= 1),
    );
  assert.equal(e.footprints.length, 32);
  keys.forEach((key, i) => {
    const [, r, c] = JSON.parse(key),
      f = e.candidateFeatures[i];
    assert.deepEqual(f.slice(0, 25), [0, 1, 0, ...Array(22).fill(0)]);
    assert.deepEqual(f.slice(25, 52), [r / 15, c / 9, 1, ...Array(24).fill(0)]);
    assert.equal(
      e.footprints[i],
      Array.from({ length: 160 }, (_, j) =>
        j === r * 10 + c ? "1" : "0",
      ).join(""),
    );
    const counts = e.state.board.map((row) => row.filter(Boolean).length);
    counts[r]++;
    const cleared = Number(counts[r] === 10);
    if (cleared) counts[r] = 0;
    assert.deepEqual(f.slice(52), [
      cleared / 16,
      0,
      0,
      0,
      counts.reduce((a, b) => a + b, 0) / 160,
      0,
      ...counts.map((n) => n / 10),
    ]);
  });
  const ref = e.reference;
  assert.equal(ref.complete, true);
  assert(ref.nodes <= 64 && ref.maxDepth <= 3);
  assert.equal(ref.best.length, 32);
  const best = Array(32).fill(null),
    prefixes = new Map();
  for (const leaf of ref.leaves) {
    assert(
      Number.isInteger(leaf.rootIndex) &&
        leaf.rootIndex >= 0 &&
        leaf.rootIndex < 32,
    );
    assert(leaf.path.length >= 1 && leaf.path.length <= 3);
    assert.equal(leaf.path[0], keys[leaf.rootIndex]);
    assert.equal(
      leaf.evaluation.allCurrentPiecesPlaced,
      leaf.path.length === 3,
    );
    assert.equal(
      leaf.evaluation.survivable,
      leaf.evaluation.allCurrentPiecesPlaced,
    );
    leaf.path.forEach((key, j) => {
      const a = JSON.parse(key);
      assert.equal(a[0], j === 0 ? "single-cell" : "place-piece");
      const k = JSON.stringify(leaf.path.slice(0, j + 1));
      if (!prefixes.has(k))
        prefixes.set(k, { rootIndex: leaf.rootIndex, depth: j + 1, key });
    });
    if (
      best[leaf.rootIndex] === null ||
      compareStoredEvaluations(leaf.evaluation, best[leaf.rootIndex]) > 0
    )
      best[leaf.rootIndex] = leaf.evaluation;
  }
  assert(best.every(Boolean));
  assert.equal(ref.nodes, 1 + prefixes.size);
  assert.deepEqual(best, ref.best);
  const optimum = best.reduce((a, b) =>
      compareStoredEvaluations(a, b) >= 0 ? a : b,
    ),
    qualityIndices = best.flatMap((v, i) =>
      compareStoredEvaluations(v, optimum) === 0 ? [i] : [],
    );
  assert.deepEqual(ref.qualityIndices, qualityIndices);
  assert.equal(e.label, qualityIndices[0]);
  const immediate = e.candidateFeatures.map((f) => f[52]);
  assert.equal(immediate.filter((v) => v === 1 / 16).length, 2);
  assert(immediate.every((v) => v === 0 || v === 1 / 16));
  const maxImmediateIndices = immediate.flatMap((v, i) =>
      v === 1 / 16 ? [i] : [],
    ),
    r = {
      rootCount: 32,
      complete: true,
      qualityIndices,
      salvageableIndices: best.flatMap((v, i) =>
        v.allCurrentPiecesPlaced ? [i] : [],
      ),
      maxImmediateIndices,
      baselineIndex: maxImmediateIndices[0],
    };
  const classification = classifyLookaheadRecord(r);
  assert.equal(
    classification,
    expected.arm === "early-aligned" ? "ORDER_LUCKY_CONTROL" : "HARD",
  );
  return {
    classification,
    qualityIndices,
    baselineIndex: r.baselineIndex,
    baselineQuality: qualityIndices.includes(r.baselineIndex),
    boardHash: digest(e.state.board),
    group: boardGroup(e.observation.planes[0]),
    ordered22Sha256: digest(projection(e, 22).map((v) => v.map(Math.fround))),
    rootCount: 32,
    complete: true,
    edges: [...prefixes.values()],
  };
}

/** Entirely saved evidence; no TypeScript import/compile, transitions or model. */
export function auditPairBuild(data, registry, ledgerRows, build) {
  assert.equal(data.schemaVersion, 1);
  assert.equal(data.examples.length, 32);
  const schedule = lookaheadSchedule(),
    rows = data.examples.map((e, i) => auditCase(e, schedule[i]));
  const oldGroups = new Set(registry.groups),
    seenGroups = new Set(),
    oldVectors = Object.fromEntries(
      [22, 24, 74].map((n) => [n, new Set(registry.vectors[n])]),
    ),
    oldTargets = Object.fromEntries(
      [22, 24, 74].map((n) => [n, new Set(registry.targets[n])]),
    );
  const novelty = Object.fromEntries(
    [22, 24, 74].map((n) => [
      n,
      {
        candidateRows: 0,
        overlapRows: 0,
        qualityTargetRows: 0,
        overlapQualityTargets: 0,
        uniqueVectors: new Set(),
      },
    ]),
  );
  let oldBagMatches = 0;
  for (const [i, e] of data.examples.entries()) {
    assert(!oldGroups.has(rows[i].group), "Prior board/D2 overlap");
    assert(!seenGroups.has(rows[i].group), "Duplicate board/D2 group");
    seenGroups.add(rows[i].group);
    const events = ledgerRows.filter((r) => r.build === build && r.index === i);
    assert.equal(events[0].kind, "state");
    assert.equal(events[1].kind, "root");
    const edgeEvents = events
      .filter((r) => r.kind === "edge")
      .map(({ rootIndex, depth, key }) => ({ rootIndex, depth, key }));
    assert.deepEqual(edgeEvents, rows[i].edges);
    const outcomes = events.filter((r) => r.kind === "outcome");
    assert.equal(outcomes.length, 32);
    outcomes.forEach((r, j) => {
      assert.equal(r.rootIndex, j);
      assert.equal(r.key, e.candidateKeys[j]);
    });
    for (const n of [22, 24, 74]) {
      const vectors = projection(e, n).map(fp),
        stats = novelty[n];
      stats.candidateRows += vectors.length;
      stats.overlapRows += vectors.filter((v) => oldVectors[n].has(v)).length;
      vectors.forEach((v) => stats.uniqueVectors.add(v));
      stats.qualityTargetRows += rows[i].qualityIndices.length;
      stats.overlapQualityTargets += rows[i].qualityIndices.filter((j) =>
        oldTargets[n].has(vectors[j]),
      ).length;
    }
    const bag = digest(
      projection(e, 22)
        .map((f, j) => [fp(f), rows[i].qualityIndices.includes(j)])
        .sort((a, b) => JSON.stringify(a).localeCompare(JSON.stringify(b))),
    );
    if (registry.bags.includes(bag)) oldBagMatches++;
  }
  let equal24 = 0,
    equal74 = 0;
  for (let i = 0; i < 32; i += 2) {
    const a = rows[i],
      b = rows[i + 1],
      ea = data.examples[i],
      eb = data.examples[i + 1];
    assert.notEqual(a.boardHash, b.boardHash);
    assert.equal(a.ordered22Sha256, b.ordered22Sha256);
    assert.deepEqual(ea.observation.metadata, eb.observation.metadata);
    assert(a.qualityIndices.every((j) => !b.qualityIndices.includes(j)));
    if (digest(projection(ea, 24)) === digest(projection(eb, 24))) equal24++;
    if (digest(projection(ea, 74)) === digest(projection(eb, 74))) equal74++;
  }
  const bound = qualityInputBound(rows);
  assert(bound.idealQualityMembershipUpperBoundCorrect <= 16);
  return {
    status: "PASS",
    states: 32,
    pairs: 16,
    initialPointCandidatesEach: 32,
    hard: rows.filter((r) => r.classification === "HARD").length,
    orderLuckyControls: rows.filter(
      (r) => r.classification === "ORDER_LUCKY_CONTROL",
    ).length,
    baselineQualityCorrect: rows.filter((r) => r.baselineQuality).length,
    completeReferenceSalvageable: rows.length,
    ordered22PairsEqual: 16,
    disjointQualityPairs: 16,
    ordered24PairsEqual: equal24,
    ordered74PairsEqual: equal74,
    boardD2Unique: seenGroups.size,
    priorBoardD2Overlap: 0,
    qualityInputBound: bound,
    novelty: Object.fromEntries(
      Object.entries(novelty).map(([n, s]) => [
        n,
        { ...s, uniqueVectors: s.uniqueVectors.size },
      ]),
    ),
    priorTagged22BagMatches: oldBagMatches,
    priorExamples: registry.examples,
    priorCandidates: registry.candidates,
  };
}

export function auditPairsStored(recorded, base = root) {
  const c = json(resolve(base, configPath)),
    report = json(resolve(recorded, "report.json"));
  assert.equal(report.status, "PASS");
  assert.equal(report.configSha256, sha(resolve(base, configPath)));
  assert.deepEqual(Object.keys(report.hashes).sort(), [
    "build1.json",
    "build2.json",
    "ledger.jsonl",
    "prior-fingerprints.json",
  ]);
  for (const [p, h] of Object.entries(report.hashes))
    assert.equal(sha(resolve(recorded, p)), h);
  const registry = json(resolve(recorded, "prior-fingerprints.json"));
  assert.deepEqual(registry.sourceInputHashes, c.inputHashes);
  const ledger = readLedger(recorded),
    counts = auditPairLedger(ledger),
    first = json(resolve(recorded, "build1.json")),
    second = json(resolve(recorded, "build2.json"));
  assert.deepEqual(first, second);
  const a = auditPairBuild(first, registry, ledger, 0),
    b = auditPairBuild(second, registry, ledger, 1);
  assert.deepEqual(a, b);
  assert(report.elapsedSeconds >= 0 && report.elapsedSeconds <= 120);
  assert(report.maxSampledRss <= 2147483648);
  assert.equal(report.instantaneousPeakMeasured, false);
  const summary = {
    schemaVersion: 1,
    id: "lookahead-pairs-result-v1",
    status: "PASS",
    ...a,
    actualCompute: counts,
    repeatedBuildsExact: true,
    additionalPythonModelLoadForwardTraining: 0,
    independentGeneralization: false,
    appUsesDfs: true,
  };
  assert.deepEqual(report.summary, summary);
  return summary;
}
import { readFileSync } from "node:fs";
export const readLedger = (recorded) =>
  readFileSync(resolve(recorded, "ledger.jsonl"), "utf8")
    .trim()
    .split("\n")
    .map(JSON.parse);
