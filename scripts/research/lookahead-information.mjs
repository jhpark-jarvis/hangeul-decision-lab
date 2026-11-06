import assert from "node:assert/strict";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  root,
  json,
  sha,
  digest,
  projection,
  auditPairsStored,
} from "./lookahead-pairs-contract.mjs";
import { qualityInputBound } from "./validate-lookahead-proposal.mjs";
import { reserveFitOutput, write } from "./controlled-point-fit-contract.mjs";

export const informationConfig =
  "research/experiments/lookahead-information-v1.json";
export const views = [
  "outcome22",
  "diagnostic24",
  "candidate74",
  "maskedSpatial",
  "fullSpatial",
];

/** Identity of one candidate's actual inputs; never an inferred embedding or score. */
export function inputRows(e, view) {
  assert(views.includes(view));
  if (view === "outcome22") return projection(e, 22);
  if (view === "diagnostic24") return projection(e, 24);
  if (view === "candidate74") return projection(e, 74);
  const planes = e.observation.planes.flat(2);
  const context = [
    ...(view === "fullSpatial"
      ? planes
      : planes.map((v, i) => (i < 160 ? 0 : v))),
    ...e.observation.metadata,
  ];
  return e.candidateFeatures.map((f, i) => [
    ...context,
    ...f,
    ...[...e.footprints[i]].map(Number),
  ]);
}

/** Exact score-inequality consistency, with first-index ties. No scores are fitted. */
export function rankingFeasibility(examples) {
  assert(examples.length > 0);
  const ids = new Map(),
    edges = [];
  const id = (k) => {
    if (!ids.has(k)) ids.set(k, ids.size);
    return ids.get(k);
  };
  let positiveNegativeAliasObservations = 0;
  for (const e of examples) {
    assert(Array.isArray(e.keys) && e.keys.length > 0);
    assert(
      Number.isInteger(e.target) && e.target >= 0 && e.target < e.keys.length,
    );
    const t = id(e.keys[e.target]);
    let alias = false;
    e.keys.forEach((key, j) => {
      const v = id(key);
      if (j !== e.target) {
        edges.push([t, v, j < e.target]);
        if (v === t) alias = true;
      }
    });
    if (alias) positiveNegativeAliasObservations++;
  }
  const graph = Array.from({ length: ids.size }, () => []),
    reverse = graph.map(() => []);
  for (const [a, b] of edges) {
    graph[a].push(b);
    reverse[b].push(a);
  }
  const visited = new Set(),
    order = [];
  // Iterative traversal avoids stack-size dependence on the number of candidates.
  for (let start = 0; start < ids.size; start++) {
    if (visited.has(start)) continue;
    visited.add(start);
    const stack = [[start, 0]];
    while (stack.length) {
      const top = stack.at(-1),
        next = graph[top[0]][top[1]++];
      if (next === undefined) {
        order.push(top[0]);
        stack.pop();
      } else if (!visited.has(next)) {
        visited.add(next);
        stack.push([next, 0]);
      }
    }
  }
  const component = new Map();
  let components = 0;
  for (const start of order.reverse()) {
    if (component.has(start)) continue;
    const stack = [start];
    component.set(start, components);
    while (stack.length)
      for (const next of reverse[stack.pop()])
        if (!component.has(next)) {
          component.set(next, components);
          stack.push(next);
        }
    components++;
  }
  const cyclic = new Set(),
    strictCyclic = new Set();
  for (const [a, b, strict] of edges)
    if (component.get(a) === component.get(b)) {
      cyclic.add(component.get(a));
      if (strict) strictCyclic.add(component.get(a));
    }
  return {
    observations: examples.length,
    uniqueCandidateInputs: ids.size,
    constraints: edges.length,
    positiveNegativeAliasObservations,
    cyclicComponents: cyclic.size,
    strictCyclicComponents: strictCyclic.size,
    allFirstIndexTop1Feasible: strictCyclic.size === 0,
    allPositiveMarginFeasible: cyclic.size === 0,
  };
}

export function analyzeInformation(examples) {
  assert.equal(examples.length, 32);
  return Object.fromEntries(
    views.map((view) => {
      const rows = examples.map((e) => {
        assert.equal(e.reference.complete, true);
        assert.equal(
          e.reference.qualityIndices.length,
          1,
          "Single-root only audit",
        );
        const values = inputRows(e, view).map((f) =>
          f.map((v) => {
            assert(Number.isFinite(v));
            return Math.fround(v);
          }),
        );
        return {
          complete: true,
          rootCount: values.length,
          keys: values.map(digest),
          target: e.reference.qualityIndices[0],
          ordered22Sha256: digest(values),
          qualityIndices: e.reference.qualityIndices,
        };
      });
      const bound = qualityInputBound(rows);
      let equalPairs = 0;
      let incompatiblePointwisePairs = 0;
      for (let i = 0; i < 32; i += 2) {
        if (!rankingFeasibility(rows.slice(i, i + 2)).allFirstIndexTop1Feasible)
          incompatiblePointwisePairs++;
        if (rows[i].ordered22Sha256 === rows[i + 1].ordered22Sha256)
          equalPairs++;
      }
      return [
        view,
        {
          orderedMatrixEqualPairs: equalPairs,
          incompatiblePointwisePairs,
          pairedPointwiseUpperBoundCorrect: 32 - incompatiblePointwisePairs,
          orderedMatrixBound: bound,
          pointwise: rankingFeasibility(rows),
        },
      ];
    }),
  );
}

export function runInformation(args) {
  assert.equal(args.length, 4);
  assert.equal(args[0], "--recorded");
  assert.equal(args[2], "--output");
  assert(args[1] && args[3]);
  const c = json(resolve(root, informationConfig)),
    recorded = resolve(root, args[1]);
  assert.equal(c.id, "lookahead-information-v1");
  assert.deepEqual(c.views, views);
  assert.equal(c.execution, "STORED_ANALYSIS_ONLY");
  for (const [p, h] of Object.entries(c.sourceHashes))
    assert.equal(sha(resolve(root, p)), h, p);
  for (const [p, h] of Object.entries(c.archiveHashes))
    assert.equal(sha(resolve(recorded, p)), h, p);
  const parent = auditPairsStored(recorded);
  const out = reserveFitOutput(args[3], root, recorded);
  const results = analyzeInformation(
    json(resolve(recorded, "build1.json")).examples,
  );
  const summary = {
    schemaVersion: 1,
    id: c.id,
    status: "PASS",
    states: 32,
    pairs: 16,
    results,
    limits: {
      informationIdentityIsNotNativeFloat32LogitIdentity: true,
      feasibleDoesNotProveNetworkCapacityOrOptimization: true,
      notAccuracyMeasurement: true,
      noIndependentGeneralization: true,
    },
    additional: {
      domain: 0,
      reference: 0,
      materialization: 0,
      python: 0,
      model: 0,
      load: 0,
      forward: 0,
      training: 0,
    },
  };
  write(resolve(out, "summary.json"), summary);
  write(resolve(out, "report.json"), {
    status: "PASS",
    atUtc: new Date().toISOString(),
    executor: "Codex",
    node: process.version,
    configSha256: sha(resolve(root, informationConfig)),
    archiveHashes: c.archiveHashes,
    parentSummarySha256: digest(parent),
    summarySha256: sha(resolve(out, "summary.json")),
  });
  return summary;
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
)
  console.log(JSON.stringify(runInformation(process.argv.slice(2))));
