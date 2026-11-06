import assert from "node:assert/strict";
import test from "node:test";
import {
  rankingFeasibility as rank,
  inputRows,
  runInformation,
  analyzeInformation,
} from "./lookahead-information.mjs";
test("Analysis adapter passes completeness and root count to ordered bounds", () => {
  const e = {
    reference: { complete: true, qualityIndices: [0] },
    observation: { planes: [Array(160).fill(0)], metadata: [0] },
    candidateFeatures: [Array(74).fill(0)],
    footprints: ["0".repeat(160)],
  };
  const rows = Array.from({ length: 32 }, () => e);
  const r = analyzeInformation(rows);
  assert.equal(
    r.outcome22.orderedMatrixBound.idealQualityMembershipUpperBoundCorrect,
    32,
  );
  assert.equal(r.fullSpatial.pairedPointwiseUpperBoundCorrect, 32);
  assert.throws(() =>
    analyzeInformation(
      rows.map((v) => ({
        ...v,
        reference: { ...v.reference, complete: false },
      })),
    ),
  );
});
test("SCC decision matches exhaustive scalar scores on small ordered lists", () => {
  const lists = [
    ["a", "b", "c"],
    ["b", "c", "a"],
    ["a", "a", "b"],
  ];
  for (const left of lists)
    for (const right of lists)
      for (let t = 0; t < 3; t++)
        for (let u = 0; u < 3; u++) {
          const examples = [
            { keys: left, target: t },
            { keys: right, target: u },
          ];
          let top1 = false,
            margin = false;
          for (let a = 0; a < 3; a++)
            for (let b = 0; b < 3; b++)
              for (let c = 0; c < 3; c++) {
                const scores = { a, b, c };
                top1 ||= examples.every(
                  (e) =>
                    e.keys.findIndex(
                      (k) =>
                        scores[k] === Math.max(...e.keys.map((k) => scores[k])),
                    ) === e.target,
                );
                margin ||= examples.every((e) =>
                  e.keys.every(
                    (k, j) =>
                      j === e.target || scores[e.keys[e.target]] > scores[k],
                  ),
                );
              }
          const actual = rank(examples);
          assert.equal(actual.allFirstIndexTop1Feasible, top1);
          assert.equal(actual.allPositiveMarginFeasible, margin);
        }
});
test("Acyclic strict orders admit all positive margins", () => {
  const r = rank([
    { keys: ["a", "b"], target: 0 },
    { keys: ["b", "c"], target: 0 },
  ]);
  assert(r.allFirstIndexTop1Feasible);
  assert(r.allPositiveMarginFeasible);
});
test("Later repeated target cannot beat identical earlier input", () => {
  const r = rank([{ keys: ["a", "a"], target: 1 }]);
  assert(!r.allFirstIndexTop1Feasible);
  assert.equal(r.positiveNegativeAliasObservations, 1);
});
test("Earlier repeated target permits a tie but not a positive margin", () => {
  const r = rank([{ keys: ["a", "a"], target: 0 }]);
  assert(r.allFirstIndexTop1Feasible);
  assert(!r.allPositiveMarginFeasible);
});
test("Cross-observation strict cycle rejects locally separable rows", () => {
  const r = rank([
    { keys: ["a", "b"], target: 1 },
    { keys: ["b", "a"], target: 1 },
  ]);
  assert(!r.allFirstIndexTop1Feasible);
  assert.equal(r.positiveNegativeAliasObservations, 0);
});
test("Weak cycle is consistent only with first-index ties", () => {
  const r = rank([
    { keys: ["a", "b"], target: 0 },
    { keys: ["b", "a"], target: 0 },
  ]);
  assert(r.allFirstIndexTop1Feasible);
  assert(!r.allPositiveMarginFeasible);
});
test("Indirect cycle with one strict edge is inconsistent", () => {
  const r = rank([
    { keys: ["a", "b"], target: 0 },
    { keys: ["b", "c"], target: 0 },
    { keys: ["a", "c"], target: 1 },
  ]);
  assert(!r.allFirstIndexTop1Feasible);
});
test("Constraint audit rejects malformed targets and empty data", () => {
  assert.throws(() => rank([]));
  assert.throws(() => rank([{ keys: ["a"], target: 1 }]));
});
test("Masked view clears only occupancy and preserves model inputs", () => {
  const e = {
    observation: {
      planes: [Array(160).fill(1), Array(160).fill(0), Array(160).fill(0)],
      metadata: [9],
    },
    candidateFeatures: [Array(74).fill(2)],
    footprints: ["1" + "0".repeat(159)],
  };
  const original = JSON.stringify(e),
    m = inputRows(e, "maskedSpatial")[0],
    f = inputRows(e, "fullSpatial")[0];
  assert.equal(m.length, 715);
  assert.deepEqual(m.slice(0, 160), Array(160).fill(0));
  assert.deepEqual(f.slice(0, 160), Array(160).fill(1));
  assert.deepEqual(m.slice(160), f.slice(160));
  assert.equal(JSON.stringify(e), original);
});
test("Unknown views and malformed CLI reject before archive use", () => {
  assert.throws(() => inputRows({}, "unknown"));
  assert.throws(() => runInformation(["--train"]));
});
