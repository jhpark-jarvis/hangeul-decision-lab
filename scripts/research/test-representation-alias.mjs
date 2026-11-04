import assert from "node:assert/strict";
import test from "node:test";
import {
  fp32Key,
  aliasBounds,
  summarizeAliases,
  analyzeRepresentations,
  parseAliasArgs,
  storedMaskSurvivors,
} from "./representation-alias.mjs";
test("FP32 collisions and multiplicity survive decimal differences", () => {
  assert.equal(fp32Key([0.5]), fp32Key([0.5 + 1e-9]));
  const b = aliasBounds([[0.5], [0.5 + 1e-9], [0.2]], 1);
  assert.deepEqual(b.members, [0, 1]);
  assert.equal(b.multiplicity, 2);
  assert.equal(b.earlierDuplicate, true);
});
test("first-legal upper bound and margin feasibility are different requirements", () => {
  const first = aliasBounds([[0.1], [0.1], [0.2]], 0),
    later = aliasBounds([[0.1], [0.1], [0.2]], 1),
    s = summarizeAliases([first, later]);
  assert.equal(s.hardLabelTop1UpperBoundCorrect, 1);
  assert.equal(s.positiveMarginPossibleExamples, 0);
  assert.equal(s.meanIdealNllLowerBound, Math.log(2));
});
test("tied target probability is bounded by inverse multiplicity without a model", () => {
  const m = 3,
    nll = Math.log(3 * Math.exp(2) + Math.exp(-20)) - 2,
    b = aliasBounds([[0.1], [0.1], [0.1], [0.2]], 0);
  assert(nll >= Math.log(m));
  assert.equal(b.idealNllLowerBound, Math.log(m));
});
test("full74 can distinguish candidates that outcome22 merges", () => {
  const a = Array(74).fill(0.1),
    b = [...a];
  b[25] = 0.2;
  assert.equal(aliasBounds([a, b], 1).multiplicity, 1);
  assert.equal(aliasBounds([a.slice(52), b.slice(52)], 1).multiplicity, 2);
});
test("empty cohorts have explicit absent mean rather than a fictitious zero bound", () => {
  assert.equal(summarizeAliases([]).meanIdealNllLowerBound, null);
  assert.equal(summarizeAliases([]).examples, 0);
});
function fixture(ordinary) {
  const f = Array(74).fill(0);
  f[56] = 0.7;
  const g = [...f];
  f[25] = 0.1;
  g[25] = 0.2;
  const features = [f, g];
  if (ordinary) {
    const h = [...g];
    h[56] = 0.8;
    features.push(h);
  }
  const e = {
    id: "synthetic",
    fixtureId: "fixture",
    labels: { A512: 1, A1024: 1 },
    candidateKeys: [
      JSON.stringify(["single-cell", 0, 0]),
      JSON.stringify(["single-cell", 0, 1]),
      ...(ordinary
        ? [JSON.stringify(["place-piece", "x", 0, "DOT", 0, 0, [[true]]])]
        : []),
    ],
    candidateFeatures: features.map((f) => f.slice(0, 52)),
    state: {
      abilities: { singleCell: 1, reroll: 0 },
      remainingPieces: [{}, {}],
      hiddenItems: [{}, {}],
      pendingReroll: null,
    },
    teachers: {
      A512: {
        result: {
          search: { searchComplete: true },
          evaluation: { survivable: false, allCurrentPiecesPlaced: false },
        },
      },
    },
  };
  return { e, a: { id: e.id, candidateFeatures: features } };
}
test("monotone dead-end and live alias cohorts are kept separate", () => {
  for (const ordinary of [false, true]) {
    const { e, a } = fixture(ordinary),
      r = analyzeRepresentations([e], [a]);
    assert.equal(r.privateCases[0].monotoneDeadEndClass, !ordinary);
    assert.equal(r.privateCases[0].eligibleLiveAlias, ordinary);
    assert.equal(r.summary.point.outcome22.earlierDuplicateTargets, 1);
    assert.equal(r.summary.point.full74.aliasTargets, 0);
  }
});
test("bad prefix, order, labels and nonfinite vectors are rejected", () => {
  const { e, a } = fixture(false);
  for (const mutate of [
    (x) => (x.id = "bad"),
    (x) => (x.candidateFeatures[0][0] = 0.3),
    (x) => (x.candidateFeatures[0][60] = NaN),
    (x) => x.candidateFeatures.pop(),
  ]) {
    const bad = structuredClone(a);
    mutate(bad);
    assert.throws(() => analyzeRepresentations([e], [bad]));
  }
  assert.throws(() => aliasBounds([[0], [1]], 2));
  assert.throws(() => aliasBounds([[0], [1, 0]], 0));
  assert.throws(() => fp32Key([Infinity]));
});
test("CLI has no execution, tuning or test-split switches", () => {
  assert.deepEqual(
    parseAliasArgs(["--source", "s", "--outcome", "o", "--output", "x"]),
    { source: "s", outcome: "o", output: "x" },
  );
  assert.throws(() =>
    parseAliasArgs(["--source", "s", "--test", "o", "--output", "x"]),
  );
  assert.throws(() =>
    parseAliasArgs([
      "--source",
      "s",
      "--outcome",
      "o",
      "--output",
      "x",
      "--train",
    ]),
  );
});

test("frozen placement mask overlap distinguishes aliases without a state transition", () => {
  const keys = [
    ["single-cell", 0, 0],
    ["single-cell", 0, 1],
    ["place-piece", "x", 0, "DOT", 0, 0, [[true]]],
  ];
  assert.deepEqual(storedMaskSurvivors(keys, [0, 1]), [0, 1]);
  assert.throws(() => storedMaskSurvivors([["place-piece"]], [0]));
});
test("24-column coordinate projection is diagnostic only and removes distinct point ties", () => {
  const { e, a } = fixture(true);
  const r = analyzeRepresentations([e], [a]);
  assert.equal(r.summary.point.outcomePlusCoordinates24.aliasTargets, 0);
  assert.equal(r.summary.storedGeometry.mixedOrdinaryAvailability, 1);
  assert.equal(r.summary.storedGeometry.storedTeacherTerminalSurvivable, 0);
});
