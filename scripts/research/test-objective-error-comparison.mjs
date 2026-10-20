import test from "node:test";
import assert from "node:assert/strict";
import {
  mkdirSync,
  mkdtempSync,
  writeFileSync,
  readFileSync,
  existsSync,
} from "node:fs";
import { resolve } from "node:path";
import { createHash } from "node:crypto";
import {
  ROOT,
  validateRow,
  alignRows,
  summarizeRows,
  privateCases,
  compareArchives,
  loadArchive,
  execute,
} from "./compare-objective-errors.mjs";

function row(
  id,
  correct,
  kind = "place-piece",
  kindOk = true,
  oracle = correct,
) {
  const selectedKind = kindOk ? kind : "reroll",
    selectedIndex = correct ? 0 : kindOk ? 1 : 2;
  const action = { kind, row: 0, col: 0 },
    selected = { kind: selectedKind, row: 0, col: selectedIndex };
  return {
    id,
    fixtureId: "private-fixture",
    family: "synthetic",
    teacherKind: kind,
    selectedKind,
    searchComplete: true,
    teacherFinalPhase: "playing",
    legalKinds: "place-piece+single-cell+reroll",
    error: correct
      ? "exact-label"
      : kindOk
        ? "different-position"
        : "wrong-kind",
    globalCorrect: correct,
    selectedIndex,
    conditional: {
      teacherKind: kind,
      teacherKindCandidates: 3,
      oracleCorrect: oracle,
      oracleIndex: oracle ? 0 : 1,
      oracleRank: oracle ? 1 : 2,
      oracleNll: oracle ? 0.1 : 1,
    },
    fullScore: {
      id,
      fixtureId: "private-fixture",
      family: "synthetic",
      candidates: 4,
      index: selectedIndex,
      selected,
      teacherActions: { A512: action, A1024: action },
      firstLegalAction: action,
      margin: 0.2,
      targets: {
        A512: { correct, rank: correct ? 1 : 2, nll: correct ? 0.1 : 1 },
      },
    },
  };
}
function eight() {
  return Object.fromEntries(
    ["control", "combined", "pointOnly"].map((m, col) => [
      m,
      Array.from({ length: 8 }, (_, i) =>
        row(`private-case-${i}`, i.toString(2).padStart(3, "0")[col] === "1"),
      ),
    ]),
  );
}
function fixtureRoot() {
  const base = resolve(ROOT, ".research-output");
  mkdirSync(base, { recursive: true });
  return mkdtempSync(resolve(base, "objective-errors-test-"));
}

test("all eight correctness patterns and overlapping gains/regressions", () => {
  const r = summarizeRows(eight());
  assert(Object.values(r.correctnessPatterns).every((n) => n === 1));
  assert.equal(r.examples, 8);
  assert.equal(r.sharedFailures.examples, 1);
  assert(Object.values(r.overlap).every((n) => n === 1));
  assert.deepEqual(r.pairs["control:combined"].actual, {
    examples: 8,
    bothCorrect: 2,
    gain: 2,
    regression: 2,
    bothWrong: 2,
    netGain: 0,
  });
});

test("error transitions conserve examples and preserve wrong-wrong selection changes", () => {
  const g = eight();
  g.pointOnly[0].selectedIndex = 3;
  g.pointOnly[0].fullScore.index = 3;
  const r = summarizeRows(g);
  for (const p of Object.values(r.pairs))
    assert.equal(
      Object.values(p.errorTransitions).reduce((a, b) => a + b, 0),
      8,
    );
  assert(r.pairs["combined:pointOnly"].changedButBothWrong > 0);
  assert.equal(
    Object.values(r.failureCohorts["control:combined"].regressions).reduce(
      (a, b) => a + b,
      0,
    ),
    2,
  );
});

test("point kind/oracle intersection is distinct from marginal recalls", () => {
  const g = {
    control: [row("p", false, "single-cell", true, false)],
    combined: [row("p", false, "single-cell", false, true)],
    pointOnly: [row("p", true, "single-cell", true, true)],
  };
  const r = summarizeRows(g);
  assert.equal(r.point.joint.control.kindOnly, 1);
  assert.equal(r.point.joint.combined.coordinateOnly, 1);
  assert.equal(r.point.joint.pointOnly.bothCorrect, 1);
  assert.deepEqual(r.point.kindPatterns, { 101: 1 });
  assert.deepEqual(r.point.oraclePatterns, { "011": 1 });
});

test("ID/order/label/candidate/meta drift and duplicates are rejected", () => {
  const mutations = [
    (g) => g.pointOnly.reverse(),
    (g) => {
      g.combined[0].id = "wrong";
    },
    (g) => {
      g.combined[0].fullScore.teacherActions.A512.row = 2;
    },
    (g) => {
      g.combined[0].fullScore.candidates = 5;
    },
    (g) => {
      g.combined[0].family = "other";
    },
    (g) => {
      g.control[1] = structuredClone(g.control[0]);
    },
  ];
  for (const mutate of mutations) {
    const g = eight();
    mutate(g);
    assert.throws(() => alignRows(g));
  }
});

test("inconsistent correctness/rank/error/kind and nonfinite scores are rejected", () => {
  const mutations = [
    (r) => {
      r.fullScore.targets.A512.nll = NaN;
    },
    (r) => {
      r.conditional.oracleRank = 1;
    },
    (r) => {
      r.globalCorrect = true;
    },
    (r) => {
      r.error = "wrong-kind";
    },
    (r) => {
      r.selectedIndex = 9;
    },
  ];
  for (const mutate of mutations) {
    const r = row("x", false);
    mutate(r);
    assert.throws(() => validateRow(r));
  }
});

test("single legal candidate preserves the original nullable margin", () => {
  const r = row("singleton", true);
  r.fullScore.candidates = 1;
  r.fullScore.margin = null;
  r.conditional.teacherKindCandidates = 1;
  validateRow(r);
  r.fullScore.margin = Infinity;
  assert.throws(() => validateRow(r));
});

test("private case details never enter public aggregate", () => {
  const g = eight(),
    detail = privateCases(g),
    summary = summarizeRows(g);
  assert(detail.length > 0 && detail[0].id.startsWith("private-"));
  assert(Object.hasOwn(detail[0].models.control, "selected"));
  const text = JSON.stringify(summary);
  assert(!text.includes("private-case") && !text.includes("private-fixture"));
  for (const key of [
    "selectedIndex",
    "teacherAction",
    "fixtureId",
    '"row"',
    '"col"',
  ])
    assert(!text.includes(key));
});

function fakeArchives() {
  const g = eight();
  const cfg = {
    source: { label: "fixed" },
    model: { parameters: 1 },
    inference: { selection: "fixed" },
    training: {
      seed: 1,
      epochs: 12,
      batchSize: 8,
      threads: 2,
      interopThreads: 1,
      optimizer: "Adam",
      learningRate: 0.001,
      weightDecay: 0.0001,
      selection: "first",
    },
    objective: { kindWeight: 0.5, pointWeight: 0.5 },
  };
  const control = {
    weightsSha256: "same",
    history: [],
    audit: {},
    selectedEpoch: 11,
    artifactHashes: { "predictions.json": "p", "logits.json": "l" },
  };
  const summary = {
    gate: { status: "FAIL" },
    mechanism: { status: "FAIL" },
    coordinateMechanism: { status: "FAIL" },
  };
  const a = {
    config: cfg,
    raw: { control: g.control, candidate: g.combined },
    arms: { control, candidate: { selectedEpoch: 12 } },
    summary,
  };
  const b = {
    config: structuredClone(cfg),
    raw: { control: structuredClone(g.control), candidate: g.pointOnly },
    arms: {
      control: structuredClone(control),
      candidate: { selectedEpoch: 11 },
    },
    summary,
  };
  b.config.objective.kindWeight = 0;
  for (const pack of [a, b])
    for (const arm of ["control", "candidate"])
      pack.raw[arm] = {
        train: pack.raw[arm],
        dev: structuredClone(pack.raw[arm]),
      };
  return [
    a,
    b,
    {
      id: "toy",
      splits: { train: 8, dev: 8 },
      archives: { combined: { hashes: {} }, pointOnly: { hashes: {} } },
      boundaries: { interpretation: "toy" },
    },
  ];
}

test("selected epochs are retained and common control is checked exactly", () => {
  const [a, b, c] = fakeArchives();
  assert.deepEqual(compareArchives(a, b, c).summary.selectedEpochs, {
    control: 11,
    combined: 12,
    pointOnly: 11,
  });
  b.arms.control.weightsSha256 = "changed";
  assert.throws(() => compareArchives(a, b, c));
});

test("changed training/input/model/selection contract is rejected", () => {
  for (const key of ["source", "model", "inference", "training"]) {
    const [a, b, c] = fakeArchives();
    b.config[key] = { changed: true };
    assert.throws(() => compareArchives(a, b, c));
  }
});

test("archive digest tampering and unexpected manifest paths are rejected before parsing", () => {
  const d = fixtureRoot(),
    hashes = {};
  for (const name of [
    "freeze.json",
    "execution.json",
    "report.json",
    "raw.json",
    "summary.json",
    "forward-count.jsonl",
    "control/report.json",
    "candidate/report.json",
  ]) {
    const p = resolve(d, name);
    mkdirSync(resolve(p, ".."), { recursive: true });
    writeFileSync(p, "{}");
    hashes[name] = createHash("sha256").update("{}").digest("hex");
  }
  writeFileSync(resolve(d, "raw.json"), "tampered");
  assert.throws(() => loadArchive(d, { hashes }), /Frozen raw.json changed/);
  assert.throws(() =>
    loadArchive(d, { hashes: { ...hashes, "../escape": "no" } }),
  );
});

test("exclusive failure output preserves original failure and has no compute ledger", () => {
  const d = fixtureRoot();
  mkdirSync(resolve(d, "research/experiments"), { recursive: true });
  writeFileSync(
    resolve(d, "research/experiments/objective-error-comparison-v1.json"),
    readFileSync(
      resolve(ROOT, "research/experiments/objective-error-comparison-v1.json"),
    ),
  );
  const opts = {
    combined: resolve(d, "missing-combined"),
    pointOnly: resolve(d, "missing-point"),
    output: ".research-output/out",
  };
  assert.throws(() => execute(opts, { fixture: "toy" }, d), /ENOENT/);
  const before = readFileSync(resolve(d, ".research-output/out/failure.json"));
  assert.throws(() => execute(opts, {}, d), /EEXIST/);
  assert.deepEqual(
    before,
    readFileSync(resolve(d, ".research-output/out/failure.json")),
  );
  assert(!existsSync(resolve(d, ".research-output/out/forward-count.jsonl")));
  assert.throws(
    () => execute({ ...opts, output: "outside" }, {}, d),
    /New owned/,
  );
});
