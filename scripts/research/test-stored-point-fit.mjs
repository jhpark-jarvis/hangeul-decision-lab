import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  readFileSync,
  writeFileSync,
  mkdirSync,
  mkdtempSync,
  existsSync,
  readdirSync,
} from "node:fs";
import { resolve } from "node:path";
import {
  ROOT,
  MODELS,
  validateCase,
  rankBucket,
  cohort,
  summarizeFit,
  loadSaved,
  execute,
  parseArgs,
} from "./analyze-stored-point-fit.mjs";
import { privateCases, summarizeRows } from "./compare-four-objectives.mjs";
const copy = (x) => structuredClone(x);
const sha = (p) => createHash("sha256").update(readFileSync(p)).digest("hex");
const write = (p, x) => writeFileSync(p, JSON.stringify(x, null, 2) + "\n");
function row(
  id,
  fixtureId,
  {
    forced = false,
    kind = true,
    oracle = false,
    rank = 2,
    family = "sparse",
    complete = false,
    teacher = "single-cell",
  } = {},
) {
  const correct = kind && oracle;
  const selected = { kind: kind ? teacher : "place-piece" };
  return {
    id,
    fixtureId,
    family,
    teacherKind: teacher,
    searchComplete: complete,
    phase: "await-next-pieces",
    legalKinds: forced ? "single-cell" : "place-piece+single-cell+reroll",
    candidates: 20,
    teacherAction: { kind: teacher },
    correctnessPattern: correct ? "1111" : "0000",
    models: Object.fromEntries(
      MODELS.map((n) => [
        n,
        {
          error: correct
            ? "exact-label"
            : kind
              ? "different-position"
              : "wrong-kind",
          selected: { ...selected },
          selectedIndex: 0,
          correct,
          kindCorrect: kind,
          oracleCorrect: oracle,
          fullRank: correct ? 1 : oracle ? 2 : rank,
          fullNll: 2,
          oracleRank: oracle ? 1 : rank,
          oracleNll: 1,
          margin: 0.1,
        },
      ]),
    ),
  };
}
function fixture() {
  const cases = {
    train: [
      row("a", "f1", { forced: true, rank: 8 }),
      row("b", "f1", { kind: false, oracle: true }),
      row("c", "f2", { complete: true }),
      row("d", "f3", { oracle: true, family: "pressure", complete: true }),
      row("e", "f4", { teacher: "place-piece", oracle: true }),
      row("f", "f5", { teacher: "reroll", oracle: true }),
    ],
    dev: [
      row("g", "f6", { forced: true, rank: 12 }),
      row("h", "f7", {
        kind: false,
        rank: 6,
        family: "pressure",
        complete: true,
      }),
    ],
  };
  const all = (n) => Object.fromEntries(MODELS.map((m) => [m, n]));
  const upstream = {
    status: "PASS",
    selectedEpochs: { control: 11, combined: 12, pointOnly: 11, kindOnly: 12 },
    additionalCompute: {
      modelCreation: 0,
      checkpointDeserialization: 0,
      forward: 0,
      training: 0,
      teacherDomainEpisodeTestRollout: 0,
    },
    splits: {
      train: {
        examples: 6,
        actualCorrect: all(3),
        byTeacherKind: {
          "place-piece": { examples: 1, actual: all(1) },
          "single-cell": { examples: 4, actual: all(1) },
          reroll: { examples: 1, actual: all(1) },
        },
        point: {
          examples: 4,
          joint: all({
            examples: 4,
            bothCorrect: 1,
            kindOnly: 2,
            coordinateOnly: 1,
            neither: 0,
          }),
          singleKindOnly: {
            examples: 1,
            models: all({ actualCorrect: 0, kindCorrect: 1, oracleCorrect: 0 }),
          },
        },
      },
      dev: {
        examples: 2,
        actualCorrect: all(0),
        byTeacherKind: {
          "place-piece": { examples: 0, actual: all(0) },
          "single-cell": { examples: 2, actual: all(0) },
          reroll: { examples: 0, actual: all(0) },
        },
        point: {
          examples: 2,
          joint: all({
            examples: 2,
            bothCorrect: 0,
            kindOnly: 1,
            coordinateOnly: 0,
            neither: 1,
          }),
          singleKindOnly: {
            examples: 1,
            models: all({ actualCorrect: 0, kindCorrect: 1, oracleCorrect: 0 }),
          },
        },
      },
    },
  };
  return { cases, upstream };
}
function diskFixture({ changed = false } = {}) {
  const base = resolve(ROOT, ".research-output");
  mkdirSync(base, { recursive: true });
  const root = mkdtempSync(resolve(base, "point-fit-test-")),
    input = resolve(root, "input");
  for (const d of [
    input,
    resolve(root, "research/experiments"),
    resolve(root, "assets/research"),
    resolve(root, "scripts/research"),
  ])
    mkdirSync(d, { recursive: true });
  const { cases, upstream } = fixture();
  const rawRow = (r, n) => {
    const m = r.models[n];
    return {
      id: r.id,
      fixtureId: r.fixtureId,
      family: r.family,
      error: m.error,
      teacherKind: r.teacherKind,
      selectedKind: m.selected.kind,
      selectedIndex: m.selectedIndex,
      globalCorrect: m.correct,
      searchComplete: r.searchComplete,
      teacherFinalPhase: r.phase,
      legalKinds: r.legalKinds,
      conditional: {
        teacherKind: r.teacherKind,
        teacherKindCandidates: r.candidates,
        oracleIndex: 0,
        oracleRank: m.oracleRank,
        oracleNll: m.oracleCorrect ? -0 : m.oracleNll,
        oracleCorrect: m.oracleCorrect,
      },
      fullScore: {
        id: r.id,
        fixtureId: r.fixtureId,
        family: r.family,
        index: m.selectedIndex,
        selected: m.selected,
        candidates: r.candidates,
        teacherActions: { A512: r.teacherAction },
        firstLegalAction: { kind: "single-cell" },
        targets: {
          A512: { rank: m.fullRank, nll: m.fullNll, correct: m.correct },
        },
        margin: m.margin,
      },
    };
  };
  const rawByModel = Object.fromEntries(
    MODELS.map((n) => [
      n,
      Object.fromEntries(
        ["train", "dev"].map((s) => [s, cases[s].map((r) => rawRow(r, n))]),
      ),
    ]),
  );
  if (changed) {
    rawByModel.combined.train[1].selectedIndex = 1;
    rawByModel.combined.train[1].fullScore.index = 1;
  }
  const selectedChanges = {};
  for (const s of ["train", "dev"]) {
    const group = Object.fromEntries(MODELS.map((n) => [n, rawByModel[n][s]]));
    selectedChanges[s] = privateCases(group);
    upstream.splits[s] = summarizeRows(group);
  }
  const archives = {};
  for (const n of ["combined", "pointOnly", "kindOnly"]) {
    archives[n] = resolve(input, n);
    mkdirSync(archives[n]);
    write(resolve(archives[n], "raw.json"), {
      control: rawByModel.control,
      candidate: rawByModel[n],
    });
    const path = resolve(archives[n], "raw.json");
    writeFileSync(
      path,
      readFileSync(path, "utf8").replaceAll(
        '"oracleNll": 0',
        '"oracleNll": -0.0',
      ),
    );
  }
  write(resolve(input, "cases.json"), selectedChanges);
  write(resolve(input, "summary.json"), upstream);
  const report = {
    status: "PASS",
    casesSha256: sha(resolve(input, "cases.json")),
    summarySha256: sha(resolve(input, "summary.json")),
    additionalCompute: upstream.additionalCompute,
  };
  write(resolve(input, "report.json"), report);
  write(resolve(root, "assets/research/upstream.json"), upstream);
  writeFileSync(
    resolve(root, "scripts/research/analyze-stored-point-fit.mjs"),
    "// Synthetic test identity only.\n",
  );
  const config = {
    schemaVersion: 1,
    id: "stored-point-fit-v1",
    method: "Synthetic test only",
    inputFiles: Object.fromEntries(
      ["cases.json", "summary.json", "report.json"].map((p) => [
        p,
        sha(resolve(input, p)),
      ]),
    ),
    rawInputs: Object.fromEntries(
      Object.entries(archives).map(([n, p]) => [
        n,
        sha(resolve(p, "raw.json")),
      ]),
    ),
    upstreamAsset: {
      path: "assets/research/upstream.json",
      sha256: sha(resolve(root, "assets/research/upstream.json")),
    },
    splits: { train: 6, dev: 2 },
    pointExamples: { train: 4, dev: 2 },
    models: MODELS,
    selectedEpochs: upstream.selectedEpochs,
    rankBuckets: ["1", "2-5", "6-10", ">10"],
    boundaries: {
      additionalModelCreationLoadForwardTraining: 0,
      teacherDomainEpisodeTestRollout: 0,
      newScoreSelectionLabelOrPolicy: false,
      publicAggregateOnly: true,
      interpretation: "Synthetic fixture",
      archiveVerification: "Synthetic fixture",
    },
  };
  write(resolve(root, "research/experiments/stored-point-fit-v1.json"), config);
  return { root, input, config, ...archives };
}

test("hand-counted joint and forced/multiple partitions differ from oracle-only success", () => {
  const { cases, upstream } = fixture(),
    s = summarizeFit(cases, upstream);
  const m = s.train.point.models.control;
  assert.deepEqual(m.joint, {
    bothCorrect: 1,
    kindOnly: 2,
    coordinateOnly: 1,
    neither: 0,
  });
  assert.equal(m.actualCorrect, 1);
  assert.equal(m.kindCorrect, 3);
  assert.equal(m.oracleCorrect, 2);
  assert.equal(s.train.byLegality.pointOnlyLegal.examples, 1);
  assert.equal(
    s.train.byLegality.multipleLegalKinds.models.control.kindCorrect,
    2,
  );
  assert.equal(
    s.train.byLegality.multipleLegalKinds.models.control.actualCorrect,
    1,
  );
  assert.equal(s.dev.point.models.control.actualCorrect, 0);
});
test("fixture counts retain repeated observations without treating them as independent fixtures", () => {
  const { cases, upstream } = fixture(),
    s = summarizeFit(cases, upstream);
  assert.equal(s.train.point.examples, 4);
  assert.equal(s.train.point.distinctFixtures, 3);
  assert.deepEqual(s.train.point.observationsPerFixtureHistogram, {
    1: 2,
    2: 1,
  });
  assert.equal(s.train.allExamples, 6);
  assert.equal(s.train.allDistinctFixtures, 5);
});
test("all separate partition axes and crossed legality/search cells conserve counts", () => {
  const { cases, upstream } = fixture(),
    s = summarizeFit(cases, upstream);
  for (const split of ["train", "dev"]) {
    const x = s[split];
    for (const axis of ["byLegality", "byFamily", "byTeacherSearch"])
      assert.equal(
        Object.values(x[axis]).reduce((a, c) => a + c.examples, 0),
        x.point.examples,
      );
    assert.equal(
      Object.values(x.legalityByTeacherSearch)
        .flatMap(Object.values)
        .reduce((a, c) => a + c.examples, 0),
      x.point.examples,
    );
  }
  assert.equal(
    s.train.legalityByTeacherSearch.multipleLegalKinds.complete.examples,
    2,
  );
});
test("rank buckets are exhaustive at fixed inclusive boundaries", () => {
  assert.deepEqual([1, 2, 5, 6, 10, 11, 4096].map(rankBucket), [
    "1",
    "2-5",
    "2-5",
    "6-10",
    "6-10",
    ">10",
    ">10",
  ]);
  for (const v of [0, -1, 1.1, NaN, Infinity, 4097])
    assert.throws(() => rankBucket(v));
  const { cases } = fixture();
  assert.deepEqual(
    cohort(cases.train.slice(0, 4)).models.control.oracleRankBuckets,
    { 1: 2, "2-5": 1, "6-10": 1, ">10": 0 },
  );
});
test("empty cohort reports zero counts without fabricating percentages", () => {
  const c = cohort([]);
  assert.equal(c.examples, 0);
  assert.equal(c.distinctFixtures, 0);
  assert.deepEqual(c.observationsPerFixtureHistogram, {});
  assert.equal(c.models.control.actualCorrect, 0);
  assert.equal(
    Object.values(c.models.control.oracleRankBuckets).reduce(
      (a, b) => a + b,
      0,
    ),
    0,
  );
});
test("duplicate IDs and cross-split fixture overlap are rejected", () => {
  for (const mutate of [
    (x) => (x.train[1].id = x.train[0].id),
    (x) => (x.dev[0].fixtureId = x.train[0].fixtureId),
  ]) {
    const { cases, upstream } = fixture();
    mutate(cases);
    assert.throws(() => summarizeFit(cases, upstream));
  }
});
test("wrong upstream denominators, accuracy, joint and forced counts fail", () => {
  for (const mutate of [
    (u) => u.splits.train.examples++,
    (u) => u.splits.train.actualCorrect.control++,
    (u) => u.splits.train.byTeacherKind["single-cell"].actual.control++,
    (u) => u.splits.train.point.joint.control.neither++,
    (u) => u.splits.train.point.singleKindOnly.examples++,
  ]) {
    const { cases, upstream } = fixture();
    mutate(upstream);
    assert.throws(() => summarizeFit(cases, upstream));
  }
});
test("inconsistent actual/oracle/kind and correctness pattern fail", () => {
  for (const mutate of [
    (r) => (r.models.control.correct = true),
    (r) => (r.models.control.oracleRank = 1),
    (r) => (r.models.control.kindCorrect = false),
    (r) => (r.correctnessPattern = "1111"),
    (r) => (r.models.control.selected.kind = "reroll"),
  ]) {
    const r = fixture().cases.train[0];
    mutate(r);
    assert.throws(() => validateCase(r));
  }
});
test("invalid rank, nonfinite values, legality and extra fields fail", () => {
  for (const mutate of [
    (r) => (r.models.control.oracleRank = 4097),
    (r) => (r.models.control.fullNll = NaN),
    (r) => (r.models.control.oracleNll = -1),
    (r) => (r.models.control.margin = Infinity),
    (r) => (r.legalKinds = "single-cell+single-cell"),
    (r) => (r.family = "other"),
    (r) => (r.searchComplete = "yes"),
    (r) => (r.candidates = 0),
    (r) => (r.extra = "private"),
    (r) => (r.models.control.extra = 1),
  ]) {
    const r = fixture().cases.train[0];
    mutate(r);
    assert.throws(() => validateCase(r));
  }
});
test("public summary contains no identifiers, actions or coordinate fields", () => {
  const { cases, upstream } = fixture(),
    s = summarizeFit(cases, upstream);
  function visit(o) {
    for (const [k, v] of Object.entries(o)) {
      assert(
        ![
          "id",
          "fixtureId",
          "teacherAction",
          "selected",
          "row",
          "col",
          "selectedIndex",
        ].includes(k),
      );
      if (v && typeof v === "object") visit(v);
    }
  }
  visit(s);
  const text = JSON.stringify(s);
  for (const id of ["f1", "f2", "f3", "f4", "f5", "f6", "f7"])
    assert(!text.includes('"' + id + '"'));
});
test("four models have independent counts, never pooled into a sample multiplier", () => {
  const { cases, upstream } = fixture();
  const m = cases.train[3].models.kindOnly;
  m.correct = false;
  m.kindCorrect = false;
  m.selected.kind = "place-piece";
  m.error = "wrong-kind";
  m.fullRank = 2;
  cases.train[3].correctnessPattern = "1110";
  // Rebuild expected mutations without sharing fixture helper object references.
  const u = copy(upstream);
  u.splits.train.actualCorrect.kindOnly = 2;
  u.splits.train.byTeacherKind["single-cell"].actual.kindOnly = 0;
  u.splits.train.point.joint.kindOnly = {
    examples: 4,
    bothCorrect: 0,
    kindOnly: 2,
    coordinateOnly: 2,
    neither: 0,
  };
  const s = summarizeFit(cases, u);
  assert.equal(s.train.point.examples, 4);
  assert.equal(s.train.point.models.kindOnly.actualCorrect, 0);
  assert.equal(s.train.point.models.control.actualCorrect, 1);
});
test("CLI accepts exactly five path options and rejects training, override and duplicate arguments", () => {
  assert.deepEqual(
    parseArgs([
      "--output",
      "o",
      "--input",
      "i",
      "--combined",
      "a",
      "--point-only",
      "b",
      "--kind-only",
      "c",
    ]),
    {
      input: "i",
      output: "o",
      combined: "a",
      pointOnly: "b",
      kindOnly: "c",
    },
  );
  for (const a of [
    [],
    ["--input", "i"],
    ["--input", "i", "--input", "j"],
    ["--train", "12", "--output", "o"],
    ["--input", "i", "--output", "o", "--scale", "2"],
  ])
    assert.throws(() => parseArgs(a));
});
test("pinned data-only input succeeds and altered bytes are rejected before aggregation", () => {
  const f = diskFixture(),
    { root, input, config } = f;
  assert.equal(loadSaved(input, config, root, f).cases.train.length, 6);
  writeFileSync(resolve(input, "cases.json"), "{}\n");
  assert.throws(() => loadSaved(input, config, root, f), /frozen hash drift/);
});
test("two exclusive executions are byte-exact, record zero compute and keep inputs intact", () => {
  const f = diskFixture(),
    { root, input, config } = f;
  for (const name of ["a", "b"])
    execute({ ...f, input, output: ".research-output/" + name }, root, {
      node: process.version,
    });
  assert.equal(
    readFileSync(resolve(root, ".research-output/a/summary.json"), "utf8"),
    readFileSync(resolve(root, ".research-output/b/summary.json"), "utf8"),
  );
  assert.deepEqual(readdirSync(resolve(root, ".research-output/a")).sort(), [
    "report.json",
    "start.json",
    "summary.json",
  ]);
  assert(
    Object.values(
      JSON.parse(readFileSync(resolve(root, ".research-output/a/report.json")))
        .additionalCompute,
    ).every((n) => n === 0),
  );
  for (const [p, h] of Object.entries(config.inputFiles))
    assert.equal(sha(resolve(input, p)), h);
  const before = sha(resolve(root, ".research-output/a/summary.json"));
  assert.throws(() =>
    execute({ ...f, input, output: ".research-output/a" }, root),
  );
  assert.equal(sha(resolve(root, ".research-output/a/summary.json")), before);
});
test("failed execution preserves input and writes failure without a success summary", () => {
  const f = diskFixture(),
    { root, input } = f;
  writeFileSync(resolve(input, "cases.json"), "{}\n");
  const before = sha(resolve(input, "cases.json"));
  assert.throws(
    () => execute({ ...f, input, output: ".research-output/fail" }, root),
    /frozen hash drift/,
  );
  assert(existsSync(resolve(root, ".research-output/fail/failure.json")));
  assert(!existsSync(resolve(root, ".research-output/fail/summary.json")));
  assert.equal(sha(resolve(input, "cases.json")), before);
});
test("output escape and input nesting are rejected before writes", () => {
  const f = diskFixture(),
    { root, input } = f;
  for (const output of [
    "outside",
    "../escape",
    input,
    resolve(input, "nested"),
  ])
    assert.throws(() => execute({ ...f, input, output }, root));
  assert(!existsSync(resolve(input, "nested")));
  assert(!existsSync(resolve(root, "outside")));
});
test("partial selected-change cases are never mistaken for all observations", () => {
  const f = diskFixture();
  assert.deepEqual(JSON.parse(readFileSync(resolve(f.input, "cases.json"))), {
    train: [],
    dev: [],
  });
  const loaded = loadSaved(f.input, f.config, f.root, f);
  assert.equal(loaded.cases.train.length, 6);
  assert.equal(loaded.cases.dev.length, 2);
  assert.equal(
    summarizeFit(loaded.cases, loaded.upstream).train.point.examples,
    4,
  );
});
test("raw byte drift and common-control identity drift fail with original files preserved", () => {
  const f = diskFixture(),
    p = resolve(f.kindOnly, "raw.json"),
    r = JSON.parse(readFileSync(p));
  r.control.train[0].id = "drift";
  write(p, r);
  assert.throws(
    () => loadSaved(f.input, f.config, f.root, f),
    /raw frozen hash drift/,
  );
  const config = copy(f.config);
  config.rawInputs.kindOnly = sha(p);
  assert.throws(() => loadSaved(f.input, config, f.root, f));
});
test("JSON-normalized partial cases and aggregates preserve raw negative-zero NLL bytes", () => {
  const f = diskFixture({ changed: true }),
    p = resolve(f.combined, "raw.json"),
    before = sha(p),
    raw = JSON.parse(readFileSync(p));
  assert(Object.is(raw.candidate.train[1].conditional.oracleNll, -0));
  assert.equal(
    JSON.parse(readFileSync(resolve(f.input, "cases.json"))).train.length,
    1,
  );
  const loaded = loadSaved(f.input, f.config, f.root, f);
  assert.equal(loaded.cases.train.length, 6);
  assert.equal(sha(p), before);
});
