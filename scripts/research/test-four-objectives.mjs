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
  models,
  pairs,
  validateRow,
  alignRows,
  summarizeRows,
  privateCases,
  compareArchives,
  loadArchive,
  validateConfigInput,
  validateLedger,
  execute,
} from "./compare-four-objectives.mjs";
const digest = (p) =>
  createHash("sha256").update(readFileSync(p)).digest("hex");
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

function sixteen() {
  return Object.fromEntries(
    models.map((m, col) => [
      m,
      Array.from({ length: 16 }, (_, i) =>
        row("private-case-" + i, i.toString(2).padStart(4, "0")[col] === "1"),
      ),
    ]),
  );
}
function fixtureRoot() {
  const base = resolve(ROOT, ".research-output");
  mkdirSync(base, { recursive: true });
  return mkdtempSync(resolve(base, "four-objective-test-"));
}

test("changed compute or privacy boundary is rejected before archive reads", () => {
  const d = fixtureRoot(),
    path = resolve(d, "research/experiments/four-objective-errors-v1.json");
  mkdirSync(resolve(path, ".."), { recursive: true });
  const original = JSON.parse(
    readFileSync(
      resolve(ROOT, "research/experiments/four-objective-errors-v1.json"),
    ),
  );
  for (const [key, value] of [
    ["newModelLoadForwardTraining", 1],
    ["newScoreOrSelection", true],
    ["publicAggregateOnly", false],
  ]) {
    const changed = structuredClone(original);
    changed.boundaries[key] = value;
    writeFileSync(path, JSON.stringify(changed));
    assert.throws(() =>
      execute(
        {
          combined: "missing",
          pointOnly: "missing",
          kindOnly: "missing",
          kindConfig: "missing",
          output: ".research-output/rejected",
        },
        {},
        d,
      ),
    );
    assert(!existsSync(resolve(d, ".research-output/rejected")));
  }
});

test("sixteen exact correctness patterns and seven gain/regression signatures", () => {
  const r = summarizeRows(sixteen());
  assert.equal(r.examples, 16);
  assert.equal(Object.keys(r.correctnessPatterns).length, 16);
  assert(Object.values(r.correctnessPatterns).every((n) => n === 1));
  assert.equal(r.sharedFailures.examples, 1);
  assert.equal(Object.keys(r.pairs).length, 6);
  for (const sig of Object.values(r.overlap))
    assert(Object.values(sig).every((n) => n === 1));
  for (const pair of Object.values(r.pairs))
    assert.deepEqual(pair.actual, {
      examples: 16,
      bothCorrect: 4,
      gain: 4,
      regression: 4,
      bothWrong: 4,
      netGain: 0,
    });
});
test("all six transition tables and failure cohorts conserve pair denominators", () => {
  const g = sixteen();
  g.kindOnly[0].selectedIndex = 3;
  g.kindOnly[0].fullScore.index = 3;
  const r = summarizeRows(g);
  for (const key of pairs.map((p) => p.join(":"))) {
    const p = r.pairs[key];
    assert.equal(
      Object.values(p.errorTransitions).reduce((a, b) => a + b, 0),
      16,
    );
    assert.equal(
      Object.values(r.failureCohorts[key].gains).reduce((a, b) => a + b, 0),
      p.actual.gain,
    );
    assert.equal(
      Object.values(r.failureCohorts[key].regressions).reduce(
        (a, b) => a + b,
        0,
      ),
      p.actual.regression,
    );
  }
  assert(r.pairs["pointOnly:kindOnly"].changedButBothWrong > 0);
});
test("point actual is same-observation kind/oracle intersection", () => {
  const g = {
    control: [row("p", false, "single-cell", true, false)],
    combined: [row("p", false, "single-cell", false, true)],
    pointOnly: [row("p", true, "single-cell", true, true)],
    kindOnly: [row("p", false, "single-cell", false, false)],
  };
  const r = summarizeRows(g);
  assert.deepEqual(r.point.kindPatterns, { 1010: 1 });
  assert.deepEqual(r.point.oraclePatterns, { "0110": 1 });
  assert.equal(r.point.joint.kindOnly.neither, 1);
  assert.equal(r.point.joint.pointOnly.bothCorrect, 1);
});
test("single-legal-kind point subgroup keeps its own denominator", () => {
  const g = Object.fromEntries(
    models.map((m) => {
      const r = row("single", false, "single-cell", true, false);
      r.legalKinds = "single-cell";
      return [m, [r]];
    }),
  );
  const r = summarizeRows(g);
  assert.equal(r.point.singleKindOnly.examples, 1);
  for (const v of Object.values(r.point.singleKindOnly.models))
    assert.deepEqual(v, { actualCorrect: 0, kindCorrect: 1, oracleCorrect: 0 });
});
test("four-way identity/order/missing model/duplicate drift is rejected", () => {
  for (const mutate of [
    (g) => g.kindOnly.reverse(),
    (g) => g.kindOnly[0].fullScore.teacherActions.A512.row++,
    (g) => g.kindOnly[0].fullScore.candidates++,
    (g) => (g.combined[0].family = "other"),
    (g) => (g.control[1] = structuredClone(g.control[0])),
    (g) => delete g.kindOnly,
  ]) {
    const g = sixteen();
    mutate(g);
    assert.throws(() => alignRows(g));
  }
});
test("inconsistent rank/error/selection and nonfinite stored values are rejected", () => {
  for (const mutate of [
    (r) => (r.fullScore.targets.A512.nll = NaN),
    (r) => (r.conditional.oracleRank = 1),
    (r) => (r.error = "wrong-kind"),
    (r) => (r.globalCorrect = true),
    (r) => (r.selectedIndex = 99),
  ]) {
    const r = row("x", false);
    mutate(r);
    assert.throws(() => validateRow(r));
  }
});
test("single candidate retains nullable original margin", () => {
  const r = row("singleton", true);
  r.fullScore.candidates = 1;
  r.fullScore.margin = null;
  r.conditional.teacherKindCandidates = 1;
  validateRow(r);
  r.fullScore.margin = Infinity;
  assert.throws(() => validateRow(r));
});
test("public aggregate contains no observation IDs/actions/coordinates", () => {
  const g = sixteen(),
    detail = privateCases(g),
    text = JSON.stringify(summarizeRows(g));
  assert(detail.length > 0);
  assert(detail[0].id.startsWith("private-"));
  for (const value of [
    "private-case",
    "private-fixture",
    "selectedIndex",
    "teacherAction",
    "fixtureId",
    '"row"',
    '"col"',
  ])
    assert(!text.includes(value));
});
function archives() {
  const g = sixteen(),
    cfg = {
      source: { label: "fixed" },
      model: { parameters: 1 },
      inference: { selection: "raw" },
      training: {
        seed: 1,
        epochs: 12,
        batchSize: 8,
        threads: 2,
        interopThreads: 1,
        optimizer: "Adam",
        learningRate: 0.001,
        weightDecay: 0.0001,
        order: "fixed",
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
  const packs = ["combined", "pointOnly", "kindOnly"].map((m, i) => ({
    config: structuredClone(cfg),
    raw: {
      control: {
        train: structuredClone(g.control),
        dev: structuredClone(g.control),
      },
      candidate: { train: g[m], dev: structuredClone(g[m]) },
    },
    arms: {
      control: structuredClone(control),
      candidate: { selectedEpoch: [12, 11, 12][i] },
    },
    summary: { gate: { status: "FAIL" }, mechanism: { status: "FAIL" } },
    optimizedFlatTrace: [[1, 339, 2, 2]],
  }));
  packs[1].config.objective.kindWeight = 0;
  packs[2].config.objective.pointWeight = 0;
  return [
    ...packs,
    {
      id: "toy",
      splits: { train: 16, dev: 16 },
      archives: {
        combined: { hashes: {} },
        pointOnly: { hashes: {} },
        kindOnly: { hashes: {} },
      },
      boundaries: { interpretation: "toy" },
    },
  ];
}
test("original selected epochs and original gate failures are retained", () => {
  const [a, b, c, spec] = archives(),
    r = compareArchives(a, b, c, spec);
  assert.deepEqual(r.summary.selectedEpochs, {
    control: 11,
    combined: 12,
    pointOnly: 11,
    kindOnly: 12,
  });
  assert(
    Object.values(r.summary.originalDecisions).every(
      (p) => p.gate.status === "FAIL",
    ),
  );
  assert(Object.values(r.summary.additionalCompute).every((n) => n === 0));
});
test("all three controls weights/history/shuffle/rows/optimized flat trace must match", () => {
  for (const mutate of [
    (p) => (p.arms.control.weightsSha256 = "changed"),
    (p) => (p.arms.control.history = [1]),
    (p) => (p.arms.control.audit = { changed: true }),
    (p) => p.raw.control.dev.reverse(),
    (p) => p.optimizedFlatTrace[0][2]++,
  ]) {
    const [a, b, c, spec] = archives();
    mutate(c);
    assert.throws(() => compareArchives(a, b, c, spec));
  }
});
test("input/model/inference/training/loss changes rejected", () => {
  for (const key of ["source", "model", "inference", "training"]) {
    const [a, b, c, spec] = archives();
    c.config[key] = { changed: true };
    assert.throws(() => compareArchives(a, b, c, spec));
  }
  const [a, b, c, spec] = archives();
  c.config.objective.pointWeight = 0.5;
  assert.throws(() => compareArchives(a, b, c, spec));
});
function ledger() {
  let call = 0;
  return [
    ["preflight", { initial: 856 }],
    ["control", { train: 4068, "epoch-dev": 1068, frozen: 856 }],
    ["candidate", { train: 4068, "epoch-dev": 1068, frozen: 856 }],
  ].flatMap(([mode, limits]) =>
    Object.entries(limits).flatMap(([phase, n]) =>
      Array.from({ length: n }, (_, i) => ({
        call: ++call,
        limit: 12840,
        mode,
        phase,
        phaseCall: i + 1,
      })),
    ),
  );
}
test("stored global and phase ledger quotas and sequence validated without calls", () => {
  validateLedger(ledger());
  for (const mutate of [
    (r) => r.pop(),
    (r) => r[900].call++,
    (r) => r[900].phaseCall++,
    (r) => (r[900].mode = "candidate"),
    (r) => (r[900].phase = "frozen"),
    (r) => r[900].limit++,
  ]) {
    const r = ledger();
    mutate(r);
    assert.throws(() => validateLedger(r));
  }
});
test("explicit original config hash plus metadata-only public projection", () => {
  const d = fixtureRoot(),
    path = "config.json",
    original = resolve(d, "original.json"),
    published = resolve(d, path);
  const c = {
    authorization: "private metadata",
    objective: { kindWeight: 0.5, pointWeight: 0 },
  };
  writeFileSync(original, JSON.stringify(c));
  writeFileSync(
    published,
    JSON.stringify({ ...c, authorization: "public metadata" }),
  );
  const spec = {
    originalConfigRequired: true,
    configPath: path,
    configSha256: digest(original),
    exportedConfigSha256: digest(published),
  };
  assert.deepEqual(validateConfigInput(spec, d, original), c);
  assert.throws(() => validateConfigInput(spec, d), /Explicit original/);
  writeFileSync(
    published,
    JSON.stringify({
      ...c,
      authorization: "public",
      objective: { kindWeight: 1, pointWeight: 0 },
    }),
  );
  assert.throws(
    () =>
      validateConfigInput(
        { ...spec, exportedConfigSha256: digest(published) },
        d,
        original,
      ),
    /Only authorization/,
  );
  writeFileSync(original, "tampered");
  assert.throws(
    () => validateConfigInput(spec, d, original),
    /Frozen computational/,
  );
});
test("archive digest tampering and unexpected paths fail before parsing", () => {
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
    hashes[name] = digest(p);
  }
  writeFileSync(resolve(d, "raw.json"), "tampered");
  assert.throws(() => loadArchive(d, { hashes }), /Frozen raw.json changed/);
  assert.throws(() =>
    loadArchive(d, { hashes: { ...hashes, "../escape": "no" } }),
  );
});
test("exclusive failure output and outside/output-inside-input paths", () => {
  const d = fixtureRoot();
  mkdirSync(resolve(d, "research/experiments"), { recursive: true });
  writeFileSync(
    resolve(d, "research/experiments/four-objective-errors-v1.json"),
    readFileSync(
      resolve(ROOT, "research/experiments/four-objective-errors-v1.json"),
    ),
  );
  const opts = {
    combined: resolve(d, "missing-combined"),
    pointOnly: resolve(d, "missing-point"),
    kindOnly: resolve(d, "missing-kind"),
    kindConfig: resolve(d, "missing-config"),
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
  assert.throws(
    () =>
      execute(
        {
          ...opts,
          combined: resolve(d, ".research-output/out"),
          output: ".research-output/out/child",
        },
        {},
        d,
      ),
    /New owned/,
  );
});
