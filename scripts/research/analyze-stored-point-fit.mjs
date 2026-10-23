import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, mkdirSync, realpathSync } from "node:fs";
import { resolve, relative, isAbsolute, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import {
  alignRows,
  summarizeRows,
  privateCases,
} from "./compare-four-objectives.mjs";

export const ROOT = fileURLToPath(new URL("../../", import.meta.url));
export const MODELS = ["control", "combined", "pointOnly", "kindOnly"];
const CONFIG = "research/experiments/stored-point-fit-v1.json";
const KINDS = ["place-piece", "single-cell", "reroll"];
const ERRORS = [
  "exact-label",
  "wrong-kind",
  "different-target",
  "different-shape",
  "different-position",
];
const hash = (p) => createHash("sha256").update(readFileSync(p)).digest("hex");
const read = (p) => JSON.parse(readFileSync(p, "utf8"));
const save = (p, value) =>
  writeFileSync(p, JSON.stringify(value, null, 2) + "\n", { flag: "wx" });
const integer = (v, min, max) =>
  assert(Number.isSafeInteger(v) && v >= min && v <= max);
const keys = (o, expected) =>
  assert.deepEqual(Object.keys(o).sort(), [...expected].sort());
const inside = (parent, p) => {
  const r = relative(parent, p);
  return r !== "" && !r.startsWith("..") && !isAbsolute(r);
};
export function rankBucket(rank) {
  integer(rank, 1, 4096);
  return rank === 1 ? "1" : rank <= 5 ? "2-5" : rank <= 10 ? "6-10" : ">10";
}

export function validateCase(row) {
  keys(row, [
    "id",
    "fixtureId",
    "family",
    "teacherKind",
    "searchComplete",
    "phase",
    "legalKinds",
    "candidates",
    "teacherAction",
    "correctnessPattern",
    "models",
  ]);
  assert(typeof row.id === "string" && row.id.length > 0);
  assert(typeof row.fixtureId === "string" && row.fixtureId.length > 0);
  assert(["sparse", "pressure"].includes(row.family));
  assert.equal(typeof row.searchComplete, "boolean");
  assert(
    [
      "playing",
      "gameover",
      "await-next-pieces",
      "await-reroll-result",
    ].includes(row.phase),
  );
  assert(KINDS.includes(row.teacherKind));
  assert.equal(row.teacherAction.kind, row.teacherKind);
  const legal = row.legalKinds.split("+");
  assert(legal.length > 0 && new Set(legal).size === legal.length);
  assert(
    legal.every((k) => KINDS.includes(k)) && legal.includes(row.teacherKind),
  );
  integer(row.candidates, 1, 4096);
  keys(row.models, MODELS);
  for (const name of MODELS) {
    const m = row.models[name];
    keys(m, [
      "error",
      "selected",
      "selectedIndex",
      "correct",
      "kindCorrect",
      "oracleCorrect",
      "fullRank",
      "fullNll",
      "oracleRank",
      "oracleNll",
      "margin",
    ]);
    assert(ERRORS.includes(m.error));
    assert(legal.includes(m.selected.kind));
    for (const p of ["correct", "kindCorrect", "oracleCorrect"])
      assert.equal(typeof m[p], "boolean");
    integer(m.selectedIndex, 0, row.candidates - 1);
    integer(m.fullRank, 1, row.candidates);
    integer(m.oracleRank, 1, row.candidates);
    assert(m.fullRank >= m.oracleRank);
    assert.equal(m.fullRank === 1, m.correct);
    assert.equal(m.oracleRank === 1, m.oracleCorrect);
    assert.equal(m.selected.kind === row.teacherKind, m.kindCorrect);
    assert.equal(m.kindCorrect && m.oracleCorrect, m.correct);
    assert.equal(m.error === "exact-label", m.correct);
    assert.equal(m.error === "wrong-kind", !m.kindCorrect);
    for (const p of ["fullNll", "oracleNll"])
      assert(Number.isFinite(m[p]) && m[p] >= 0);
    if (row.candidates === 1) assert.equal(m.margin, null);
    else assert(Number.isFinite(m.margin));
    if (legal.length === 1) assert(m.kindCorrect);
  }
  assert.equal(
    row.correctnessPattern,
    MODELS.map((n) => Number(row.models[n].correct)).join(""),
  );
}

export function cohort(rows) {
  const fixtureCounts = new Map();
  for (const r of rows)
    fixtureCounts.set(r.fixtureId, (fixtureCounts.get(r.fixtureId) ?? 0) + 1);
  const multiplicity = {};
  for (const count of fixtureCounts.values())
    multiplicity[count] = (multiplicity[count] ?? 0) + 1;
  const result = {
    examples: rows.length,
    distinctFixtures: fixtureCounts.size,
    observationsPerFixtureHistogram: multiplicity,
    models: {},
  };
  for (const name of MODELS) {
    const count = (f) => rows.filter((r) => f(r.models[name])).length;
    const ranks = (key) =>
      Object.fromEntries(
        ["1", "2-5", "6-10", ">10"].map((b) => [
          b,
          count((m) => rankBucket(m[key]) === b),
        ]),
      );
    result.models[name] = {
      actualCorrect: count((m) => m.correct),
      kindCorrect: count((m) => m.kindCorrect),
      oracleCorrect: count((m) => m.oracleCorrect),
      joint: {
        bothCorrect: count((m) => m.kindCorrect && m.oracleCorrect),
        kindOnly: count((m) => m.kindCorrect && !m.oracleCorrect),
        coordinateOnly: count((m) => !m.kindCorrect && m.oracleCorrect),
        neither: count((m) => !m.kindCorrect && !m.oracleCorrect),
      },
      fullRankBuckets: ranks("fullRank"),
      oracleRankBuckets: ranks("oracleRank"),
    };
  }
  return result;
}

export function summarizeFit(cases, upstream) {
  keys(cases, ["train", "dev"]);
  const seen = new Set(),
    fixtureSplits = new Map(),
    splits = {};
  for (const split of ["train", "dev"]) {
    const rows = cases[split];
    assert(Array.isArray(rows) && rows.length > 0);
    assert.equal(rows.length, upstream.splits[split].examples);
    for (const row of rows) {
      validateCase(row);
      assert(!seen.has(row.id), "Duplicate observation");
      seen.add(row.id);
      const previous = fixtureSplits.get(row.fixtureId);
      assert(
        previous === undefined || previous === split,
        "Fixture overlaps splits",
      );
      fixtureSplits.set(row.fixtureId, split);
    }
    const total = cohort(rows),
      pointRows = rows.filter((r) => r.teacherKind === "single-cell");
    for (const name of MODELS)
      assert.equal(
        total.models[name].actualCorrect,
        upstream.splits[split].actualCorrect[name],
      );
    for (const kind of KINDS) {
      const selected = rows.filter((r) => r.teacherKind === kind),
        expected = upstream.splits[split].byTeacherKind[kind];
      assert.equal(selected.length, expected.examples);
      for (const name of MODELS)
        assert.equal(
          selected.filter((r) => r.models[name].correct).length,
          expected.actual[name],
        );
    }
    const point = cohort(pointRows);
    assert.equal(point.examples, upstream.splits[split].point.examples);
    for (const name of MODELS) {
      const j = upstream.splits[split].point.joint[name];
      const { examples, ...counts } = j;
      assert.equal(examples, point.examples);
      assert.deepEqual(point.models[name].joint, counts);
    }
    const by = (values, select) =>
      Object.fromEntries(
        values.map((v) => [
          v,
          cohort(pointRows.filter((r) => select(r) === v)),
        ]),
      );
    const legalGroup = (r) =>
      r.legalKinds === "single-cell" ? "pointOnlyLegal" : "multipleLegalKinds";
    const byLegality = by(["pointOnlyLegal", "multipleLegalKinds"], legalGroup);
    const forced = upstream.splits[split].point.singleKindOnly;
    assert.equal(byLegality.pointOnlyLegal.examples, forced.examples);
    for (const name of MODELS)
      for (const p of ["actualCorrect", "kindCorrect", "oracleCorrect"])
        assert.equal(
          byLegality.pointOnlyLegal.models[name][p],
          forced.models[name][p],
        );
    splits[split] = {
      allExamples: total.examples,
      allDistinctFixtures: total.distinctFixtures,
      allActualCorrect: Object.fromEntries(
        MODELS.map((n) => [n, total.models[n].actualCorrect]),
      ),
      point,
      byLegality,
      byFamily: by(["sparse", "pressure"], (r) => r.family),
      byTeacherSearch: by(["complete", "incomplete"], (r) =>
        r.searchComplete ? "complete" : "incomplete",
      ),
      legalityByTeacherSearch: Object.fromEntries(
        ["pointOnlyLegal", "multipleLegalKinds"].map((k) => [
          k,
          Object.fromEntries(
            [true, false].map((complete) => [
              complete ? "complete" : "incomplete",
              cohort(
                pointRows.filter(
                  (r) => legalGroup(r) === k && r.searchComplete === complete,
                ),
              ),
            ]),
          ),
        ]),
      ),
    };
  }
  return splits;
}

export function projectFullRows(group) {
  alignRows(group);
  return group.control.map((r, i) => ({
    id: r.id,
    fixtureId: r.fixtureId,
    family: r.family,
    teacherKind: r.teacherKind,
    searchComplete: r.searchComplete,
    phase: r.teacherFinalPhase,
    legalKinds: r.legalKinds,
    candidates: r.fullScore.candidates,
    teacherAction: r.fullScore.teacherActions.A512,
    correctnessPattern: MODELS.map((n) =>
      Number(group[n][i].globalCorrect),
    ).join(""),
    models: Object.fromEntries(
      MODELS.map((n) => {
        const x = group[n][i];
        return [
          n,
          {
            error: x.error,
            selected: x.fullScore.selected,
            selectedIndex: x.selectedIndex,
            correct: x.globalCorrect,
            kindCorrect: x.selectedKind === x.teacherKind,
            oracleCorrect: x.conditional.oracleCorrect,
            fullRank: x.fullScore.targets.A512.rank,
            fullNll: x.fullScore.targets.A512.nll,
            oracleRank: x.conditional.oracleRank,
            oracleNll: x.conditional.oracleNll,
            margin: x.fullScore.margin,
          },
        ];
      }),
    ),
  }));
}

export function loadSaved(directory, config, root = ROOT, archives) {
  for (const [name, sha] of Object.entries(config.inputFiles))
    assert.equal(
      hash(resolve(directory, name)),
      sha,
      name + " frozen hash drift",
    );
  const selectedChanges = read(resolve(directory, "cases.json")),
    upstream = read(resolve(directory, "summary.json")),
    report = read(resolve(directory, "report.json"));
  assert.equal(report.status, "PASS");
  assert.equal(upstream.status, "PASS");
  assert.equal(report.casesSha256, config.inputFiles["cases.json"]);
  assert.equal(report.summarySha256, config.inputFiles["summary.json"]);
  assert(Object.values(report.additionalCompute).every((n) => n === 0));
  assert(Object.values(upstream.additionalCompute).every((n) => n === 0));
  assert.deepEqual(upstream.selectedEpochs, config.selectedEpochs);
  keys(config.rawInputs, ["combined", "pointOnly", "kindOnly"]);
  const raw = {};
  for (const name of ["combined", "pointOnly", "kindOnly"]) {
    const path = resolve(archives[name], "raw.json");
    assert.equal(
      hash(path),
      config.rawInputs[name],
      name + " raw frozen hash drift",
    );
    raw[name] = read(path);
  }
  const cases = {};
  for (const split of ["train", "dev"]) {
    const group = {
      control: raw.combined.control[split],
      combined: raw.combined.candidate[split],
      pointOnly: raw.pointOnly.candidate[split],
      kindOnly: raw.kindOnly.candidate[split],
    };
    assert.deepEqual(raw.combined.control[split], raw.pointOnly.control[split]);
    assert.deepEqual(raw.combined.control[split], raw.kindOnly.control[split]);
    cases[split] = projectFullRows(group);
    assert.deepEqual(
      JSON.parse(JSON.stringify(privateCases(group))),
      selectedChanges[split],
      "Partial selected-change cases drift",
    );
    assert.deepEqual(
      JSON.parse(JSON.stringify(summarizeRows(group))),
      upstream.splits[split],
      "Original full four-model aggregate drift",
    );
    assert.equal(cases[split].length, config.splits[split]);
    assert.equal(
      upstream.splits[split].point.examples,
      config.pointExamples[split],
    );
  }
  const asset = resolve(root, config.upstreamAsset.path);
  assert.equal(hash(asset), config.upstreamAsset.sha256);
  assert.deepEqual(
    read(asset),
    upstream,
    "Original public aggregate semantic drift",
  );
  return { cases, upstream };
}

export function execute(
  { input, output, combined, pointOnly, kindOnly },
  root = ROOT,
  metadata = {},
) {
  const configPath = resolve(root, CONFIG),
    config = read(configPath);
  assert.equal(config.id, "stored-point-fit-v1");
  assert.equal(config.schemaVersion, 1);
  assert.deepEqual(config.models, MODELS);
  assert.deepEqual(config.rankBuckets, ["1", "2-5", "6-10", ">10"]);
  assert.equal(config.boundaries.additionalModelCreationLoadForwardTraining, 0);
  assert.equal(config.boundaries.teacherDomainEpisodeTestRollout, 0);
  assert.equal(config.boundaries.newScoreSelectionLabelOrPolicy, false);
  assert.equal(config.boundaries.publicAggregateOnly, true);
  keys(config.inputFiles, ["cases.json", "summary.json", "report.json"]);
  const base = resolve(root, ".research-output"),
    out = resolve(root, output),
    source = realpathSync(resolve(input));
  const sources = [
    source,
    ...[combined, pointOnly, kindOnly].map((p) => realpathSync(resolve(p))),
  ];
  assert(
    inside(base, out) && !inside(source, out) && out !== source,
    "New owned ignored output only",
  );
  mkdirSync(base, { recursive: true });
  assert(inside(realpathSync(root), realpathSync(base)));
  const parent = realpathSync(dirname(out));
  assert(parent === realpathSync(base) || inside(realpathSync(base), parent));
  assert(
    sources.every(
      (p) => !inside(p, out) && p !== out && !inside(p, parent) && p !== parent,
    ),
    "Do not write inside source",
  );
  mkdirSync(out);
  save(resolve(out, "start.json"), {
    atUtc: new Date().toISOString(),
    configSha256: hash(configPath),
    ...metadata,
  });
  try {
    const { cases, upstream } = loadSaved(source, config, root, {
      combined,
      pointOnly,
      kindOnly,
    });
    const result = {
      schemaVersion: 1,
      id: config.id,
      status: "PASS",
      method: config.method,
      inputFiles: config.inputFiles,
      rawInputs: config.rawInputs,
      selectedEpochs: config.selectedEpochs,
      splits: summarizeFit(cases, upstream),
      additionalCompute: {
        modelCreation: 0,
        checkpointDeserialization: 0,
        forward: 0,
        training: 0,
        teacherDomainEpisodeTestRollout: 0,
      },
      limits: [
        config.boundaries.interpretation,
        config.boundaries.archiveVerification,
        "Counts are observations; multiple observations may share a fixture. Splits and strata are descriptive, not randomized treatments.",
        "Oracle rank is within teacher kind; full rank is among all candidates. No new score, gate, epoch, label, routing or policy is selected.",
        "Search complete is limited to original search scope, not an optimality or label-quality certificate.",
        "Identifiers, actions, coordinates and observation-level predictions remain private; original model gates FAIL and app DFS remain unchanged.",
      ],
    };
    save(resolve(out, "summary.json"), result);
    save(resolve(out, "report.json"), {
      status: "PASS",
      atUtc: new Date().toISOString(),
      configSha256: hash(configPath),
      programSha256: hash(
        resolve(root, "scripts/research/analyze-stored-point-fit.mjs"),
      ),
      summarySha256: hash(resolve(out, "summary.json")),
      inputFiles: config.inputFiles,
      additionalCompute: result.additionalCompute,
      ...metadata,
    });
    return result;
  } catch (error) {
    save(resolve(out, "failure.json"), {
      status: "FAIL",
      error: String(error),
      restartAllowed: false,
      originalInputsRewritten: false,
    });
    throw error;
  }
}

export function parseArgs(args) {
  assert.equal(
    args.length,
    10,
    "Expected --input, --output and three frozen archive directories only",
  );
  const result = {};
  for (let i = 0; i < args.length; i += 2) {
    assert(
      [
        "--input",
        "--output",
        "--combined",
        "--point-only",
        "--kind-only",
      ].includes(args[i]) &&
        args[i + 1] &&
        !result[args[i]],
      "Unknown, duplicate or missing argument",
    );
    result[args[i]] = args[i + 1];
  }
  return {
    input: result["--input"],
    output: result["--output"],
    combined: result["--combined"],
    pointOnly: result["--point-only"],
    kindOnly: result["--kind-only"],
  };
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  if (process.argv.length === 3 && process.argv[2] === "--help")
    console.log(
      "node scripts/research/analyze-stored-point-fit.mjs --input <frozen-four-objective-output> --combined <frozen-run> --point-only <frozen-run> --kind-only <frozen-run> --output .research-output/<new-directory>",
    );
  else {
    const args = parseArgs(process.argv.slice(2));
    const git = (a) =>
      execFileSync("git", a, {
        cwd: ROOT,
        encoding: "utf8",
        windowsHide: true,
      }).trim();
    assert.equal(
      git(["status", "--porcelain"]),
      "",
      "Commit source/config before execution",
    );
    const summary = execute(args, ROOT, {
      sourceHead: git(["rev-parse", "HEAD"]),
      node: process.version,
    });
    console.log(
      JSON.stringify({
        status: summary.status,
        pointExamples: Object.fromEntries(
          ["train", "dev"].map((s) => [s, summary.splits[s].point.examples]),
        ),
        additionalCompute: summary.additionalCompute,
      }),
    );
  }
}
