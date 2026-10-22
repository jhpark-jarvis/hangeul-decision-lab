import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, mkdirSync, realpathSync } from "node:fs";
import { resolve, relative, isAbsolute, dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { summarizeRows as legacySummary } from "./compare-objective-errors.mjs";

export const ROOT = fileURLToPath(new URL("../../", import.meta.url));
const CONFIG = "research/experiments/four-objective-errors-v1.json";
export const models = ["control", "combined", "pointOnly", "kindOnly"];
export const pairs = models.flatMap((a, i) =>
  models.slice(i + 1).map((b) => [a, b]),
);
const kinds = ["place-piece", "single-cell", "reroll"];
const errors = [
  "exact-label",
  "wrong-kind",
  "different-target",
  "different-shape",
  "different-position",
];
const read = (p) => JSON.parse(readFileSync(p, "utf8"));
const sha = (p) => createHash("sha256").update(readFileSync(p)).digest("hex");
const save = (p, v) =>
  writeFileSync(p, JSON.stringify(v, null, 2) + "\n", { flag: "wx" });
const histogram = (values) =>
  Object.fromEntries(
    [...new Set(values)]
      .sort()
      .map((v) => [v, values.filter((x) => x === v).length]),
  );
const finite = (n) => assert(typeof n === "number" && Number.isFinite(n));
const integer = (n, low, high) =>
  assert(Number.isInteger(n) && n >= low && n <= high);
const inside = (parent, path) => {
  const p = relative(parent, path);
  return p !== "" && !p.startsWith("..") && !isAbsolute(p);
};

export function validateRow(row) {
  assert(
    typeof row.id === "string" && row.id && typeof row.fixtureId === "string",
  );
  assert(
    typeof row.family === "string" && typeof row.searchComplete === "boolean",
  );
  assert(
    typeof row.teacherFinalPhase === "string" &&
      typeof row.legalKinds === "string",
  );
  assert(
    kinds.includes(row.teacherKind) &&
      kinds.includes(row.selectedKind) &&
      errors.includes(row.error),
  );
  assert.equal(typeof row.globalCorrect, "boolean");
  const f = row.fullScore,
    c = row.conditional,
    target = f.targets.A512;
  integer(f.candidates, 1, 4096);
  integer(row.selectedIndex, 0, f.candidates - 1);
  assert.equal(f.index, row.selectedIndex);
  assert.equal(f.id, row.id);
  assert.equal(f.fixtureId, row.fixtureId);
  assert.equal(f.family, row.family);
  assert.equal(f.selected.kind, row.selectedKind);
  assert.equal(f.teacherActions.A512.kind, row.teacherKind);
  assert.equal(c.teacherKind, row.teacherKind);
  assert.equal(target.correct, row.globalCorrect);
  integer(target.rank, 1, f.candidates);
  assert.equal(target.rank === 1, row.globalCorrect);
  integer(c.teacherKindCandidates, 1, f.candidates);
  integer(c.oracleRank, 1, c.teacherKindCandidates);
  integer(c.oracleIndex, 0, f.candidates - 1);
  assert.equal(typeof c.oracleCorrect, "boolean");
  assert.equal(c.oracleRank === 1, c.oracleCorrect);
  for (const n of [target.nll, c.oracleNll]) finite(n);
  if (f.candidates === 1) assert.equal(f.margin, null);
  else finite(f.margin);
  assert(target.nll >= 0 && c.oracleNll >= 0);
  assert.equal(row.error === "exact-label", row.globalCorrect);
  assert.equal(
    row.error === "wrong-kind",
    row.selectedKind !== row.teacherKind,
  );
  assert.equal(
    row.globalCorrect,
    row.selectedKind === row.teacherKind && c.oracleCorrect,
  );
}

function identity(r) {
  return [
    r.id,
    r.fixtureId,
    r.family,
    r.teacherKind,
    r.searchComplete,
    r.teacherFinalPhase,
    r.legalKinds,
    r.fullScore.candidates,
    r.fullScore.teacherActions,
    r.fullScore.firstLegalAction,
  ];
}

export function alignRows(group) {
  assert.deepEqual(Object.keys(group).sort(), [...models].sort());
  const before = group.control;
  assert(
    before.length > 0 &&
      new Set(before.map((r) => r.id)).size === before.length,
  );
  for (const model of models) {
    assert.equal(group[model].length, before.length);
    for (let i = 0; i < before.length; i++) {
      validateRow(group[model][i]);
      assert.deepEqual(
        identity(group[model][i]),
        identity(before[i]),
        "Observation/label/candidate identity drift",
      );
    }
  }
}

const kindCorrect = (r) => r.selectedKind === r.teacherKind;
const oracleCorrect = (r) => r.conditional.oracleCorrect;
function paired(left, right, correct = (r) => r.globalCorrect) {
  const result = {
    examples: left.length,
    bothCorrect: 0,
    gain: 0,
    regression: 0,
    bothWrong: 0,
  };
  left.forEach((l, i) => {
    const a = correct(l),
      b = correct(right[i]);
    result[
      a && b ? "bothCorrect" : b ? "gain" : a ? "regression" : "bothWrong"
    ]++;
  });
  result.netGain = result.gain - result.regression;
  return result;
}

function transitions(left, right) {
  const counts = {};
  left.forEach((l, i) => {
    const key = `${l.teacherKind}: ${l.error} -> ${right[i].error}`;
    counts[key] = (counts[key] ?? 0) + 1;
  });
  return Object.fromEntries(
    Object.entries(counts).sort(([a], [b]) => a.localeCompare(b, "en")),
  );
}

function diagnosticDeltas(left, right) {
  const delta = (field) => {
    const values = left.map((r, i) => field(right[i]) - field(r));
    values.forEach(finite);
    return {
      mean: values.reduce((a, b) => a + b, 0) / values.length,
      decreased: values.filter((x) => x < 0).length,
      unchanged: values.filter((x) => x === 0).length,
      increased: values.filter((x) => x > 0).length,
    };
  };
  return {
    fullNll: delta((r) => r.fullScore.targets.A512.nll),
    fullRank: delta((r) => r.fullScore.targets.A512.rank),
    oracleNll: delta((r) => r.conditional.oracleNll),
    oracleRank: delta((r) => r.conditional.oracleRank),
  };
}

export function summarizeRows(group) {
  alignRows(group);
  const patterns = {};
  for (let i = 0; i < 16; i++) patterns[i.toString(2).padStart(4, "0")] = 0;
  for (let i = 0; i < group.control.length; i++)
    patterns[models.map((m) => Number(group[m][i].globalCorrect)).join("")]++;
  const pairResults = Object.fromEntries(
    pairs.map(([a, b]) => {
      const l = group[a],
        r = group[b],
        changed = l.filter(
          (row, i) => row.selectedIndex !== r[i].selectedIndex,
        ).length;
      return [
        `${a}:${b}`,
        {
          actual: paired(l, r),
          kind: paired(l, r, kindCorrect),
          oracle: paired(l, r, oracleCorrect),
          changedSelection: changed,
          changedButBothWrong: l.filter(
            (row, i) =>
              row.selectedIndex !== r[i].selectedIndex &&
              !row.globalCorrect &&
              !r[i].globalCorrect,
          ).length,
          errorTransitions: transitions(l, r),
          savedDiagnosticDeltas: diagnosticDeltas(l, r),
        },
      ];
    }),
  );
  const points = Object.fromEntries(
    models.map((m) => [
      m,
      group[m].filter((r) => r.teacherKind === "single-cell"),
    ]),
  );
  const pointPatterns = (predicate) =>
    histogram(
      points.control.map((_, i) =>
        models.map((m) => Number(predicate(points[m][i]))).join(""),
      ),
    );
  const pointJoint = Object.fromEntries(
    models.map((m) => [
      m,
      {
        examples: points[m].length,
        bothCorrect: points[m].filter((r) => kindCorrect(r) && oracleCorrect(r))
          .length,
        kindOnly: points[m].filter((r) => kindCorrect(r) && !oracleCorrect(r))
          .length,
        coordinateOnly: points[m].filter(
          (r) => !kindCorrect(r) && oracleCorrect(r),
        ).length,
        neither: points[m].filter((r) => !kindCorrect(r) && !oracleCorrect(r))
          .length,
      },
    ]),
  );
  const signatures = (gain) =>
    Object.fromEntries(
      Array.from({ length: 7 }, (_, j) => {
        const mask = (j + 1).toString(2).padStart(3, "0");
        const names = models
          .slice(1)
          .filter((_, i) => mask[i] === "1")
          .join("+");
        const key = gain
          ? "0" + mask
          : "1" + [...mask].map((v) => (v === "1" ? "0" : "1")).join("");
        return [names, patterns[key]];
      }),
    );
  const overlap = {
    exactGainSignatures: signatures(true),
    exactRegressionSignatures: signatures(false),
  };
  const byTeacherKind = Object.fromEntries(
    kinds.map((k) => [
      k,
      {
        examples: group.control.filter((r) => r.teacherKind === k).length,
        actual: Object.fromEntries(
          models.map((m) => [
            m,
            group[m].filter((r) => r.teacherKind === k && r.globalCorrect)
              .length,
          ]),
        ),
      },
    ]),
  );
  const failureCohorts = Object.fromEntries(
    pairs.map(([a, b]) => {
      const l = group[a],
        r = group[b];
      return [
        `${a}:${b}`,
        {
          gains: transitions(
            l.filter((x, i) => !x.globalCorrect && r[i].globalCorrect),
            r.filter((x, i) => x.globalCorrect && !l[i].globalCorrect),
          ),
          regressions: transitions(
            l.filter((x, i) => x.globalCorrect && !r[i].globalCorrect),
            r.filter((x, i) => !x.globalCorrect && l[i].globalCorrect),
          ),
        },
      ];
    }),
  );
  const sharedFailures = group.control
    .map((_, i) => models.map((m) => group[m][i]))
    .filter((rows) => rows.every((r) => !r.globalCorrect));
  return {
    examples: group.control.length,
    modelOrder: models,
    correctnessPatterns: patterns,
    overlap,
    byTeacherKind,
    actualCorrect: Object.fromEntries(
      models.map((m) => [m, group[m].filter((r) => r.globalCorrect).length]),
    ),
    errors: Object.fromEntries(
      models.map((m) => [m, histogram(group[m].map((r) => r.error))]),
    ),
    pairs: pairResults,
    failureCohorts,
    point: {
      examples: points.control.length,
      joint: pointJoint,
      kindPatterns: pointPatterns(kindCorrect),
      oraclePatterns: pointPatterns(oracleCorrect),
      singleKindOnly: {
        examples: points.control.filter((r) => r.legalKinds === "single-cell")
          .length,
        models: Object.fromEntries(
          models.map((m) => {
            const rows = points[m].filter(
              (r) => r.legalKinds === "single-cell",
            );
            return [
              m,
              {
                actualCorrect: rows.filter((r) => r.globalCorrect).length,
                kindCorrect: rows.filter(kindCorrect).length,
                oracleCorrect: rows.filter(oracleCorrect).length,
              },
            ];
          }),
        ),
      },
    },
    sharedFailures: {
      examples: sharedFailures.length,
      byTeacherKind: histogram(
        sharedFailures.map((rows) => rows[0].teacherKind),
      ),
      byFamily: histogram(sharedFailures.map((rows) => rows[0].family)),
      errorPatterns: histogram(
        sharedFailures.map((rows) => rows.map((r) => r.error).join(" | ")),
      ),
    },
  };
}

export function privateCases(group) {
  alignRows(group);
  return group.control.flatMap((row, i) => {
    const rows = models.map((m) => group[m][i]);
    if (
      rows.every(
        (r) =>
          r.selectedIndex === row.selectedIndex &&
          r.conditional.oracleIndex === row.conditional.oracleIndex,
      )
    )
      return [];
    return [
      {
        id: row.id,
        fixtureId: row.fixtureId,
        family: row.family,
        teacherKind: row.teacherKind,
        searchComplete: row.searchComplete,
        phase: row.teacherFinalPhase,
        legalKinds: row.legalKinds,
        candidates: row.fullScore.candidates,
        teacherAction: row.fullScore.teacherActions.A512,
        correctnessPattern: rows.map((r) => Number(r.globalCorrect)).join(""),
        models: Object.fromEntries(
          models.map((m, j) => [
            m,
            {
              error: rows[j].error,
              selected: rows[j].fullScore.selected,
              selectedIndex: rows[j].selectedIndex,
              correct: rows[j].globalCorrect,
              kindCorrect: kindCorrect(rows[j]),
              oracleCorrect: oracleCorrect(rows[j]),
              fullRank: rows[j].fullScore.targets.A512.rank,
              fullNll: rows[j].fullScore.targets.A512.nll,
              oracleRank: rows[j].conditional.oracleRank,
              oracleNll: rows[j].conditional.oracleNll,
              margin: rows[j].fullScore.margin,
            },
          ]),
        ),
      },
    ];
  });
}

export function validateConfigInput(spec, root = ROOT, originalConfigPath) {
  const exportedPath = resolve(root, spec.configPath);
  const configPath = spec.originalConfigRequired
    ? originalConfigPath
    : exportedPath;
  assert(
    typeof configPath === "string" && configPath,
    "Explicit original private config required",
  );
  assert.equal(
    sha(configPath),
    spec.configSha256,
    "Frozen computational config changed",
  );
  const config = read(configPath);
  if (spec.originalConfigRequired) {
    assert(
      [spec.configSha256, spec.exportedConfigSha256].includes(
        sha(exportedPath),
      ),
      "Known original or projected config only",
    );
    const exported = read(exportedPath),
      a = { ...config },
      b = { ...exported };
    assert.equal(typeof a.authorization, "string");
    assert.equal(typeof b.authorization, "string");
    delete a.authorization;
    delete b.authorization;
    assert.deepEqual(a, b, "Only authorization export metadata may differ");
  }
  return config;
}

export function loadArchive(directory, spec, root = ROOT, originalConfigPath) {
  const required = [
    "freeze.json",
    "execution.json",
    "report.json",
    "raw.json",
    "summary.json",
    "forward-count.jsonl",
    "control/report.json",
    "candidate/report.json",
  ];
  assert.deepEqual(Object.keys(spec.hashes).sort(), required.sort());
  for (const [name, digest] of Object.entries(spec.hashes))
    assert.equal(
      sha(resolve(directory, name)),
      digest,
      `Frozen ${name} changed`,
    );
  const config = validateConfigInput(spec, root, originalConfigPath);
  const freeze = read(resolve(directory, "freeze.json")),
    execution = read(resolve(directory, "execution.json"));
  const report = read(resolve(directory, "report.json")),
    summary = read(resolve(directory, "summary.json")),
    raw = read(resolve(directory, "raw.json"));
  assert.equal(config.approval, "USER_APPROVED");
  assert.equal(config.status, "FROZEN_BEFORE_EXECUTION");
  assert.equal(execution.status, "FROZEN_BEFORE_EXECUTION");
  assert.equal(freeze.status, "FROZEN_DEV_ONLY");
  assert.equal(report.status, "PASS");
  assert.equal(summary.status, "PASS");
  assert.equal(execution.configSha256, spec.configSha256);
  for (const body of [freeze, report, summary.identity])
    assert.equal(body.executionSha256, spec.hashes["execution.json"]);
  assert.equal(freeze.reportSha256, spec.hashes["report.json"]);
  assert.equal(report.rawSha256, spec.hashes["raw.json"]);
  assert.equal(report.summarySha256, spec.hashes["summary.json"]);
  assert.equal(report.ledgerSha256, spec.hashes["forward-count.jsonl"]);
  assert.equal(report.accounting.originalTopLevelForwards, 12840);
  assert.equal(report.accounting.trainingRuns, 2);
  assert.equal(report.accounting.teacherDomainTestRolloutGeometry, 0);
  assert.deepEqual(summary.identity.sourceHashes, execution.sourceHashes);
  const ledger = readFileSync(resolve(directory, "forward-count.jsonl"), "utf8")
    .trim()
    .split("\n")
    .map(JSON.parse);
  assert.equal(ledger.length, 12840);
  assert(ledger.every((r, i) => r.call === i + 1 && r.limit === 12840));
  validateLedger(ledger);
  assert.deepEqual(Object.keys(raw).sort(), ["candidate", "control"]);
  const arms = {};
  for (const arm of ["control", "candidate"]) {
    assert.deepEqual(Object.keys(raw[arm]).sort(), ["dev", "train"]);
    const r = read(resolve(directory, arm, "report.json"));
    assert.equal(r.status, "PASS");
    assert.equal(r.executionSha256, spec.hashes["execution.json"]);
    assert.deepEqual(Object.keys(r.artifactHashes).sort(), [
      "checkpoint.pt",
      "logits.json",
      "loss-history.json",
      "predictions.json",
    ]);
    for (const [name, digest] of Object.entries(r.artifactHashes)) {
      assert.equal(
        sha(resolve(directory, arm, name)),
        digest,
        `${arm} ${name} changed`,
      );
      assert.equal(summary.identity.artifactHashes[arm][name], digest);
    }
    assert.equal(
      summary.identity.artifactHashes[arm]["report.json"],
      spec.hashes[`${arm}/report.json`],
    );
    assert.equal(r.history.length, 12);
    const best = Math.max(...r.history.map((h) => h.dev.correct));
    assert.equal(
      r.selectedEpoch,
      r.history.findIndex((h) => h.dev.correct === best) + 1,
    );
    assert.equal(r.selectedEpoch, summary.models[arm].selectedEpoch);
    assert.equal(r.modelForwards, 5992);
    assert.equal(r.trainingRuns, 1);
    assert.deepEqual(r.phaseCounts, {
      train: 4068,
      "epoch-dev": 1068,
      frozen: 856,
    });
    for (const split of ["train", "dev"]) {
      const rows = raw[arm][split];
      rows.forEach(validateRow);
      assert.equal(
        rows.filter((row) => row.globalCorrect).length,
        summary.models[arm].splits[split].fullList.overall.correct,
      );
      const recorded =
        summary.models[arm].splits[split].fullList.overall.errors;
      for (const e of errors)
        assert.equal(
          rows.filter((row) => row.error === e).length,
          recorded[e] ?? 0,
        );
    }
    arms[arm] = r;
  }
  const losses = read(resolve(directory, "control/loss-history.json"));
  assert.equal(losses.length, 12);
  assert(
    losses.every(
      (r) => Number.isFinite(r.flatMean) && r.flatMean === r.optimizedMean,
    ),
  );
  return {
    config,
    summary,
    raw,
    arms,
    optimizedFlatTrace: losses.map((r) => [
      r.epoch,
      r.examples,
      r.flatMean,
      r.optimizedMean,
    ]),
  };
}

export function validateLedger(records) {
  assert.equal(records.length, 12840);
  let offset = 0;
  for (const [mode, limits] of [
    ["preflight", { initial: 856 }],
    ["control", { train: 4068, "epoch-dev": 1068, frozen: 856 }],
    ["candidate", { train: 4068, "epoch-dev": 1068, frozen: 856 }],
  ]) {
    const counts = Object.fromEntries(Object.keys(limits).map((p) => [p, 0]));
    for (const row of records.slice(
      offset,
      offset + Object.values(limits).reduce((a, b) => a + b, 0),
    )) {
      assert.equal(row.call, ++offset);
      assert.equal(row.limit, 12840);
      assert.equal(row.mode, mode);
      assert(Object.hasOwn(limits, row.phase));
      assert.equal(row.phaseCall, ++counts[row.phase]);
      assert(counts[row.phase] <= limits[row.phase]);
    }
    assert.deepEqual(counts, limits);
  }
}

export function compareArchives(combined, pointOnly, kindOnly, config) {
  const archives = { combined, pointOnly, kindOnly };
  for (const [name, pack] of Object.entries(archives)) {
    for (const key of ["source", "model", "inference"])
      assert.deepEqual(
        pack.config[key],
        combined.config[key],
        name + " " + key,
      );
    for (const key of [
      "seed",
      "epochs",
      "batchSize",
      "threads",
      "interopThreads",
      "optimizer",
      "learningRate",
      "weightDecay",
      "order",
      "selection",
    ])
      assert.deepEqual(
        pack.config.training[key],
        combined.config.training[key],
        name + " " + key,
      );
    assert.deepEqual(
      pack.raw.control,
      combined.raw.control,
      "Shared control rows differ",
    );
    for (const key of ["weightsSha256", "history", "audit", "selectedEpoch"])
      assert.deepEqual(
        pack.arms.control[key],
        combined.arms.control[key],
        name + " shared control " + key,
      );
    for (const key of ["predictions.json", "logits.json"])
      assert.equal(
        pack.arms.control.artifactHashes[key],
        combined.arms.control.artifactHashes[key],
      );
    assert.deepEqual(
      pack.optimizedFlatTrace,
      combined.optimizedFlatTrace,
      "Shared optimized flat trace differs",
    );
  }
  assert.deepEqual(
    [
      combined.config.objective.kindWeight,
      combined.config.objective.pointWeight,
    ],
    [0.5, 0.5],
  );
  assert.deepEqual(
    [
      pointOnly.config.objective.kindWeight,
      pointOnly.config.objective.pointWeight,
    ],
    [0, 0.5],
  );
  assert.deepEqual(
    [
      kindOnly.config.objective.kindWeight,
      kindOnly.config.objective.pointWeight,
    ],
    [0.5, 0],
  );
  const splits = {},
    cases = {};
  for (const split of ["train", "dev"]) {
    const group = {
      control: combined.raw.control[split],
      combined: combined.raw.candidate[split],
      pointOnly: pointOnly.raw.candidate[split],
      kindOnly: kindOnly.raw.candidate[split],
    };
    assert.equal(group.control.length, config.splits[split]);
    splits[split] = summarizeRows(group);
    cases[split] = privateCases(group);
  }
  return {
    summary: {
      schemaVersion: 1,
      id: config.id,
      status: "PASS",
      method:
        "Stored controlled rows only; four historical selected-epoch outcomes, no scoring or model calls",
      inputIdentity: {
        archives: Object.fromEntries(
          Object.entries(config.archives).map(([m, s]) => [m, s.hashes]),
        ),
        threeSharedControlsExact: true,
        sourceModelLabelCandidatesEqual: true,
        optimizedFlatTraceExact: true,
      },
      selectedEpochs: {
        control: combined.arms.control.selectedEpoch,
        ...Object.fromEntries(
          Object.entries(archives).map(([m, p]) => [
            m,
            p.arms.candidate.selectedEpoch,
          ]),
        ),
      },
      originalDecisions: Object.fromEntries(
        Object.entries(archives).map(([m, p]) => [
          m,
          Object.fromEntries(
            [
              "gate",
              "mechanism",
              "coordinateMechanism",
              "kindMechanism",
              "tradeoffGuards",
            ]
              .filter((k) => Object.hasOwn(p.summary, k))
              .map((k) => [k, p.summary[k]]),
          ),
        ]),
      ),
      splits,
      additionalCompute: {
        modelCreation: 0,
        checkpointDeserialization: 0,
        forward: 0,
        training: 0,
        teacherDomainEpisodeTestRollout: 0,
      },
      limits: [
        config.boundaries.interpretation,
        "No fixed-epoch factorial interaction, causal effect, routing or ensemble is evaluated",
        "Oracle and actual action success are separate; ID/actions/coordinates remain private",
        "Original gate/label/epoch/model and app DFS remain unchanged",
      ],
    },
    cases,
  };
}

export function execute(
  { combined, pointOnly, kindOnly, kindConfig, output },
  metadata = {},
  root = ROOT,
) {
  const started = performance.now(),
    configPath = resolve(root, CONFIG),
    config = read(configPath);
  assert.equal(config.approval, "STORED_VALIDATION_AND_PUBLICATION_ONLY");
  assert.deepEqual(Object.keys(config.archives).sort(), [
    "combined",
    "kindOnly",
    "pointOnly",
  ]);
  assert.equal(config.boundaries.newModelLoadForwardTraining, 0);
  assert.equal(config.boundaries.teacherDomainEpisodeTestRollout, 0);
  assert.equal(config.boundaries.newScoreOrSelection, false);
  assert.equal(config.boundaries.originalFilesImmutable, true);
  assert.equal(config.boundaries.detailCasesPrivate, true);
  assert.equal(config.boundaries.publicAggregateOnly, true);
  assert.deepEqual(config.splits, { train: 339, dev: 89 });
  assert.deepEqual(config.models, models);
  assert.deepEqual(config.pairs, pairs);
  const out = resolve(root, output),
    privateRoot = resolve(root, ".research-output");
  assert(
    inside(privateRoot, out) &&
      [combined, pointOnly, kindOnly].every(
        (p) => !inside(resolve(p), out) && resolve(p) !== out,
      ),
    "New owned ignored output only",
  );
  mkdirSync(privateRoot, { recursive: true });
  assert(inside(realpathSync(root), realpathSync(privateRoot)));
  assert(
    inside(realpathSync(privateRoot), realpathSync(dirname(out))) ||
      realpathSync(dirname(out)) === realpathSync(privateRoot),
  );
  mkdirSync(out);
  save(resolve(out, "start.json"), {
    atUtc: new Date().toISOString(),
    configSha256: sha(configPath),
    ...metadata,
  });
  try {
    const a = loadArchive(resolve(combined), config.archives.combined, root),
      b = loadArchive(resolve(pointOnly), config.archives.pointOnly, root),
      c = loadArchive(
        resolve(kindOnly),
        config.archives.kindOnly,
        root,
        resolve(kindConfig),
      );
    const result = compareArchives(a, b, c, config);
    const legacyPath = resolve(root, config.legacyComparison.path);
    assert.equal(sha(legacyPath), config.legacyComparison.sha256);
    const legacy = read(legacyPath);
    for (const split of ["train", "dev"])
      assert.deepEqual(
        legacySummary({
          control: a.raw.control[split],
          combined: a.raw.candidate[split],
          pointOnly: b.raw.candidate[split],
        }),
        legacy.splits[split],
        "Original three-model aggregate drift",
      );
    result.summary.inputIdentity.originalThreeModelProjectionExact = true;
    save(resolve(out, "cases.json"), result.cases);
    save(resolve(out, "summary.json"), result.summary);
    save(resolve(out, "report.json"), {
      status: "PASS",
      atUtc: new Date().toISOString(),
      configSha256: sha(configPath),
      programSha256: sha(
        resolve(root, "scripts/research/compare-four-objectives.mjs"),
      ),
      legacyProgramSha256: sha(
        resolve(root, "scripts/research/compare-objective-errors.mjs"),
      ),
      originalKindConfigSha256: sha(kindConfig),
      summarySha256: sha(resolve(out, "summary.json")),
      casesSha256: sha(resolve(out, "cases.json")),
      additionalCompute: result.summary.additionalCompute,
      durationMs: performance.now() - started,
      rssAfterBytes: process.memoryUsage().rss,
      instantaneousPeakMeasured: false,
      ...metadata,
    });
    return result.summary;
  } catch (error) {
    save(resolve(out, "failure.json"), {
      status: "FAIL",
      error: String(error),
      restartAllowed: false,
      inputPreservation: "Separate original byte-hash audit required",
    });
    throw error;
  }
}

if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  const args = process.argv.slice(2);
  if (args.length === 1 && args[0] === "--help")
    console.log(
      "node scripts/research/compare-four-objectives.mjs --combined <frozen-run> --point-only <frozen-run> --kind-only <frozen-run> --kind-config <original-private-config.json> --output <new-ignored-directory>",
    );
  else {
    const opts = {};
    for (let i = 0; i < args.length; i += 2) {
      assert(
        [
          "--combined",
          "--point-only",
          "--kind-only",
          "--kind-config",
          "--output",
        ].includes(args[i]) &&
          args[i + 1] &&
          !opts[args[i]],
        "Unknown, duplicate or missing argument",
      );
      opts[args[i]] = args[i + 1];
    }
    assert.equal(Object.keys(opts).length, 5);
    const git = (args) =>
      execFileSync("git", args, {
        cwd: ROOT,
        encoding: "utf8",
        windowsHide: true,
      }).trim();
    assert.equal(
      git(["status", "--porcelain"]),
      "",
      "Commit source/config before execution",
    );
    const summary = execute(
      {
        combined: opts["--combined"],
        pointOnly: opts["--point-only"],
        kindOnly: opts["--kind-only"],
        kindConfig: opts["--kind-config"],
        output: opts["--output"],
      },
      { sourceHead: git(["rev-parse", "HEAD"]), node: process.version },
    );
    console.log(
      JSON.stringify({
        status: summary.status,
        dev: summary.splits.dev.actualCorrect,
        additionalCompute: summary.additionalCompute,
      }),
    );
  }
}
