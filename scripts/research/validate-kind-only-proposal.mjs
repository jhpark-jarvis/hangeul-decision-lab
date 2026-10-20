import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const defaultRoot = fileURLToPath(new URL("../../", import.meta.url));
const proposalPath =
  "research/experiments/kind-only-objective-proposal-v1.json";
const read = (root, path) => readFileSync(resolve(root, path));
const json = (root, path) => JSON.parse(read(root, path).toString("utf8"));
const hash = (root, path) =>
  createHash("sha256").update(read(root, path)).digest("hex");

// Synthetic scalar arithmetic only. This is not a training criterion or runner.
function lse(values) {
  assert(values.length > 0 && values.every(Number.isFinite));
  const max = Math.max(...values);
  return max + Math.log(values.reduce((sum, x) => sum + Math.exp(x - max), 0));
}
function ce(values, target) {
  assert(Number.isInteger(target) && target >= 0 && target < values.length);
  return lse(values) - values[target];
}
function kindTerms(scores, target, kinds) {
  ce(scores, target);
  assert.equal(scores.length, kinds.length);
  assert(kinds.every((k) => Number.isInteger(k) && k >= 0 && k < 3));
  const present = [0, 1, 2].filter((g) => kinds.includes(g));
  const groups = present.map((g) =>
    kinds.flatMap((k, i) => (k === g ? [i] : [])),
  );
  const logits = groups.map(
    (indices) => lse(indices.map((i) => scores[i])) - Math.log(indices.length),
  );
  const targetKind = present.indexOf(kinds[target]);
  return { present, groups, logits, targetKind, loss: ce(logits, targetKind) };
}
const near = (a, b, epsilon = 1e-9) => assert(Math.abs(a - b) <= epsilon);

export function validateProposal(proposal, root = defaultRoot) {
  const base = json(
    root,
    "research/experiments/point-objective-proposal-v1.json",
  );
  assert.equal(proposal.schemaVersion, 1);
  assert.equal(proposal.id, "kind-only-objective-proposal-v1");
  assert.equal(proposal.status, "PROPOSED_NOT_RUN");
  assert.equal(proposal.approval, "DESIGN_AND_PUBLICATION_ONLY");
  assert.equal(proposal.executionRequiresSeparateUserDecision, true);
  for (const key of [
    "source",
    "model",
    "inference",
    "forwardBudget",
    "resources",
  ])
    assert.deepEqual(proposal[key], base[key], key);
  assert.equal(
    proposal.objective.baseline,
    "Original native cross_entropy(s, y)",
  );
  assert.equal(proposal.objective.candidate, "L_flat + 0.5 * L_kind");
  assert.equal(proposal.objective.kindWeight, 0.5);
  assert.equal(proposal.objective.pointWeight, 0);
  assert.equal(
    proposal.objective.kind,
    "For present legal kinds g: a[g] = logsumexp(s[G_g]) - log(n_g); L_kind = native cross_entropy(a, target_kind)",
  );
  assert.equal(
    proposal.objective.point,
    "Disabled; do not evaluate point conditional CE, including on teacher-point observations",
  );
  assert.equal(
    proposal.objective.normalization,
    "Mean per observation in the actual batch including zero kind terms; final batch uses actual length. No class reweighting, oversampling or separate kind-only denominator",
  );
  assert.equal(proposal.objective.numeric, base.objective.numeric);
  assert.match(
    proposal.objective.absentKinds,
    /One present kind gives exactly zero/,
  );
  assert.match(proposal.objective.absentKinds, /STOP/);
  assert.equal(proposal.training.runs, 2);
  assert.deepEqual(proposal.training.arms, [
    "original-spatial-flat-reproduction",
    "same-spatial-kind-only-aux",
  ]);
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
    "initialControl",
  ])
    assert.deepEqual(proposal.training[key], base.training[key], key);
  assert.match(
    proposal.training.baselineControl,
    /both completed flat controls/,
  );
  assert.match(proposal.training.baselineControl, /epoch11/);
  assert.match(proposal.training.baselineControl, /STOP/);
  for (const row of Object.values(proposal.priorControlHashes)) {
    assert.deepEqual(Object.keys(row).sort(), [
      "control/checkpoint.pt",
      "control/logits.json",
      "control/loss-history.json",
      "control/predictions.json",
      "control/report.json",
    ]);
    for (const sha of Object.values(row)) assert.match(sha, /^[0-9a-f]{64}$/);
    assert.equal(
      row["control/checkpoint.pt"],
      base.source.spatialHashes["spatial/checkpoint.pt"],
    );
    assert.equal(
      row["control/logits.json"],
      base.source.spatialHashes["spatial/logits.json"],
    );
    assert.equal(
      row["control/predictions.json"],
      base.source.spatialHashes["spatial/predictions.json"],
    );
  }
  assert.deepEqual(
    proposal.screening.originalFourGate,
    base.screening.originalFourGate,
  );
  assert.deepEqual(proposal.screening.kindMechanism, {
    pointCorrectKindAtLeast: 7,
    minimumGainVsConcurrentControl: 3,
    pointExamples: 13,
  });
  assert.deepEqual(proposal.screening.tradeoffGuards, {
    pointOracleAtLeastConcurrentControl: true,
    actualRerollAtLeastConcurrentControl: true,
    pointExamples: 13,
    rerollExamples: 8,
  });
  assert.deepEqual(proposal.screening.jointImprovementClaim, {
    pointCorrectKindAtLeast: 7,
    pointTeacherKindOracleCorrectAtLeast: 5,
    bothRequired: true,
  });
  assert.equal(
    proposal.boundaries.designActualModelCreationLoadForwardTraining,
    0,
  );
  assert.equal(proposal.boundaries.teacherDomainEpisodeTestRolloutGeometry, 0);
  for (const key of [
    "originalFilesImmutable",
    "appUsesDfs",
    "weightsRawComponentsPrivate",
  ])
    assert.equal(proposal.boundaries[key], true);
  for (const key of [
    "newDataLabelsFeaturesArchitecture",
    "scalePolicyAdoption",
  ])
    assert.equal(proposal.boundaries[key], false);
  assert.match(proposal.boundaries.implementation, /not implemented/);
  assert.match(
    proposal.boundaries.futureExecutionAndPublication,
    /Separate explicit execution decision/,
  );

  const paths = {
    comparison: "assets/research/objective-error-comparison-20261020-v1.json",
    combined: "assets/research/point-objective-train-dev-20261018-v1.json",
    pointOnly: "assets/research/point-only-train-dev-20261019-v1.json",
  };
  for (const [key, path] of Object.entries(paths)) {
    assert.equal(proposal.evidence[key].path, path);
    assert.equal(proposal.evidence[key].sha256, hash(root, path));
  }
  const comparison = json(root, paths.comparison);
  const dev = comparison.splits.dev;
  assert.deepEqual(comparison.selectedEpochs, {
    control: 11,
    combined: 12,
    pointOnly: 11,
  });
  assert.deepEqual(dev.actualCorrect, {
    control: 29,
    combined: 29,
    pointOnly: 28,
  });
  assert.deepEqual(dev.overlap, {
    gainsBoth: 0,
    gainsCombinedOnly: 2,
    gainsPointOnly: 0,
    regressionsBoth: 0,
    regressionsCombinedOnly: 2,
    regressionsPointOnly: 1,
  });
  assert.equal(
    dev.failureCohorts["control:combined"].gains[
      "place-piece: wrong-kind -> exact-label"
    ],
    2,
  );
  assert.equal(dev.sharedFailures.examples, 58);
  assert.equal(dev.sharedFailures.byTeacherKind["single-cell"], 13);
  assert.deepEqual(dev.point.kindPatterns, { 111: 4, "011": 1, "000": 8 });
  const point = json(root, paths.pointOnly);
  const control = point.models.control.splits.dev;
  assert.equal(control.pointCorrectKind, 4);
  assert.equal(
    control.conditional.groups.teacherKind["single-cell"]
      .teacherKindOracleCorrect,
    2,
  );
  assert.equal(control.fullList.groups.teacherKind.reroll.correct, 7);
  assert.equal(control.fullList.groups.legalKinds["single-cell"].examples, 4);
  assert.deepEqual(proposal.evidence.observedDev, {
    examples: 89,
    pointExamples: 13,
    combinedOnlyGains: 2,
    combinedOnlyRegressions: 2,
    pointOnlyRegressions: 1,
    overlappingGains: 0,
    overlappingRegressions: 0,
    sharedFailures: 58,
    sharedPointFailures: 13,
    sharedPointKindRecovery: 1,
    controlPointKind: 4,
    controlPointOracle: 2,
    controlReroll: 7,
    pointOnlyLegalKindExamples: 4,
  });
  for (const path of [paths.combined, paths.pointOnly]) {
    const result = json(root, path);
    assert.equal(result.gate.status, "FAIL");
    assert.equal(result.accounting.originalTopLevelForwards, 12840);
    assert.equal(result.accounting.trainingRuns, 2);
  }
  const { train, dev: devCount } = proposal.source.examples;
  assert.deepEqual([train, devCount], [339, 89]);
  const budget = proposal.forwardBudget;
  assert.equal(budget.preflight, 2 * (train + devCount));
  assert.equal(budget.perArmTraining, 12 * train);
  assert.equal(budget.perArmEpochDev, 12 * devCount);
  assert.equal(budget.perArmFrozenTwoRepeats, 2 * (train + devCount));
  assert.equal(
    budget.perArmTotal,
    budget.perArmTraining +
      budget.perArmEpochDev +
      budget.perArmFrozenTwoRepeats,
  );
  assert.equal(
    budget.totalTopLevelMax,
    budget.preflight + 2 * budget.perArmTotal,
  );
  return { contract: "PASS", evidence: "PASS", proposedForwardCap: 12840 };
}

export function validate(root = defaultRoot) {
  const proposal = json(root, proposalPath);
  const result = validateProposal(proposal, root);
  const checks = [];
  const check = (name, fn) => {
    fn();
    checks.push(name);
  };
  check("kind logits use log mean exp, not sum mass", () => {
    const row = kindTerms([0, 0, 0], 2, [0, 0, 1]);
    near(row.logits[0], 0);
    near(row.logits[1], 0);
    near(row.loss, Math.log(2));
    assert.notEqual(ce([Math.log(2), 0], 1), row.loss);
  });
  check(
    "duplicating every candidate in a kind preserves its mean logit",
    () => {
      near(
        kindTerms([2, -1, 0], 2, [0, 0, 1]).loss,
        kindTerms([2, -1, 2, -1, 0], 4, [0, 0, 0, 0, 1]).loss,
      );
    },
  );
  check("empty kinds excluded, with original kind target mapping", () => {
    const row = kindTerms([2, 1, 0], 2, [2, 2, 0]);
    assert.deepEqual(row.present, [0, 2]);
    assert.equal(row.targetKind, 0);
    near(row.loss, ce([0, lse([2, 1]) - Math.log(2)], 0));
  });
  check("one present kind has zero extra loss and gradient", () => {
    const scores = [0.2, 0.7, -0.3];
    assert.equal(kindTerms(scores, 2, [1, 1, 1]).loss, 0);
    for (let i = 0; i < scores.length; i++) {
      const shifted = [...scores];
      shifted[i] += 1e-5;
      assert.equal(kindTerms(shifted, 2, [1, 1, 1]).loss, 0);
    }
    assert(ce(scores, 2) > 0);
  });
  check("same target kind has no coordinate-specific extra supervision", () => {
    near(
      kindTerms([2, 1, 0], 0, [1, 1, 0]).loss,
      kindTerms([2, 1, 0], 1, [1, 1, 0]).loss,
    );
    assert.notEqual(ce([2, 1, 0], 0), ce([2, 1, 0], 1));
  });
  check("stable common score shift", () => {
    near(
      kindTerms([2, 1, 0], 1, [0, 1, 1]).loss,
      kindTerms([1002, 1001, 1000], 1, [0, 1, 1]).loss,
    );
  });
  check("kind gradient and total loss match finite differences", () => {
    const scores = [0.2, 0.7, -0.3, 0.1],
      kinds = [0, 1, 1, 2],
      target = 2;
    const row = kindTerms(scores, target, kinds);
    const total = (s) => ce(s, target) + 0.5 * kindTerms(s, target, kinds).loss;
    for (let i = 0; i < scores.length; i++) {
      const g = row.present.indexOf(kinds[i]);
      const group = row.groups[g].map((j) => scores[j]);
      const auxiliary =
        (Math.exp(row.logits[g] - lse(row.logits)) -
          Number(g === row.targetKind)) *
        Math.exp(scores[i] - lse(group));
      const flat = Math.exp(scores[i] - lse(scores)) - Number(i === target);
      const plus = [...scores],
        minus = [...scores],
        epsilon = 1e-5;
      plus[i] += epsilon;
      minus[i] -= epsilon;
      near(
        (total(plus) - total(minus)) / (2 * epsilon),
        flat + 0.5 * auxiliary,
        1e-8,
      );
      near(
        (kindTerms(plus, target, kinds).loss -
          kindTerms(minus, target, kinds).loss) /
          (2 * epsilon),
        auxiliary,
        1e-8,
      );
    }
  });
  check("actual batch mean includes singleton-kind zeros", () => {
    const loss = kindTerms([2, 1], 1, [0, 1]).loss;
    const zero = kindTerms([1, 0], 0, [2, 2]).loss;
    near((loss + zero + zero) / 3, loss / 3);
    assert.notEqual(loss / 3, loss);
  });
  check("tail batch uses three instead of eight", () => {
    assert.equal(339 % proposal.training.batchSize, 3);
    assert.equal([1, 2, 3].reduce((a, b) => a + b, 0) / 3, 2);
    assert.notEqual(6 / 8, 2);
  });
  check("actual raw argmax and auxiliary kind routing differ", () => {
    const scores = [4, -10, 3.5],
      row = kindTerms(scores, 0, [0, 0, 1]);
    assert.equal(scores.indexOf(Math.max(...scores)), 0);
    assert.equal(row.present[row.logits.indexOf(Math.max(...row.logits))], 1);
    assert.equal([4, 4, 1].indexOf(Math.max(4, 4, 1)), 0);
  });
  check("invalid target, empty/nonfinite scores and masks rejected", () => {
    for (const args of [
      [[1], 1, [0]],
      [[], 0, []],
      [[Infinity], 0, [0]],
      [[NaN], 0, [0]],
      [[1, 2], 1, [0]],
      [[1], 0, [-1]],
      [[1], 0, [3]],
      [[1], 0, [0.5]],
    ])
      assert.throws(() => kindTerms(...args));
  });
  for (const [name, mutate] of [
    ["changed kind weight", (p) => (p.objective.kindWeight = 1)],
    ["enabled point term", (p) => (p.objective.pointWeight = 0.5)],
    [
      "changed original gate",
      (p) => (p.screening.originalFourGate.minimumActualPointCorrect = 1),
    ],
    [
      "relaxed coordinate guard",
      (p) =>
        (p.screening.tradeoffGuards.pointOracleAtLeastConcurrentControl = false),
    ],
    ["changed model", (p) => p.model.parameters++],
    ["unapproved execution", (p) => (p.status = "USER_APPROVED")],
    ["extra run budget", (p) => p.forwardBudget.totalTopLevelMax++],
    [
      "changed evidence hash",
      (p) => (p.evidence.comparison.sha256 = "0".repeat(64)),
    ],
  ])
    check(`${name} rejected`, () => {
      const changed = structuredClone(proposal);
      mutate(changed);
      assert.throws(() => validateProposal(changed, root));
    });
  return {
    status: "PASS",
    ...result,
    proposalSha256: hash(root, proposalPath),
    syntheticChecks: checks.length,
    checks,
    actual: { modelCreation: 0, checkpointLoad: 0, forward: 0, training: 0 },
    limitation:
      "Node double scalar checks do not verify PyTorch FP32 autograd, model execution or training reproducibility.",
  };
}

if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
) {
  assert.equal(
    process.argv.length,
    2,
    "No execution or training arguments supported",
  );
  process.stdout.write(JSON.stringify(validate(), null, 2) + "\n");
}
