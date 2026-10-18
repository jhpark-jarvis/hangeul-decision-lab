import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

const defaultRoot = fileURLToPath(new URL("../../", import.meta.url));
const proposalPath =
  "research/experiments/point-only-objective-proposal-v1.json";
const read = (root, path) => readFileSync(resolve(root, path));
const json = (root, path) => JSON.parse(read(root, path).toString("utf8"));
const hash = (root, path) =>
  createHash("sha256").update(read(root, path)).digest("hex");

// Synthetic scalar arithmetic only; this file is not a training criterion.
function lse(values) {
  assert(values.length > 0 && values.every(Number.isFinite));
  const max = Math.max(...values);
  return max + Math.log(values.reduce((sum, x) => sum + Math.exp(x - max), 0));
}
function ce(values, target) {
  assert(Number.isInteger(target) && target >= 0 && target < values.length);
  return lse(values) - values[target];
}
function conditional(scores, target, points) {
  ce(scores, target); // Validate the full target even when the extra term is zero.
  assert(new Set(points).size === points.length);
  assert(
    points.every((i) => Number.isInteger(i) && i >= 0 && i < scores.length),
  );
  if (!points.includes(target)) return 0;
  return ce(
    points.map((i) => scores[i]),
    points.indexOf(target),
  );
}
const near = (a, b, epsilon = 1e-9) => assert(Math.abs(a - b) <= epsilon);

export function validateProposal(proposal, root = defaultRoot) {
  const base = json(
    root,
    "research/experiments/point-objective-proposal-v1.json",
  );
  assert.equal(proposal.schemaVersion, 1);
  assert.equal(proposal.id, "point-only-objective-proposal-v1");
  assert.equal(proposal.status, "PROPOSED_NOT_RUN");
  assert.equal(proposal.approval, "DESIGN_AND_PUBLICATION_ONLY");
  assert.equal(proposal.executionRequiresSeparateUserDecision, true);
  assert.deepEqual(proposal.source, base.source);
  assert.deepEqual(proposal.model, base.model);
  assert.deepEqual(proposal.inference, base.inference);
  assert.deepEqual(proposal.forwardBudget, base.forwardBudget);
  assert.deepEqual(proposal.resources, base.resources);
  assert.equal(proposal.objective.kindWeight, 0);
  assert.equal(proposal.objective.pointWeight, 0.5);
  assert.equal(proposal.objective.candidate, "L_flat + 0.5 * L_point");
  assert.equal(proposal.objective.normalization, base.objective.normalization);
  assert.equal(proposal.objective.numeric, base.objective.numeric);
  assert.equal(proposal.training.runs, 2);
  assert.deepEqual(proposal.training.arms, [
    "original-spatial-flat-reproduction",
    "same-spatial-point-only-aux",
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
  assert.match(proposal.training.baselineControl, /epoch11/);
  assert.match(proposal.training.baselineControl, /STOP/);
  assert.deepEqual(
    proposal.screening.originalFourGate,
    base.screening.originalFourGate,
  );
  assert.deepEqual(proposal.screening.coordinateMechanism, {
    pointTeacherKindOracleCorrectAtLeast: 5,
    minimumGainVsConcurrentControl: 3,
    pointExamples: 13,
    claim:
      "Coordinate screening only. Actual point, kind selection and original four gates must still be reported separately.",
  });
  assert.deepEqual(proposal.screening.historicalCombinedMechanism, {
    pointCorrectKindAtLeast: 7,
    pointTeacherKindOracleCorrectAtLeast: 5,
    pointExamples: 13,
    requiredForClaimingBothImprove: true,
  });
  assert.equal(
    proposal.boundaries.designActualModelCreationLoadForwardTraining,
    0,
  );
  assert.equal(proposal.boundaries.teacherDomainEpisodeTestRolloutGeometry, 0);
  assert.equal(proposal.boundaries.originalFilesImmutable, true);
  assert.equal(proposal.boundaries.newDataLabelsFeaturesArchitecture, false);
  assert.equal(proposal.boundaries.scalePolicyAdoption, false);
  assert.equal(proposal.boundaries.appUsesDfs, true);
  assert.equal(proposal.boundaries.weightsRawComponentsPrivate, true);

  const expectedPaths = {
    summary: "assets/research/point-objective-train-dev-20261018-v1.json",
    joint: "assets/research/point-objective-joint-errors-20261018-v1.json",
  };
  for (const [key, path] of Object.entries(expectedPaths)) {
    assert.equal(proposal.evidence[key].path, path);
    assert.equal(proposal.evidence[key].sha256, hash(root, path));
  }
  assert.deepEqual(proposal.evidence.observedDev, {
    control: {
      actual: 29,
      pointActual: 0,
      pointKind: 4,
      pointOracle: 2,
      reroll: 7,
    },
    combined: {
      actual: 29,
      pointActual: 0,
      pointKind: 5,
      pointOracle: 1,
      reroll: 6,
    },
    denominators: { all: 89, point: 13, reroll: 8 },
  });
  const joint = json(root, expectedPaths.joint);
  for (const [arm, expected] of [
    ["control", [0, 4, 2, 7]],
    ["candidate", [0, 5, 1, 7]],
  ]) {
    const row = joint.models[arm].dev;
    assert.equal(row.examples, 13);
    const partition = [
      row.bothCorrect,
      row.kindOnly,
      row.coordinateOnly,
      row.neither,
    ];
    assert.deepEqual(partition, expected);
    assert.equal(
      partition.reduce((a, b) => a + b, 0),
      13,
    );
  }
  const summary = json(root, expectedPaths.summary);
  assert.equal(summary.gate.status, "FAIL");
  assert.equal(summary.mechanism.status, "FAIL");
  assert.equal(summary.accounting.originalTopLevelForwards, 12840);
  assert.equal(summary.accounting.trainingRuns, 2);
  const { train, dev } = proposal.source.examples;
  assert.deepEqual([train, dev], [339, 89]);
  const budget = proposal.forwardBudget;
  assert.equal(budget.preflight, 2 * (train + dev));
  assert.equal(budget.perArmTraining, proposal.training.epochs * train);
  assert.equal(budget.perArmEpochDev, proposal.training.epochs * dev);
  assert.equal(budget.perArmFrozenTwoRepeats, 2 * (train + dev));
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
  check("non-point observation contributes exactly zero", () => {
    assert.equal(conditional([2, 3, 1], 0, [1, 2]), 0);
  });
  check("absent point group contributes zero", () => {
    assert.equal(conditional([2, 3], 1, []), 0);
  });
  check("singleton conditional CE is zero", () => {
    assert.equal(conditional([2, 3], 1, [1]), 0);
  });
  check("point CE uses original local index", () => {
    near(conditional([2, 3, 1], 2, [1, 2]), ce([3, 1], 1));
  });
  check("extra point loss ignores non-point logits", () => {
    near(
      conditional([2, 3, 1], 2, [1, 2]),
      conditional([200, 3, 1], 2, [1, 2]),
    );
  });
  check("stable shifted conditional scores", () => {
    near(
      conditional([2, 3, 1], 2, [1, 2]),
      conditional([2, 1003, 1001], 2, [1, 2]),
    );
  });
  check("batch mean includes zero terms", () => {
    const point = conditional([2, 3, 1], 2, [1, 2]);
    near([point, 0, 0].reduce((a, b) => a + b, 0) / 3, point / 3);
    assert.notEqual(point / 3, point);
  });
  check("tail batch uses length three instead of eight", () => {
    const terms = [1, 2, 3];
    assert.equal(terms.reduce((a, b) => a + b, 0) / terms.length, 2);
    assert.notEqual(6 / 8, 2);
  });
  check("full CE decomposes with sum-based kind mass", () => {
    const scores = [2, 3, 1];
    near(ce(scores, 2), ce([scores[0], lse([3, 1])], 1) + ce([3, 1], 1));
  });
  check(
    "synthetic loss finite differences match analytical derivatives",
    () => {
      const scores = [0.2, 0.7, -0.3],
        target = 2,
        points = [1, 2];
      const total = (s) => ce(s, target) + 0.5 * conditional(s, target, points);
      for (let i = 0; i < scores.length; i++) {
        const plus = [...scores],
          minus = [...scores],
          epsilon = 1e-5;
        plus[i] += epsilon;
        minus[i] -= epsilon;
        const full = Math.exp(scores[i] - lse(scores)) - Number(i === target);
        const point = points.includes(i)
          ? Math.exp(scores[i] - lse(points.map((j) => scores[j]))) -
            Number(i === target)
          : 0;
        near(
          (total(plus) - total(minus)) / (2 * epsilon),
          full + 0.5 * point,
          1e-8,
        );
      }
    },
  );
  check("raw argmax retains first-index ties", () => {
    const scores = [4, 4, 1];
    assert.equal(scores.indexOf(Math.max(...scores)), 0);
  });
  check("invalid target/group/nonfinite/empty scores are rejected", () => {
    assert.throws(() => conditional([1, 2], 2, [1]));
    assert.throws(() => conditional([1, 2], 1, [1, 1]));
    assert.throws(() => conditional([1, 2], 1, [-1]));
    assert.throws(() => ce([Infinity, 0], 0));
    assert.throws(() => ce([], 0));
  });
  check("changed loss weight is rejected", () => {
    const changed = structuredClone(proposal);
    changed.objective.pointWeight = 1;
    assert.throws(() => validateProposal(changed, root));
  });
  check("changed gate is rejected", () => {
    const changed = structuredClone(proposal);
    changed.screening.originalFourGate.minimumActualPointCorrect = 1;
    assert.throws(() => validateProposal(changed, root));
  });
  check("unapproved execution status is rejected", () => {
    const changed = structuredClone(proposal);
    changed.status = "USER_APPROVED";
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
      "Node double scalar checks do not verify PyTorch FP32 autograd or training reproducibility.",
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
