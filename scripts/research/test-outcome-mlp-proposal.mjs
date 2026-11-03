import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { analyzeOutcomeInputs } from "./analyze-outcome-equivalence.mjs";
import {
  validateMlpProposal,
  mlpBudget,
} from "./validate-outcome-mlp-proposal.mjs";
const features = (x, prefix = 0) => {
  const v = Array(74).fill(0);
  v[0] = prefix;
  v[52] = x;
  return v;
};
const fit = () => [
  {
    candidateFeatures: [features(0.1), features(0.2), features(0.2)],
    label: 0,
  },
];
const proposal = () =>
  JSON.parse(
    readFileSync(
      new URL(
        "../../research/experiments/outcome-mlp-proposal-v1.json",
        import.meta.url,
      ),
      "utf8",
    ),
  );
test("label-preserving outcome permutations produce a bijection despite changed full candidate inputs", () => {
  const e = {
    candidateFeatures: [features(0.2, 1), features(0.1, 1), features(0.2, 1)],
    label: 1,
  };
  const r = analyzeOutcomeInputs(fit(), [e]);
  assert.equal(r.summary.labelPreservingBagMatches, 1);
  assert.equal(r.summary.transferUnseenOutcomeVectors, 0);
  assert.equal(r.summary.transferUnseenFull74Vectors, 2);
  assert.deepEqual(r.witnesses[0], { fitIndex: 0, permutation: [1, 0, 2] });
});
test("set coverage is insufficient when duplicate multiplicities or target tags differ", () => {
  const duplicate = {
    candidateFeatures: [features(0.1), features(0.1), features(0.2)],
    label: 0,
  };
  const swapped = {
    candidateFeatures: [features(0.1), features(0.2), features(0.2)],
    label: 1,
  };
  for (const e of [duplicate, swapped]) {
    const r = analyzeOutcomeInputs(fit(), [e]);
    assert.equal(r.summary.transferUnseenOutcomeVectors, 0);
    assert.equal(r.summary.labelPreservingBagMatches, 0);
    assert.equal(r.witnesses[0], null);
  }
});
test("new vectors, ambiguous fit bags, invalid feature shapes/nonfinite values and labels are caught", () => {
  const r = analyzeOutcomeInputs(fit(), [
    { candidateFeatures: [features(0.1), features(0.3)], label: 0 },
  ]);
  assert.equal(r.summary.transferUnseenOutcomeVectors, 1);
  assert.throws(() => analyzeOutcomeInputs([...fit(), ...fit()], fit()));
  for (const mutate of [
    (e) => (e.label = -1),
    (e) => e.candidateFeatures[0].pop(),
    (e) => (e.candidateFeatures[0][52] = NaN),
    (e) => (e.candidateFeatures[0][1] = true),
  ]) {
    const e = fit()[0];
    mutate(e);
    assert.throws(() => analyzeOutcomeInputs(fit(), [e]));
  }
});
test("conflicting outcome labels are reported rather than mistaken for a learned success", () => {
  const e = {
    candidateFeatures: [features(0.1), features(0.1), features(0.2)],
    label: 0,
  };
  const r = analyzeOutcomeInputs([e], [e]);
  assert.equal(r.summary.fitOutcomeLabelConflicts, 1);
  assert.equal(r.summary.labelPreservingBagMatches, 1);
});
test("static proposal fixes ReLU/bias/769 parameters, frozen data and the complete 32672-call arithmetic", () => {
  assert.equal(mlpBudget().total, 16336);
  const r = validateMlpProposal(proposal());
  assert.equal(r.proposedTotalForwards, 32672);
  assert.equal(r.execution, "NOT_RUN");
  assert.equal(r.actualPythonModelLoadForwardTrainingDomainDataset, 0);
});
test("static CLI fails closed on an execution argument before loading any data", () => {
  const result = spawnSync(
    process.execPath,
    [
      fileURLToPath(
        new URL("./validate-outcome-mlp-proposal.mjs", import.meta.url),
      ),
      "--run",
    ],
    { encoding: "utf8", windowsHide: true, timeout: 10000 },
  );
  assert.equal(result.status, 1);
  assert.equal(result.stdout, "");
  assert.match(result.stderr, /accepts no execution\/tuning arguments/);
});
test("activation/input/training/ledger/control/gate/scope or source/evidence drift is rejected", () => {
  for (const mutate of [
    (p) => (p.contrast.candidate.activation = "Tanh"),
    (p) => (p.contrast.candidate.layers[0] = 74),
    (p) => p.contrast.candidate.parameters++,
    (p) => (p.shared.training.learningRate = 0.002),
    (p) => (p.shared.training.updates = 2000),
    (p) => p.forwardBudget.totalTopLevelMax++,
    (p) => (p.forwardBudget.extraTelemetryOrWarmup = 1),
    (p) => (p.controls.order = "candidate first"),
    (p) => (p.screening.transfer.correct = 109),
    (p) => (p.boundaries.modelRunnerImplemented = true),
    (p) => (p.status = "APPROVED"),
    (p) => (p.evidence.sha256 = "0".repeat(64)),
    (p) => (p.sourceHashes["research/cnn/model.py"] = "0".repeat(64)),
  ]) {
    const p = proposal();
    mutate(p);
    assert.throws(() => validateMlpProposal(p));
  }
});
