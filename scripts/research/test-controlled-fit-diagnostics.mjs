import assert from "node:assert/strict";
import test from "node:test";
import { storedFitDiagnostics } from "./controlled-fit-diagnostics.mjs";
const examples = Array.from({ length: 16 }, (_, i) => ({
  id: `controlled-fit-${i}`,
  label: 1,
}));
function row(e, correct) {
  const logits = Array(31).fill(0);
  logits[correct ? 1 : 0] = 2;
  const max = 2,
    nll = Math.fround(
      Math.log(logits.reduce((a, v) => a + Math.exp(v - max), 0)) +
        max -
        logits[1],
    );
  return {
    id: e.id,
    label: 1,
    index: correct ? 1 : 0,
    correct,
    nll,
    rank: correct ? 1 : 2,
    margin: correct ? 2 : -2,
    logits,
  };
}
const initial = examples.map((e) => row(e, false)),
  final = examples.map((e, i) => row(e, i < 15));
const history = Array.from({ length: 200 }, (_, i) => ({
  step: i + 1,
  meanTrainingNll: 3 - i / 100,
}));
test("saved transitions conserve the fit denominator without pooling repeats", () => {
  const d = storedFitDiagnostics(initial, final, history, examples);
  assert.deepEqual(d.transitions, {
    bothCorrect: 0,
    recovered: 15,
    regressed: 0,
    bothWrong: 1,
  });
  assert.equal(
    Object.values(d.finalTargetRanks).reduce((a, b) => a + b),
    16,
  );
  assert.equal(d.finalNonpositiveTargetMargins, 1);
});
test("correct argmax and stringent NLL are separate diagnostics", () => {
  const d = storedFitDiagnostics(initial, final, history, examples);
  assert.equal(d.finalTargetRanks.one, 15);
  assert.equal(d.finalNllAboveAcceptanceThreshold, 16);
  assert(d.targetProbabilityFromStoredNll.mean < 1);
});
test("loss milestones use saved pre-update rows only and reject missing or reordered steps", () => {
  const d = storedFitDiagnostics(initial, final, history, examples);
  assert.deepEqual(
    d.savedTrainingLossMilestones.map((r) => r.step),
    [1, 25, 50, 75, 100, 125, 150, 175, 200],
  );
  assert.throws(() =>
    storedFitDiagnostics(initial, final, history.slice(1), examples),
  );
  const h = structuredClone(history);
  h[1].step = 1;
  assert.throws(() => storedFitDiagnostics(initial, final, h, examples));
});
test("public aggregate excludes observation IDs, actions and coordinate predictions", () => {
  const serialized = JSON.stringify(
    storedFitDiagnostics(initial, final, history, examples),
  );
  for (const text of [
    "controlled-fit-0",
    "candidateKeys",
    '"logits"',
    '"row"',
    '"col"',
  ])
    assert(!serialized.includes(text));
  const invalid = structuredClone(final);
  invalid[0].rank = 2;
  assert.throws(() =>
    storedFitDiagnostics(initial, invalid, history, examples),
  );
});
