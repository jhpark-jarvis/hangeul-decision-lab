import assert from "node:assert/strict";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  root,
  auditStored,
  auditRows,
  json,
  write,
  reserveFitOutput,
} from "./controlled-point-fit-contract.mjs";
import { parseReplayArgs } from "./replay-controlled-point-fit.mjs";

/** Descriptive saved-output analysis, never a training criterion or policy. */
export function storedFitDiagnostics(initial, final, history, examples) {
  auditRows(initial, examples);
  auditRows(final, examples);
  assert.equal(history.length, 200);
  history.forEach((r, i) => {
    assert.equal(r.step, i + 1);
    assert(Number.isFinite(r.meanTrainingNll) && r.meanTrainingNll >= 0);
  });
  const transitions = {
    bothCorrect: 0,
    recovered: 0,
    regressed: 0,
    bothWrong: 0,
  };
  for (let i = 0; i < 16; i++) {
    const a = initial[i].correct,
      b = final[i].correct;
    transitions[
      a ? (b ? "bothCorrect" : "regressed") : b ? "recovered" : "bothWrong"
    ]++;
  }
  const probabilities = final.map((r) => Math.exp(-r.nll));
  return {
    examples: 16,
    transitions,
    finalTargetRanks: {
      one: final.filter((r) => r.rank === 1).length,
      twoToFive: final.filter((r) => r.rank >= 2 && r.rank <= 5).length,
      sixToTen: final.filter((r) => r.rank >= 6 && r.rank <= 10).length,
      aboveTen: final.filter((r) => r.rank > 10).length,
    },
    finalNllAboveAcceptanceThreshold: final.filter((r) => r.nll > 0.05).length,
    nllThresholdInterpretation:
      "0.05 is the mean acceptance threshold; the per-example exceedance count is descriptive, not an additional individual gate.",
    finalNonpositiveTargetMargins: final.filter((r) => r.margin <= 0).length,
    targetProbabilityFromStoredNll: {
      minimum: Math.min(...probabilities),
      maximum: Math.max(...probabilities),
      mean: probabilities.reduce((a, b) => a + b, 0) / 16,
      geometricMean: Math.exp(-final.reduce((a, r) => a + r.nll, 0) / 16),
      interpretation:
        "exp(-stored float32 native NLL); target mass on 31 legal point candidates, no calibration or held-out claim",
    },
    savedTrainingLossMilestones: [1, 25, 50, 75, 100, 125, 150, 175, 200].map(
      (step) => history[step - 1],
    ),
    lossInterpretation:
      "Per-update mean training NLL before that optimizer update; milestones are descriptive, not extra inference or checkpoint reselection",
    scope:
      "Post-run fit-set description only; fixed gates and final step preserved. No model/weights load, forward, training or tuning.",
  };
}
export function diagnose(args) {
  const recorded = resolve(root, args.recorded),
    output = reserveFitOutput(args.output, root, recorded);
  try {
    const verified = auditStored(recorded);
    const diagnostics = storedFitDiagnostics(
      json(resolve(recorded, "initial.json")),
      json(resolve(recorded, "final1.json")),
      json(resolve(recorded, "history.json")),
      json(resolve(recorded, "dataset.json")).examples,
    );
    const owner = json(resolve(recorded, "report.json")),
      child = json(resolve(recorded, "child-report.json"));
    assert(owner.child.elapsedMs <= 600000);
    const result = {
      ...verified,
      researchVerdict: verified.summary.fitGate.status,
      diagnostics,
      executionTiming: {
        ownerWallSeconds: owner.elapsedSeconds,
        childProcessWallSeconds: owner.child.elapsedMs / 1000,
        pythonBodyWallSeconds: child.budget.wallTimeSeconds,
        pythonBodyExcludesInterpreterImport: true,
        inheritedBudgetChildWallSecondsMeans:
          "Python body only; the Node child process timer and timeout include interpreter/import startup",
      },
    };
    write(resolve(output, "summary.json"), result);
    return result;
  } catch (error) {
    write(resolve(output, "failure.json"), {
      status: "FAIL",
      error: String(error),
    });
    throw error;
  }
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
)
  console.log(JSON.stringify(diagnose(parseReplayArgs(process.argv.slice(2)))));
