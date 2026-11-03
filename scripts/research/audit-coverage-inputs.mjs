import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  root,
  json,
  hash,
  write,
  reserveFitOutput,
  parseTransferArgs,
  auditMlpStored,
} from "./outcome-mlp-contract.mjs";

const digest = (value) =>
  createHash("sha256").update(JSON.stringify(value)).digest("hex");
/** Hash grouping only: reflecting a saved board neither generates fixtures nor validates augmentation. */
export function boardGroup(board) {
  assert(Array.isArray(board) && board.length === 16);
  board.forEach((r) =>
    assert(
      Array.isArray(r) && r.length === 10 && r.every((v) => v === 0 || v === 1),
    ),
  );
  return [false, true]
    .flatMap((vertical) =>
      [false, true].map((horizontal) =>
        digest(
          Array.from({ length: 16 }, (_, r) =>
            Array.from(
              { length: 10 },
              (_, c) => board[vertical ? 15 - r : r][horizontal ? 9 - c : c],
            ),
          ),
        ),
      ),
    )
    .sort()[0];
}
function checked(rows) {
  assert(Array.isArray(rows) && rows.length > 0);
  return rows.map((e) => {
    assert(
      typeof e.boardGroup === "string" && /^[a-f0-9]{64}$/.test(e.boardGroup),
    );
    assert(
      Array.isArray(e.candidates) &&
        e.candidates.length > 1 &&
        Number.isInteger(e.label) &&
        e.label >= 0 &&
        e.label < e.candidates.length,
    );
    const vectors = e.candidates.map((v) => {
      assert(
        Array.isArray(v) &&
          v.length === 22 &&
          v.every(
            (x) =>
              typeof x === "number" && Number.isFinite(x) && x >= 0 && x <= 1,
          ),
      );
      return JSON.stringify(v.map(Math.fround));
    });
    return {
      group: e.boardGroup,
      vectors,
      target: vectors[e.label],
      label: e.label,
      bag: JSON.stringify(
        vectors.map((v, i) => JSON.stringify([i === e.label, v])).sort(),
      ),
    };
  });
}
/** Model-visible FP32 novelty, preserving duplicates, label tags and each denominator. */
export function auditCoverageSplits(training, evaluations) {
  const train = checked(training),
    vectors = new Set(train.flatMap((e) => e.vectors)),
    groups = new Set(train.map((e) => e.group)),
    bags = new Set(train.map((e) => e.bag));
  assert(
    evaluations &&
      typeof evaluations === "object" &&
      !Array.isArray(evaluations) &&
      Object.keys(evaluations).length > 0,
  );
  const labels = new Map();
  for (const e of train)
    e.vectors.forEach((v, i) => {
      const s = labels.get(v) || new Set();
      s.add(i === e.label);
      labels.set(v, s);
    });
  const result = {
    training: {
      examples: train.length,
      candidateRows: train.reduce((n, e) => n + e.vectors.length, 0),
      uniqueBoardGroups: groups.size,
      uniqueFp32Vectors: vectors.size,
      conflictingFp32Labels: [...labels.values()].filter((s) => s.size > 1)
        .length,
    },
    evaluations: {},
    evaluationPairs: [],
  };
  const all = {};
  for (const [name, raw] of Object.entries(evaluations)) {
    assert(/^[a-z][a-z0-9-]*$/.test(name));
    const rows = checked(raw);
    all[name] = rows;
    const novel = rows.flatMap((e) => e.vectors).filter((v) => !vectors.has(v));
    result.evaluations[name] = {
      examples: rows.length,
      candidateRows: rows.reduce((n, e) => n + e.vectors.length, 0),
      uniqueBoardGroups: new Set(rows.map((e) => e.group)).size,
      boardGroupOverlap: rows.filter((e) => groups.has(e.group)).length,
      labelPreservingBagMatches: rows.filter((e) => bags.has(e.bag)).length,
      unseenTargetRows: rows.filter((e) => !vectors.has(e.target)).length,
      unseenCandidateRows: novel.length,
      unseenUniqueFp32Vectors: new Set(novel).size,
    };
  }
  for (const [a, rows] of Object.entries(all))
    for (const [b, other] of Object.entries(all))
      if (a < b) {
        const bg = new Set(rows.map((e) => e.group)),
          v = new Set(rows.flatMap((e) => e.vectors));
        result.evaluationPairs.push({
          a,
          b,
          boardGroupOverlap: other.filter((e) => bg.has(e.group)).length,
          candidateRowsSeenInOther: other
            .flatMap((e) => e.vectors)
            .filter((x) => v.has(x)).length,
        });
      }
  return result;
}
export function coveragePreflightGate(
  audit,
  expectedEvaluationExamples = 16,
  candidatesEach = 46,
) {
  const checks = {
    splitNames:
      JSON.stringify(Object.keys(audit.evaluations).sort()) ===
      JSON.stringify(["dev", "final"]),
    training32: audit.training.examples === 32,
    trainingCandidateRows: audit.training.candidateRows === 32 * candidatesEach,
    trainingGroups32: audit.training.uniqueBoardGroups === 32,
    noConflictingTrainingLabels: audit.training.conflictingFp32Labels === 0,
    allEvaluationExamplesAndCandidates: true,
    allBoardGroupsUnique: true,
    noTrainingGroupOverlap: true,
    noTrainingBagMatches: true,
    allTargetsUnseen: true,
    allCandidateRowsUnseen: true,
    noDevFinalGroupOrFp32Overlap: true,
  };
  for (const e of Object.values(audit.evaluations)) {
    checks.allEvaluationExamplesAndCandidates &&=
      e.examples === expectedEvaluationExamples &&
      e.candidateRows === expectedEvaluationExamples * candidatesEach;
    checks.allBoardGroupsUnique &&= e.uniqueBoardGroups === e.examples;
    checks.noTrainingGroupOverlap &&= e.boardGroupOverlap === 0;
    checks.noTrainingBagMatches &&= e.labelPreservingBagMatches === 0;
    checks.allTargetsUnseen &&= e.unseenTargetRows === e.examples;
    checks.allCandidateRowsUnseen &&= e.unseenCandidateRows === e.candidateRows;
  }
  checks.noDevFinalGroupOrFp32Overlap =
    audit.evaluationPairs.length === 1 &&
    audit.evaluationPairs.every(
      (p) => p.boardGroupOverlap === 0 && p.candidateRowsSeenInOther === 0,
    );
  return {
    status: Object.values(checks).every(Boolean) ? "PASS" : "FAIL",
    checks,
  };
}
export function storedCoverageDiagnosis(args) {
  const recorded = resolve(root, args.recorded),
    output = reserveFitOutput(args.output, root, recorded);
  try {
    const prior = auditMlpStored(recorded);
    const convert = (e) => ({
      boardGroup: boardGroup(e.observation.planes[0]),
      label: e.label,
      candidates: e.candidateFeatures.map((v) => v.slice(52)),
    });
    const audit = auditCoverageSplits(
      json(resolve(recorded, "fit-dataset.json")).examples.map(convert),
      {
        diagnostic: json(
          resolve(recorded, "transfer-dataset.json"),
        ).examples.map(convert),
      },
    );
    const summary = {
      schemaVersion: 1,
      id: "coverage-input-diagnosis-v1",
      status: "PASS_STORED_AUDIT",
      sourceResultSha256: hash(
        resolve(root, "assets/research/outcome-mlp-20261103-v1.json"),
      ),
      previousCandidateCorrect: prior.candidate.transfer.correct,
      audit,
      interpretation:
        "Distinct board groups can share every model-visible FP32 input; old144 remains a known-input diagnostic, never a new holdout.",
      additionalPythonModelLoadForwardTrainingDomainDataset: 0,
    };
    write(resolve(output, "summary.json"), summary);
    return summary;
  } catch (error) {
    write(resolve(output, "failure.json"), {
      status: "FAIL",
      error: String(error),
      automaticRetry: false,
    });
    throw error;
  }
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
)
  console.log(
    JSON.stringify(
      storedCoverageDiagnosis(
        parseTransferArgs("replay", process.argv.slice(2)),
      ),
    ),
  );
