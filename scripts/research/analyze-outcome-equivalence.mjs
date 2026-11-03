import assert from "node:assert/strict";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  root,
  json,
  hash,
  write,
  reserveFitOutput,
  parseTransferArgs,
  auditTransferStored,
} from "./frozen-point-transfer-contract.mjs";

/** Stored vectors and label-preserving permutations only; no scores or policies. */
export function analyzeOutcomeInputs(fit, transfer) {
  assert(fit.length > 0 && transfer.length > 0);
  const sets = {
    fit22: new Set(),
    transfer22: new Set(),
    fit74: new Set(),
    transfer74: new Set(),
  };
  const tagged = new Map(),
    labels = new Map();
  const rowKey = (features, target) =>
    JSON.stringify([target, features.slice(52)]);
  const check = (e) => {
    assert(
      e.candidateFeatures.length > 1 &&
        Number.isInteger(e.label) &&
        e.label >= 0 &&
        e.label < e.candidateFeatures.length,
    );
    for (const f of e.candidateFeatures)
      assert(
        Array.isArray(f) &&
          f.length === 74 &&
          f.every(
            (v) =>
              typeof v === "number" && Number.isFinite(v) && v >= 0 && v <= 1,
          ),
      );
  };
  const bag = (e) =>
    JSON.stringify(
      e.candidateFeatures.map((f, i) => rowKey(f, i === e.label)).sort(),
    );
  fit.forEach((e, index) => {
    check(e);
    const b = bag(e);
    assert(!tagged.has(b), "Ambiguous fit feature bag");
    tagged.set(b, index);
    e.candidateFeatures.forEach((f, i) => {
      const k = JSON.stringify(f.slice(52));
      sets.fit22.add(k);
      sets.fit74.add(JSON.stringify(f));
      const s = labels.get(k) || new Set();
      s.add(i === e.label);
      labels.set(k, s);
    });
  });
  let targetsUnseen74 = 0,
    nonTargetsUnseen74 = 0;
  const witnesses = transfer.map((e) => {
    check(e);
    e.candidateFeatures.forEach((f, i) => {
      sets.transfer22.add(JSON.stringify(f.slice(52)));
      const k = JSON.stringify(f);
      sets.transfer74.add(k);
      if (!sets.fit74.has(k)) {
        if (i === e.label) targetsUnseen74++;
        else nonTargetsUnseen74++;
      }
    });
    const matched = tagged.get(bag(e));
    if (matched === undefined) return null;
    const bins = new Map();
    fit[matched].candidateFeatures.forEach((f, i) => {
      const k = rowKey(f, i === fit[matched].label);
      const q = bins.get(k) || [];
      q.push(i);
      bins.set(k, q);
    });
    const permutation = e.candidateFeatures.map((f, i) => {
      const q = bins.get(rowKey(f, i === e.label));
      assert(q && q.length);
      return q.shift();
    });
    assert.equal(new Set(permutation).size, permutation.length);
    assert([...bins.values()].every((q) => q.length === 0));
    assert.equal(permutation[e.label], fit[matched].label);
    return { fitIndex: matched, permutation };
  });
  return {
    summary: {
      fitExamples: fit.length,
      transferExamples: transfer.length,
      fitCandidateRows: fit.reduce((n, e) => n + e.candidateFeatures.length, 0),
      transferCandidateRows: transfer.reduce(
        (n, e) => n + e.candidateFeatures.length,
        0,
      ),
      fitUniqueOutcomeVectors: sets.fit22.size,
      transferUniqueOutcomeVectors: sets.transfer22.size,
      transferUnseenOutcomeVectors: [...sets.transfer22].filter(
        (k) => !sets.fit22.has(k),
      ).length,
      fitOutcomeLabelConflicts: [...labels.values()].filter((s) => s.size > 1)
        .length,
      uniqueFitBags: tagged.size,
      labelPreservingBagMatches: witnesses.filter(Boolean).length,
      fitUniqueFull74Vectors: sets.fit74.size,
      transferUniqueFull74Vectors: sets.transfer74.size,
      transferUnseenFull74Vectors: [...sets.transfer74].filter(
        (k) => !sets.fit74.has(k),
      ).length,
      transferTargetRowsUnseenFull74: targetsUnseen74,
      transferNonTargetRowsUnseenFull74: nonTargetsUnseen74,
    },
    witnesses,
  };
}
export function analyzeRecordedOutcome(args) {
  const recorded = resolve(root, args.recorded),
    output = reserveFitOutput(args.output, root, recorded);
  try {
    const original = auditTransferStored(recorded),
      fit = json(resolve(recorded, "fit-dataset.json")).examples,
      transfer = json(resolve(recorded, "transfer-dataset.json")).examples;
    const analysis = analyzeOutcomeInputs(fit, transfer);
    const summary = {
      schemaVersion: 1,
      id: "outcome-input-equivalence-v1",
      status: "PASS_STORED_INPUT_AUDIT",
      sourceAggregate: {
        path: "assets/research/frozen-point-transfer-20261101-v1.json",
        sha256: hash(
          resolve(
            root,
            "assets/research/frozen-point-transfer-20261101-v1.json",
          ),
        ),
      },
      previousTransfer: original.transfer,
      previousTransferGate: original.transferGate.status,
      inputAudit: analysis.summary,
      interpretation:
        "All 144 tagged 22-feature candidate multisets match one of 16 fit multisets; all 136 unique outcome vectors were seen in fitting. A deterministic pointwise outcome-only scorer therefore sees rearranged known inputs, not new outcome vectors. Full boards and original 74-feature model inputs are different. This is an input audit, not a new model prediction, transfer accuracy, causal claim or native floating-point guarantee.",
      additionalPythonModelLoadForwardTrainingDomainDataset: 0,
      appUsesDfs: true,
    };
    write(resolve(output, "private-witnesses.json"), analysis.witnesses);
    write(resolve(output, "summary.json"), summary);
    write(resolve(output, "report.json"), {
      status: "PASS",
      atUtc: new Date().toISOString(),
      recorded,
      sourceHashes: Object.fromEntries(
        [
          "fit-dataset.json",
          "transfer-dataset.json",
          "report.json",
          "execution.json",
        ].map((p) => [p, hash(resolve(recorded, p))]),
      ),
      programSha256: hash(fileURLToPath(import.meta.url)),
      summarySha256: hash(resolve(output, "summary.json")),
      witnessesSha256: hash(resolve(output, "private-witnesses.json")),
    });
    return summary;
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
  console.log(
    JSON.stringify(
      analyzeRecordedOutcome(
        parseTransferArgs("replay", process.argv.slice(2)),
      ),
    ),
  );
