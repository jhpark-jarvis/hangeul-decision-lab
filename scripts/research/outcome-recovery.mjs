/** Reuse encoded data only after preflight failed before either training child. */
export function validatePretrainingReuse(input) {
  const {
    execution,
    configSha256,
    originalHashes,
    currentSources,
    augmentedSha256,
    progress,
    failure,
    hasTrainingOutput,
  } = input;
  const fail = (ok, message) => {
    if (!ok) throw new Error(message);
  };
  fail(
    execution.status === "FROZEN_BEFORE_EXECUTION" &&
      execution.configSha256 === configSha256,
    "Recovery config must be unchanged",
  );
  fail(
    execution.augmentedSha256 === augmentedSha256 &&
      execution.transitions === 149597,
    "Recovery encoded identity/count",
  );
  fail(
    JSON.stringify(execution.originalHashes) === JSON.stringify(originalHashes),
    "Recovery originals changed",
  );
  fail(
    !hasTrainingOutput &&
      failure.status === "FAIL" &&
      progress.length > 0 &&
      progress.at(-1).phase === "preflight" &&
      progress.every((p) => ["encoding", "preflight"].includes(p.phase)),
    "Recovery forbidden after any training began",
  );
  const allowed = new Set([
    "scripts/research/outcome-cnn.mjs",
    "research/cnn/outcome.py",
  ]);
  for (const [path, digest] of Object.entries(execution.sourceHashes))
    fail(
      currentSources[path] === digest || allowed.has(path),
      "Recovery feature producer/source changed: " + path,
    );
  return {
    stage: "PREFLIGHT_FAILURE_BEFORE_TRAINING",
    totalDomainTransitions: 149597,
    additionalDomainTransitions: 0,
    priorTrainingRuns: 0,
    initialControlRequiredAgain: true,
  };
}
