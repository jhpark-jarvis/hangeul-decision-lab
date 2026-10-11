import { describe, expect, it } from "vitest";
// @ts-expect-error Local Node CLI helper deliberately uses native ESM.
import { validatePretrainingReuse } from "../../scripts/research/outcome-recovery.mjs";

const fixture = () => ({
  execution: {
    status: "FROZEN_BEFORE_EXECUTION",
    configSha256: "config",
    augmentedSha256: "encoded",
    transitions: 149597,
    originalHashes: { original: "same" },
    sourceHashes: {
      "src/research/action-outcome.ts": "producer",
      "research/cnn/outcome.py": "old",
    },
  },
  configSha256: "config",
  augmentedSha256: "encoded",
  originalHashes: { original: "same" },
  currentSources: {
    "src/research/action-outcome.ts": "producer",
    "research/cnn/outcome.py": "fixed",
  },
  progress: [{ phase: "encoding" }, { phase: "preflight" }],
  failure: { status: "FAIL" },
  hasTrainingOutput: false,
});
describe("pretraining encoded reuse", () => {
  it("reuses the same encoded bytes with zero new transitions and no prior training", () => {
    expect(validatePretrainingReuse(fixture())).toMatchObject({
      additionalDomainTransitions: 0,
      priorTrainingRuns: 0,
      initialControlRequiredAgain: true,
    });
  });
  it("rejects training, source, config, original and encoded drift", () => {
    for (const input of [
      { ...fixture(), hasTrainingOutput: true },
      { ...fixture(), progress: [{ phase: "baseline" }] },
      { ...fixture(), configSha256: "changed" },
      { ...fixture(), augmentedSha256: "changed" },
      { ...fixture(), originalHashes: { original: "changed" } },
      {
        ...fixture(),
        currentSources: { "src/research/action-outcome.ts": "changed" },
      },
    ])
      expect(() => validatePretrainingReuse(input)).toThrow();
  });
});
