import { describe, expect, it } from "vitest";
import protocol from "../../research/cnn/paired-protocol.json";
import { getAvailableActions } from "../../src/domain/game/actions";
import { getInitialCatalog } from "../../src/domain/pieces/catalog";
import { solveTurn } from "../../src/domain/solver/solver";
import { qualityFixtures } from "../../src/research/teacher-quality";
import { runEpisode, stateKey } from "../../src/research/episode";
import { createSyntheticFixture } from "../../src/research/fixture";
import {
  pairDecision,
  pauseEpisode,
  registerPair,
  validatePairedProtocol,
  type Pair,
  type RegistryEntry,
} from "../../src/research/paired-teacher";
function paired() {
  const state = qualityFixtures()[0].state,
    legal = getAvailableActions(state);
  if (!legal.ok) throw new Error(legal.error.message);
  const teachers = {} as Pair["teachers"];
  for (const name of ["A512", "A1024"] as const) {
    const solved = solveTurn(
      state,
      getInitialCatalog(),
      protocol.teachers[name],
    );
    if (!solved.ok) throw new Error(solved.error.message);
    teachers[name] = { result: solved.result, latencyMs: 0 };
  }
  const p = pairDecision(state, legal.actions, teachers, {
    id: "unit-pair",
    fixtureId: "unit",
    seed: 1,
    split: "train",
    actionIndex: 0,
  });
  if (!p) throw new Error("Expected nonempty pair");
  return p;
}
describe("paired teacher contract", () => {
  it("freezes the approved experiment and rejects seed/freeze/teacher/training drift", () => {
    expect(validatePairedProtocol(protocol)).toEqual(protocol);
    for (const mutate of [
      (p: typeof protocol) => p.dataset.trainSeeds.push(3000),
      (p: typeof protocol) => p.dataset.reservedTestSeeds.pop(),
      (p: typeof protocol) => (p.dataset.testGate = "early-test"),
      (p: typeof protocol) => p.training.epochs++,
      (p: typeof protocol) => (p.teachers.A1024.maxNodes = 3072),
      (p: typeof protocol) => (p.boundaries.appModel = true),
    ]) {
      const changed = structuredClone(protocol);
      mutate(changed);
      expect(() => validatePairedProtocol(changed)).toThrow(
        "Invalid frozen paired protocol",
      );
    }
    expect(() =>
      validatePairedProtocol({ ...protocol, extra: true }),
    ).toThrow();
  });
  it("retains the full shared encoding while allowing different labels", () => {
    const p = paired();
    expect(p.candidateFeatures.length).toBe(p.candidateKeys.length);
    for (const n of ["A512", "A1024"] as const)
      expect(p.labels[n]).toBeGreaterThanOrEqual(0);
    const original = stateKey(p.state);
    p.teachers.A512.result.finalState.abilities.reroll++;
    expect(stateKey(p.state)).toBe(original);
  });
  it("skips a pair when one teacher abstains, without inventing a label", () => {
    const p = paired(),
      legal = getAvailableActions(p.state);
    if (!legal.ok) throw new Error("legal");
    const empty = structuredClone(p.teachers);
    empty.A1024.result.actions = [];
    empty.A1024.result.alternatives = [];
    empty.A1024.result.finalState = structuredClone(p.state);
    // An abstention must still have a valid replay/evaluation; use a real zero-action frontier.
    const zero = solveTurn(p.state, getInitialCatalog(), {
      maxNodes: 1,
      maxAlternatives: 3,
      useMemoization: true,
    });
    if (!zero.ok) throw new Error("zero");
    empty.A1024.result = {
      ...zero.result,
      search: { ...zero.result.search, maxNodes: 1024 },
    };
    expect(pairDecision(p.state, legal.actions, empty, p)).toBeNull();
  });
  it("keeps the first duplicate and records label conflicts, rejecting split leakage", () => {
    const p = paired(),
      registry: RegistryEntry[] = [];
    expect(registerPair(p, registry, (s) => s).keep).toBe(true);
    const duplicate = structuredClone(p);
    duplicate.id = "later";
    duplicate.labels.A1024 =
      (duplicate.labels.A1024 + 1) % duplicate.candidateKeys.length;
    expect(registerPair(duplicate, registry, (s) => s)).toMatchObject({
      keep: false,
      priorId: p.id,
      labelConflict: true,
    });
    duplicate.split = "dev";
    expect(() => registerPair(duplicate, registry, (s) => s)).toThrow(
      "leaked across splits",
    );
  });
  it("semantic duplicates ignore instance naming even when forward slot inputs differ", () => {
    const p = paired(),
      registry: RegistryEntry[] = [];
    registerPair(p, registry, (s) => s);
    const renamed = structuredClone(p);
    renamed.state.remainingPieces.forEach(
      (piece) => (piece.instanceId = "other:" + piece.instanceId),
    );
    renamed.observation.metadata[0] = 1 - renamed.observation.metadata[0];
    renamed.split = "test";
    expect(() => registerPair(renamed, registry, (s) => s)).toThrow(
      "leaked across splits",
    );
  });
  it("pauses and replays domain prefixes exactly without exposing future tape", () => {
    const f = createSyntheticFixture(1, "sparse", 2);
    if (!f.ok) throw new Error(f.error);
    const first = pauseEpisode(f.value, []);
    expect(first.pending).not.toBeNull();
    expect(Object.keys(first.pending!)).toEqual([
      "state",
      "legalActions",
      "actionIndex",
    ]);
    const h = [{ action: first.pending!.legalActions[0] }],
      second = pauseEpisode(f.value, h);
    expect(second.pending?.actionIndex).toBe(1);
    h.push({ action: second.pending!.legalActions[0] });
    const done = pauseEpisode(f.value, h),
      direct = runEpisode(f.value, (o) => h[o.actionIndex]);
    expect(done.pending).toBeNull();
    expect(direct.ok && direct.value).toEqual(done.result);
  });
});
