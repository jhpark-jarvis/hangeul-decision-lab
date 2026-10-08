import { describe, expect, it } from "vitest";
import {
  buildImitationDataset,
  actionKey,
  encodeObservation,
} from "../../src/research/imitation";
import { getAvailableActions } from "../../src/domain/game/actions";
import { solveTurn } from "../../src/domain/solver/solver";
import { getInitialCatalog } from "../../src/domain/pieces/catalog";
import { game } from "../game/fixtures";
import {
  describeDataset,
  describeExamples,
  physicalActionKey,
  runTeacherAudit,
  transformAuditState,
  validateAuditProtocol,
  verifyCandidatePath,
  type AuditProtocol,
} from "../../src/research/teacher-audit";
const spec = {
  trainSeeds: [1],
  devSeeds: [2],
  testSeeds: [3],
  families: ["sparse"] as "sparse"[],
  maxActions: 1,
  teacher: { maxNodes: 8, maxAlternatives: 3, useMemoization: true },
};
function fixture() {
  const r = buildImitationDataset(spec, () => 0);
  if (!r.ok || r.value.status !== "PASS") throw new Error("Invalid fixture");
  return r.value;
}
const protocol = (): AuditProtocol => ({
  schemaVersion: 1,
  id: "test-only",
  trainSeeds: [1],
  devSeeds: [2],
  families: ["sparse"],
  actionIndex: 0,
  solver: spec.teacher,
  variants: ["baseline", "reverse-array", "cycle-slots"],
});
describe("read-only teacher audit", () => {
  it("preserves all physical legal actions and pending target identity under a slot bijection", () => {
    const state = game({ abilities: { singleCell: 1, reroll: 1 } });
    const snapshot = structuredClone(state);
    const a = getAvailableActions(state),
      b = getAvailableActions(transformAuditState(state, "cycle-slots"));
    if (!a.ok || !b.ok) throw new Error("Invalid state");
    expect(a.actions.map(physicalActionKey).sort()).toEqual(
      b.actions.map(physicalActionKey).sort(),
    );
    expect(a.actions.map(actionKey)).not.toEqual(b.actions.map(actionKey));
    state.pendingReroll = {
      instanceId: state.remainingPieces[0].instanceId,
      pieceIndex: 0,
    };
    expect(transformAuditState(state, "cycle-slots").pendingReroll).toEqual({
      ...state.pendingReroll,
      pieceIndex: 1,
    });
    state.pendingReroll = null;
    expect(state).toEqual(snapshot);
  });
  it("counts exact ranks and empty subgroups without pretending empty is zero accuracy", () => {
    const pack = fixture(),
      e = structuredClone(pack.examples.train[0]);
    e.label = 0;
    const second = structuredClone(e);
    second.label = second.candidateKeys.length - 1;
    const result = describeExamples([e, second]);
    expect(result.count).toBe(2);
    expect(result.firstLegalFraction).toBe(0.5);
    expect(result.meanNormalizedLabelRank).toBe(0.5);
    expect(describeExamples([]).firstLegalFraction).toBeNull();
    second.label = NaN;
    expect(() => describeExamples([second])).toThrow("Invalid label");
  });
  it("detects model-observation overlap despite different fixture/instance audit names", () => {
    const pack = fixture(),
      base = structuredClone(pack.examples.train[0]);
    const dev = structuredClone(base);
    dev.split = "dev";
    dev.seed = 2;
    dev.id = "dev-independent-id";
    dev.state.remainingPieces.forEach((p) => {
      p.instanceId += "different";
    });
    const obs = encodeObservation(dev.state);
    if (!obs.ok) throw new Error(obs.error);
    dev.observation = obs.value;
    pack.examples.dev = [dev];
    expect(
      describeDataset(pack).exactModelObservationOverlap["train-dev"],
    ).toBe(1);
  });
  it("checks returned path/physical state/rewards and rejects forged evaluation", () => {
    const state = game(),
      result = solveTurn(state, getInitialCatalog(), spec.teacher);
    if (!result.ok) throw new Error(result.error.message);
    verifyCandidatePath(state, result.result);
    result.result.evaluation.clearedRows++;
    expect(() => verifyCandidatePath(state, result.result)).toThrow("mismatch");
  });
  it("runs actual DFS probes with an unchanged array control, no test probes and no mutations", () => {
    const pack = fixture(),
      before = JSON.stringify(pack);
    const completed: unknown[] = [];
    const result = runTeacherAudit(
      pack,
      protocol(),
      () => 0,
      (n, row) => {
        expect(n).toBe(completed.length + 1);
        completed.push(row);
      },
    );
    expect(completed).toEqual(result.rows);
    expect(completed[0]).not.toBe(result.rows[0]);
    expect(result.summary.states).toBe(2);
    expect(result.summary.probes).toBe(6);
    expect(result.summary.arrayControlsUnchanged).toBe(2);
    expect(result.rows.every((r) => r.split !== ("test" as string))).toBe(true);
    expect(result.summary.returnedPathsVerified).toBeGreaterThanOrEqual(6);
    expect(JSON.stringify(pack)).toBe(before);
    expect(() => runTeacherAudit(pack, protocol(), () => NaN)).toThrow(
      "probe/clock",
    );
    pack.examples.train[0].teacher.search.maxNodes++;
    expect(() => runTeacherAudit(pack, protocol(), () => 0)).toThrow(
      "Stored teacher",
    );
  });
  it("rejects test seeds, duplicates, noninitial states and oversized protocols", () => {
    const pack = fixture();
    for (const bad of [
      null,
      { ...protocol(), trainSeeds: [3] },
      { ...protocol(), trainSeeds: [1, 1] },
      { ...protocol(), actionIndex: 1 },
      { ...protocol(), solver: { ...spec.teacher, maxNodes: 1024 } },
    ])
      expect(() => validateAuditProtocol(bad, pack)).toThrow();
  });
});
