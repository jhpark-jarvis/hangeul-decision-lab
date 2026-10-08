import { describe, expect, it } from "vitest";
import proposal from "../../research/experiments/teacher-quality-v1.json";
import { applyAction } from "../../src/domain/game/actions";
import {
  assertCoverage,
  exhaustiveOrdinaryReference,
  qualityFixtures,
  runQualityPreflight,
  validateQualityProtocol,
} from "../../src/research/teacher-quality";
import { physicalActionKey } from "../../src/research/teacher-audit";
describe("teacher quality design and scripted preflight", () => {
  it("validates a proposed-only split with untouched detached reserved seeds", () => {
    const p = validateQualityProtocol(proposal);
    expect(p.status).toBe("PROPOSED_NOT_RUN");
    p.splits.reservedTest.reverse();
    expect(proposal.splits.reservedTest).toEqual([6000, 6001, 6002, 6003]);
  });
  it("rejects old/overlapping/fractional seeds and dangerous status/scope/budget/time changes", () => {
    for (const bad of [
      null,
      { ...proposal, id: 1 },
      { ...proposal, status: "ACCEPTED" },
      { ...proposal, scope: "model-training" },
      { ...proposal, nodeBudgets: [512, 8192] },
      { ...proposal, timeComparison: "EQUAL_NODE_MEANS_EQUAL_TIME" },
      { ...proposal, splits: { ...proposal.splits, train: [3000] } },
      { ...proposal, splits: { ...proposal.splits, train: [4000, 4000] } },
      { ...proposal, splits: { ...proposal.splits, reservedTest: [5000] } },
      { ...proposal, splits: { ...proposal.splits, dev: [5000.5] } },
      { ...proposal, reference: { ...proposal.reference, maxNodes: 100000 } },
    ])
      expect(() => validateQualityProtocol(bad)).toThrow();
  });
  it("enumerates all one-dot leaves, roots and legal transitions without changing input", () => {
    const f = qualityFixtures()[0],
      before = JSON.stringify(f);
    const result = exhaustiveOrdinaryReference(f.state);
    expect(result.complete).toBe(true);
    expect(result.visitedNodes).toBe(17);
    expect(result.leaves).toBe(16);
    expect(result.optimalFirstActions!.length).toBeGreaterThan(0);
    expect(JSON.stringify(f)).toBe(before);
    for (const key of result.optimalFirstActions!) {
      const data = JSON.parse(key);
      expect(data[0]).toBe("place-piece");
      expect(data[1]).toBe(f.state.remainingPieces[0].instanceId);
    }
  });
  it("never exposes an optimal action set from a truncated traversal, even with a found leaf", () => {
    const state = qualityFixtures()[0].state;
    expect(exhaustiveOrdinaryReference(state, 1)).toMatchObject({
      complete: false,
      visitedNodes: 1,
      bestObserved: null,
      optimalFirstActions: null,
    });
    expect(exhaustiveOrdinaryReference(state, 2)).toMatchObject({
      complete: false,
      leaves: 1,
      optimalFirstActions: null,
    });
    expect(exhaustiveOrdinaryReference(state, 17).complete).toBe(true);
  });
  it("rejects unsupported scope and invalid limits before traversal", () => {
    const state = qualityFixtures()[0].state;
    for (const cap of [0, 1025, NaN])
      expect(() => exhaustiveOrdinaryReference(state, cap)).toThrow();
    state.pendingReroll = {
      instanceId: state.remainingPieces[0].instanceId,
      pieceIndex: 0,
    };
    expect(() => exhaustiveOrdinaryReference(state)).toThrow(
      "Reference requires",
    );
    state.pendingReroll = null;
    state.remainingPieces[0].piece.shape = [[true, true]];
    expect(() => exhaustiveOrdinaryReference(state)).toThrow("canonical");
  });
  it("checks acquisition, retention, spending and pending reroll; rejects forged coverage", () => {
    const fixtures = qualityFixtures();
    for (const f of fixtures.filter((f) => f.action)) {
      const r = applyAction(f.state, f.action);
      if (!r.ok) throw new Error(r.error.message);
      assertCoverage(f.id, r.state, r.info);
      if (f.id === "item-acquisition") {
        r.info.acquiredItems = [];
        expect(() => assertCoverage(f.id, r.state, r.info)).toThrow(
          "Coverage mismatch",
        );
      }
    }
    const blocked = exhaustiveOrdinaryReference(fixtures[2].state);
    expect(blocked.optimalFirstActions).toEqual([]);
    expect(blocked.leaves).toBe(1);
  });
  it("matches complete production trees with memo on/off, replaying every returned path", () => {
    const r = runQualityPreflight(proposal);
    expect(r.status).toBe("PASS");
    expect(r.summary).toMatchObject({
      scriptedCases: 7,
      completeReferences: 7,
      productionComparisons: 14,
      scriptedTransitions: 4,
      generatedReservedTestStates: 0,
      trainingRuns: 0,
    });
    expect(r.summary.returnedPathsVerified).toBeGreaterThanOrEqual(14);
    for (const row of r.rows)
      for (const c of row.comparisons)
        if (c.result.actions.length)
          expect(row.reference.optimalFirstActions).toContain(
            physicalActionKey(c.result.actions[0]),
          );
  }, 20000);
});
