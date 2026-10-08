import { describe, expect, it, vi, afterEach } from "vitest";
import proposal from "../../research/experiments/teacher-quality-v1.json";
import {
  freezeMatrixProtocol,
  pathCoverage,
  runTeacherMatrix,
} from "../../src/research/teacher-matrix";
import * as solver from "../../src/domain/solver/solver";
import * as fixture from "../../src/research/fixture";
import { qualityFixtures } from "../../src/research/teacher-quality";
import { evaluateState } from "../../src/domain/solver/evaluator";
import { getInitialCatalog } from "../../src/domain/pieces/catalog";
import { applyAction } from "../../src/domain/game/actions";
import type { GameState } from "../../src/domain/game/types";
import type { SolverConfig } from "../../src/domain/solver/types";

afterEach(() => vi.restoreAllMocks());
// A valid empty candidate isolates the harness from expensive DFS. Actual CLI runs real DFS.
function stub() {
  return vi
    .spyOn(solver, "solveTurn")
    .mockImplementation((input, _catalog, inputConfig) => {
      const state = input as GameState,
        config = inputConfig as SolverConfig;
      const e = evaluateState(state, getInitialCatalog(), {
        clearedRows: 0,
        acquiredItems: 0,
      });
      if (!e.ok) throw new Error(e.error.message);
      return {
        ok: true,
        result: {
          actions: [],
          alternatives: [],
          evaluation: e.evaluation,
          finalState: structuredClone(state),
          usedAbilities: { singleCell: 0, reroll: 0 },
          search: {
            scope: "pieces-and-abilities",
            searchComplete: false,
            optimalWithinScope: false,
            specialAbilitiesSearched: true,
            alternativesMayOmitEquivalentPaths: true,
            stopReason: "node-budget",
            visitedNodes: config!.maxNodes,
            maxNodes: config!.maxNodes,
            memoPrunedNodes: 0,
            memoEntries: 0,
            discoveredCompletePaths: 0,
            evaluatedCandidates: 1,
          },
        },
      };
    });
}
describe("train/dev diagnostic matrix", () => {
  it("freezes detached execution details without adopting a training contract", () => {
    const before = JSON.stringify(proposal),
      p = freezeMatrixProtocol(proposal);
    p.proposal.splits.train.reverse();
    expect(JSON.stringify(proposal)).toBe(before);
    expect(p.execution).toMatchObject({
      actionIndex: 0,
      heldOut: "NOT_RUN",
      training: "NOT_RUN",
      useMemoization: true,
    });
    expect(() =>
      freezeMatrixProtocol({
        ...proposal,
        splits: { ...proposal.splits, train: [3000] },
      }),
    ).toThrow();
  });
  it("runs only the frozen train/dev grid, rotates order, and does not inflate coverage with repetitions", () => {
    stub();
    const generated = vi.spyOn(fixture, "createSyntheticFixture");
    let time = 0;
    const r = runTeacherMatrix(proposal, () => time++);
    expect(generated.mock.calls.map((c) => c[0])).toEqual([
      4000, 4000, 4001, 4001, 4002, 4002, 4003, 4003, 5000, 5000, 5001, 5001,
    ]);
    expect(r.summary).toMatchObject({
      states: 12,
      probes: 144,
      uniqueConditions: 72,
      repeatedConditionsChecked: 72,
      arrayControlsChecked: 24,
      returnedPathsVerified: 144,
      generatedReservedTestStates: 0,
      trainingRuns: 0,
    });
    expect(r.rows[0]).toMatchObject({
      budget: 128,
      variant: "baseline",
      repetition: 0,
    });
    expect(r.rows[6]).toMatchObject({
      budget: 128,
      variant: "reverse-array",
      repetition: 1,
    });
    expect(r.rows[12]).toMatchObject({
      budget: 128,
      variant: "reverse-array",
      repetition: 0,
    });
    expect(r.summary.groups[0].timing).toMatchObject({
      samples: 8,
      meanMs: 1,
      p95Ms: 1,
    });
    expect(r.summary.groups[0].states).toBe(4);
  }, 20000);
  it("rejects nonfinite or backward clocks and node budget breaches", () => {
    const mock = stub();
    for (const clock of [
      () => NaN,
      () => Infinity,
      (() => {
        let t = 2;
        return () => t--;
      })(),
    ])
      expect(() => runTeacherMatrix(proposal, clock)).toThrow("clock");
    const original = mock.getMockImplementation()!;
    mock.mockImplementation((...args) => {
      const r = original(...args);
      if (r.ok) r.result.search.visitedNodes++;
      return r;
    });
    expect(() => runTeacherMatrix(proposal, () => 0)).toThrow("budget");
  });
  it("rejects forged paths even when the candidate looks plausible", () => {
    const mock = stub(),
      original = mock.getMockImplementation()!;
    mock.mockImplementation((...args) => {
      const r = original(...args);
      if (r.ok) r.result.evaluation.emptyCells++;
      return r;
    });
    expect(() => runTeacherMatrix(proposal, () => 0)).toThrow("mismatch");
  });
  it("rejects drift in repeat or reverse controls", () => {
    for (const target of [1, 6]) {
      const mock = stub(),
        original = mock.getMockImplementation()!;
      let count = 0;
      mock.mockImplementation((...args) => {
        const r = original(...args);
        if (count++ === target && r.ok) r.result.search.evaluatedCandidates++;
        return r;
      });
      expect(() => runTeacherMatrix(proposal, () => 0)).toThrow(
        /control|Repeated/,
      );
      vi.restoreAllMocks();
    }
  }, 20000);
  it("counts acquired/retained/spent events from the selected path only", () => {
    for (const f of qualityFixtures().filter((f) => f.action)) {
      const t = applyAction(f.state, f.action!);
      if (!t.ok) throw new Error(t.error.message);
      const e = evaluateState(t.state, getInitialCatalog(), {
        clearedRows: t.info.clearedRows.length,
        acquiredItems: t.info.acquiredItems.length,
      });
      if (!e.ok) throw new Error(e.error.message);
      const real = solver.solveTurn(f.state, getInitialCatalog(), {
        maxNodes: 1,
        maxAlternatives: 3,
        useMemoization: true,
      });
      if (!real.ok) throw new Error(real.error.message);
      const counts = pathCoverage(f.state, {
        ...real.result,
        actions: [f.action! as (typeof real.result.actions)[number]],
        finalState: t.state,
        evaluation: e.evaluation,
      });
      expect(counts.acquiredItems).toBe(t.info.acquiredItems.length);
      expect(counts.retainedItemEvents).toBe(t.info.retainedItems.length);
      expect(counts.singleCellSpends).toBe(
        Number(t.info.spentAbility === "single-cell"),
      );
      expect(counts.rerollSpends).toBe(
        Number(t.info.spentAbility === "reroll"),
      );
      expect(counts.pendingReroll).toBe(t.state.pendingReroll !== null);
    }
  });
});
