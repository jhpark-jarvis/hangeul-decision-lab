import { describe, it, expect, vi, afterEach } from "vitest";
import { applyAction } from "../../src/domain/game/actions";
import {
  ITEM_COVERAGE_PROTOCOL,
  itemCoverageFixtures,
  validateItemProtocol,
  verifyItemTransition,
  runItemCoverage,
} from "../../src/research/item-coverage";
import * as solver from "../../src/domain/solver/solver";
afterEach(() => vi.restoreAllMocks());
describe("separated item coverage diagnostic", () => {
  it("accepts reordered object keys but rejects changed budget/scope/test/training/cases", () => {
    expect(
      validateItemProtocol(
        Object.fromEntries(Object.entries(ITEM_COVERAGE_PROTOCOL).reverse()),
      ),
    ).toEqual(ITEM_COVERAGE_PROTOCOL);
    for (const bad of [
      null,
      { ...ITEM_COVERAGE_PROTOCOL, nodeBudgets: [512, 2048] },
      { ...ITEM_COVERAGE_PROTOCOL, scope: "train" },
      { ...ITEM_COVERAGE_PROTOCOL, heldOut: "RUN" },
      { ...ITEM_COVERAGE_PROTOCOL, cases: [] },
      { ...ITEM_COVERAGE_PROTOCOL, training: "RUN" },
    ])
      expect(() => validateItemProtocol(bad)).toThrow("protocol");
  });
  it("verifies eight explicit transitions without mutating fixtures; separate calls are detached", () => {
    const fixtures = itemCoverageFixtures(),
      before = JSON.stringify(fixtures);
    for (const f of fixtures) {
      const t = applyAction(f.state, f.action);
      if (!t.ok) throw new Error(t.error.message);
      verifyItemTransition(f, t.state, t.info);
    }
    expect(JSON.stringify(fixtures)).toBe(before);
    fixtures[0].state.board[0][0] = false;
    expect(itemCoverageFixtures()[0].state.board[0][0]).toBe(true);
  });
  it("acquires by row/col across simultaneous rows and keeps overflow at capacity", () => {
    const f = itemCoverageFixtures().find(
        (f) => f.id === "simultaneous-row-order",
      )!,
      t = applyAction(f.state, f.action);
    if (!t.ok) throw new Error(t.error.message);
    expect(t.info.clearedRows).toEqual([0, 1, 2, 3, 4]);
    expect(t.info.acquiredItems).toEqual([
      { row: 0, col: 0, type: "single-cell" },
    ]);
    expect(t.info.retainedItems).toEqual([
      { row: 0, col: 9, type: "reroll" },
      { row: 1, col: 8, type: "single-cell" },
    ]);
    expect(t.state.abilities).toEqual({ singleCell: 7, reroll: 0 });
  });
  it("spends at capacity before collection, and reroll waits without clearing or collecting", () => {
    const fs = itemCoverageFixtures();
    for (const id of ["spend-at-capacity", "capacity-reroll-boundary"]) {
      const f = fs.find((f) => f.id === id)!,
        t = applyAction(f.state, f.action);
      if (!t.ok) throw new Error(t.error.message);
      verifyItemTransition(f, t.state, t.info);
      if (id === "spend-at-capacity") {
        expect(t.info.acquiredItems.length).toBe(1);
        expect(t.info.retainedItems.length).toBe(1);
        expect(t.state.abilities.singleCell).toBe(7);
      } else {
        expect(t.info.acquiredItems).toEqual([]);
        expect(t.state.hiddenItems).toEqual(f.state.hiddenItems);
        expect(t.state.pendingReroll).not.toBeNull();
      }
    }
  });
  it("rejects forged transition order, board and remaining pieces", () => {
    const f = itemCoverageFixtures()[0],
      t = applyAction(f.state, f.action);
    if (!t.ok) throw new Error(t.error.message);
    const info = structuredClone(t.info);
    info.acquiredItems.reverse();
    expect(() => verifyItemTransition(f, t.state, info)).toThrow("transition");
    const state = structuredClone(t.state);
    state.board[0][0] = true;
    expect(() => verifyItemTransition(f, state, t.info)).toThrow("board");
    state.board[0][0] = false;
    state.remainingPieces = f.state.remainingPieces;
    expect(() => verifyItemTransition(f, state, t.info)).toThrow("piece");
  });
  it("runs actual ordinary and full paths with repeat checks, rejecting invalid clocks and forged solver paths", () => {
    let time = 0;
    const r = runItemCoverage(ITEM_COVERAGE_PROTOCOL, () => time++);
    expect(r.summary).toMatchObject({
      scriptedCases: 8,
      scriptedTransitions: 8,
      completeOrdinaryReferences: 8,
      ordinaryComparisons: 8,
      fullProbes: 32,
      repeatedConditionsChecked: 16,
      generatedTestStates: 0,
      trainingRuns: 0,
    });
    expect(r.summary.returnedPathsVerified).toBeGreaterThan(32);
    expect(r.summary.selectedByBudget.every((g) => g.states === 8)).toBe(true);
    expect(() => runItemCoverage(ITEM_COVERAGE_PROTOCOL, () => NaN)).toThrow(
      "clock",
    );
    const original = solver.solveTurn;
    vi.spyOn(solver, "solveTurn").mockImplementation((...args) => {
      const r = original(...args);
      if (r.ok) r.result.evaluation.emptyCells++;
      return r;
    });
    expect(() => runItemCoverage(ITEM_COVERAGE_PROTOCOL, () => 0)).toThrow(
      "mismatch",
    );
  }, 120000);
});
