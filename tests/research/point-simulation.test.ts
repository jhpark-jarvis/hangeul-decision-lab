import { describe, it, expect } from "vitest";
import {
  getAvailableActions,
  applyAction,
} from "../../src/domain/game/actions";
import { actionKey, encodeDecision } from "../../src/research/imitation";
import {
  POINT_SIMULATION_PROTOCOL,
  validatePointProtocol,
  pointFixtures,
  verifyPointFixture,
  pointReference,
  replayPointPath,
  pointGreedy,
  inspectPointFixture,
  summarizePointSimulation,
} from "../../src/research/point-simulation";

describe("controlled point coordinate simulation", () => {
  it("accepts key reordering but rejects changed count/budget/scope/model or extra fields", () => {
    expect(
      validatePointProtocol(
        Object.fromEntries(Object.entries(POINT_SIMULATION_PROTOCOL).reverse()),
      ),
    ).toEqual(POINT_SIMULATION_PROTOCOL);
    for (const bad of [
      null,
      { ...POINT_SIMULATION_PROTOCOL, fixtureCount: 159 },
      { ...POINT_SIMULATION_PROTOCOL, nodeBudgets: [512] },
      { ...POINT_SIMULATION_PROTOCOL, modelCreationLoadForwardTraining: 1 },
      { ...POINT_SIMULATION_PROTOCOL, randomSupplyOrNewItems: true },
      { ...POINT_SIMULATION_PROTOCOL, training: 1 },
    ])
      expect(() => validatePointProtocol(bad)).toThrow("protocol");
  });
  it("covers all 160 coordinates with detached valid states and 31 point-only actions", () => {
    const fs = pointFixtures(),
      before = JSON.stringify(fs);
    expect(new Set(fs.map((f) => JSON.stringify(f.target))).size).toBe(160);
    for (const f of fs) expect(verifyPointFixture(f)).toHaveLength(31);
    expect(JSON.stringify(fs)).toBe(before);
    fs[0].state.board[0][0] = true;
    expect(pointFixtures()[0].state.board[0][0]).toBe(false);
    expect(fs[1].state.board[0][1]).toBe(false);
  });
  it("requires the unique one-hole row, blocked piece, no extra items/abilities and valid target", () => {
    for (const mutate of [
      (f: ReturnType<typeof pointFixtures>[number]) => (f.target.row = 16),
      (f: ReturnType<typeof pointFixtures>[number]) => (f.target.col = 1),
      (f: ReturnType<typeof pointFixtures>[number]) =>
        (f.state.abilities.singleCell = 2),
      (f: ReturnType<typeof pointFixtures>[number]) =>
        f.state.hiddenItems.push({ row: 1, col: 1, type: "reroll" }),
    ]) {
      const f = pointFixtures()[0];
      mutate(f);
      expect(() => verifyPointFixture(f)).toThrow();
    }
  });
  it("wrong points end the turn in gameover; the constructed point unlocks exactly eight placements", () => {
    const f = pointFixtures()[67],
      legal = verifyPointFixture(f);
    let wins = 0;
    for (const a of legal) {
      const t = applyAction(f.state, a);
      if (!t.ok) throw new Error(t.error.message);
      const next = getAvailableActions(t.state);
      if (!next.ok) throw new Error(next.error.message);
      if (t.info.clearedRows.length) {
        wins++;
        expect(next.actions).toHaveLength(8);
        expect(next.actions.every((x) => x.type === "place-piece")).toBe(true);
      } else expect(next.phase).toBe("gameover");
    }
    expect(wins).toBe(1);
  });
  it("exhaustive reference visits all 40 nodes, 38 leaves and reports the unique target", () => {
    const f = pointFixtures()[159],
      before = JSON.stringify(f.state),
      r = pointReference(f.state);
    expect(r).toMatchObject({
      complete: true,
      visitedNodes: 40,
      leaves: 38,
      maxDepth: 2,
    });
    expect(r.optimalFirstActions).toEqual([
      actionKey({ type: "single-cell", ...f.target }),
    ]);
    expect(r.bestObserved?.allCurrentPiecesPlaced).toBe(true);
    expect(JSON.stringify(f.state)).toBe(before);
  });
  it("incomplete reference never publishes an optimum and rejects unsupported scope/budgets", () => {
    const f = pointFixtures()[0];
    expect(pointReference(f.state, 1)).toMatchObject({
      complete: false,
      visitedNodes: 1,
      optimalFirstActions: null,
      bestObserved: null,
    });
    for (const n of [0, 257, 1.5, NaN])
      expect(() => pointReference(f.state, n)).toThrow();
    const changed = structuredClone(f.state);
    changed.abilities.reroll = 1;
    expect(() => pointReference(changed)).toThrow("Reference requires");
    expect(() => pointReference({})).toThrow();
  });
  it("baseline policies only use current state; first-legal succeeds on the first target row only", () => {
    const fs = pointFixtures();
    for (const i of [0, 9, 10, 81, 159]) {
      const f = fs[i],
        first = pointGreedy(f.state, "first-legal"),
        greedy = pointGreedy(f.state, "immediate-clear");
      expect(
        actionKey(first[0]) === actionKey({ type: "single-cell", ...f.target }),
      ).toBe(i < 10);
      expect(greedy[0]).toEqual({ type: "single-cell", ...f.target });
      expect(
        replayPointPath(f.state, greedy).evaluation.allCurrentPiecesPlaced,
      ).toBe(true);
    }
  });
  it("base encoder distinguishes the 160 contexts and every coordinate candidate", () => {
    const observationKeys = new Set<string>();
    for (const f of pointFixtures()) {
      const legal = verifyPointFixture(f),
        e = encodeDecision(f.state, legal, {
          type: "single-cell",
          ...f.target,
        });
      if (!e.ok) throw new Error(e.error);
      observationKeys.add(JSON.stringify(e.value.observation));
      expect(e.value.candidateFeatures).toHaveLength(31);
      expect(
        new Set(e.value.candidateFeatures.map((x) => JSON.stringify(x))).size,
      ).toBe(31);
      expect(e.value.candidateFeatures[e.value.label].slice(25, 27)).toEqual([
        f.target.row / 15,
        f.target.col / 9,
      ]);
    }
    expect(observationKeys.size).toBe(160);
  });
  it("DFS128/512 best and alternatives replay to their original values and match the tiny reference", () => {
    for (const i of [0, 87, 159]) {
      const r = inspectPointFixture(pointFixtures()[i]);
      for (const name of ["dfs-128", "dfs-512"])
        expect(r.policies[name]).toMatchObject({
          correctFirst: true,
          allPiecesPlaced: true,
          referenceEvaluationMatch: true,
          searchComplete: true,
        });
    }
  });
  it("repeated inspection is exact and aggregates observations without treating repetitions as new fixtures", () => {
    const f = pointFixtures()[33],
      a = inspectPointFixture(f),
      b = inspectPointFixture(f);
    expect(a).toEqual(b);
    const s = summarizePointSimulation([a]);
    expect(s.fixtures).toBe(1);
    expect(s.distinctTargets).toBe(1);
    expect(s.reference.complete).toBe(1);
    expect(s.policies["immediate-clear"].allPiecesPlaced).toBe(1);
  });
  it("replay rejects illegal, stale or overlong paths", () => {
    const f = pointFixtures()[0];
    expect(() =>
      replayPointPath(f.state, [{ type: "single-cell", row: 0, col: 1 }]),
    ).toThrow();
    expect(() =>
      replayPointPath(f.state, [
        { type: "single-cell", row: 0, col: 0 },
        { type: "single-cell", row: 0, col: 0 },
      ]),
    ).toThrow();
    expect(() =>
      replayPointPath(
        f.state,
        Array(3).fill({ type: "single-cell", row: 0, col: 0 }),
      ),
    ).toThrow();
  });
  it("summary rejects repeated fixtures as independent samples and empty populations", () => {
    expect(() => summarizePointSimulation([])).toThrow("distinct nonempty");
    const row = inspectPointFixture(pointFixtures()[0]);
    expect(() => summarizePointSimulation([row, row])).toThrow("repetitions");
  });
});
