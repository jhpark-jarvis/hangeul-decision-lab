import { createHash } from "node:crypto";
import { afterEach, describe, expect, it, vi } from "vitest";
import protocol from "../../research/experiments/permutation-teacher-v1.json";
import {
  diagnosticFixtures,
  permuteSlots,
  restoreCandidate,
  runPermutationDiagnostic,
  selectReturnedPath,
  semanticPath,
  semanticStateKey,
  validatePermutationProtocol,
} from "../../src/research/permutation-teacher";
import * as solver from "../../src/domain/solver/solver";
import * as generator from "../../src/research/fixture";
import { itemCoverageFixtures } from "../../src/research/item-coverage";
import { qualityFixtures } from "../../src/research/teacher-quality";
import { stateKey } from "../../src/research/episode";
import {
  physicalActionKey,
  verifyCandidatePath,
} from "../../src/research/teacher-audit";
import { getInitialCatalog } from "../../src/domain/pieces/catalog";
import { evaluateState } from "../../src/domain/solver/evaluator";
import {
  applyAction,
  getAvailableActions,
} from "../../src/domain/game/actions";
import type { GameAction, GameState } from "../../src/domain/game/types";
import type {
  SolverCandidate,
  SolverConfig,
} from "../../src/domain/solver/types";

afterEach(() => vi.restoreAllMocks());
const hash = (text: string) => createHash("sha256").update(text).digest("hex");
const catalog = getInitialCatalog();
function fromAction(state: GameState, action: GameAction): SolverCandidate {
  const t = applyAction(state, action);
  if (!t.ok) throw new Error(t.error.message);
  const e = evaluateState(t.state, catalog, {
    clearedRows: t.info.clearedRows.length,
    acquiredItems: t.info.acquiredItems.length,
  });
  if (!e.ok) throw new Error(e.error.message);
  return {
    actions: [
      action.type === "place-piece"
        ? { ...action, rotation: 0, flipped: false }
        : action.type === "reroll"
          ? { ...action, reason: "blocked-piece" }
          : action,
    ],
    finalState: t.state,
    evaluation: e.evaluation,
    usedAbilities: {
      singleCell: Number(action.type === "single-cell"),
      reroll: Number(action.type === "reroll"),
    },
  };
}
function stubHarness() {
  const original = generator.createSyntheticFixture;
  const tiny = qualityFixtures()[0].state;
  const generated = vi
    .spyOn(generator, "createSyntheticFixture")
    .mockImplementation((...args) => {
      const f = original(...args);
      if (f.ok) f.value.initialState = structuredClone(tiny);
      return f;
    });
  const solved = vi
    .spyOn(solver, "solveTurn")
    .mockImplementation((input, _catalog, rawConfig) => {
      const state = input as GameState,
        config = rawConfig as SolverConfig;
      const e = evaluateState(state, catalog, {
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
            visitedNodes: config.maxNodes,
            maxNodes: config.maxNodes,
            memoPrunedNodes: 0,
            memoEntries: 0,
            discoveredCompletePaths: 0,
            evaluatedCandidates: 1,
          },
        },
      };
    });
  return { generated, solved };
}
describe("permutation teacher diagnostic", () => {
  it("freezes only the approved grid and preserves the historical input states", () => {
    const detached = validatePermutationProtocol(protocol);
    detached.inputs.reverse();
    expect(detached.inputs).not.toEqual(protocol.inputs);
    const fixtures = diagnosticFixtures();
    expect(
      fixtures.map((f) => ({ id: f.id, stateSha256: hash(stateKey(f.state)) })),
    ).toEqual(protocol.inputs);
    expect(fixtures.filter((f) => f.stratum === "generated")).toHaveLength(12);
    expect(fixtures.filter((f) => f.stratum === "scripted")).toHaveLength(8);
    for (const change of [
      { trainSeeds: [6000] },
      { status: "TRAINING" },
      { perCall: { ...protocol.perCall, sharedMemo: true } },
      { repetitions: 3 },
      { injected: true },
    ])
      expect(() =>
        validatePermutationProtocol({ ...protocol, ...change }),
      ).toThrow("protocol");
    expect(() =>
      validatePermutationProtocol({
        ...protocol,
        inputs: protocol.inputs.slice(1),
      }),
    ).toThrow();
  });
  it("maps sparse slots and pending targets bijectively and restores real domain transitions", () => {
    for (const fixture of itemCoverageFixtures().filter((f) =>
      [
        "reroll-boundary",
        "single-spend-acquire",
        "simultaneous-row-order",
      ].includes(f.id),
    )) {
      const before = JSON.stringify(fixture.state),
        candidate = fromAction(fixture.state, fixture.action);
      for (const p of protocol.permutations) {
        const state = permuteSlots(fixture.state, p),
          mapped = structuredClone(candidate);
        for (const action of mapped.actions)
          if (action.type !== "single-cell")
            action.pieceIndex = Number(p[action.pieceIndex]) as 0 | 1 | 2;
        mapped.finalState = permuteSlots(candidate.finalState, p);
        verifyCandidatePath(state, mapped);
        const restored = restoreCandidate(mapped, p);
        verifyCandidatePath(fixture.state, restored);
        expect(stateKey(restored.finalState)).toBe(
          stateKey(candidate.finalState),
        );
        expect(restored.actions).toEqual(candidate.actions);
      }
      expect(JSON.stringify(fixture.state)).toBe(before);
    }
    for (const p of ["001", "123", "01", "abc"])
      expect(() => permuteSlots(qualityFixtures()[0].state, p)).toThrow(
        "bijection",
      );
    expect(() => permuteSlots({}, "012")).toThrow();
  });
  it("selects identical-piece ties independently of enumeration while retaining physical identity", () => {
    const state = structuredClone(qualityFixtures()[0].state),
      dot = catalog.find((p) => p.id === "DOT")!;
    state.remainingPieces = [
      { instanceId: "z", pieceIndex: 0, piece: dot },
      { instanceId: "a", pieceIndex: 2, piece: dot },
    ];
    const legal = getAvailableActions(state);
    if (!legal.ok) throw new Error(legal.error.message);
    const actions = legal.actions.filter((a) => a.type === "place-piece");
    const a = fromAction(
        state,
        actions.find((a) => a.instanceId === "a")!,
      ),
      z = fromAction(
        state,
        actions.find((a) => a.instanceId === "z")!,
      );
    expect(semanticPath(state, a)).toEqual(semanticPath(state, z));
    expect(semanticStateKey(a.finalState)).toBe(semanticStateKey(z.finalState));
    const paths = [z, a].map((candidate, i) => ({
      candidate,
      source: { callIndex: i, candidateIndex: 0, permutation: "012" },
    }));
    const selected = selectReturnedPath(state, paths),
      reverse = selectReturnedPath(state, [...paths].reverse());
    expect(selected.candidate.actions.map(physicalActionKey)).toEqual(
      reverse.candidate.actions.map(physicalActionKey),
    );
    expect(selected.candidate.actions[0]).toMatchObject({ instanceId: "a" });
    expect(selected.candidate).not.toHaveProperty("search");
    expect(() => selectReturnedPath(state, [])).toThrow("No returned paths");
  });
  it("replays every path from six real bounded DFS calls against original slots", () => {
    const f = itemCoverageFixtures()[0];
    const original = getAvailableActions(f.state);
    if (!original.ok) throw new Error(original.error.message);
    for (const p of protocol.permutations) {
      const state = permuteSlots(f.state, p),
        legal = getAvailableActions(state);
      if (!legal.ok) throw new Error(legal.error.message);
      expect(legal.actions.map(physicalActionKey).sort()).toEqual(
        original.actions.map(physicalActionKey).sort(),
      );
      const result = solver.solveTurn(state, catalog, {
        maxNodes: 16,
        maxAlternatives: 3,
        useMemoization: true,
      });
      if (!result.ok) throw new Error(result.error.message);
      expect(restoreCandidate(result.result, p)).not.toHaveProperty(
        "alternatives",
      );
      expect(restoreCandidate(result.result, p)).not.toHaveProperty("search");
      for (const c of [result.result, ...result.result.alternatives])
        verifyCandidatePath(f.state, restoreCandidate(c, p));
    }
  });
  it("runs the bounded grid, separates warmup/repeats, and never generates test seeds", () => {
    const { generated, solved } = stubHarness(),
      checked = vi.fn(),
      emitted: string[] = [];
    let time = 0;
    const r = runPermutationDiagnostic(
      protocol,
      () => time++,
      checked,
      (e) => emitted.push(e.type),
    );
    expect(checked).toHaveBeenCalledTimes(20);
    expect(generated.mock.calls.map((c) => c[0])).toEqual([
      4000, 4000, 4001, 4001, 4002, 4002, 4003, 4003, 5000, 5000, 5001, 5001,
    ]);
    expect(generated.mock.calls.every((c) => c[2] === 1)).toBe(true);
    expect(solved).toHaveBeenCalledTimes(608);
    expect(emitted.filter((e) => e === "call")).toHaveLength(608);
    expect(r.warmup.reduce((n, row) => n + row.calls.length, 0)).toBe(8);
    expect(r.summary).toMatchObject({
      states: 20,
      uniqueConditions: 100,
      measuredCalls: 600,
      repeatedConditionsChecked: 100,
      arrayControlsChecked: 20,
      returnedPathsVerified: 600,
      cycleControls: {
        states: 20,
        semanticFirstSame: 20,
        evaluationSame: 20,
        finalSemanticSame: 20,
      },
      generatedTestStates: 0,
      trainingRuns: 0,
    });
    expect(r.rows[0].condition).toBe("baseline");
    expect(r.rows[5].condition).toBe("largeSingle");
    expect(r.rows[2].uniqueTransformedStates).toBe(3);
    expect(r.summary.groups[0].states).toBe(12);
    expect(r.summary.groups[0].timing.samples).toBe(24);
    expect(
      r.rows.every((row) => row.latencyMs >= 0 && row.loggingMs >= 0),
    ).toBe(true);
  }, 30000);
  it("rejects historical drift before any solve and rejects invalid clocks or budget reports", () => {
    const { solved } = stubHarness();
    expect(() =>
      runPermutationDiagnostic(
        protocol,
        () => 0,
        () => {
          throw new Error("Historical input mismatch");
        },
      ),
    ).toThrow("Historical input");
    expect(solved).not.toHaveBeenCalled();
    for (const clock of [
      () => NaN,
      () => Infinity,
      (() => {
        let n = 1;
        return () => n--;
      })(),
    ])
      expect(() => runPermutationDiagnostic(protocol, clock, () => {})).toThrow(
        "clock",
      );
    const original = solved.getMockImplementation()!;
    solved.mockImplementation((...args) => {
      const r = original(...args);
      if (r.ok) r.result.search.visitedNodes++;
      return r;
    });
    expect(() =>
      runPermutationDiagnostic(
        protocol,
        () => 0,
        () => {},
      ),
    ).toThrow("budget");
  });
  it("rejects forged path metadata, solver errors and input mutation", () => {
    for (const kind of ["evaluation", "mutation", "error"] as const) {
      const { solved } = stubHarness(),
        original = solved.getMockImplementation()!;
      solved.mockImplementation((...args) => {
        if (kind === "error")
          return {
            ok: false,
            error: {
              code: "INVALID_SOLVER_CONFIG",
              message: "Injected solver error",
            },
          };
        const r = original(...args);
        if (kind === "mutation")
          (args[0] as GameState).board[0][0] = !(args[0] as GameState)
            .board[0][0];
        else if (r.ok) r.result.evaluation.emptyCells++;
        return r;
      });
      expect(() =>
        runPermutationDiagnostic(
          protocol,
          () => 0,
          () => {},
        ),
      ).toThrow(/mismatch|mutated|Injected/);
      vi.restoreAllMocks();
    }
  });
  it("rejects repeat drift even when every returned empty path remains legal", () => {
    const { solved } = stubHarness(),
      original = solved.getMockImplementation()!;
    let call = 0;
    solved.mockImplementation((...args) => {
      const r = original(...args);
      if (call++ === 37 && r.ok) r.result.search.evaluatedCandidates++;
      return r;
    });
    expect(() =>
      runPermutationDiagnostic(
        protocol,
        () => 0,
        () => {},
      ),
    ).toThrow("Repeated condition");
  }, 30000);
});
