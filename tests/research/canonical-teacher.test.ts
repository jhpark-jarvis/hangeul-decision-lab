import { createHash } from "node:crypto";
import { afterEach, describe, expect, it, vi } from "vitest";
import protocol from "../../research/experiments/canonical-teacher-v1.json";
import {
  CANONICAL_CONDITIONS,
  assertCanonicalLegality,
  canonicalizeState,
  canonicalPieceKey,
  ordinal,
  prepareCanonicalInput,
  restoreCanonicalCandidate,
  runCanonicalDiagnostic,
  validateCanonicalProtocol,
} from "../../src/research/canonical-teacher";
import {
  diagnosticFixtures,
  semanticPath,
  semanticStateKey,
  permuteSlots,
} from "../../src/research/permutation-teacher";
import { stateKey } from "../../src/research/episode";
import { qualityFixtures } from "../../src/research/teacher-quality";
import { getInitialCatalog } from "../../src/domain/pieces/catalog";
import { getUniqueVariants } from "../../src/domain/pieces/transforms";
import { verifyCandidatePath } from "../../src/research/teacher-audit";
import { evaluateState } from "../../src/domain/solver/evaluator";
import * as solver from "../../src/domain/solver/solver";
import type { GameState, PieceIndex } from "../../src/domain/game/types";
import type { TurnSolution } from "../../src/domain/solver/types";

afterEach(() => vi.restoreAllMocks());
const hash = (text: string) => createHash("sha256").update(text).digest("hex");
const checker = (fixture: { id: string }, key: string, expected: string) => {
  if (hash(key) !== expected) throw new Error(`Hash mismatch ${fixture.id}`);
};
const tiny = () => structuredClone(qualityFixtures()[0].state);
function zeroPath(state: GameState): TurnSolution {
  const e = evaluateState(state, getInitialCatalog(), {
    clearedRows: 0,
    acquiredItems: 0,
  });
  if (!e.ok) throw new Error(e.error.message);
  return {
    actions: [],
    alternatives: [],
    evaluation: e.evaluation,
    finalState: structuredClone(state),
    usedAbilities: { reroll: 0, singleCell: 0 },
    search: {
      scope: "pieces-and-abilities",
      searchComplete: false,
      optimalWithinScope: false,
      specialAbilitiesSearched: true,
      alternativesMayOmitEquivalentPaths: true,
      stopReason: "node-budget",
      visitedNodes: 512,
      maxNodes: 512,
      memoPrunedNodes: 0,
      memoEntries: 0,
      discoveredCompletePaths: 0,
      evaluatedCandidates: 1,
    },
  };
}
describe("canonical research contract", () => {
  it("freezes the design contract and historical states without consuming new seeds", () => {
    const p = validateCanonicalProtocol(protocol);
    diagnosticFixtures().forEach((f, i) => {
      expect(f.id).toBe(p.inputs[i].id);
      checker(f, stateKey(f.state), p.inputs[i].stateSha256);
    });
    for (const mutate of [
      (p: typeof protocol) => p.perCall.maxNodes++,
      (p: typeof protocol) => (p.inputs[0].stateSha256 = "bad"),
      (p: typeof protocol) => (p.inputs[0] = p.inputs[1]),
      (p: typeof protocol) => p.sortKey.reverse(),
    ]) {
      const drift = structuredClone(protocol);
      mutate(drift);
      expect(() => validateCanonicalProtocol(drift)).toThrow();
    }
    expect(() =>
      validateCanonicalProtocol({ ...protocol, extra: true }),
    ).toThrow();
  });
  it("orders UTF-16 keys without name, instance identity or locale dependence", () => {
    const state = tiny(),
      p = state.remainingPieces[0],
      key = canonicalPieceKey(p, null);
    const renamed = structuredClone(p);
    renamed.instanceId = "가";
    renamed.piece.name = "other";
    renamed.pieceIndex = 2;
    expect(canonicalPieceKey(renamed, null)).toEqual(key);
    expect(ordinal("Z", "a")).toBe(-1);
    expect(ordinal("😀", "\uE000")).toBe(-1);
    const variants = getUniqueVariants([
      [true, true],
      [true, false],
    ]);
    if (!variants.ok) throw new Error("shape");
    const keys = variants.variants.map((v) =>
      canonicalPieceKey({ ...p, piece: { ...p.piece, shape: v.shape } }, null),
    );
    expect(new Set(keys.map((k) => k[0])).size).toBe(1);
    expect(new Set(keys.map((k) => k[1])).size).toBeGreaterThan(1);
    expect(
      canonicalPieceKey(
        {
          ...p,
          piece: {
            ...p.piece,
            shape: [
              [false, false, false],
              [false, true, false],
              [false, false, false],
            ],
          },
        },
        null,
      )[1],
    ).toBe("1x1:1");
  });
  it.each([0, 1, 2, 3])(
    "maps %i pieces including gaps through a complete bijection",
    (count) => {
      const state = tiny(),
        base = state.remainingPieces[0];
      state.remainingPieces = ([2, 0, 1] as PieceIndex[])
        .slice(0, count)
        .map((slot, i) => ({
          ...structuredClone(base),
          pieceIndex: slot,
          instanceId: `p${i}`,
        }));
      const before = structuredClone(state),
        canonical = canonicalizeState(state);
      expect(new Set(canonical.permutation).size).toBe(3);
      expect(canonical.state.remainingPieces.map((p) => p.pieceIndex)).toEqual(
        Array.from({ length: count }, (_, i) => i),
      );
      const round = permuteSlots(
        canonical.state,
        [0, 1, 2].map((i) => canonical.permutation.indexOf(String(i))).join(""),
      );
      expect(stateKey(round)).toBe(stateKey(state));
      expect(state).toEqual(before);
      assertCanonicalLegality(
        state,
        prepareCanonicalInput(state, "canonical-210"),
      );
    },
  );
  it("keeps duplicate IDs and original orientation separate, with pending first and physical slot tie", () => {
    const state = tiny(),
      base = state.remainingPieces[0];
    state.remainingPieces = ([2, 0, 1] as PieceIndex[]).map((slot) => ({
      ...structuredClone(base),
      pieceIndex: slot,
      instanceId: `id${slot}`,
    }));
    state.pendingReroll = { instanceId: "id2", pieceIndex: 2 };
    const mapped = prepareCanonicalInput(state, "canonical-rename-instances");
    expect(mapped.state.pendingReroll?.pieceIndex).toBe(0);
    const restored = restoreCanonicalCandidate(zeroPath(mapped.state), mapped);
    expect(stateKey(restored.finalState)).toBe(stateKey(state));
    verifyCandidatePath(state, restored);
    state.pendingReroll = null;
    expect(
      canonicalizeState(state).state.remainingPieces.map((p) => p.instanceId),
    ).toEqual(["id0", "id1", "id2"]);
    state.remainingPieces[0].piece.id = "Z";
    state.remainingPieces[1].piece.id = "a";
    state.remainingPieces[2].piece.id = "zz";
    expect(canonicalizeState(state).state.remainingPieces[0].piece.id).toBe(
      "Z",
    );
    expect(() =>
      canonicalizeState({
        ...state,
        pendingReroll: { instanceId: "absent", pieceIndex: 2 },
      }),
    ).toThrow();
  });
  it("restores real best and every alternative under all nine controls", () => {
    const input = tiny(),
      before = structuredClone(input),
      paths: string[][] = [],
      finals: string[] = [];
    for (const condition of CANONICAL_CONDITIONS.slice(1)) {
      const mapping = prepareCanonicalInput(input, condition);
      assertCanonicalLegality(input, mapping);
      const result = solver.solveTurn(mapping.state, getInitialCatalog(), {
        maxNodes: 512,
        maxAlternatives: 3,
        useMemoization: true,
      });
      if (!result.ok) throw new Error(result.error.message);
      for (const raw of [result.result, ...result.result.alternatives]) {
        verifyCandidatePath(mapping.state, raw);
        verifyCandidatePath(input, restoreCanonicalCandidate(raw, mapping));
      }
      const restored = restoreCanonicalCandidate(result.result, mapping);
      paths.push(semanticPath(input, restored));
      finals.push(semanticStateKey(restored.finalState));
    }
    expect(
      paths.every((p) => JSON.stringify(p) === JSON.stringify(paths[0])),
    ).toBe(true);
    expect(new Set(finals).size).toBe(1);
    expect(input).toEqual(before);
  });
  it("rejects corrupted inverse targets through exact legal-action checks", () => {
    const state = tiny(),
      mapping = prepareCanonicalInput(state, "canonical-012");
    mapping.inverseNames[state.remainingPieces[0].instanceId] = "missing";
    expect(() => assertCanonicalLegality(state, mapping)).toThrow("bijection");
    expect(() => prepareCanonicalInput(state, "invalid" as never)).toThrow();
  });
  it("executes exact nine-condition scheduling, warmup and full-repeat checks with a no-action harness", () => {
    const stub = vi.spyOn(solver, "solveTurn").mockImplementation((state) => ({
      ok: true,
      result: zeroPath(state as GameState),
    }));
    let t = 0;
    const events: string[] = [];
    const report = runCanonicalDiagnostic(
      protocol,
      () => ++t,
      checker,
      (event) => events.push(event.type),
    );
    expect(stub).toHaveBeenCalledTimes(362);
    expect(report.status).toBe("PASS");
    expect(report.summary.measuredCalls).toBe(360);
    expect(report.summary.nominalNodeCap).toBe(184320);
    expect(report.summary.invariance).toMatchObject({
      uniquePairs: 140,
      checks: 280,
      passed: true,
    });
    expect(report.summary.repeated).toMatchObject({
      conditions: 180,
      passed: true,
    });
    expect(report.summary.candidateForUserReview).toBe(false);
    expect(events).toHaveLength(724);
    expect(report.rows[9].condition).toBe("canonical-012");
    expect(report.rows.map((r) => r.calls[0].sequence)).toEqual(
      Array.from({ length: 360 }, (_, i) => i + 3),
    );
  }, 30000);
  it("rejects all hash drift before any solve and guards invalid clocks", () => {
    const spy = vi.spyOn(solver, "solveTurn");
    expect(() =>
      runCanonicalDiagnostic(
        protocol,
        () => 0,
        () => {
          throw new Error("drift");
        },
      ),
    ).toThrow("drift");
    expect(spy).not.toHaveBeenCalled();
    expect(() => runCanonicalDiagnostic(protocol, () => NaN, checker)).toThrow(
      "Clock",
    );
    expect(spy).not.toHaveBeenCalled();
    let i = 0;
    expect(() =>
      runCanonicalDiagnostic(protocol, () => (i++ === 0 ? 1 : 0), checker),
    ).toThrow("Clock");
    expect(spy).not.toHaveBeenCalled();
  });
  it("detects mutated solver inputs and illegal node accounting", () => {
    const spy = vi.spyOn(solver, "solveTurn").mockImplementation((input) => {
      const state = input as GameState,
        result = zeroPath(state);
      state.abilities.reroll++;
      return { ok: true, result };
    });
    expect(() => runCanonicalDiagnostic(protocol, () => 0, checker)).toThrow(
      "mutation",
    );
    spy.mockImplementation((input) => {
      const result = zeroPath(input as GameState);
      result.search.visitedNodes = 513;
      return { ok: true, result };
    });
    expect(() => runCanonicalDiagnostic(protocol, () => 0, checker)).toThrow(
      "cap",
    );
  });
});
