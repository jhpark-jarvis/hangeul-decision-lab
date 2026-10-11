import { describe, expect, it } from "vitest";
import protocol from "../../research/experiments/teacher-root-cause-v1.json";
import { getInitialCatalog } from "../../src/domain/pieces/catalog";
import { solveTurn } from "../../src/domain/solver/solver";
import { compareEvaluations } from "../../src/domain/solver/evaluator";
import { qualityFixtures } from "../../src/research/teacher-quality";
import { physicalActionKey } from "../../src/research/teacher-audit";
import { stateKey } from "../../src/research/episode";
import {
  assertTraceAccounting,
  buildPassiveTrace,
  firstEvaluationDifference,
  incumbentPortfolio,
  logicalTurnSignature,
  replayPath,
  traceWitness,
  validateRootCauseProtocol,
  type TraceEvent,
} from "../../src/research/teacher-root-cause";
const input = qualityFixtures()[0].state;
const solved = solveTurn(input, getInitialCatalog(), {
  maxNodes: 16,
  maxAlternatives: 3,
  useMemoization: true,
});
if (!solved.ok) throw new Error(solved.error.message);
const turn = solved.result;
describe("root-cause diagnostic boundary", () => {
  it("refuses missing or ambiguous passive trace insertion points", () => {
    const code = "search.visitedNodes++; search.evaluatedCandidates++;";
    expect(buildPassiveTrace(code)).toContain("export function tracedTurn");
    for (const drift of [
      "",
      code + code,
      code.replace("search.visitedNodes++;", ""),
      code.replace("search.evaluatedCandidates++;", ""),
    ])
      expect(() => buildPassiveTrace(drift)).toThrow(
        "Passive trace insertion drift",
      );
  });
  it("rejects budget, seed, path and unexpected-field drift", () => {
    expect(validateRootCauseProtocol(protocol)).toEqual(protocol);
    for (const drift of [
      { ...protocol, baselineBudget: 2048 },
      { ...protocol, noNewSeeds: false },
      { ...protocol, extra: true },
      { ...protocol, inputs: [null, ...protocol.inputs.slice(1)] },
      {
        ...protocol,
        inputs: [protocol.inputs[1], ...protocol.inputs.slice(1)],
      },
      { ...protocol, reports: { ...protocol.reports, canonicalSha256: "bad" } },
    ])
      expect(() => validateRootCauseProtocol(drift)).toThrow(
        "Invalid root-cause protocol",
      );
  });
  it("reports the first lexicographic axis even if a later reward improves", () => {
    const a = structuredClone(turn.evaluation),
      b = structuredClone(a);
    b.mobility.totalPlacements++;
    b.clearedRows = 0;
    a.clearedRows = 100;
    expect(firstEvaluationDifference(a, b)).toEqual({
      axis: "totalPlacements",
      left: a.mobility.totalPlacements,
      right: b.mobility.totalPlacements,
      comparison: -1,
    });
    expect(compareEvaluations(a, b)).toBe(-1);
  });
  it("uses fewer isolated cells as better and reports exact evaluation ties", () => {
    const a = structuredClone(turn.evaluation),
      b = structuredClone(a);
    expect(firstEvaluationDifference(a, b)).toBeNull();
    b.isolatedEmptyCells++;
    expect(firstEvaluationDifference(a, b)?.comparison).toBe(1);
    expect(firstEvaluationDifference(a, b)?.axis).toBe("isolatedEmptyCells");
  });
  it("keeps the incumbent path on ties and never mutates either solution", () => {
    const a = structuredClone(turn),
      d = structuredClone(turn);
    d.actions = [];
    const before = JSON.stringify([a, d]);
    const selected = incumbentPortfolio(a, d);
    expect(selected.source).toBe("A512");
    expect(selected.result.actions).toEqual(a.actions);
    selected.result.finalState.abilities.singleCell++;
    expect(JSON.stringify([a, d])).toBe(before);
    d.evaluation.clearedRows++;
    expect(incumbentPortfolio(a, d).source).toBe("D512");
    d.evaluation.mobility.totalPlacements--;
    expect(incumbentPortfolio(a, d).source).toBe("A512");
  });
  it("signatures ignore only array order and retain search information", () => {
    const other = structuredClone(turn);
    other.finalState.remainingPieces.reverse();
    expect(logicalTurnSignature(other)).toBe(logicalTurnSignature(turn));
    other.search.visitedNodes++;
    expect(logicalTurnSignature(other)).not.toBe(logicalTurnSignature(turn));
  });
  it("replays the actual returned path into its exact final state", () => {
    expect(stateKey(replayPath(input, turn.actions))).toBe(
      stateKey(turn.finalState),
    );
    expect(stateKey(input)).toBe(stateKey(qualityFixtures()[0].state));
  });
  it("distinguishes visited prefixes from evaluated paths and physical identity", () => {
    expect(turn.actions.length).toBeGreaterThan(0);
    const keys = turn.actions.map(physicalActionKey);
    const trace: TraceEvent[] = [
      {
        type: "visit",
        scope: "ordinary-pieces",
        node: 2,
        path: keys.slice(0, 1),
      },
    ];
    const witness = traceWitness(trace, turn);
    expect(witness.rootVisited).toBe(true);
    expect(witness.prefixes[0].ordinaryNode).toBe(2);
    expect(witness.evaluations).toHaveLength(0);
    trace.push({
      type: "consider",
      scope: "pieces-and-abilities",
      node: 9,
      path: keys,
      evaluation: turn.evaluation,
    });
    expect(traceWitness(trace, turn).evaluations).toHaveLength(1);
    trace[0].path = ["different physical instance"];
    expect(traceWitness(trace, turn).rootVisited).toBe(false);
  });
  it("rejects incomplete visit or evaluation accounting", () => {
    const empty = structuredClone(turn);
    empty.search.visitedNodes = 1;
    empty.search.evaluatedCandidates = 1;
    delete empty.search.abilitySearch;
    const trace: TraceEvent[] = [
      { type: "visit", scope: "pieces-and-abilities", node: 1, path: [] },
      {
        type: "consider",
        scope: "pieces-and-abilities",
        node: 1,
        path: [],
        evaluation: empty.evaluation,
      },
    ];
    expect(() => assertTraceAccounting(trace, empty)).not.toThrow();
    expect(() => assertTraceAccounting(trace.slice(0, 1), empty)).toThrow(
      "accounting mismatch",
    );
    trace[0].scope = "ordinary-pieces";
    expect(() => assertTraceAccounting(trace, empty)).toThrow(
      "accounting mismatch",
    );
  });
});
