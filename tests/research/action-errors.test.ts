import { describe, expect, it } from "vitest";
import { actionKey } from "../../src/research/imitation";
import {
  auditStored,
  decodeStoredAction,
  errorKind,
  inspectStored,
  storedEqualFirstActions,
  type StoredExample,
  type StoredPrediction,
} from "../../src/research/action-errors";
import type { GameAction } from "../../src/domain/game/types";
import type {
  SolverEvaluation,
  TurnSolution,
} from "../../src/domain/solver/types";

const point = (
  row: number,
  col = 0,
): Extract<GameAction, { type: "single-cell" }> => ({
  type: "single-cell",
  row,
  col,
});
const evaluation = (): SolverEvaluation => ({
  survivable: true,
  allCurrentPiecesPlaced: false,
  phase: "playing",
  clearedRows: 0,
  acquiredItems: 0,
  mobility: {
    playablePieceTypes: 19,
    totalPieceTypes: 19,
    totalPlacements: 100,
  },
  remainingAbilities: { reroll: 1, singleCell: 0 },
  isolatedEmptyCells: 0,
  unfillableGaps: 0,
  emptyCells: 50,
});
function fixture(index = 0) {
  const actions: GameAction[] = [
    point(0),
    point(3),
    { type: "reroll", instanceId: "a", pieceIndex: 0 },
  ];
  const path = (action: GameAction, e = evaluation()) => ({
    actions: [action],
    evaluation: e,
    finalState: {} as TurnSolution["finalState"],
    usedAbilities: { singleCell: 0, reroll: 0 },
  });
  const solution = {
    ...path(actions[1]),
    alternatives: [path(actions[0])],
    search: { searchComplete: false, alternativesMayOmitEquivalentPaths: true },
  } as TurnSolution;
  const example: StoredExample = {
    id: "dev:pressure-1:0",
    fixtureId: "pressure-1",
    candidateKeys: actions.map(actionKey),
    labels: { A512: 1 },
    teachers: { A512: { result: solution } },
  };
  const selected =
    index === 2
      ? { kind: "reroll" as const }
      : { kind: "single-cell" as const, row: index === 0 ? 0 : 3, col: 0 };
  const prediction: StoredPrediction = {
    id: example.id,
    fixtureId: example.fixtureId,
    family: "pressure",
    index,
    candidates: 3,
    selected,
    teacherActions: { A512: { kind: "single-cell", row: 3, col: 0 } },
    complete: { A512: false },
    targets: {
      A512: {
        correct: index === 1,
        top5: true,
        rank: index === 1 ? 1 : 2,
        nll: 1,
        firstLegal: false,
      },
    },
  };
  return { example, prediction };
}
describe("stored action error analysis", () => {
  it("separates wrong-kind from coordinate errors and keeps conditional distances", () => {
    const { example, prediction } = fixture();
    const a = inspectStored(example, prediction);
    expect(a.error).toBe("different-position");
    expect(a.singleCellDistance).toBe(3);
    const b = fixture(2);
    expect(inspectStored(b.example, b.prediction).error).toBe("wrong-kind");
    expect(
      inspectStored(b.example, b.prediction).singleCellDistance,
    ).toBeNull();
    const summary = auditStored([example], [prediction]).summary.overall;
    expect(summary.singleCell).toMatchObject({
      examples: 1,
      exact: 0,
      wrongPosition: 1,
      wrongKind: 0,
      sameKindDistanceSamples: 1,
    });
  });
  it("uses the production comparator for returned full-path ties without relabeling", () => {
    const { example, prediction } = fixture();
    const equal = example.teachers.A512.result.alternatives[0].evaluation;
    equal.phase = "await-reroll-result";
    equal.remainingAbilities = { singleCell: 1, reroll: 0 };
    expect(storedEqualFirstActions(example).indices).toEqual([1, 0]);
    const summary = auditStored([example], [prediction]).summary.overall;
    expect(summary.correct).toBe(0);
    expect(summary.storedTies.additionalStoredTieHits).toBe(1);
    expect(example.labels.A512).toBe(1);
  });
  it("does not count a worse alternative or duplicate first action as a new tie action", () => {
    const { example } = fixture();
    const solution = example.teachers.A512.result;
    solution.alternatives.push({
      ...solution.alternatives[0],
      actions: [point(3)],
    });
    expect(storedEqualFirstActions(example).indices).toEqual([1, 0]);
    solution.alternatives[0].evaluation = { ...evaluation(), emptyCells: 49 };
    expect(storedEqualFirstActions(example).indices).toEqual([1]);
  });
  it("rejects nonfinite evaluations, illegal stored alternatives and corrupted best ordering", () => {
    for (const fault of ["finite", "mask", "better"] as const) {
      const { example } = fixture();
      const a = example.teachers.A512.result.alternatives[0];
      if (fault === "finite") a.evaluation.emptyCells = NaN;
      if (fault === "mask") a.actions = [point(10)];
      if (fault === "better") a.evaluation.emptyCells = 51;
      expect(() => storedEqualFirstActions(example)).toThrow();
    }
  });
  it("rejects id/order/index/rank/description drift rather than dropping records", () => {
    for (const fault of [
      "id",
      "index",
      "rank",
      "description",
      "complete",
    ] as const) {
      const { example, prediction } = fixture();
      if (fault === "id") prediction.id = "other";
      if (fault === "index") prediction.index = 9;
      if (fault === "rank") prediction.targets.A512.rank = 1;
      if (fault === "description") prediction.selected.row = 8;
      if (fault === "complete") prediction.complete.A512 = true;
      expect(() => inspectStored(example, prediction)).toThrow();
    }
  });
  it("partitions every observation and preserves frozen inputs", () => {
    const { example, prediction } = fixture();
    const original = structuredClone({ example, prediction });
    const { summary } = auditStored([example], [prediction]);
    expect(
      Object.values(summary.overall.errors).reduce((a, b) => a + b, 0),
    ).toBe(1);
    for (const group of Object.values(summary.groups))
      expect(Object.values(group).reduce((a, b) => a + b.examples, 0)).toBe(1);
    expect({ example, prediction }).toEqual(original);
    expect(() =>
      auditStored([example, example], [prediction, prediction]),
    ).toThrow();
  });
  it("distinguishes ordinary target, shape and position and validates stored key schema", () => {
    const a: GameAction = {
      type: "place-piece",
      instanceId: "a",
      pieceIndex: 0,
      pieceId: "LINE_3",
      row: 0,
      col: 0,
      variant: [[true, true, true]],
    };
    expect(errorKind(a, { ...a, instanceId: "b" })).toBe("different-target");
    expect(errorKind(a, { ...a, variant: [[true], [true], [true]] })).toBe(
      "different-shape",
    );
    expect(errorKind(a, { ...a, col: 1 })).toBe("different-position");
    expect(errorKind(a, a)).toBe("exact-label");
    expect(() => decodeStoredAction('["single-cell",16,0]')).toThrow();
    expect(() => decodeStoredAction('["reroll","a",3]')).toThrow();
  });
});
