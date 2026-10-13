import { compareEvaluations } from "../domain/solver/evaluator";
import type { SolverEvaluation, TurnSolution } from "../domain/solver/types";
import type { GameAction } from "../domain/game/types";
import { actionKey } from "./imitation";

export const KINDS = ["place-piece", "single-cell", "reroll"] as const;
type Kind = (typeof KINDS)[number];
type Description = { kind: Kind; piece?: string; row?: number; col?: number };
export type StoredExample = {
  id: string;
  fixtureId: string;
  candidateKeys: string[];
  labels: { A512: number };
  teachers: { A512: { result: TurnSolution } };
};
export type StoredPrediction = {
  id: string;
  fixtureId: string;
  family: string;
  index: number;
  candidates: number;
  selected: Description;
  teacherActions: { A512: Description };
  complete: { A512: boolean };
  targets: {
    A512: {
      correct: boolean;
      top5: boolean;
      rank: number;
      nll: number;
      firstLegal: boolean;
    };
  };
};
function check(ok: unknown, reason: string): asserts ok {
  if (!ok) throw new Error(reason);
}
function integer(value: number, max: number) {
  return Number.isInteger(value) && value >= 0 && value <= max;
}
export function decodeStoredAction(key: string): GameAction {
  const a = JSON.parse(key);
  check(Array.isArray(a) && KINDS.includes(a[0]), "Stored action kind");
  const kind: Kind = a[0];
  check(a.length === (kind === "place-piece" ? 7 : 3), "Stored action length");
  if (kind === "single-cell") {
    check(integer(a[1], 15) && integer(a[2], 9), "Stored cell coordinate");
    return { type: kind, row: a[1], col: a[2] };
  }
  check(
    typeof a[1] === "string" && a[1].length > 0 && integer(a[2], 2),
    "Stored target",
  );
  if (kind === "reroll")
    return { type: kind, instanceId: a[1], pieceIndex: a[2] };
  check(
    typeof a[3] === "string" &&
      integer(a[4], 15) &&
      integer(a[5], 9) &&
      Array.isArray(a[6]) &&
      a[6].length > 0 &&
      a[6].length <= 5 &&
      a[6].every(
        (r: unknown) =>
          Array.isArray(r) &&
          r.length > 0 &&
          r.length <= 5 &&
          r.length === a[6][0].length &&
          r.every((c: unknown) => typeof c === "boolean"),
      ),
    "Stored placement schema",
  );
  return {
    type: kind,
    instanceId: a[1],
    pieceIndex: a[2],
    pieceId: a[3],
    row: a[4],
    col: a[5],
    variant: a[6],
  };
}
function description(a: GameAction): Description {
  if (a.type === "reroll") return { kind: a.type };
  return {
    kind: a.type,
    ...(a.type === "place-piece" ? { piece: a.pieceId } : {}),
    row: a.row,
    col: a.col,
  };
}
function sameDescription(a: Description, b: Description) {
  return ["kind", "piece", "row", "col"].every(
    (k) => a[k as keyof Description] === b[k as keyof Description],
  );
}
export function errorKind(target: GameAction, selected: GameAction) {
  if (actionKey(target) === actionKey(selected)) return "exact-label";
  if (target.type !== selected.type) return "wrong-kind";
  if (target.type === "single-cell") return "different-position";
  if (selected.type === "single-cell")
    throw new Error("Unreachable action kind");
  if (
    target.instanceId !== selected.instanceId ||
    target.pieceIndex !== selected.pieceIndex
  )
    return "different-target";
  if (
    target.type === "place-piece" &&
    selected.type === "place-piece" &&
    (target.pieceId !== selected.pieceId ||
      JSON.stringify(target.variant) !== JSON.stringify(selected.variant))
  )
    return "different-shape";
  if (target.type === "place-piece") return "different-position";
  throw new Error("Stored reroll identity mismatch");
}
function validEvaluation(e: SolverEvaluation) {
  check(
    e &&
      typeof e.survivable === "boolean" &&
      typeof e.allCurrentPiecesPlaced === "boolean" &&
      [
        "playing",
        "gameover",
        "await-next-pieces",
        "await-reroll-result",
      ].includes(e.phase) &&
      e.mobility &&
      e.remainingAbilities,
    "Stored evaluation schema",
  );
  const values = [
    e.mobility.playablePieceTypes,
    e.mobility.totalPieceTypes,
    e.mobility.totalPlacements,
    e.clearedRows,
    e.acquiredItems,
    e.remainingAbilities.reroll,
    e.remainingAbilities.singleCell,
    e.isolatedEmptyCells,
    e.unfillableGaps,
    e.emptyCells,
  ];
  check(
    values.every((n) => Number.isSafeInteger(n) && n >= 0),
    "Stored finite evaluation",
  );
}
export function storedEqualFirstActions(example: StoredExample) {
  const solution = example.teachers.A512.result;
  check(
    solution.actions.length > 0 && solution.alternatives.length <= 3,
    "Stored teacher paths",
  );
  validEvaluation(solution.evaluation);
  const best = example.candidateKeys.indexOf(actionKey(solution.actions[0]));
  check(best === example.labels.A512, "Stored teacher label identity");
  const equal = new Set([best]);
  let equalPaths = 0;
  for (const a of solution.alternatives) {
    check(a.actions.length > 0, "Empty stored alternative");
    validEvaluation(a.evaluation);
    const index = example.candidateKeys.indexOf(actionKey(a.actions[0]));
    check(index >= 0, "Stored alternative outside mask");
    const comparison = compareEvaluations(a.evaluation, solution.evaluation);
    check(comparison <= 0, "Stored alternative better than best");
    if (comparison === 0) {
      equal.add(index);
      equalPaths++;
    }
  }
  return {
    indices: [...equal],
    equalPaths,
    storedAlternatives: solution.alternatives.length,
  };
}
export function inspectStored(
  example: StoredExample,
  prediction: StoredPrediction,
) {
  const n = example.candidateKeys.length,
    label = example.labels.A512;
  check(
    n > 0 &&
      n <= 4096 &&
      new Set(example.candidateKeys).size === n &&
      integer(label, n - 1),
    "Stored mask/label",
  );
  check(
    prediction.id === example.id &&
      prediction.fixtureId === example.fixtureId &&
      prediction.candidates === n &&
      integer(prediction.index, n - 1),
    "Stored prediction identity",
  );
  const target = decodeStoredAction(example.candidateKeys[label]);
  const selected = decodeStoredAction(example.candidateKeys[prediction.index]);
  check(
    sameDescription(prediction.selected, description(selected)) &&
      sameDescription(prediction.teacherActions.A512, description(target)),
    "Stored prediction description drift",
  );
  const correct = prediction.index === label,
    p = prediction.targets.A512;
  check(
    p.correct === correct &&
      p.firstLegal === (label === 0) &&
      integer(p.rank - 1, n - 1) &&
      p.top5 === p.rank <= 5 &&
      correct === (p.rank === 1) &&
      Number.isFinite(p.nll) &&
      p.nll >= 0,
    "Stored rank/correct drift",
  );
  const solution = example.teachers.A512.result;
  check(
    prediction.complete.A512 === solution.search.searchComplete &&
      typeof solution.search.searchComplete === "boolean",
    "Stored search identity",
  );
  check(
    /^(sparse|pressure)-[0-9]+$/.test(example.fixtureId) &&
      example.fixtureId.startsWith(prediction.family + "-"),
    "Stored family identity",
  );
  const ties = storedEqualFirstActions(example);
  const legalKinds = [
    ...new Set(example.candidateKeys.map((k) => decodeStoredAction(k).type)),
  ]
    .sort()
    .join("+");
  const distance =
    target.type === "single-cell" && selected.type === "single-cell"
      ? Math.abs(target.row - selected.row) +
        Math.abs(target.col - selected.col)
      : null;
  return {
    id: example.id,
    fixtureId: example.fixtureId,
    family: prediction.family,
    teacherKind: target.type,
    selectedKind: selected.type,
    error: errorKind(target, selected),
    correct,
    legalKinds,
    searchComplete: solution.search.searchComplete,
    teacherFinalPhase: solution.evaluation.phase,
    teacherRank: p.rank,
    teacherNll: p.nll,
    top5: p.top5,
    singleCellDistance: distance,
    ties: {
      ...ties,
      selectedIsStoredEqualFirst: ties.indices.includes(prediction.index),
      alternativesMayOmitEquivalentPaths:
        solution.search.alternativesMayOmitEquivalentPaths,
    },
  };
}
export type ErrorRow = ReturnType<typeof inspectStored>;
export function aggregateErrors(rows: ErrorRow[]) {
  const kinds = Object.fromEntries(
    KINDS.map((kind) => [
      kind,
      Object.fromEntries(KINDS.map((selected) => [selected, 0])),
    ]),
  );
  const errors = Object.fromEntries(
    [
      "exact-label",
      "wrong-kind",
      "different-target",
      "different-shape",
      "different-position",
    ].map((e) => [e, 0]),
  );
  for (const r of rows) {
    kinds[r.teacherKind][r.selectedKind]++;
    errors[r.error]++;
  }
  const point = rows.filter((r) => r.teacherKind === "single-cell");
  const distances = point.flatMap((r) =>
    r.singleCellDistance === null ? [] : [r.singleCellDistance],
  );
  const tied = rows.filter((r) => r.ties.indices.length > 1);
  return {
    examples: rows.length,
    fixtures: new Set(rows.map((r) => r.fixtureId)).size,
    correct: rows.filter((r) => r.correct).length,
    confusion: kinds,
    errors,
    singleCell: {
      examples: point.length,
      exact: point.filter((r) => r.correct).length,
      wrongKind: point.filter((r) => r.error === "wrong-kind").length,
      wrongPosition: point.filter((r) => r.error === "different-position")
        .length,
      sameKindDistanceSamples: distances.length,
      meanManhattanDistance: distances.length
        ? distances.reduce((a, b) => a + b, 0) / distances.length
        : null,
      maxManhattanDistance: distances.length ? Math.max(...distances) : null,
      storedEqualFirstHits: point.filter(
        (r) => r.ties.selectedIsStoredEqualFirst,
      ).length,
    },
    storedTies: {
      examplesWithDistinctEqualFirst: tied.length,
      returnedEqualAlternativePaths: rows.reduce(
        (sum, r) => sum + r.ties.equalPaths,
        0,
      ),
      selectedInReturnedEqualFirstSet: rows.filter(
        (r) => r.ties.selectedIsStoredEqualFirst,
      ).length,
      additionalStoredTieHits: rows.filter(
        (r) => !r.correct && r.ties.selectedIsStoredEqualFirst,
      ).length,
      omissionFlagExamples: rows.filter(
        (r) => r.ties.alternativesMayOmitEquivalentPaths,
      ).length,
    },
  };
}
export function auditStored(
  examples: StoredExample[],
  predictions: StoredPrediction[],
) {
  check(
    examples.length > 0 &&
      examples.length === predictions.length &&
      new Set(examples.map((e) => e.id)).size === examples.length,
    "Stored observation join",
  );
  const rows = examples.map((e, i) => inspectStored(e, predictions[i]));
  const groups: Record<
    string,
    Record<string, ReturnType<typeof aggregateErrors>>
  > = {};
  const selectors = {
    teacherKind: (r: ErrorRow) => r.teacherKind,
    family: (r: ErrorRow) => r.family,
    searchComplete: (r: ErrorRow) => String(r.searchComplete),
    legalKinds: (r: ErrorRow) => r.legalKinds,
    teacherFinalPhase: (r: ErrorRow) => r.teacherFinalPhase,
  };
  for (const [axis, select] of Object.entries(selectors)) {
    const buckets = new Map<string, ErrorRow[]>();
    for (const row of rows) {
      const key = select(row);
      const bucket = buckets.get(key) ?? [];
      bucket.push(row);
      buckets.set(key, bucket);
    }
    groups[axis] = Object.fromEntries(
      [...buckets.entries()]
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([key, bucket]) => [key, aggregateErrors(bucket)]),
    );
  }
  return { rows, summary: { overall: aggregateErrors(rows), groups } };
}
