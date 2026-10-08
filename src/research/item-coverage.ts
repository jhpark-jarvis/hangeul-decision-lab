import { applyAction } from "../domain/game/actions";
import type {
  GameAction,
  GameState,
  TransitionInfo,
} from "../domain/game/types";
import { getInitialCatalog } from "../domain/pieces/catalog";
import { compareEvaluations } from "../domain/solver/evaluator";
import { solveOrdinaryTurn, solveTurn } from "../domain/solver/solver";
import { physicalActionKey, verifyCandidatePath } from "./teacher-audit";
import { pathCoverage } from "./teacher-matrix";
import {
  exhaustiveOrdinaryReference,
  qualityFixtures,
} from "./teacher-quality";

export const ITEM_COVERAGE_PROTOCOL = {
  schemaVersion: 1,
  id: "item-coverage-20261008-v1",
  provenance: "synthetic-scripted",
  scope: "current-turn-diagnostic-only",
  nodeBudgets: [128, 512],
  repetitions: 2,
  useMemoization: true,
  maxAlternatives: 3,
  ordinaryReferenceMaxNodes: 1024,
  cases: [
    "item-acquisition",
    "item-retention",
    "single-spend-acquire",
    "reroll-boundary",
    "capacity-six-row-order",
    "spend-at-capacity",
    "simultaneous-row-order",
    "capacity-reroll-boundary",
  ],
  heldOut: "NOT_RUN",
  training: "NOT_RUN",
} as const;
export function validateItemProtocol(input: unknown) {
  if (
    !input ||
    typeof input !== "object" ||
    Object.keys(input).length !== Object.keys(ITEM_COVERAGE_PROTOCOL).length ||
    Object.entries(ITEM_COVERAGE_PROTOCOL).some(
      ([key, value]) =>
        JSON.stringify((input as Record<string, unknown>)[key]) !==
        JSON.stringify(value),
    )
  )
    throw new Error("Invalid item diagnostic protocol");
  return structuredClone(ITEM_COVERAGE_PROTOCOL);
}
type Expected = {
  clearedRows: number[];
  acquiredItems: GameState["hiddenItems"];
  retainedItems: GameState["hiddenItems"];
  hiddenItems: GameState["hiddenItems"];
  abilities: GameState["abilities"];
  spentAbility: TransitionInfo["spentAbility"];
  pendingReroll: GameState["pendingReroll"];
};
export type ItemCoverageFixture = {
  id: string;
  state: GameState;
  action: GameAction;
  expected: Expected;
};
const normalize = (value: unknown): unknown =>
  Array.isArray(value)
    ? value.map(normalize)
    : value && typeof value === "object"
      ? Object.fromEntries(
          Object.entries(value)
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([k, v]) => [k, normalize(v)]),
        )
      : value;
const same = (a: unknown, b: unknown) =>
  JSON.stringify(normalize(a)) === JSON.stringify(normalize(b));
export function itemCoverageFixtures(): ItemCoverageFixture[] {
  const originals = qualityFixtures()
    .filter((f) => f.action)
    .map((f) => structuredClone(f));
  const [acquire, retain, single, reroll] = originals;
  const six = structuredClone(acquire);
  six.state.abilities.singleCell = 6;
  const spend = structuredClone(retain);
  spend.action = { type: "single-cell", row: 0, col: 1 };
  const simultaneous = structuredClone(six),
    line = getInitialCatalog().find((p) => p.id === "LINE_5")!;
  simultaneous.state.remainingPieces[0].piece = structuredClone(line);
  for (let r = 0; r < 5; r++)
    simultaneous.state.board[r] = Array.from({ length: 10 }, (_, c) => c !== 1);
  const point = { row: 0, col: 0, type: "single-cell" } as const,
    roll = { row: 0, col: 9, type: "reroll" } as const,
    later = { row: 1, col: 8, type: "single-cell" } as const;
  simultaneous.state.hiddenItems = [later, roll, point].map((i) => ({ ...i }));
  const target = simultaneous.state.remainingPieces[0];
  simultaneous.action = {
    type: "place-piece",
    instanceId: target.instanceId,
    pieceIndex: target.pieceIndex,
    pieceId: line.id,
    variant: structuredClone(line.shape),
    row: 0,
    col: 1,
  };
  const capRoll = structuredClone(reroll);
  capRoll.state.abilities.singleCell = 6;
  capRoll.state.hiddenItems = [{ row: 0, col: 0, type: "single-cell" }];
  const expected = (
    clearedRows: number[],
    acquiredItems: GameState["hiddenItems"],
    retainedItems: GameState["hiddenItems"],
    hiddenItems: GameState["hiddenItems"],
    singleCell: number,
    rollCount: number,
    spentAbility: TransitionInfo["spentAbility"] = null,
    pendingReroll: GameState["pendingReroll"] = null,
  ): Expected => ({
    clearedRows,
    acquiredItems,
    retainedItems,
    hiddenItems,
    abilities: { singleCell, reroll: rollCount },
    spentAbility,
    pendingReroll,
  });
  const items = acquire.state.hiddenItems;
  return [
    {
      ...acquire,
      action: acquire.action!,
      expected: expected([0], items.slice(0, 2), [], items.slice(2), 1, 1),
    },
    {
      ...retain,
      action: retain.action!,
      expected: expected([0], [], items.slice(0, 2), items.slice(0, 2), 7, 0),
    },
    {
      ...single,
      action: single.action!,
      expected: expected(
        [0],
        single.state.hiddenItems,
        [],
        [],
        0,
        1,
        "single-cell",
      ),
    },
    {
      ...reroll,
      action: reroll.action!,
      expected: expected([], [], [], [], 0, 0, "reroll", {
        instanceId: reroll.state.remainingPieces[0].instanceId,
        pieceIndex: 0,
      }),
    },
    {
      ...six,
      id: "capacity-six-row-order",
      action: six.action!,
      expected: expected(
        [0],
        items.slice(0, 1),
        items.slice(1, 2),
        items.slice(1),
        7,
        0,
      ),
    },
    {
      ...spend,
      id: "spend-at-capacity",
      action: spend.action!,
      expected: expected(
        [0],
        items.slice(0, 1),
        items.slice(1, 2),
        items.slice(1, 2),
        7,
        0,
        "single-cell",
      ),
    },
    {
      ...simultaneous,
      id: "simultaneous-row-order",
      action: simultaneous.action!,
      expected: expected(
        [0, 1, 2, 3, 4],
        [point],
        [roll, later],
        [roll, later],
        7,
        0,
      ),
    },
    {
      ...capRoll,
      id: "capacity-reroll-boundary",
      action: capRoll.action!,
      expected: expected(
        [],
        [],
        [],
        capRoll.state.hiddenItems,
        6,
        0,
        "reroll",
        {
          instanceId: capRoll.state.remainingPieces[0].instanceId,
          pieceIndex: 0,
        },
      ),
    },
  ].map((f) => structuredClone(f));
}
export function verifyItemTransition(
  f: ItemCoverageFixture,
  state: GameState,
  info: TransitionInfo,
) {
  const actual: Expected = {
    clearedRows: info.clearedRows,
    acquiredItems: info.acquiredItems,
    retainedItems: info.retainedItems,
    hiddenItems: state.hiddenItems,
    abilities: state.abilities,
    spentAbility: info.spentAbility,
    pendingReroll: state.pendingReroll,
  };
  if (!same(actual, f.expected))
    throw new Error(`Scripted item transition mismatch: ${f.id}`);
  for (let r = 0; r < 16; r++) {
    const expected = f.expected.clearedRows.includes(r)
      ? Array<boolean>(10).fill(false)
      : f.state.board[r];
    if (!same(state.board[r], expected))
      throw new Error(`Scripted board mismatch: ${f.id}`);
  }
  const action = f.action;
  const remaining =
    action.type === "place-piece"
      ? f.state.remainingPieces.filter(
          (p) => p.instanceId !== action.instanceId,
        )
      : f.state.remainingPieces;
  if (!same(state.remainingPieces, remaining))
    throw new Error(`Scripted piece mismatch: ${f.id}`);
}
export function runItemCoverage(
  input: unknown,
  clock: () => number,
  progress?: (row: ReturnType<typeof runCase>) => void,
) {
  const protocol = validateItemProtocol(input),
    rows = itemCoverageFixtures().map((f) => {
      const row = runCase(f, clock);
      progress?.(structuredClone(row));
      return row;
    });
  const unique = rows.flatMap((r) =>
    r.probes.filter((p) => p.repetition === 0),
  );
  return {
    status: "PASS" as const,
    protocol,
    rows,
    summary: {
      scriptedCases: rows.length,
      scriptedTransitions: rows.length,
      completeOrdinaryReferences: rows.length,
      ordinaryComparisons: rows.length,
      fullProbes: rows.reduce((n, r) => n + r.probes.length, 0),
      repeatedConditionsChecked: unique.length,
      returnedPathsVerified: rows.reduce(
        (n, r) =>
          n +
          1 +
          r.ordinary.alternatives.length +
          r.probes.reduce((a, p) => a + p.returnedPaths, 0),
        0,
      ),
      generatedTestStates: 0,
      trainingRuns: 0,
      selectedByBudget: protocol.nodeBudgets.map((budget) => {
        const ps = unique.filter((p) => p.budget === budget);
        return {
          budget,
          states: ps.length,
          incomplete: ps.filter((p) => !p.result.search.searchComplete).length,
          acquisitionPaths: ps.filter((p) => p.coverage.acquiredItems > 0)
            .length,
          retentionPaths: ps.filter((p) => p.coverage.retainedItemEvents > 0)
            .length,
          singleSpendPaths: ps.filter((p) => p.coverage.singleCellSpends > 0)
            .length,
          rerollSpendPaths: ps.filter((p) => p.coverage.rerollSpends > 0)
            .length,
          scriptedFirstMatches: ps.filter((p) => p.scriptedFirstMatches).length,
        };
      }),
    },
  };
}
function runCase(fixture: ItemCoverageFixture, clock: () => number) {
  const before = JSON.stringify(fixture),
    applied = applyAction(fixture.state, fixture.action);
  if (!applied.ok) throw new Error(applied.error.message);
  verifyItemTransition(fixture, applied.state, applied.info);
  const reference = exhaustiveOrdinaryReference(fixture.state, 1024);
  const ordinary = solveOrdinaryTurn(fixture.state, getInitialCatalog(), {
    maxNodes: 1024,
    maxAlternatives: 3,
    useMemoization: true,
  });
  if (
    !reference.complete ||
    !reference.bestObserved ||
    !ordinary.ok ||
    !ordinary.result.search.searchComplete
  )
    throw new Error("Incomplete ordinary coverage reference/comparison");
  if (
    compareEvaluations(ordinary.result.evaluation, reference.bestObserved) !==
      0 ||
    (ordinary.result.actions.length &&
      !reference.optimalFirstActions!.includes(
        physicalActionKey(ordinary.result.actions[0]),
      ))
  )
    throw new Error("Ordinary coverage reference mismatch");
  [ordinary.result, ...ordinary.result.alternatives].forEach((c) =>
    verifyCandidatePath(fixture.state, c),
  );
  const probes = [];
  for (const budget of ITEM_COVERAGE_PROTOCOL.nodeBudgets) {
    let original: string | null = null;
    for (let repetition = 0; repetition < 2; repetition++) {
      const state = structuredClone(fixture.state),
        stateBefore = JSON.stringify(state);
      const start = clock(),
        solved = solveTurn(state, getInitialCatalog(), {
          maxNodes: budget,
          maxAlternatives: 3,
          useMemoization: true,
        }),
        end = clock(),
        latencyMs = end - start;
      if (![start, end, latencyMs].every(Number.isFinite) || latencyMs < 0)
        throw new Error("Invalid coverage clock");
      if (!solved.ok) throw new Error(solved.error.message);
      const result = solved.result;
      if (
        result.search.visitedNodes > budget ||
        result.search.maxNodes !== budget
      )
        throw new Error("Coverage node budget mismatch");
      [result, ...result.alternatives].forEach((c) =>
        verifyCandidatePath(state, c),
      );
      if (original !== null && original !== JSON.stringify(result))
        throw new Error("Repeated coverage result changed");
      original = JSON.stringify(result);
      if (JSON.stringify(state) !== stateBefore)
        throw new Error("Coverage solver mutated input");
      probes.push({
        budget,
        repetition,
        result,
        latencyMs,
        returnedPaths: 1 + result.alternatives.length,
        coverage: pathCoverage(state, result),
        scriptedFirstMatches:
          !!result.actions.length &&
          physicalActionKey(result.actions[0]) ===
            physicalActionKey(fixture.action),
      });
    }
  }
  if (JSON.stringify(fixture) !== before)
    throw new Error("Coverage fixture mutated");
  return {
    fixture,
    transition: { state: applied.state, info: applied.info },
    reference,
    ordinary: ordinary.result,
    probes,
  };
}
