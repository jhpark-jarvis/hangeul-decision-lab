import type {
  GameAction,
  GameState,
  TransitionInfo,
} from "../domain/game/types";
import { validateGameState } from "../domain/game/game-state";
import { applyAction, getAvailableActions } from "../domain/game/actions";
import { getInitialCatalog } from "../domain/pieces/catalog";
import { compareEvaluations, evaluateState } from "../domain/solver/evaluator";
import { solveOrdinaryTurn } from "../domain/solver/solver";
import type { PathRewards, SolverEvaluation } from "../domain/solver/types";
import { physicalActionKey, verifyCandidatePath } from "./teacher-audit";

export const COVERAGE_CASES = [
  "ordinary-dot",
  "ordinary-two-dots",
  "blocked-line",
  "item-acquisition",
  "item-retention",
  "single-spend-acquire",
  "reroll-boundary",
] as const;
export type QualityProtocol = {
  schemaVersion: 1;
  id: string;
  status: "PROPOSED_NOT_RUN";
  scope: "offline-teacher-diagnostics";
  splits: { train: number[]; dev: number[]; reservedTest: number[] };
  families: ["sparse", "pressure"];
  nodeBudgets: [128, 512];
  slotVariants: ["baseline", "reverse-array", "cycle-slots"];
  repetitions: 2;
  timeComparison: "OBSERVE_ONLY_UNTIL_COOPERATIVE_DEADLINE";
  reference: { scope: "ordinary-only"; maxPieces: 2; maxNodes: 1024 };
  coverageCases: string[];
  gates: {
    completeReferenceMustMatch: true;
    allReturnedPathsMustReplay: true;
    coverageTransitionsMustMatch: true;
    heldOutAfterFreezeOnly: true;
    equalNodeIsNotEqualTime: true;
  };
};
export function validateQualityProtocol(input: unknown): QualityProtocol {
  const p = input as QualityProtocol;
  const same = (a: unknown, b: unknown) =>
    JSON.stringify(a) === JSON.stringify(b);
  if (
    !p ||
    p.schemaVersion !== 1 ||
    typeof p.id !== "string" ||
    !/^[a-z0-9-]+$/.test(p.id) ||
    p.status !== "PROPOSED_NOT_RUN" ||
    p.scope !== "offline-teacher-diagnostics" ||
    !same(p.families, ["sparse", "pressure"]) ||
    !same(p.nodeBudgets, [128, 512]) ||
    !same(p.slotVariants, ["baseline", "reverse-array", "cycle-slots"]) ||
    p.repetitions !== 2 ||
    p.timeComparison !== "OBSERVE_ONLY_UNTIL_COOPERATIVE_DEADLINE" ||
    !p.reference ||
    p.reference.scope !== "ordinary-only" ||
    p.reference.maxPieces !== 2 ||
    p.reference.maxNodes !== 1024 ||
    !same(p.coverageCases, COVERAGE_CASES) ||
    !p.gates ||
    [
      "completeReferenceMustMatch",
      "allReturnedPathsMustReplay",
      "coverageTransitionsMustMatch",
      "heldOutAfterFreezeOnly",
      "equalNodeIsNotEqualTime",
    ].some((k) => p.gates[k as keyof QualityProtocol["gates"]] !== true)
  )
    throw new Error("Invalid proposed quality protocol");
  const seen = new Set<number>();
  for (const [i, split] of (
    ["train", "dev", "reservedTest"] as const
  ).entries()) {
    const seeds = p.splits?.[split];
    if (!Array.isArray(seeds) || !seeds.length || seeds.length > 8)
      throw new Error("Invalid proposed seed split");
    for (const seed of seeds) {
      if (
        !Number.isInteger(seed) ||
        seed < 4000 + i * 1000 ||
        seed >= 5000 + i * 1000 ||
        seen.has(seed)
      )
        throw new Error("Seeds must be new, disjoint and reserved by split");
      seen.add(seed);
    }
  }
  return structuredClone(p);
}
export type ReferenceResult = {
  scope: "ordinary-only";
  complete: boolean;
  visitedNodes: number;
  leaves: number;
  bestObserved: SolverEvaluation | null;
  optimalFirstActions: string[] | null;
};
/** Independent traversal, shared domain transitions/evaluator, no production memo/pruning. */
export function exhaustiveOrdinaryReference(
  input: unknown,
  maxNodes = 1024,
): ReferenceResult {
  const valid = validateGameState(input);
  if (!valid.ok) throw new Error(valid.error.message);
  if (
    valid.state.pendingReroll ||
    valid.state.remainingPieces.length > 2 ||
    !Number.isSafeInteger(maxNodes) ||
    maxNodes < 1 ||
    maxNodes > 1024
  )
    throw new Error(
      "Reference requires at most two pieces, no pending reroll and 1..1024 nodes",
    );
  const catalog = getInitialCatalog();
  if (
    valid.state.remainingPieces.some(
      (p) =>
        JSON.stringify(p.piece) !==
        JSON.stringify(catalog.find((c) => c.id === p.piece.id)),
    )
  )
    throw new Error("Reference requires canonical catalog pieces");
  let visitedNodes = 0,
    leaves = 0,
    complete = true,
    best: SolverEvaluation | null = null;
  const roots = new Set<string>();
  function visit(state: GameState, rewards: PathRewards, root: string | null) {
    if (visitedNodes >= maxNodes) {
      complete = false;
      return;
    }
    visitedNodes++;
    const available = getAvailableActions(state);
    if (!available.ok) throw new Error(available.error.message);
    const actions = available.actions.filter((a) => a.type === "place-piece");
    if (!actions.length) {
      leaves++;
      const result = evaluateState(state, catalog, rewards);
      if (!result.ok) throw new Error(result.error.message);
      const comparison = best ? compareEvaluations(result.evaluation, best) : 1;
      if (comparison > 0) {
        best = result.evaluation;
        roots.clear();
      }
      if (comparison >= 0 && root) roots.add(root);
      return;
    }
    for (const action of actions) {
      if (visitedNodes >= maxNodes) {
        complete = false;
        break;
      }
      const next = applyAction(state, action);
      if (!next.ok) throw new Error(next.error.message);
      visit(
        next.state,
        {
          clearedRows: rewards.clearedRows + next.info.clearedRows.length,
          acquiredItems: rewards.acquiredItems + next.info.acquiredItems.length,
        },
        root ?? physicalActionKey(action),
      );
    }
  }
  visit(valid.state, { clearedRows: 0, acquiredItems: 0 }, null);
  return {
    scope: "ordinary-only",
    complete,
    visitedNodes,
    leaves,
    bestObserved: best,
    optimalFirstActions: complete ? [...roots].sort() : null,
  };
}
export type QualityFixture = {
  id: (typeof COVERAGE_CASES)[number];
  state: GameState;
  action?: GameAction;
};
export function qualityFixtures(): QualityFixture[] {
  const catalog = getInitialCatalog();
  const base = (
    id: QualityFixture["id"],
    count = 1,
    pieceId = "DOT",
  ): QualityFixture => ({
    id,
    state: {
      board: Array.from({ length: 16 }, (_, r) =>
        Array.from({ length: 10 }, (_, c) => c !== (r * 3 + 1) % 10),
      ),
      remainingPieces: Array.from({ length: count }, (_, i) => ({
        instanceId: `${id}-${i}`,
        pieceIndex: i as 0 | 1,
        piece: structuredClone(catalog.find((p) => p.id === pieceId)!),
      })),
      abilities: { singleCell: 0, reroll: 0 },
      hiddenItems: [],
      pendingReroll: null,
    },
  });
  const dot = base("ordinary-dot"),
    two = base("ordinary-two-dots", 2),
    blocked = base("blocked-line", 1, "LINE_3");
  const acquisition = base("item-acquisition"),
    retention = base("item-retention"),
    single = base("single-spend-acquire"),
    reroll = base("reroll-boundary", 1, "LINE_3");
  acquisition.state.hiddenItems = [
    { row: 0, col: 0, type: "single-cell" },
    { row: 0, col: 3, type: "reroll" },
    { row: 1, col: 4, type: "single-cell" },
  ];
  retention.state.hiddenItems = acquisition.state.hiddenItems
    .slice(0, 2)
    .map((i) => ({ ...i }));
  retention.state.abilities.singleCell = 7;
  for (const fixture of [acquisition, retention]) {
    const target = fixture.state.remainingPieces[0];
    fixture.action = {
      type: "place-piece",
      instanceId: target.instanceId,
      pieceIndex: target.pieceIndex,
      pieceId: target.piece.id,
      variant: [[true]],
      row: 0,
      col: 1,
    };
  }
  single.state.abilities.singleCell = 1;
  single.state.hiddenItems = [{ row: 0, col: 4, type: "reroll" }];
  single.action = { type: "single-cell", row: 0, col: 1 };
  reroll.state.abilities.reroll = 1;
  reroll.action = {
    type: "reroll",
    instanceId: reroll.state.remainingPieces[0].instanceId,
    pieceIndex: 0,
  };
  return [dot, two, blocked, acquisition, retention, single, reroll];
}
export function assertCoverage(
  id: QualityFixture["id"],
  state: GameState,
  info: TransitionInfo,
): void {
  const expect = (ok: boolean) => {
    if (!ok) throw new Error(`Coverage mismatch: ${id}`);
  };
  switch (id) {
    case "item-acquisition":
      expect(
        info.acquiredItems.length === 2 &&
          info.retainedItems.length === 0 &&
          state.hiddenItems.length === 1 &&
          state.abilities.singleCell === 1 &&
          state.abilities.reroll === 1,
      );
      break;
    case "item-retention":
      expect(
        info.acquiredItems.length === 0 &&
          info.retainedItems.length === 2 &&
          state.hiddenItems.length === 2 &&
          state.abilities.singleCell === 7,
      );
      break;
    case "single-spend-acquire":
      expect(
        info.spentAbility === "single-cell" &&
          info.acquiredItems.length === 1 &&
          state.abilities.singleCell === 0 &&
          state.abilities.reroll === 1,
      );
      break;
    case "reroll-boundary":
      expect(
        info.spentAbility === "reroll" &&
          state.abilities.reroll === 0 &&
          state.pendingReroll !== null,
      );
      break;
    default:
      throw new Error("No scripted transition for this case");
  }
}
export function runQualityPreflight(input: unknown) {
  const protocol = validateQualityProtocol(input);
  const fixtures = qualityFixtures();
  const rows = fixtures.map((fixture) => {
    const before = JSON.stringify(fixture);
    const reference = exhaustiveOrdinaryReference(
      fixture.state,
      protocol.reference.maxNodes,
    );
    if (!reference.complete || !reference.bestObserved)
      throw new Error("Preflight reference is incomplete");
    const comparisons = [false, true].map((useMemoization) => {
      const solved = solveOrdinaryTurn(fixture.state, getInitialCatalog(), {
        maxNodes: 1024,
        maxAlternatives: 3,
        useMemoization,
      });
      if (!solved.ok) throw new Error(solved.error.message);
      const result = solved.result;
      if (
        !result.search.searchComplete ||
        compareEvaluations(result.evaluation, reference.bestObserved!) !== 0 ||
        (result.actions.length &&
          !reference.optimalFirstActions!.includes(
            physicalActionKey(result.actions[0]),
          ))
      )
        throw new Error("Complete production/reference mismatch");
      [result, ...result.alternatives].forEach((c) =>
        verifyCandidatePath(fixture.state, c),
      );
      return { useMemoization, result };
    });
    let transition = null;
    if (fixture.action) {
      const applied = applyAction(fixture.state, fixture.action);
      if (!applied.ok) throw new Error(applied.error.message);
      assertCoverage(fixture.id, applied.state, applied.info);
      transition = {
        action: fixture.action,
        state: applied.state,
        info: applied.info,
      };
    }
    if (JSON.stringify(fixture) !== before)
      throw new Error("Preflight mutated fixture");
    return { fixture, reference, comparisons, transition };
  });
  return {
    status: "PASS" as const,
    experimentStatus: protocol.status,
    protocol,
    rows,
    summary: {
      scriptedCases: rows.length,
      completeReferences: rows.length,
      productionComparisons: rows.length * 2,
      scriptedTransitions: rows.filter((r) => r.transition).length,
      returnedPathsVerified: rows.reduce(
        (n, r) =>
          n +
          r.comparisons.reduce(
            (a, c) => a + 1 + c.result.alternatives.length,
            0,
          ),
        0,
      ),
      referenceNodes: rows.reduce((n, r) => n + r.reference.visitedNodes, 0),
      largestOptimalRootSet: Math.max(
        ...rows.map((r) => r.reference.optimalFirstActions!.length),
      ),
      generatedReservedTestStates: 0,
      trainingRuns: 0,
    },
  };
}
