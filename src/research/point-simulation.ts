import { applyAction, getAvailableActions } from "../domain/game/actions";
import { validateGameState } from "../domain/game/game-state";
import type { GameAction, GameState } from "../domain/game/types";
import { getInitialCatalog } from "../domain/pieces/catalog";
import { compareEvaluations, evaluateState } from "../domain/solver/evaluator";
import { solveTurn } from "../domain/solver/solver";
import type { PathRewards, SolverEvaluation } from "../domain/solver/types";
import { actionKey, encodeDecision } from "./imitation";

export const POINT_SIMULATION_PROTOCOL = {
  schemaVersion: 1,
  id: "point-simulation-v1",
  profile: "controlled-single-point-line-unlock",
  fixtureCount: 160,
  rows: 16,
  cols: 10,
  repetitions: 2,
  nodeBudgets: [128, 512],
  referenceMaxNodes: 256,
  maxActions: 2,
  useMemoization: true,
  maxAlternatives: 3,
  modelCreationLoadForwardTraining: 0,
  existingDatasetLabelChanges: false,
  randomSupplyOrNewItems: false,
  realGameAcceptance: "BLOCKED_NO_GAME_SCREEN",
} as const;
export function validatePointProtocol(input: unknown) {
  if (
    !input ||
    typeof input !== "object" ||
    Object.keys(input).length !==
      Object.keys(POINT_SIMULATION_PROTOCOL).length ||
    Object.entries(POINT_SIMULATION_PROTOCOL).some(
      ([k, v]) =>
        JSON.stringify((input as Record<string, unknown>)[k]) !==
        JSON.stringify(v),
    )
  )
    throw new Error("Invalid point simulation protocol");
  return structuredClone(POINT_SIMULATION_PROTOCOL);
}
export type PointFixture = {
  id: string;
  state: GameState;
  target: { row: number; col: number };
};
export function pointFixtures(): PointFixture[] {
  const line = getInitialCatalog().find((p) => p.id === "LINE_3")!;
  return Array.from({ length: 160 }, (_, i) => {
    const row = Math.floor(i / 10),
      col = i % 10;
    const board = Array.from({ length: 16 }, (_, r) =>
      Array.from({ length: 10 }, (_, c) =>
        r === row ? c !== col : c !== (3 * r) % 10 && c !== (3 * r + 1) % 10,
      ),
    );
    return {
      id: `point-${i}`,
      target: { row, col },
      state: {
        board,
        remainingPieces: [
          {
            instanceId: "controlled-line",
            pieceIndex: 0,
            piece: structuredClone(line),
          },
        ],
        hiddenItems: [],
        abilities: { singleCell: 1, reroll: 0 },
        pendingReroll: null,
      },
    };
  });
}
function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}
const normalize = (v: unknown): unknown =>
  Array.isArray(v)
    ? v.map(normalize)
    : v && typeof v === "object"
      ? Object.fromEntries(
          Object.entries(v)
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([k, x]) => [k, normalize(x)]),
        )
      : v;
const same = (a: unknown, b: unknown) =>
  JSON.stringify(normalize(a)) === JSON.stringify(normalize(b));
function available(state: GameState) {
  const a = getAvailableActions(state);
  if (!a.ok) throw new Error(a.error.message);
  return a;
}
function transition(state: GameState, action: GameAction) {
  const t = applyAction(state, action);
  if (!t.ok) throw new Error(t.error.message);
  return t;
}
function evaluation(state: GameState, rewards: PathRewards) {
  const e = evaluateState(state, getInitialCatalog(), rewards);
  if (!e.ok) throw new Error(e.error.message);
  return e.evaluation;
}
export function verifyPointFixture(f: PointFixture) {
  const valid = validateGameState(f.state);
  if (!valid.ok) throw new Error(valid.error.message);
  const { row, col } = f.target;
  assert(
    Number.isInteger(row) &&
      row >= 0 &&
      row < 16 &&
      Number.isInteger(col) &&
      col >= 0 &&
      col < 10,
    "Invalid target",
  );
  const holes = f.state.board.map((r) => r.filter((v) => !v).length);
  assert(
    holes.every((n, r) => n === (r === row ? 1 : 2)) &&
      !f.state.board[row][col],
    "Target must be the only single-hole row",
  );
  assert(
    f.state.hiddenItems.length === 0 &&
      f.state.pendingReroll === null &&
      f.state.remainingPieces.length === 1 &&
      f.state.remainingPieces[0].piece.id === "LINE_3" &&
      f.state.abilities.singleCell === 1 &&
      f.state.abilities.reroll === 0,
    "Controlled fixture boundary",
  );
  const a = available(f.state);
  assert(
    a.phase === "playing" &&
      a.actions.length === 31 &&
      a.actions.every((x) => x.type === "single-cell"),
    "Expected exactly 31 point-only legal actions",
  );
  const expected: GameAction = { type: "single-cell", row, col };
  const next = transition(f.state, expected);
  assert(
    JSON.stringify(next.info.clearedRows) === JSON.stringify([row]) &&
      next.state.abilities.singleCell === 0,
    "Expected one horizontal clear and one spend",
  );
  assert(
    available(next.state).actions.length === 8 &&
      available(next.state).actions.every((x) => x.type === "place-piece"),
    "Expected eight unlocked line placements",
  );
  return a.actions;
}

export type PointReference = {
  complete: boolean;
  visitedNodes: number;
  leaves: number;
  maxDepth: number;
  bestObserved: SolverEvaluation | null;
  optimalFirstActions: string[] | null;
};
/** Full traversal for one piece plus one point, no items/reroll/supply. Shared domain/evaluator, no production search/pruning. */
export function pointReference(input: unknown, maxNodes = 256): PointReference {
  const v = validateGameState(input);
  if (!v.ok) throw new Error(v.error.message);
  const state = v.state,
    line = getInitialCatalog().find((p) => p.id === "LINE_3")!;
  assert(
    Number.isSafeInteger(maxNodes) && maxNodes >= 1 && maxNodes <= 256,
    "Reference budget must be 1..256",
  );
  assert(
    state.remainingPieces.length === 1 &&
      JSON.stringify(state.remainingPieces[0].piece) === JSON.stringify(line) &&
      state.hiddenItems.length === 0 &&
      state.pendingReroll === null &&
      state.abilities.singleCell === 1 &&
      state.abilities.reroll === 0,
    "Reference requires one canonical line, one point and no items/reroll",
  );
  let visitedNodes = 0,
    leaves = 0,
    maxDepth = 0,
    complete = true,
    best: SolverEvaluation | null = null;
  const roots = new Set<string>();
  function visit(
    s: GameState,
    rewards: PathRewards,
    root: string | null,
    depth: number,
  ) {
    if (visitedNodes >= maxNodes) {
      complete = false;
      return;
    }
    visitedNodes++;
    maxDepth = Math.max(maxDepth, depth);
    assert(depth <= 2, "Reference depth bound violated");
    const a = available(s);
    if (a.phase !== "playing") {
      leaves++;
      const e = evaluation(s, rewards),
        order = best ? compareEvaluations(e, best) : 1;
      if (order > 0) {
        best = e;
        roots.clear();
      }
      if (order >= 0 && root) roots.add(root);
      return;
    }
    for (const action of a.actions) {
      if (visitedNodes >= maxNodes) {
        complete = false;
        break;
      }
      const t = transition(s, action);
      visit(
        t.state,
        {
          clearedRows: rewards.clearedRows + t.info.clearedRows.length,
          acquiredItems: rewards.acquiredItems + t.info.acquiredItems.length,
        },
        root ?? actionKey(action),
        depth + 1,
      );
    }
  }
  visit(state, { clearedRows: 0, acquiredItems: 0 }, null, 0);
  return {
    complete,
    visitedNodes,
    leaves,
    maxDepth,
    bestObserved: best,
    optimalFirstActions: complete ? [...roots].sort() : null,
  };
}
export function replayPointPath(input: GameState, actions: GameAction[]) {
  assert(actions.length <= 2, "Controlled path exceeds two actions");
  let state = structuredClone(input),
    clearedRows = 0,
    acquiredItems = 0;
  for (const a of actions) {
    const t = transition(state, a);
    state = t.state;
    clearedRows += t.info.clearedRows.length;
    acquiredItems += t.info.acquiredItems.length;
  }
  return {
    state,
    evaluation: evaluation(state, { clearedRows, acquiredItems }),
  };
}
export function pointGreedy(
  input: GameState,
  mode: "first-legal" | "immediate-clear",
): GameAction[] {
  assert(
    mode === "first-legal" || mode === "immediate-clear",
    "Unknown control policy",
  );
  let state = structuredClone(input);
  const actions: GameAction[] = [];
  while (available(state).phase === "playing") {
    assert(actions.length < 2, "Controlled greedy depth exceeded");
    const legal = available(state).actions;
    const action =
      mode === "first-legal"
        ? legal[0]
        : legal.reduce((best, a) =>
            transition(state, a).info.clearedRows.length >
            transition(state, best).info.clearedRows.length
              ? a
              : best,
          );
    actions.push(action);
    state = transition(state, action).state;
  }
  return actions;
}
export function inspectPointFixture(f: PointFixture) {
  const before = JSON.stringify(f),
    legal = verifyPointFixture(f),
    reference = pointReference(f.state);
  assert(
    reference.complete &&
      reference.bestObserved !== null &&
      reference.optimalFirstActions?.length === 1,
    "Complete unique reference required",
  );
  const target: GameAction = { type: "single-cell", ...f.target };
  assert(
    reference.optimalFirstActions[0] === actionKey(target),
    "Constructed target differs from exhaustive reference",
  );
  const encoded = encodeDecision(f.state, legal, target);
  if (!encoded.ok) throw new Error(encoded.error);
  assert(
    new Set(encoded.value.candidateFeatures.map((v) => JSON.stringify(v)))
      .size === legal.length,
    "Candidate feature alias detected",
  );
  const policies: Record<
    string,
    {
      actions: GameAction[];
      correctFirst: boolean;
      allPiecesPlaced: boolean;
      clearedRows: number;
      referenceEvaluationMatch: boolean;
      searchComplete: boolean | null;
      visitedNodes: number | null;
    }
  > = {};
  const record = (
    name: string,
    actions: GameAction[],
    searchComplete: boolean | null = null,
    visitedNodes: number | null = null,
  ) => {
    const r = replayPointPath(f.state, actions);
    policies[name] = {
      actions,
      correctFirst: !!actions[0] && actionKey(actions[0]) === actionKey(target),
      allPiecesPlaced: r.evaluation.allCurrentPiecesPlaced,
      clearedRows: r.evaluation.clearedRows,
      referenceEvaluationMatch:
        compareEvaluations(r.evaluation, reference.bestObserved!) === 0,
      searchComplete,
      visitedNodes,
    };
    return r;
  };
  for (const mode of ["first-legal", "immediate-clear"] as const)
    record(mode, pointGreedy(f.state, mode));
  for (const maxNodes of POINT_SIMULATION_PROTOCOL.nodeBudgets) {
    const result = solveTurn(f.state, getInitialCatalog(), {
      maxNodes,
      maxAlternatives: 3,
      useMemoization: true,
    });
    if (!result.ok) throw new Error(result.error.message);
    const r = record(
      `dfs-${maxNodes}`,
      result.result.actions,
      result.result.search.searchComplete,
      result.result.search.visitedNodes,
    );
    assert(
      same(r.state, result.result.finalState) &&
        same(r.evaluation, result.result.evaluation),
      "DFS path replay differs",
    );
    for (const alt of result.result.alternatives) {
      const replay = replayPointPath(f.state, alt.actions);
      assert(
        same(replay.state, alt.finalState) &&
          same(replay.evaluation, alt.evaluation),
        "DFS alternative replay differs",
      );
    }
  }
  assert(JSON.stringify(f) === before, "Fixture mutated");
  return {
    id: f.id,
    target: f.target,
    reference,
    legalCandidates: legal.length,
    observationKey: JSON.stringify(encoded.value.observation),
    candidateFeaturesDistinct: true,
    targetIndex: encoded.value.label,
    policies,
  };
}
export type PointSimulationRow = ReturnType<typeof inspectPointFixture>;
export function summarizePointSimulation(rows: PointSimulationRow[]) {
  assert(
    rows.length > 0 && new Set(rows.map((r) => r.id)).size === rows.length,
    "Summary requires distinct nonempty fixtures; repetitions are not new fixtures",
  );
  const policyNames = ["first-legal", "immediate-clear", "dfs-128", "dfs-512"];
  return {
    fixtures: rows.length,
    distinctTargets: new Set(rows.map((r) => JSON.stringify(r.target))).size,
    distinctBaseObservations: new Set(rows.map((r) => r.observationKey)).size,
    distinctCandidateFeaturesEveryFixture: rows.every(
      (r) => r.candidateFeaturesDistinct,
    ),
    legalCandidateRange: [
      Math.min(...rows.map((r) => r.legalCandidates)),
      Math.max(...rows.map((r) => r.legalCandidates)),
    ],
    reference: {
      complete: rows.filter((r) => r.reference.complete).length,
      uniqueOptimalFirstActions: rows.filter(
        (r) => r.reference.optimalFirstActions?.length === 1,
      ).length,
      nodeRange: [
        Math.min(...rows.map((r) => r.reference.visitedNodes)),
        Math.max(...rows.map((r) => r.reference.visitedNodes)),
      ],
      maxDepth: Math.max(...rows.map((r) => r.reference.maxDepth)),
    },
    policies: Object.fromEntries(
      policyNames.map((name) => [
        name,
        {
          correctFirst: rows.filter((r) => r.policies[name].correctFirst)
            .length,
          allPiecesPlaced: rows.filter((r) => r.policies[name].allPiecesPlaced)
            .length,
          clearedRows: rows.reduce(
            (n, r) => n + r.policies[name].clearedRows,
            0,
          ),
          referenceEvaluationMatch: rows.filter(
            (r) => r.policies[name].referenceEvaluationMatch,
          ).length,
          searchComplete: name.startsWith("dfs-")
            ? rows.filter((r) => r.policies[name].searchComplete).length
            : null,
          visitedNodes: name.startsWith("dfs-")
            ? rows.reduce((n, r) => n + (r.policies[name].visitedNodes ?? 0), 0)
            : null,
        },
      ]),
    ),
  };
}
export function runPointSimulation(
  protocol: unknown,
  emit: (row: { repetition: number } & PointSimulationRow) => void = () => {},
) {
  const p = validatePointProtocol(protocol),
    first: PointSimulationRow[] = [];
  for (let repetition = 0; repetition < p.repetitions; repetition++)
    for (const [i, f] of pointFixtures().entries()) {
      const row = inspectPointFixture(f);
      if (repetition === 0) first.push(row);
      else
        assert(
          JSON.stringify(first[i]) === JSON.stringify(row),
          "Simulation repeat differs",
        );
      emit({ repetition, ...row });
    }
  const summary = summarizePointSimulation(first);
  assert(
    summary.distinctTargets === 160 && summary.distinctBaseObservations === 160,
    "Expected 160 unique contexts",
  );
  return {
    schemaVersion: 1,
    id: p.id,
    status: "PASS",
    protocol: p,
    summary,
    repetitionsExact: true,
    simulationCalls: { reference: 320, dfs: 640 },
    additionalCompute: {
      modelCreation: 0,
      checkpointDeserialization: 0,
      forward: 0,
      training: 0,
      existingTeacherLabelChanges: 0,
    },
    limits: [
      "Constructed two-action coordinate diagnostic; no actual game distribution, capture or native sharing validation",
      "Exhaustive reference shares domain transitions/evaluator; no independent proof of those game rules",
      "Base encoder has distinct input contexts/candidates here; no neural representability, fitting or generalization conclusion",
      "Current-turn completion is not long-term survival; no supply, random outcomes, new items, held-out evaluation or learned-policy adoption",
      "Existing CNNs were not run and their previous gates remain FAIL; application policy remains DFS",
    ],
  };
}
