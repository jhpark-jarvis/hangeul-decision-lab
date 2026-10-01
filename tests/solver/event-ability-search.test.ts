import { describe, expect, it } from "vitest";
import reference from "../fixtures/pieces/catalog-v1.json";
import { applyAction } from "../../src/domain/game/actions";
import { enterNextPieces } from "../../src/domain/game/game-state";
import type { GameState } from "../../src/domain/game/types";
import { getInitialCatalog } from "../../src/domain/pieces/catalog";
import { getValidPlacements } from "../../src/domain/pieces/placement";
import {
  compareEvaluations,
  evaluateState,
} from "../../src/domain/solver/evaluator";
import { solveTurn } from "../../src/domain/solver/solver";
import {
  DEFAULT_SOLVER_CONFIG,
  type PathRewards,
  type SolverCandidate,
  type SolverEvaluation,
} from "../../src/domain/solver/types";
import {
  applyStep,
  createSession,
  inputReroll,
  installAnalysis,
} from "../../src/features/puzzle/session";
import { freezeDeep, game, instance } from "../game/fixtures";
import { staggeredBoard, success } from "./turn-fixtures";

const catalog = getInitialCatalog();
const exact = {
  ...DEFAULT_SOLVER_CONFIG,
  maxNodes: 10000,
  useMemoization: false,
};
function boardFor(rows: string[], gate = false) {
  const board = staggeredBoard();
  rows.forEach((line, r) => {
    board[5 + r].fill(true);
    [...line].forEach((cell, c) => {
      if (cell === "1") board[5 + r][2 + c] = false;
    });
  });
  if (gate) {
    board[5].fill(true);
    board[5][9] = false;
  }
  return board;
}

// Separate coordinate and single-cell enumeration, with no placement/transition
// calls. Evaluator is shared; reroll/spawn/probabilities are outside this oracle.
function variants(rows: string[]) {
  const cells = rows.flatMap((line, r) =>
    [...line].flatMap((cell, c) => (cell === "1" ? [[r, c]] : [])),
  );
  const shapes = new Map<string, number[][]>();
  for (const sign of [1, -1]) {
    let rotated = cells.map(([r, c]) => [r, c * sign]);
    for (let turn = 0; turn < 4; turn++) {
      const minR = Math.min(...rotated.map(([r]) => r)),
        minC = Math.min(...rotated.map(([, c]) => c));
      const shape = rotated
        .map(([r, c]) => [r - minR, c - minC])
        .sort(([r, c], [s, d]) => r - s || c - d);
      shapes.set(JSON.stringify(shape), shape);
      rotated = rotated.map(([r, c]) => [c, -r]);
    }
  }
  return [...shapes.values()];
}
function singleOracle(input: GameState, rows: string[]) {
  let best: SolverEvaluation | null = null;
  let bestStates: GameState[] = [];
  let completePaths = 0;
  const shapes = variants(rows);
  function fill(
    state: GameState,
    cells: number[][],
    single: boolean,
    rewards: PathRewards,
  ) {
    const next = structuredClone(state);
    for (const [r, c] of cells) next.board[r][c] = true;
    if (single) next.abilities.singleCell--;
    else next.remainingPieces = [];
    const cleared = next.board.flatMap((line, r) =>
      line.every(Boolean) ? [r] : [],
    );
    for (const row of cleared) next.board[row].fill(false);
    visit(next, {
      clearedRows: rewards.clearedRows + cleared.length,
      acquiredItems: 0,
    });
  }
  function visit(state: GameState, rewards: PathRewards) {
    let children = 0;
    if (state.remainingPieces.length) {
      for (const shape of shapes)
        for (let row = 0; row < 16; row++)
          for (let col = 0; col < 10; col++) {
            const cells = shape.map(([r, c]) => [r + row, c + col]);
            if (cells.some(([r, c]) => r >= 16 || c >= 10 || state.board[r][c]))
              continue;
            children++;
            fill(state, cells, false, rewards);
          }
      if (state.abilities.singleCell)
        for (let row = 0; row < 16; row++)
          for (let col = 0; col < 10; col++)
            if (!state.board[row][col]) {
              children++;
              fill(state, [[row, col]], true, rewards);
            }
    }
    if (!children) {
      if (!state.remainingPieces.length) completePaths++;
      const evaluation = success(
        evaluateState(state, catalog, rewards),
      ).evaluation;
      const comparison = best ? compareEvaluations(evaluation, best) : 1;
      if (comparison > 0) {
        best = evaluation;
        bestStates = [state];
      } else if (comparison === 0) bestStates.push(state);
    }
  }
  visit(input, { clearedRows: 0, acquiredItems: 0 });
  return { best, bestStates, completePaths };
}
function replay(input: GameState, candidate: SolverCandidate) {
  let state = input;
  const rewards = { clearedRows: 0, acquiredItems: 0 };
  const uses = { singleCell: 0, reroll: 0 };
  for (const [index, action] of candidate.actions.entries()) {
    if (action.type === "reroll")
      expect(index).toBe(candidate.actions.length - 1);
    const step = success(applyAction(state, action));
    if (step.info.spentAbility)
      uses[step.info.spentAbility === "reroll" ? "reroll" : "singleCell"]++;
    rewards.clearedRows += step.info.clearedRows.length;
    rewards.acquiredItems += step.info.acquiredItems.length;
    state = step.state;
  }
  expect(state).toEqual(candidate.finalState);
  expect(success(evaluateState(state, catalog, rewards)).evaluation).toEqual(
    candidate.evaluation,
  );
  expect(candidate.usedAbilities).toEqual(uses);
  return state;
}
function verifyAll(input: GameState, result: ReturnType<typeof solveTurn>) {
  const solution = success(result).result;
  for (const plan of [solution, ...solution.alternatives]) replay(input, plan);
  return solution;
}

describe("actual19 catalog and ability integration", () => {
  it("reacquires between duplicate C_5 slots, retains overflow, and collects it only on a later clear", () => {
    const board = staggeredBoard();
    for (const row of [3, 5, 9, 11]) {
      board[row].fill(true);
      board[row].fill(false, 2, 4);
    }
    for (const row of [4, 10]) {
      board[row].fill(true);
      board[row][9] = false;
    }
    const input = freezeDeep(
      game({
        board,
        remainingPieces: [instance(0, "C_5"), instance(2, "C_5")],
        abilities: { reroll: 6, singleCell: 1 },
        hiddenItems: [
          { row: 3, col: 0, type: "single-cell" },
          { row: 5, col: 0, type: "reroll" },
          { row: 9, col: 0, type: "single-cell" },
        ],
      }),
    );
    const result = verifyAll(input, solveTurn(input, catalog));
    expect(result.evaluation.allCurrentPiecesPlaced).toBe(true);
    expect(result.evaluation.acquiredItems).toBeGreaterThanOrEqual(1);
    expect(result.evaluation.acquiredItems).toBeLessThanOrEqual(2);
    expect(result.usedAbilities).toEqual({ reroll: 0, singleCell: 2 });
    expect(result.actions.map((action) => action.type)).toEqual([
      "single-cell",
      "place-piece",
      "single-cell",
      "place-piece",
    ]);
    expect(result.finalState.hiddenItems).toContainEqual({
      row: 5,
      col: 0,
      type: "reroll",
    });
    // A hand-selected legal route is a replay fixture, not a claim that the
    // bounded solver finds it. Check both actual solver paths and this route.
    const actions: SolverCandidate["actions"] = [
      { type: "single-cell", row: 4, col: 9 },
      {
        type: "place-piece",
        pieceId: "C_5",
        instanceId: "set1-0",
        pieceIndex: 0,
        variant: [
          [true, true],
          [true, false],
          [true, true],
        ],
        rotation: 0,
        flipped: false,
        row: 3,
        col: 2,
      },
      { type: "single-cell", row: 10, col: 9 },
      {
        type: "place-piece",
        pieceId: "C_5",
        instanceId: "set1-2",
        pieceIndex: 2,
        variant: [
          [true, true],
          [true, false],
          [true, true],
        ],
        rotation: 0,
        flipped: false,
        row: 9,
        col: 2,
      },
    ];
    let referenceState: GameState = input;
    const rewards = { clearedRows: 0, acquiredItems: 0 };
    for (const action of actions) {
      const transition = success(applyAction(referenceState, action));
      referenceState = transition.state;
      rewards.clearedRows += transition.info.clearedRows.length;
      rewards.acquiredItems += transition.info.acquiredItems.length;
    }
    expect(rewards).toEqual({ clearedRows: 6, acquiredItems: 2 });
    expect(referenceState.abilities).toEqual({ reroll: 6, singleCell: 1 });
    expect(referenceState.hiddenItems).toEqual([
      { row: 5, col: 0, type: "reroll" },
    ]);
    const evaluation = success(
      evaluateState(referenceState, catalog, rewards),
    ).evaluation;
    if (compareEvaluations(evaluation, result.evaluation) > 0) {
      expect(result.search.searchComplete).toBe(false);
      expect(result.search.optimalWithinScope).toBe(false);
    }
    const referencePlan = {
      ...result,
      actions,
      finalState: referenceState,
      evaluation,
      usedAbilities: { reroll: 0, singleCell: 2 },
      alternatives: [],
    };
    const initial = createSession(input);
    let session = installAnalysis(
      initial,
      initial.version,
      { ok: true, result: referencePlan },
      0,
    );
    expect(session.error).toBeNull();
    expect(
      session.analysis!.plans[0].info.flatMap((info) => info.retainedItems),
    ).toContainEqual({ row: 5, col: 0, type: "reroll" });
    for (let step = 0; step < referencePlan.actions.length; step++) {
      session = applyStep(session, session.version, step);
      expect(session.error).toBeNull();
    }
    expect(session.game).toEqual(referenceState);
    const next = success(
      enterNextPieces(session.game, [
        instance(0, "BRANCH_4", "set2-0"),
        instance(1, "LINE_5", "set2-1"),
        instance(2, "DIAGONAL_3", "set2-2"),
      ]),
    ).state;
    next.board[5].fill(true);
    next.board[5][9] = false;
    const recovered = success(
      applyAction(freezeDeep(next), { type: "single-cell", row: 5, col: 9 }),
    );
    expect(recovered.info.acquiredItems).toEqual([
      { row: 5, col: 0, type: "reroll" },
    ]);
    expect(recovered.info.retainedItems).toEqual([]);
    expect(recovered.state.abilities).toEqual({ reroll: 7, singleCell: 0 });
    expect(recovered.state.hiddenItems).toEqual([]);
    expect(next.hiddenItems).toHaveLength(1);
    expect(input.hiddenItems).toHaveLength(3);
  });
  it.each([1, 2, 4, 20, 512])(
    "19-type ability budget %i preserves valid frontier/alternatives and reports limits",
    (maxNodes) => {
      const ref = reference.pieces.find(
        (piece) => piece.id === "DOUBLE_BRANCH_7",
      )!;
      const input = freezeDeep(
        game({
          board: boardFor(ref.rows, true),
          remainingPieces: [instance(2, ref.id)],
          abilities: { reroll: 0, singleCell: 1 },
        }),
      );
      const result = verifyAll(
        input,
        solveTurn(input, catalog, { ...DEFAULT_SOLVER_CONFIG, maxNodes }),
      );
      expect(result.search.visitedNodes).toBeLessThanOrEqual(maxNodes);
      expect(result.search.memoEntries).toBeLessThanOrEqual(
        result.search.visitedNodes,
      );
      if (!result.search.searchComplete) {
        expect(result.search.optimalWithinScope).toBe(false);
        expect(result.search.stopReason).toBe("node-budget");
      }
      expect(
        result.search.abilitySearch?.ordinaryProbe.noCompletePathProven,
      ).toBe(
        result.search.abilitySearch?.ordinaryProbe.searchComplete === true &&
          result.search.abilitySearch.ordinaryProbe.discoveredCompletePaths ===
            0,
      );
    },
  );
  it.each(reference.pieces)(
    "$id: single-cell search agrees with coordinate enumeration and memo",
    (ref) => {
      const input = freezeDeep(
        game({
          board: boardFor(ref.rows, true),
          remainingPieces: [instance(2, ref.id)],
          abilities: { reroll: 0, singleCell: 1 },
        }),
      );
      const expected = singleOracle(input, ref.rows);
      const result = verifyAll(input, solveTurn(input, catalog, exact));
      const memo = verifyAll(
        input,
        solveTurn(input, catalog, { ...exact, useMemoization: true }),
      );
      expect(result.search.searchComplete).toBe(true);
      expect(memo.search.searchComplete).toBe(true);
      expect(result.evaluation).toEqual(expected.best);
      expect(memo.evaluation).toEqual(expected.best);
      expect(expected.bestStates).toContainEqual(result.finalState);
      expect(expected.bestStates).toContainEqual(memo.finalState);
      // The ordinary probe counts some paths separately; only the ability DFS
      // count is comparable to this reference enumeration.
      expect(result.search.discoveredCompletePaths).toBe(
        expected.completePaths,
      );
      expect(result.usedAbilities.singleCell).toBe(1);
      expect(result.evaluation.allCurrentPiecesPlaced).toBe(true);
      expect(result.search.abilitySearch?.rerollFutureEvaluated).toBe(false);
    },
  );
  it.each(reference.pieces)(
    "actual reroll result $id resolves fixedslot and replays through session",
    (ref) => {
      const original = ref.id === "STAR_9" ? "PI_10" : "STAR_9";
      const input = freezeDeep(
        game({
          board: boardFor(ref.rows),
          remainingPieces: [instance(2, original)],
          abilities: { reroll: 1, singleCell: 0 },
        }),
      );
      expect(
        success(getValidPlacements(input.board, input.remainingPieces[0].piece))
          .placements,
      ).toHaveLength(0);
      const first = verifyAll(input, solveTurn(input, catalog));
      expect(first.actions).toEqual([
        expect.objectContaining({
          type: "reroll",
          pieceIndex: 2,
          reason: "blocked-piece",
        }),
      ]);
      const initial = createSession(input);
      const installed = installAnalysis(
        initial,
        initial.version,
        { ok: true, result: first },
        0,
      );
      expect(installed.error).toBeNull();
      const waiting = applyStep(installed, installed.version, 0);
      expect(waiting.analysis).toBeNull();
      expect(waiting.game.abilities.reroll).toBe(0);
      expect(waiting.game.pendingReroll).toEqual({
        pieceIndex: 2,
        instanceId: "set1-2",
      });
      const same = inputReroll(waiting, original);
      expect(same.error).not.toBeNull();
      expect(same.game).toEqual(waiting.game);
      const resolved = inputReroll(same, ref.id);
      expect(resolved.error).toBeNull();
      expect(resolved.game.pendingReroll).toBeNull();
      expect(resolved.game.remainingPieces[0]).toMatchObject({
        pieceIndex: 2,
        instanceId: "set1-2",
        piece: { id: ref.id },
      });
      expect(applyStep(resolved, installed.version, 0).error).not.toBeNull();
      const solution = verifyAll(
        resolved.game,
        solveTurn(resolved.game, catalog),
      );
      expect(solution.evaluation.allCurrentPiecesPlaced).toBe(true);
      const analyzed = installAnalysis(
        resolved,
        resolved.version,
        { ok: true, result: solution },
        0,
      );
      expect(analyzed.error).toBeNull();
      const applied = applyStep(analyzed, analyzed.version, 0);
      expect(applied.error).toBeNull();
      expect(applied.game).toEqual(solution.finalState);
      expect(applied.game.remainingPieces).toEqual([]);
      expect(input.remainingPieces[0].piece.id).toBe(original);
      expect(input.abilities.reroll).toBe(1);
    },
  );
});
