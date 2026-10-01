import { describe, expect, it } from "vitest";
import {
  applyAction,
  getAvailableActions,
} from "../../src/domain/game/actions";
import { resolveReroll } from "../../src/domain/game/game-state";
import { getRegressionCatalog } from "../game/fixtures";
import { getAbilityCandidates } from "../../src/domain/solver/ability-candidates";
import { evaluateState } from "../../src/domain/solver/evaluator";
import { solveTurn } from "../../src/domain/solver/solver";
import {
  DEFAULT_SOLVER_CONFIG,
  type SolverCandidate,
} from "../../src/domain/solver/types";
import { freezeDeep, game, instance, piece } from "../game/fixtures";
import { singleRescue, reacquireRescue } from "./ability-fixtures";
import { staggeredBoard, success } from "./turn-fixtures";

const exact = {
  ...DEFAULT_SOLVER_CONFIG,
  maxNodes: 10000,
  useMemoization: false,
};
const catalog = getRegressionCatalog();

function replay(input: ReturnType<typeof game>, candidate: SolverCandidate) {
  let state = input;
  let clearedRows = 0,
    acquiredItems = 0,
    singleCell = 0,
    reroll = 0;
  for (const [index, action] of candidate.actions.entries()) {
    if (action.type === "reroll")
      expect(index).toBe(candidate.actions.length - 1);
    const transition = success(applyAction(state, action));
    state = transition.state;
    clearedRows += transition.info.clearedRows.length;
    acquiredItems += transition.info.acquiredItems.length;
    if (transition.info.spentAbility === "single-cell") singleCell++;
    if (transition.info.spentAbility === "reroll") reroll++;
  }
  expect(state).toEqual(candidate.finalState);
  expect(
    success(evaluateState(state, catalog, { clearedRows, acquiredItems }))
      .evaluation,
  ).toEqual(candidate.evaluation);
  expect(candidate.usedAbilities).toEqual({ singleCell, reroll });
  return state;
}

describe("ability candidates and immutable replay", () => {
  it("agrees with a handwritten exhaustive two-single-cell ring oracle and memo on/off", () => {
    const input = freezeDeep(singleRescue(2));
    const footprint = [
      [0, 0],
      [0, 1],
      [0, 2],
      [1, 0],
      [1, 2],
      [2, 0],
      [2, 1],
      [2, 2],
    ];
    let completePaths = 0;
    function clear(board: boolean[][]) {
      let count = 0;
      for (const row of board)
        if (row.every(Boolean)) {
          row.fill(false);
          count++;
        }
      return count;
    }
    // No solver/placement/transforms/game transition APIs in this reference.
    function enumerate(board: boolean[][], charges: number, cleared: number) {
      for (let row = 0; row < 14; row++)
        for (let col = 0; col < 8; col++) {
          if (footprint.some(([dr, dc]) => board[row + dr][col + dc])) continue;
          const next = board.map((cells) => cells.slice());
          for (const [dr, dc] of footprint) next[row + dr][col + dc] = true;
          expect(cleared + clear(next)).toBe(3);
          expect(charges).toBe(0);
          completePaths++;
        }
      if (!charges) return;
      for (let row = 0; row < 16; row++)
        for (let col = 0; col < 10; col++) {
          if (board[row][col]) continue;
          const next = board.map((cells) => cells.slice());
          next[row][col] = true;
          enumerate(next, charges - 1, cleared + clear(next));
        }
    }
    enumerate(input.board, 2, 0);
    // Filling row4's two holes, or clearing rows1+2 / rows6+7, each has two orders.
    expect(completePaths).toBe(6);
    const exhaustive = success(solveTurn(input, catalog, exact)).result;
    expect(exhaustive.search.searchComplete).toBe(true);
    expect(exhaustive.search.discoveredCompletePaths).toBe(completePaths);
    const memoized = success(
      solveTurn(input, catalog, { ...exact, useMemoization: true }),
    ).result;
    expect(memoized.actions).toEqual(exhaustive.actions);
    expect(memoized.evaluation).toEqual(exhaustive.evaluation);
    expect(memoized.search.memoPrunedNodes).toBeGreaterThan(0);
    expect(memoized.search.visitedNodes).toBeLessThan(
      exhaustive.search.visitedNodes,
    );
  });
  it("spends a reacquired single cell between ordinary pieces while keeping finite depth", () => {
    const input = freezeDeep(reacquireRescue());
    const result = success(solveTurn(input, catalog)).result;
    expect(result.evaluation.allCurrentPiecesPlaced).toBe(true);
    expect(result.usedAbilities.singleCell).toBe(2);
    expect(result.evaluation.acquiredItems).toBe(1);
    const types = result.actions.map((action) => action.type);
    expect(types).toEqual([
      "single-cell",
      "place-piece",
      "single-cell",
      "place-piece",
    ]);
    replay(input, result);
    expect(result.search.abilitySearch?.maxActionDepth).toBeLessThanOrEqual(4);
  });
  it("allows more than seven lifetime uses through reacquisition without an arbitrary depth cutoff", () => {
    const input = freezeDeep(
      game({
        board: staggeredBoard(),
        remainingPieces: [
          {
            ...instance(2),
            piece: {
              id: "SYNTHETIC_OVERSIZE",
              name: "synthetic test only",
              shape: [Array<boolean>(11).fill(true)],
            },
          },
        ],
        abilities: { singleCell: 7, reroll: 0 },
        hiddenItems: Array.from({ length: 16 }, (_, row) => ({
          row,
          col: 0,
          type: "single-cell" as const,
        })),
      }),
    );
    const result = success(
      solveTurn(input, catalog, { ...DEFAULT_SOLVER_CONFIG, maxNodes: 32 }),
    ).result;
    expect(result.search.abilitySearch?.maxActionDepth).toBeGreaterThan(7);
    expect(result.search.abilitySearch?.maxActionDepth).toBeLessThanOrEqual(24);
    expect(result.usedAbilities.singleCell).toBeGreaterThan(7);
    replay(input, result);
    expect(result.finalState.abilities.singleCell).toBeLessThanOrEqual(7);
  });
  it("propagates bad state/catalog/config and accepts a corrected retry", () => {
    expect(solveTurn({}, catalog)).toMatchObject({ ok: false });
    expect(solveTurn(singleRescue(), [piece(), piece()])).toMatchObject({
      ok: false,
      error: { code: "INVALID_CATALOG" },
    });
    expect(
      solveTurn(singleRescue(), catalog, {
        ...DEFAULT_SOLVER_CONFIG,
        maxNodes: 0,
      }),
    ).toMatchObject({ ok: false, error: { code: "INVALID_SOLVER_CONFIG" } });
    expect(
      success(solveTurn(singleRescue(), catalog)).result.evaluation
        .allCurrentPiecesPlaced,
    ).toBe(true);
  });
  it("offers no actions after full consumption or while reroll input is pending", () => {
    const next = success(
      solveTurn(
        game({ remainingPieces: [], abilities: { reroll: 2, singleCell: 3 } }),
        catalog,
      ),
    ).result;
    expect(next.actions).toEqual([]);
    expect(next.search.stopReason).toBe("await-next-pieces");
    expect(next.search.visitedNodes).toBe(1);
    expect(next.search.specialAbilitiesSearched).toBe(false);
    expect(
      success(getAbilityCandidates(next.finalState, catalog, true)).actions,
    ).toEqual([]);
    const pending = game({
      pendingReroll: { instanceId: "set1-2", pieceIndex: 2 },
      abilities: { reroll: 1, singleCell: 1 },
    });
    const result = success(solveTurn(pending, catalog)).result;
    expect(result.actions).toEqual([]);
    expect(result.finalState).toEqual(pending);
    expect(result.search.stopReason).toBe("await-reroll-result");
    expect(result.search.specialAbilitiesSearched).toBe(false);
    expect(
      success(getAbilityCandidates(pending, catalog, true)).actions,
    ).toEqual([]);
  });
  it("retains all legal single cells, prioritizes immediate clears, and reports excluded rerolls", () => {
    const input = game({
      remainingPieces: [instance(2)],
      abilities: { reroll: 1, singleCell: 1 },
    });
    input.board[9].fill(true);
    input.board[9][4] = false;
    const candidates = success(
      getAbilityCandidates(freezeDeep(input), catalog, false),
    );
    const legal = success(getAvailableActions(input)).actions;
    expect(
      candidates.actions.filter((a) => a.type === "single-cell"),
    ).toHaveLength(151);
    expect(candidates.actions[0]).toEqual({
      type: "single-cell",
      row: 9,
      col: 4,
    });
    expect(candidates.actions.filter((a) => a.type === "single-cell")).toEqual(
      expect.arrayContaining(legal.filter((a) => a.type === "single-cell")),
    );
    expect(candidates.excludedRerollTargets).toBe(1);
    expect(candidates.actions.some((a) => a.type === "reroll")).toBe(false);
    const severe = success(
      getAbilityCandidates(input, [piece("MIEUM")], false),
    );
    expect(severe.excludedRerollTargets).toBe(1); // Empty area still fits MIEUM.
  });
  it.each([1, 2])(
    "uses %i single cells to clear a row and rescue the remaining ring",
    (holes) => {
      const input = freezeDeep(singleRescue(holes));
      const result = success(solveTurn(input, catalog)).result;
      expect(result.evaluation.allCurrentPiecesPlaced).toBe(true);
      expect(
        result.actions.filter((a) => a.type === "single-cell"),
      ).toHaveLength(holes);
      expect(result.actions.at(-1)?.type).toBe("place-piece");
      expect(result.evaluation.clearedRows).toBe(3);
      for (const candidate of [result, ...result.alternatives])
        replay(input, candidate);
      expect(result.search).toMatchObject({
        scope: "pieces-and-abilities",
        specialAbilitiesSearched: true,
      });
    },
  );
  it("ends blocked-piece reroll at pending input and resumes only after the actual new piece", () => {
    const input = freezeDeep(
      game({
        board: staggeredBoard(),
        remainingPieces: [instance(2, "MIEUM")],
        abilities: { reroll: 1, singleCell: 0 },
      }),
    );
    const result = success(solveTurn(input, catalog, exact)).result;
    expect(result.actions).toEqual([
      {
        type: "reroll",
        instanceId: "set1-2",
        pieceIndex: 2,
        reason: "blocked-piece",
      },
    ]);
    expect(result.evaluation).toMatchObject({
      phase: "await-reroll-result",
      survivable: true,
      allCurrentPiecesPlaced: false,
      remainingAbilities: { reroll: 0, singleCell: 0 },
    });
    const pending = replay(input, result);
    expect(success(solveTurn(pending, catalog)).result.actions).toEqual([]);
    const resolved = success(resolveReroll(pending, piece("DOT"))).state;
    expect(
      success(solveTurn(resolved, catalog)).result.evaluation
        .allCurrentPiecesPlaced,
    ).toBe(true);
    expect(input.abilities.reroll).toBe(1);
  });
  it("offers an unblocked target when the initial ordinary probe finds no full path", () => {
    const input = game({
      remainingPieces: [instance(0), instance(2)],
      abilities: { reroll: 1, singleCell: 0 },
    });
    const actions = success(getAbilityCandidates(input, catalog, true)).actions;
    expect(actions.filter((a) => a.type === "reroll")).toEqual([
      {
        type: "reroll",
        instanceId: "set1-0",
        pieceIndex: 0,
        reason: "initial-ordinary-probe-no-complete-path",
      },
      {
        type: "reroll",
        instanceId: "set1-2",
        pieceIndex: 2,
        reason: "initial-ordinary-probe-no-complete-path",
      },
    ]);
  });
  it("handles zero catalog mobility separately from current target placement", () => {
    const input = game({
      board: staggeredBoard(),
      remainingPieces: [instance(1)],
      abilities: { reroll: 1, singleCell: 0 },
    });
    const candidates = success(
      getAbilityCandidates(input, [piece("MIEUM")], false),
    ).actions;
    expect(candidates).toContainEqual({
      type: "reroll",
      instanceId: "set1-1",
      pieceIndex: 1,
      reason: "zero-catalog-mobility",
    });
    expect(
      success(getAbilityCandidates(input, [], false)).actions.some(
        (a) => a.type === "reroll",
      ),
    ).toBe(false);
  });
  it.each([1, 2, 4, 20, 512])(
    "accounts probe and ability visits together within %i nodes",
    (maxNodes) => {
      const input = freezeDeep(singleRescue(2));
      const result = success(
        solveTurn(input, catalog, { ...DEFAULT_SOLVER_CONFIG, maxNodes }),
      ).result;
      expect(result.search.visitedNodes).toBeLessThanOrEqual(maxNodes);
      expect(
        result.search.abilitySearch?.ordinaryProbe.visitedNodes,
      ).toBeLessThanOrEqual(result.search.visitedNodes);
      expect(result.search.memoEntries).toBeLessThanOrEqual(
        result.search.visitedNodes,
      );
      replay(input, result);
      if (!result.search.searchComplete)
        expect(result.search.optimalWithinScope).toBe(false);
      expect(result.search.abilitySearch?.rerollFutureEvaluated).toBe(false);
    },
  );
  it("does not label a truncated ordinary probe as proof that completion is impossible", () => {
    const result = success(
      solveTurn(game(), catalog, { ...DEFAULT_SOLVER_CONFIG, maxNodes: 8 }),
    ).result;
    expect(result.search.abilitySearch?.ordinaryProbe).toMatchObject({
      searchComplete: false,
      noCompletePathProven: false,
    });
  });
  it("detaches the best plan and alternatives and remains deterministic", () => {
    const input = freezeDeep(singleRescue());
    const result = success(solveTurn(input, catalog)).result;
    expect(success(solveTurn(input, catalog)).result).toEqual(result);
    const copy = structuredClone(result.alternatives);
    result.finalState.board[0][0] = !result.finalState.board[0][0];
    result.usedAbilities.singleCell = 99;
    result.actions.push({ type: "single-cell", row: 0, col: 0 });
    expect(result.alternatives).toEqual(copy);
  });
});
