import { describe, expect, it } from "vitest";
import {
  applyAction,
  getAvailableActions,
} from "../../src/domain/game/actions";
import { getInitialCatalog } from "../../src/domain/pieces/catalog";
import { getValidPlacements } from "../../src/domain/pieces/placement";
import { evaluateState } from "../../src/domain/solver/evaluator";
import { solveOrdinaryTurn as solveTurn } from "../../src/domain/solver/solver";
import { createSearchKey } from "../../src/domain/solver/search";
import { DEFAULT_SOLVER_CONFIG } from "../../src/domain/solver/types";
import { freezeDeep, game, instance, piece } from "../game/fixtures";
import { lineTurn, rescueTurn, staggeredBoard, success } from "./turn-fixtures";

const exact = {
  ...DEFAULT_SOLVER_CONFIG,
  maxNodes: 10000,
  useMemoization: false,
};

describe("ordinary turn search and replay", () => {
  it("preserves a required rotated L variant and replays its clear", () => {
    const board = staggeredBoard();
    board[6].fill(true);
    board[7].fill(true);
    board[6][4] = board[6][5] = board[7][5] = false;
    const input = game({ board, remainingPieces: [instance(1, "L_3")] });
    const result = success(solveTurn(input, [piece("L_3")], exact)).result;
    expect(result.actions).toEqual([
      expect.objectContaining({
        pieceIndex: 1,
        row: 6,
        col: 4,
        rotation: 180,
        flipped: false,
        variant: [
          [true, true],
          [false, true],
        ],
      }),
    ]);
    const replay = success(applyAction(input, result.actions[0]));
    expect(replay.info.clearedRows).toEqual([6, 7]);
    expect(result.finalState).toEqual(replay.state);
  });
  it("matches a handwritten exhaustive LINE_3 oracle and consumes duplicate slots separately", () => {
    const input = freezeDeep(lineTurn());
    // Reference recursively occupies/clears the board; no solver, transforms, placement or game API.
    let completePaths = 0;
    function enumerate(board: boolean[][], remaining: number[]) {
      if (!remaining.length) {
        completePaths++;
        expect(board[0].every((cell) => !cell)).toBe(true);
        return;
      }
      for (const slot of remaining)
        for (const footprint of [
          [
            [0, 0],
            [0, 1],
            [0, 2],
          ],
          [
            [0, 0],
            [1, 0],
            [2, 0],
          ],
        ]) {
          for (let row = 0; row < 16; row++)
            for (let col = 0; col < 10; col++) {
              const cells = footprint.map(([dr, dc]) => [row + dr, col + dc]);
              if (cells.some(([r, c]) => r >= 16 || c >= 10 || board[r][c]))
                continue;
              const next = board.map((cells) => cells.slice());
              for (const [r, c] of cells) next[r][c] = true;
              next.forEach((cells) => {
                if (cells.every(Boolean)) cells.fill(false);
              });
              enumerate(
                next,
                remaining.filter((index) => index !== slot),
              );
            }
        }
    }
    enumerate(input.board, [0, 1, 2]);
    expect(completePaths).toBe(36);
    const reference = success(solveTurn(input, [piece()], exact)).result;
    const memoized = success(
      solveTurn(input, [piece()], { ...exact, useMemoization: true }),
    ).result;
    expect(reference.search.discoveredCompletePaths).toBe(completePaths);
    expect(reference.search.searchComplete).toBe(true);
    expect(reference.evaluation).toMatchObject({
      allCurrentPiecesPlaced: true,
      clearedRows: 1,
      mobility: { totalPlacements: 25 },
    });
    expect(reference.actions.map((action) => action.pieceIndex).sort()).toEqual(
      [0, 1, 2],
    );
    expect(memoized.evaluation).toEqual(reference.evaluation);
    expect(memoized.actions).toEqual(reference.actions);
    expect(memoized.search.memoPrunedNodes).toBeGreaterThan(0);
    expect(memoized.search.visitedNodes).toBeLessThan(
      reference.search.visitedNodes,
    );
    expect(memoized.search.memoEntries).toBeLessThanOrEqual(
      memoized.search.visitedNodes,
    );
    expect(memoized.search.alternativesMayOmitEquivalentPaths).toBe(true);
    expect(memoized.search.discoveredCompletePaths).toBeLessThan(
      reference.search.discoveredCompletePaths,
    );
  });
  it("places a later slot first to clear space for an initially blocked MIEUM", () => {
    const input = freezeDeep(rescueTurn());
    expect(
      success(getValidPlacements(input.board, input.remainingPieces[0].piece))
        .placements,
    ).toHaveLength(0);
    const result = success(solveTurn(input, getInitialCatalog(), exact)).result;
    expect(result.actions.map((action) => action.pieceIndex)).toEqual([2, 0]);
    expect(result.evaluation).toMatchObject({
      allCurrentPiecesPlaced: true,
      survivable: true,
      clearedRows: 3,
      acquiredItems: 3,
      remainingAbilities: { reroll: 2, singleCell: 1 },
      phase: "await-next-pieces",
    });
    let current = input;
    for (const action of result.actions)
      current = success(applyAction(current, action)).state;
    expect(current).toEqual(result.finalState);
    expect(current.hiddenItems).toEqual([]);
    expect(input.hiddenItems).toHaveLength(3);
  });
  it("replays best and every alternative and re-evaluates their cumulative rewards", () => {
    const input = freezeDeep(lineTurn());
    const catalog = freezeDeep(getInitialCatalog());
    const result = success(solveTurn(input, catalog, exact)).result;
    expect(result.alternatives).toHaveLength(2);
    for (const candidate of [result, ...result.alternatives]) {
      let state = input;
      let clearedRows = 0;
      let acquiredItems = 0;
      for (const action of candidate.actions) {
        const transition = success(applyAction(state, freezeDeep(action)));
        state = transition.state;
        clearedRows += transition.info.clearedRows.length;
        acquiredItems += transition.info.acquiredItems.length;
      }
      expect(state).toEqual(candidate.finalState);
      expect(
        success(evaluateState(state, catalog, { clearedRows, acquiredItems }))
          .evaluation,
      ).toEqual(candidate.evaluation);
    }
    expect(result.search).toMatchObject({
      scope: "ordinary-pieces",
      searchComplete: true,
      optimalWithinScope: true,
      specialAbilitiesSearched: false,
      stopReason: "exhausted",
    });
  });
  it("detaches candidates from input and each other and ignores remaining-array order for deterministic traversal", () => {
    const input = freezeDeep(lineTurn());
    const result = success(solveTurn(input, [piece()], exact)).result;
    const again = success(
      solveTurn(
        { ...input, remainingPieces: [...input.remainingPieces].reverse() },
        [piece()],
        exact,
      ),
    ).result;
    expect(result.actions).toEqual(again.actions);
    expect(result.evaluation).toEqual(again.evaluation);
    const alternativeBefore = structuredClone(result.alternatives[0]);
    result.actions[0].variant[0][0] = false;
    result.finalState.board[0][0] = true;
    result.evaluation.remainingAbilities.reroll = 7;
    expect(result.alternatives[0]).toEqual(alternativeBefore);
    expect(input.remainingPieces[0].piece.shape).toEqual([[true, true, true]]);
    expect(again.actions[0].variant[0][0]).toBe(true);
  });
});

describe("budgets, actual phase and ingress recovery", () => {
  it("reports complete when the last required visit exactly reaches the budget", () => {
    const reference = success(solveTurn(lineTurn(), [piece()], exact)).result;
    const result = success(
      solveTurn(lineTurn(), [piece()], {
        ...exact,
        maxNodes: reference.search.visitedNodes,
      }),
    ).result;
    expect(result.search.searchComplete).toBe(true);
    expect(result.actions).toEqual(reference.actions);
    const cut = success(
      solveTurn(lineTurn(), [piece()], {
        ...exact,
        maxNodes: reference.search.visitedNodes - 1,
      }),
    ).result;
    expect(cut.search.searchComplete).toBe(false);
  });
  it.each([1, 2, 5, 20])(
    "never visits more than %i nodes or claims a truncated search optimal",
    (maxNodes) => {
      const input = freezeDeep(lineTurn());
      const result = success(
        solveTurn(input, [piece()], {
          ...exact,
          maxNodes,
          useMemoization: true,
        }),
      ).result;
      expect(result.search.visitedNodes).toBeLessThanOrEqual(maxNodes);
      expect(result.search).toMatchObject({
        searchComplete: false,
        optimalWithinScope: false,
        stopReason: "node-budget",
      });
      let state = input;
      for (const action of result.actions)
        state = success(applyAction(state, action)).state;
      expect(state).toEqual(result.finalState);
      expect(result.search.memoEntries).toBeLessThanOrEqual(maxNodes);
      if (maxNodes === 1) expect(result.actions).toEqual([]);
    },
  );
  it.each([
    {
      abilities: { reroll: 0, singleCell: 0 },
      survivable: false,
      phase: "gameover",
    },
    {
      abilities: { reroll: 1, singleCell: 0 },
      survivable: true,
      phase: "playing",
    },
    {
      abilities: { reroll: 0, singleCell: 1 },
      survivable: true,
      phase: "playing",
    },
  ])(
    "keeps special legal-action semantics when ordinary search finds no placement %#",
    ({ abilities, survivable, phase }) => {
      const input = game({
        board: staggeredBoard(),
        remainingPieces: [instance(2, "MIEUM")],
        abilities,
      });
      const result = success(solveTurn(input, [piece()], exact)).result;
      expect(result.actions).toEqual([]);
      expect(result.evaluation).toMatchObject({
        survivable,
        phase,
        allCurrentPiecesPlaced: false,
      });
      expect(result.search).toMatchObject({
        searchComplete: true,
        specialAbilitiesSearched: false,
      });
      expect(success(getAvailableActions(result.finalState)).phase).toBe(phase);
    },
  );
  it("returns input-wait phases without generating new sets or random reroll pieces", () => {
    const next = success(
      solveTurn(game({ remainingPieces: [] }), [], exact),
    ).result;
    expect(next.actions).toEqual([]);
    expect(next.search.stopReason).toBe("await-next-pieces");
    const pending = game({
      abilities: { reroll: 1, singleCell: 0 },
      pendingReroll: { instanceId: "set1-0", pieceIndex: 0 },
    });
    const result = success(solveTurn(freezeDeep(pending), [], exact)).result;
    expect(result.actions).toEqual([]);
    expect(result.finalState).toEqual(pending);
    expect(result.search.stopReason).toBe("await-reroll-result");
  });
  it.each([
    null,
    {},
    { ...exact, maxNodes: 0 },
    { ...exact, maxNodes: 0.5 },
    { ...exact, maxNodes: Infinity },
    { ...exact, maxNodes: NaN },
    { ...exact, maxAlternatives: 4 },
    { ...exact, maxAlternatives: 0 },
    { ...exact, useMemoization: 1 },
  ])("rejects malformed config %#", (config) => {
    expect(solveTurn(lineTurn(), [], config)).toMatchObject({
      ok: false,
      error: { code: "INVALID_SOLVER_CONFIG" },
    });
  });
  it("propagates bad state/catalog and allows corrected retry without mutation", () => {
    const input = freezeDeep(lineTurn());
    expect(solveTurn({}, [], exact)).toMatchObject({ ok: false });
    expect(solveTurn(input, [piece(), piece()], exact)).toMatchObject({
      ok: false,
      error: { code: "INVALID_CATALOG" },
    });
    expect(
      success(solveTurn(input, [piece()], exact)).result.evaluation
        .allCurrentPiecesPlaced,
    ).toBe(true);
    expect(input.remainingPieces).toHaveLength(3);
  });
  it("respects maxAlternatives and the public default config", () => {
    const result = success(
      solveTurn(lineTurn(), [piece()], { ...exact, maxAlternatives: 1 }),
    ).result;
    expect(result.alternatives).toEqual([]);
    const bounded = success(solveTurn(lineTurn(), [piece()])).result;
    expect(bounded.search.maxNodes).toBe(DEFAULT_SOLVER_CONFIG.maxNodes);
  });
});

describe("memo identity includes every transition/evaluation-relevant field", () => {
  it("distinguishes rewards, abilities, items, shape, kind, slot, instance and pending input", () => {
    const base = game();
    const reward = { clearedRows: 0, acquiredItems: 0 };
    const key = createSearchKey(freezeDeep(base), reward);
    const different = [
      {
        ...base,
        board: base.board.map((row, r) =>
          row.map((cell, c) => (r === 0 && c === 0 ? true : cell)),
        ),
      },
      { ...base, abilities: { reroll: 1, singleCell: 0 } },
      { ...base, abilities: { reroll: 0, singleCell: 1 } },
      { ...base, hiddenItems: [{ row: 0, col: 0, type: "reroll" as const }] },
      { ...base, remainingPieces: [instance(0, "LINE_3")] },
      { ...base, remainingPieces: [instance(2)] },
      { ...base, remainingPieces: [instance(0, "DOT", "other")] },
      {
        ...base,
        remainingPieces: base.remainingPieces.map((entry, index) =>
          index === 0
            ? { ...entry, piece: { ...entry.piece, shape: [[true, true]] } }
            : entry,
        ),
      },
      {
        ...base,
        pendingReroll: { instanceId: "set1-0", pieceIndex: 0 as const },
      },
    ];
    for (const state of different)
      expect(createSearchKey(state, reward)).not.toBe(key);
    expect(
      createSearchKey(base, { clearedRows: 1, acquiredItems: 0 }),
    ).not.toBe(key);
    expect(
      createSearchKey(base, { clearedRows: 0, acquiredItems: 1 }),
    ).not.toBe(key);
    expect(
      createSearchKey(
        { ...base, remainingPieces: [...base.remainingPieces].reverse() },
        reward,
      ),
    ).toBe(key);
    const items = [
      { row: 1, col: 2, type: "reroll" as const },
      { row: 0, col: 3, type: "single-cell" as const },
    ];
    expect(createSearchKey({ ...base, hiddenItems: items }, reward)).toBe(
      createSearchKey({ ...base, hiddenItems: [...items].reverse() }, reward),
    );
  });
});
