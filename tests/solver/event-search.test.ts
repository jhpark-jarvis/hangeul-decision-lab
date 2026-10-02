import { describe, expect, it } from "vitest";
import reference from "../fixtures/pieces/catalog-v2.json";
import { applyAction } from "../../src/domain/game/actions";
import type { GameState } from "../../src/domain/game/types";
import { getInitialCatalog } from "../../src/domain/pieces/catalog";
import {
  compareEvaluations,
  evaluateState,
} from "../../src/domain/solver/evaluator";
import { solveOrdinaryTurn } from "../../src/domain/solver/solver";
import {
  DEFAULT_SOLVER_CONFIG,
  type PathRewards,
  type SolverEvaluation,
} from "../../src/domain/solver/types";
import { freezeDeep, game, instance } from "../game/fixtures";
import { staggeredBoard, success } from "./turn-fixtures";

// Independent coordinate transforms and occupancy/clear/retention transitions.
// Evaluation is shared deliberately: this checks search coverage and replay,
// not the evaluator itself (covered by metric/evaluator/catalog oracle tests).
function variants(rows: string[]) {
  const cells = rows.flatMap((line, r) =>
    [...line].flatMap((cell, c) => (cell === "1" ? [[r, c]] : [])),
  );
  const unique = new Map<string, number[][]>();
  for (const sign of [1, -1]) {
    let rotated = cells.map(([r, c]) => [r, c * sign]);
    for (let turn = 0; turn < 4; turn++) {
      const minR = Math.min(...rotated.map(([r]) => r)),
        minC = Math.min(...rotated.map(([, c]) => c));
      const normalized = rotated
        .map(([r, c]) => [r - minR, c - minC])
        .sort(([r, c], [s, d]) => r - s || c - d);
      unique.set(JSON.stringify(normalized), normalized);
      rotated = rotated.map(([r, c]) => [c, -r]);
    }
  }
  return [...unique.values()];
}
const footprints = new Map(
  reference.pieces.map((piece) => [piece.id, variants(piece.rows)]),
);
const catalog = getInitialCatalog();
function oracle(input: GameState) {
  let best: SolverEvaluation | null = null;
  let bestStates: GameState[] = [];
  let completePaths = 0,
    leaves = 0;
  function visit(state: GameState, rewards: PathRewards) {
    let children = 0;
    for (const target of state.remainingPieces) {
      for (const footprint of footprints.get(target.piece.id)!) {
        for (let row = 0; row < 16; row++)
          for (let col = 0; col < 10; col++) {
            const cells = footprint.map(([r, c]) => [row + r, col + c]);
            if (cells.some(([r, c]) => r >= 16 || c >= 10 || state.board[r][c]))
              continue;
            children++;
            const next = structuredClone(state);
            for (const [r, c] of cells) next.board[r][c] = true;
            const cleared = next.board.flatMap((line, r) =>
              line.every(Boolean) ? [r] : [],
            );
            for (const r of cleared) next.board[r].fill(false);
            next.remainingPieces = next.remainingPieces.filter(
              (piece) => piece.instanceId !== target.instanceId,
            );
            let acquired = 0;
            next.hiddenItems.sort((a, b) => a.row - b.row || a.col - b.col);
            next.hiddenItems = next.hiddenItems.filter((item) => {
              if (
                !cleared.includes(item.row) ||
                next.abilities.reroll + next.abilities.singleCell === 7
              )
                return true;
              next.abilities[
                item.type === "reroll" ? "reroll" : "singleCell"
              ]++;
              acquired++;
              return false;
            });
            visit(next, {
              clearedRows: rewards.clearedRows + cleared.length,
              acquiredItems: rewards.acquiredItems + acquired,
            });
          }
      }
    }
    if (!children) {
      leaves++;
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
  return { best, bestStates, completePaths, leaves };
}
function compare(input: GameState) {
  const expected = oracle(input);
  expect(expected.leaves).toBeGreaterThan(0);
  const config = { ...DEFAULT_SOLVER_CONFIG, maxNodes: 10000 };
  const exact = success(
    solveOrdinaryTurn(freezeDeep(input), catalog, {
      ...config,
      useMemoization: false,
    }),
  ).result;
  const memo = success(
    solveOrdinaryTurn(input, catalog, { ...config, useMemoization: true }),
  ).result;
  expect(exact.search.searchComplete).toBe(true);
  expect(memo.search.searchComplete).toBe(true);
  expect(exact.search.discoveredCompletePaths).toBe(expected.completePaths);
  expect(exact.evaluation).toEqual(expected.best);
  expect(memo.evaluation).toEqual(expected.best);
  expect(expected.bestStates).toContainEqual(exact.finalState);
  expect(expected.bestStates).toContainEqual(memo.finalState);
  for (const candidate of [
    exact,
    ...exact.alternatives,
    memo,
    ...memo.alternatives,
  ]) {
    let state = input;
    const rewards = { clearedRows: 0, acquiredItems: 0 };
    for (const action of candidate.actions) {
      const replay = success(applyAction(state, action));
      rewards.clearedRows += replay.info.clearedRows.length;
      rewards.acquiredItems += replay.info.acquiredItems.length;
      state = replay.state;
    }
    expect(state).toEqual(candidate.finalState);
    expect(success(evaluateState(state, catalog, rewards)).evaluation).toEqual(
      candidate.evaluation,
    );
  }
  return exact;
}

describe("19-type catalog search against separate coordinate enumeration", () => {
  it.each(reference.pieces)(
    "$id: exhausts constrained legal placements and replays clears",
    (ref) => {
      const board = staggeredBoard();
      ref.rows.forEach((line, r) => {
        board[5 + r].fill(true);
        [...line].forEach((cell, c) => {
          if (cell === "1") board[5 + r][2 + c] = false;
        });
      });
      const result = compare(
        game({ board, remainingPieces: [instance(2, ref.id)] }),
      );
      expect(result.evaluation.allCurrentPiecesPlaced).toBe(true);
      expect(result.evaluation.clearedRows).toBe(ref.rows.length);
    },
  );
  it.each([6, 7])(
    "mixed slots and3 icons retain overflow at capacity %i",
    (capacity) => {
      const board = staggeredBoard();
      board[3].fill(true);
      board[3][2] = false;
      for (let row = 7; row < 10; row++) board[row].fill(true);
      for (const [r, c] of [
        [0, 0],
        [1, 0],
        [1, 1],
        [2, 0],
      ])
        board[7 + r][2 + c] = false;
      const input = game({
        board,
        remainingPieces: [instance(0), instance(2, "BRANCH_4")],
        abilities: { reroll: capacity, singleCell: 0 },
        hiddenItems: [
          { row: 3, col: 0, type: "single-cell" },
          { row: 7, col: 0, type: "reroll" },
          { row: 9, col: 0, type: "single-cell" },
        ],
      });
      const result = compare(input);
      expect(result.evaluation.allCurrentPiecesPlaced).toBe(true);
      expect(result.evaluation.acquiredItems).toBe(7 - capacity);
      expect(result.finalState.hiddenItems).toHaveLength(3 - (7 - capacity));
      expect(input.hiddenItems).toHaveLength(3);
    },
  );
});
