import { describe, expect, it } from "vitest";
import {
  compareEvaluations,
  evaluateState,
} from "../../src/domain/solver/evaluator";
import type { SolverEvaluation } from "../../src/domain/solver/types";
import { freezeDeep, game, piece } from "../game/fixtures";
import { success } from "./turn-fixtures";

describe("lexicographic evaluation", () => {
  const base: SolverEvaluation = {
    survivable: true,
    allCurrentPiecesPlaced: true,
    phase: "await-next-pieces",
    clearedRows: 0,
    acquiredItems: 0,
    mobility: { playablePieceTypes: 1, totalPieceTypes: 4, totalPlacements: 1 },
    remainingAbilities: { reroll: 0, singleCell: 0 },
    isolatedEmptyCells: 1,
    unfillableGaps: 1,
    emptyCells: 1,
  };
  it("survival outranks completion; completion outranks high mobility/rewards", () => {
    expect(
      compareEvaluations(
        { ...base, survivable: true, allCurrentPiecesPlaced: false },
        { ...base, survivable: false, clearedRows: 100 },
      ),
    ).toBeGreaterThan(0);
    expect(
      compareEvaluations(base, {
        ...base,
        allCurrentPiecesPlaced: false,
        mobility: {
          playablePieceTypes: 4,
          totalPieceTypes: 4,
          totalPlacements: 1000,
        },
        clearedRows: 100,
      }),
    ).toBeGreaterThan(0);
  });
  it("more playable types outrank placements and mobility outranks row clears", () => {
    expect(
      compareEvaluations(
        {
          ...base,
          mobility: {
            playablePieceTypes: 2,
            totalPieceTypes: 4,
            totalPlacements: 2,
          },
        },
        {
          ...base,
          mobility: {
            playablePieceTypes: 1,
            totalPieceTypes: 4,
            totalPlacements: 1000,
          },
        },
      ),
    ).toBeGreaterThan(0);
    expect(
      compareEvaluations(
        { ...base, mobility: { ...base.mobility, totalPlacements: 2 } },
        { ...base, clearedRows: 100 },
      ),
    ).toBeGreaterThan(0);
  });
  it("orders rows, acquired items, ability preservation and separate shape metrics", () => {
    const improvements = [
      { ...base, clearedRows: 1 },
      { ...base, acquiredItems: 1 },
      { ...base, remainingAbilities: { reroll: 1, singleCell: 0 } },
      { ...base, isolatedEmptyCells: 0 },
      { ...base, unfillableGaps: 0 },
      { ...base, emptyCells: 2 },
    ];
    for (const evaluation of improvements) {
      expect(compareEvaluations(evaluation, base)).toBeGreaterThan(0);
      expect(compareEvaluations(base, evaluation)).toBeLessThan(0);
    }
    for (let index = 0; index < improvements.length - 1; index++)
      expect(
        compareEvaluations(improvements[index], improvements[index + 1]),
      ).toBeGreaterThan(0);
    expect(compareEvaluations(base, structuredClone(base))).toBe(0);
  });
  it("preserves inputs and interprets survivable as current non-gameover only", () => {
    const input = freezeDeep(game({ remainingPieces: [] }));
    const result = success(
      evaluateState(input, [piece()], { clearedRows: 2, acquiredItems: 3 }),
    ).evaluation;
    expect(result).toMatchObject({
      survivable: true,
      allCurrentPiecesPlaced: true,
      phase: "await-next-pieces",
      clearedRows: 2,
      acquiredItems: 3,
      mobility: {
        playablePieceTypes: 1,
        totalPieceTypes: 1,
        totalPlacements: 160,
      },
      emptyCells: 160,
    });
    result.remainingAbilities.reroll = 7;
    expect(input.abilities.reroll).toBe(0);
  });
  it.each([
    { clearedRows: -1, acquiredItems: 0 },
    { clearedRows: 0.5, acquiredItems: 0 },
    { clearedRows: 0, acquiredItems: NaN },
  ])("rejects invalid reward counts %#", (rewards) => {
    expect(evaluateState(game(), [], rewards)).toMatchObject({
      ok: false,
      error: { code: "INVALID_REWARDS" },
    });
  });
});
