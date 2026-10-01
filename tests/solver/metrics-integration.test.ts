import { describe, expect, it } from "vitest";
import { createEmptyBoard } from "../../src/domain/board/board";
import {
  calculateRowProgress,
  countEmptyCells,
  countIsolatedEmptyCells,
  countOccupiedCells,
} from "../../src/domain/board/board-metrics";
import { applyAction } from "../../src/domain/game/actions";
import {
  calculateMobility,
  countUnfillableGaps,
} from "../../src/domain/solver/mobility";
import {
  freezeDeep,
  game,
  instance,
  piece,
  placement,
  success,
} from "../game/fixtures";

describe("shared transition followed by metric recalculation", () => {
  it("a row clear makes previously uncovered cells fillable; gaps are not permanent impossibility", () => {
    const input = game({
      remainingPieces: [instance(0, "LINE_3"), instance(2, "LINE_3")],
    });
    input.board.forEach((row) => {
      row.fill(true);
      row[9] = false;
    });
    input.board[5].fill(true);
    input.board[5].fill(false, 0, 3);
    input.board[6][5] = false;
    input.board[7][5] = false;
    freezeDeep(input);
    const catalog = freezeDeep([piece("LINE_3")]);
    expect(calculateMobility(input.board, catalog)).toEqual({
      ok: true,
      mobility: {
        playablePieceTypes: 1,
        totalPieceTypes: 1,
        totalPlacements: 12,
      },
    });
    expect(countEmptyCells(input.board)).toEqual({ ok: true, count: 20 });
    expect(countIsolatedEmptyCells(input.board)).toEqual({
      ok: true,
      count: 0,
    });
    expect(countUnfillableGaps(input.board, catalog)).toEqual({
      ok: true,
      count: 2,
    });
    const output = success(
      applyAction(input, placement(input.remainingPieces[0], 5, 0)),
    );
    expect(output.info.clearedRows).toEqual([5]);
    expect(
      output.state.remainingPieces.map((entry) => entry.pieceIndex),
    ).toEqual([2]);
    expect(calculateMobility(output.state.board, catalog)).toEqual({
      ok: true,
      mobility: {
        playablePieceTypes: 1,
        totalPieceTypes: 1,
        totalPlacements: 23,
      },
    });
    expect(countUnfillableGaps(output.state.board, catalog)).toEqual({
      ok: true,
      count: 0,
    });
    expect(countEmptyCells(output.state.board)).toEqual({
      ok: true,
      count: 27,
    });
    expect(countOccupiedCells(output.state.board)).toEqual({
      ok: true,
      count: 133,
    });
    const progress = calculateRowProgress(output.state.board);
    if (!progress.ok) throw new Error(progress.error.message);
    expect(progress.rows[5].filledRatio).toBe(0);
    expect(input.board[5].filter(Boolean)).toHaveLength(7);
    expect(countUnfillableGaps(input.board, catalog)).toEqual({
      ok: true,
      count: 2,
    });
  });
  it("single-cell without a clear cannot improve current placement counts", () => {
    const input = freezeDeep(
      game({
        board: createEmptyBoard(),
        abilities: { reroll: 0, singleCell: 1 },
      }),
    );
    const catalog = freezeDeep([piece()]);
    const output = success(
      applyAction(input, { type: "single-cell", row: 8, col: 5 }),
    );
    expect(output.info.clearedRows).toEqual([]);
    expect(calculateMobility(input.board, catalog)).toEqual({
      ok: true,
      mobility: {
        playablePieceTypes: 1,
        totalPieceTypes: 1,
        totalPlacements: 160,
      },
    });
    expect(calculateMobility(output.state.board, catalog)).toEqual({
      ok: true,
      mobility: {
        playablePieceTypes: 1,
        totalPieceTypes: 1,
        totalPlacements: 159,
      },
    });
    expect(countEmptyCells(input.board)).toEqual({ ok: true, count: 160 });
    expect(countEmptyCells(output.state.board)).toEqual({
      ok: true,
      count: 159,
    });
  });
});
