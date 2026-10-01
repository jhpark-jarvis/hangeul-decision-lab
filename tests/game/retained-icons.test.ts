import { describe, expect, it } from "vitest";
import { applyAction } from "../../src/domain/game/actions";
import { resolveReroll } from "../../src/domain/game/game-state";
import { getInitialCatalog } from "../../src/domain/pieces/catalog";
import { solveOrdinaryTurn } from "../../src/domain/solver/solver";
import { createSearchKey } from "../../src/domain/solver/search";
import {
  applyStep,
  createSession,
  installAnalysis,
} from "../../src/features/puzzle/session";
import {
  freezeDeep,
  game,
  instance,
  piece,
  placement,
  success,
} from "./fixtures";

function capacityGame() {
  const input = game({
    remainingPieces: [instance(0), instance(1)],
    abilities: { reroll: 7, singleCell: 0 },
    hiddenItems: [{ row: 0, col: 0, type: "single-cell" }],
  });
  input.board[0].fill(true);
  input.board[0][9] = false;
  return input;
}
describe("official unacquired icon retention", () => {
  it("clears occupancy while retaining an unacquired icon on the same empty coordinate", () => {
    const input = freezeDeep(capacityGame());
    const output = success(
      applyAction(input, placement(input.remainingPieces[0], 0, 9)),
    );
    expect(output.state.board[0].some(Boolean)).toBe(false);
    expect(output.state.hiddenItems).toEqual(input.hiddenItems);
    expect(output.state.abilities).toEqual(input.abilities);
    expect(output.info.acquiredItems).toEqual([]);
    expect(output.info.retainedItems).toEqual(input.hiddenItems);
    output.info.retainedItems[0].col = 8;
    expect(output.state.hiddenItems[0].col).toBe(0);
    expect(input.board[0][0]).toBe(true);
  });
  it("spends single-cell before collecting a retained icon on a later row clear", () => {
    const input = capacityGame();
    input.abilities = { reroll: 6, singleCell: 1 };
    const held = success(
      applyAction(input, placement(input.remainingPieces[0], 0, 9)),
    ).state;
    held.board[0].fill(true);
    held.board[0][9] = false; // User re-enters the later actual board.
    const output = success(
      applyAction(freezeDeep(held), { type: "single-cell", row: 0, col: 9 }),
    );
    expect(output.info.spentAbility).toBe("single-cell");
    expect(output.info.acquiredItems).toEqual(input.hiddenItems);
    expect(output.info.retainedItems).toEqual([]);
    expect(output.state.hiddenItems).toEqual([]);
    expect(output.state.abilities).toEqual({ reroll: 6, singleCell: 1 });
  });
  it("retains the icon through reroll and acquires after the real replacement clears its row", () => {
    const input = capacityGame();
    const held = success(
      applyAction(input, placement(input.remainingPieces[0], 0, 9)),
    ).state;
    const target = held.remainingPieces[0];
    const waiting = success(
      applyAction(held, {
        type: "reroll",
        pieceIndex: target.pieceIndex,
        instanceId: target.instanceId,
      }),
    ).state;
    expect(waiting.abilities.reroll).toBe(6);
    expect(waiting.hiddenItems).toEqual(input.hiddenItems);
    const resolved = success(resolveReroll(waiting, piece("LINE_3"))).state;
    resolved.board[0].fill(true, 0, 7);
    const output = success(
      applyAction(resolved, placement(resolved.remainingPieces[0], 0, 7)),
    );
    expect(output.state.abilities).toEqual({ reroll: 6, singleCell: 1 });
    expect(output.state.hiddenItems).toEqual([]);
    expect(output.info.acquiredItems).toEqual(input.hiddenItems);
  });
  it("solver replay and session Apply retain the same icon without counting it as acquired", () => {
    const input = capacityGame();
    input.remainingPieces = [instance(0)];
    const solution = solveOrdinaryTurn(freezeDeep(input), getInitialCatalog());
    if (!solution.ok) throw new Error(solution.error.message);
    expect(solution.result.evaluation.clearedRows).toBe(1);
    expect(solution.result.evaluation.acquiredItems).toBe(0);
    expect(solution.result.finalState.hiddenItems).toEqual(input.hiddenItems);
    const session = createSession(input);
    const installed = installAnalysis(session, session.version, solution, 0);
    expect(installed.error).toBeNull();
    expect(installed.analysis!.plans[0].info[0].retainedItems).toEqual(
      input.hiddenItems,
    );
    const applied = applyStep(installed, installed.version, 0);
    expect(applied.game.hiddenItems).toEqual(input.hiddenItems);
    expect(applied.notice).toContain("상한으로 남김 1개");
    const rewards = { clearedRows: 1, acquiredItems: 0 };
    expect(createSearchKey(applied.game, rewards)).not.toEqual(
      createSearchKey({ ...applied.game, hiddenItems: [] }, rewards),
    );
    const forged = structuredClone(solution);
    forged.result.finalState.hiddenItems = [];
    expect(
      installAnalysis(session, session.version, forged, 0).error,
    ).not.toBeNull();
  });
});
