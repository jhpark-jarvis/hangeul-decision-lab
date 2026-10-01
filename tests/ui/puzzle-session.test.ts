import { describe, expect, it } from "vitest";
import { createEmptyBoard } from "../../src/domain/board/board";
import { getAvailableActions } from "../../src/domain/game/actions";
import { getInitialCatalog } from "../../src/domain/pieces/catalog";
import { solveTurn } from "../../src/domain/solver/solver";
import {
  applyStep,
  createSession,
  editCell,
  editPiece,
  inputReroll,
  installAnalysis,
  loadNextPieces,
  nextPreview,
  replaceGame,
  selectPlan,
  type PuzzleSession,
} from "../../src/features/puzzle/session";
import { freezeDeep, game, instance } from "../game/fixtures";
import { singleRescue, reacquireRescue } from "../solver/ability-fixtures";

function analyze(session: PuzzleSession) {
  return installAnalysis(
    session,
    session.version,
    solveTurn(session.game, getInitialCatalog()),
    12.5,
  );
}
function finish(session: PuzzleSession) {
  let current = session;
  while (current.analysis && nextPreview(current).action)
    current = applyStep(current, current.version, current.analysis.step);
  return current;
}

describe("input and session boundary", () => {
  it("starts at next-set input and preserves board/abilities when loading exactly three", () => {
    const initial = editCell(createSession(), 0, 0, "filled");
    const loaded = loadNextPieces(initial, ["DOT", "DOT", "MIEUM"]);
    expect(loaded.error).toBeNull();
    expect(loaded.game.board[0][0]).toBe(true);
    expect(
      loaded.game.remainingPieces.map((entry) => entry.pieceIndex),
    ).toEqual([0, 1, 2]);
    expect(
      new Set(loaded.game.remainingPieces.map((entry) => entry.instanceId))
        .size,
    ).toBe(3);
    expect(loadNextPieces(loaded, ["DOT", "DOT", "DOT"]).game).toEqual(
      loaded.game,
    );
    expect(loadNextPieces(initial, ["DOT", "DOT"]).error).toBeTruthy();
  });
  it("edits filled/empty independently of hidden items, replaces one item and removes it", () => {
    let session = editCell(createSession(), 2, 4, "filled");
    session = editCell(session, 2, 4, "reroll");
    session = editCell(session, 2, 4, "single-cell");
    expect(session.game.board[2][4]).toBe(true);
    expect(session.game.hiddenItems).toEqual([
      { row: 2, col: 4, type: "single-cell" },
    ]);
    session = editCell(session, 2, 4, "empty");
    expect(session.game.hiddenItems).toHaveLength(1);
    session = editCell(session, 2, 4, "remove-item");
    expect(session.game.board[2][4]).toBe(false);
    expect(session.game.hiddenItems).toEqual([]);
    expect(editCell(session, 16, 0, "filled").game).toEqual(session.game);
  });
  it("preserves last valid state for malformed board and counts, then recovers", () => {
    const session = analyze(createSession(game()));
    for (const abilities of [
      { reroll: 7, singleCell: 1 },
      { reroll: -1, singleCell: 0 },
      { reroll: 1.5, singleCell: 0 },
    ]) {
      const failed = replaceGame(
        session,
        { ...session.game, abilities },
        "edit",
      );
      expect(failed.game).toEqual(session.game);
      expect(failed.analysis).toBeNull();
      expect(failed.error).toBeTruthy();
    }
    expect(
      replaceGame(session, { ...session.game, board: [] }, "bad").game,
    ).toEqual(session.game);
    expect(replaceGame(session, session.game, "retry").error).toBeNull();
  });
  it("keeps fixed slots with duplicate types and changes identity on edit", () => {
    const session = createSession(game());
    const edited = editPiece(session, 1, "MIEUM");
    expect(edited.game.remainingPieces[1].piece.id).toBe("MIEUM");
    expect(edited.game.remainingPieces[1].instanceId).not.toBe(
      session.game.remainingPieces[1].instanceId,
    );
    expect(
      editPiece(edited, 0, "").game.remainingPieces.map(
        (entry) => entry.pieceIndex,
      ),
    ).toEqual([1, 2]);
    expect(editPiece(edited, 0, "BAD").game).toEqual(edited.game);
  });
});

describe("analysis, replay and revision protection", () => {
  it("detaches every candidate and previews only true shape cells", () => {
    const session = createSession(
      game({ remainingPieces: [instance(2, "MIEUM")] }),
    );
    const result = solveTurn(session.game, getInitialCatalog());
    const installed = installAnalysis(session, 0, freezeDeep(result), 1);
    expect(installed.analysis?.plans.length).toBeGreaterThan(1);
    expect(nextPreview(installed).cells).toHaveLength(8);
    const alternative = selectPlan(installed, 1);
    expect(alternative.analysis?.selected).toBe(1);
    expect(alternative.game).toEqual(session.game);
    installed.analysis!.plans[0].snapshots[0].board[0][0] = true;
    expect(session.game.board[0][0]).toBe(false);
    expect(installed.analysis!.plans[1].snapshots[0].board[0][0]).toBe(false);
  });
  it.each(["board", "piece", "ability", "item"])(
    "invalidates analysis after %s edit and rejects old apply",
    (field) => {
      const analyzed = analyze(createSession(game()));
      const edited =
        field === "piece"
          ? editPiece(analyzed, 2, "L_3")
          : field === "ability"
            ? replaceGame(
                analyzed,
                { ...analyzed.game, abilities: { reroll: 1, singleCell: 0 } },
                "edit",
              )
            : editCell(analyzed, 0, 0, field === "item" ? "reroll" : "filled");
      expect(edited.analysis).toBeNull();
      const failed = applyStep(edited, analyzed.version, 0);
      expect(failed.game).toEqual(edited.game);
      expect(failed.error).toBeTruthy();
      expect(analyze(edited).analysis).not.toBeNull();
    },
  );
  it("rejects late solver completion, including after an invalid edit", () => {
    const session = createSession(game());
    const result = solveTurn(session.game, getInitialCatalog());
    for (const edited of [
      editCell(session, 0, 0, "filled"),
      replaceGame(
        session,
        { ...session.game, abilities: { reroll: 8, singleCell: 0 } },
        "bad",
      ),
    ]) {
      const late = installAnalysis(edited, session.version, result, 1);
      expect(late.analysis).toBeNull();
      expect(late.game).toEqual(edited.game);
    }
  });
  it("rejects duplicate click tokens while preserving the first applied state", () => {
    const initial = analyze(createSession(game()));
    const applied = applyStep(initial, initial.version, 0);
    expect(applied.game.remainingPieces).toHaveLength(2);
    const duplicate = applyStep(applied, initial.version, 0);
    expect(duplicate.game).toEqual(applied.game);
    expect(duplicate.analysis).toBeNull();
    expect(duplicate.error).toBeTruthy();
  });
  it("rejects matching version with stale step or changed snapshot", () => {
    const initial = analyze(createSession(game()));
    const applied = applyStep(initial, initial.version, 0);
    expect(applyStep(applied, applied.version, 0).game).toEqual(applied.game);
    const altered = structuredClone(initial);
    altered.game.board[15][9] = true;
    expect(applyStep(altered, altered.version, 0).game).toEqual(altered.game);
  });
  it("rejects invalid solver actions and final-state mismatch at installation", () => {
    const session = createSession(game());
    const result = solveTurn(session.game, getInitialCatalog());
    if (!result.ok) throw new Error(result.error.message);
    const invalid = structuredClone(result);
    invalid.result.actions[0].type = "reroll";
    expect(installAnalysis(session, 0, invalid, 1).analysis).toBeNull();
    const wrong = structuredClone(result);
    wrong.result.finalState.board[15][9] =
      !wrong.result.finalState.board[15][9];
    expect(installAnalysis(session, 0, wrong, 1).analysis).toBeNull();
    expect(installAnalysis(session, 0, result, Number.NaN).game).toEqual(
      session.game,
    );
    expect(
      installAnalysis(
        session,
        0,
        {
          ok: false,
          error: { code: "INVALID_SOLVER_CONFIG", message: "test failure" },
        },
        1,
      ).error,
    ).toBe("test failure");
    expect(analyze(session).analysis).not.toBeNull();
  });
  it("rechecks transitions at apply time and leaves game untouched on failure", () => {
    const session = analyze(createSession(game()));
    const bad = structuredClone(session);
    const action = bad.analysis!.plans[0].candidate.actions[0];
    if (action.type !== "place-piece") throw new Error("expected piece");
    action.col = 99;
    const failed = applyStep(bad, bad.version, 0);
    expect(failed.error).toMatch(/적용 실패/);
    expect(failed.game).toEqual(session.game);
  });
  it("freezes ingress, applies exactly the selected path and waits for next set", () => {
    const initial = freezeDeep(analyze(createSession(game())));
    const selected = selectPlan(initial, 1);
    const final = finish(selected);
    expect(final.game).toEqual(
      selected.analysis!.plans[1].candidate.finalState,
    );
    expect(final.game.remainingPieces).toHaveLength(0);
    expect(nextPreview(final).action).toBeUndefined();
    expect(
      selectPlan(applyStep(selected, selected.version, 0), 0).error,
    ).toBeTruthy();
    const loaded = loadNextPieces(final, ["L_3", "DOT", "DOT"]);
    expect(loaded.game.board).toEqual(final.game.board);
    expect(loaded.analysis).toBeNull();
  });
});

describe("ability and lifecycle UI contracts", () => {
  it("previews single-cell clear and applies subsequent general placement", () => {
    const session = analyze(createSession(singleRescue()));
    expect(nextPreview(session).action?.type).toBe("single-cell");
    expect(nextPreview(session).clearedRows).toEqual([4]);
    const applied = applyStep(session, session.version, 0);
    expect(applied.game.abilities.singleCell).toBe(0);
    expect(applied.game.board[4].every((cell) => !cell)).toBe(true);
    expect(finish(applied).game.remainingPieces).toEqual([]);
  });
  it("reacquires a single-cell between pieces and matches final predicted state", () => {
    const session = analyze(createSession(reacquireRescue()));
    expect(session.analysis!.plans[0].candidate.usedAbilities.singleCell).toBe(
      2,
    );
    expect(finish(session).game).toEqual(
      session.analysis!.plans[0].candidate.finalState,
    );
  });
  it("stops at fixed-slot reroll, rejects same type, then accepts real result and reanalyzes", () => {
    const board = createEmptyBoard().map((row) => row.map(() => true));
    board[0][0] = false;
    const initial = analyze(
      createSession(
        game({
          board,
          remainingPieces: [instance(2, "MIEUM")],
          abilities: { reroll: 1, singleCell: 0 },
        }),
      ),
    );
    expect(nextPreview(initial).action?.type).toBe("reroll");
    const pending = applyStep(initial, initial.version, 0);
    expect(pending.analysis).toBeNull();
    expect(pending.game.pendingReroll?.pieceIndex).toBe(2);
    expect(pending.game.abilities.reroll).toBe(0);
    expect(inputReroll(pending, "MIEUM").game).toEqual(pending.game);
    expect(editPiece(pending, 2, "DOT").game).toEqual(pending.game);
    const resolved = inputReroll(pending, "DOT");
    expect(resolved.game.pendingReroll).toBeNull();
    expect(resolved.game.remainingPieces[0].pieceIndex).toBe(2);
    expect(finish(analyze(resolved)).game.remainingPieces).toEqual([]);
  });
  it("distinguishes no legal action from next-set and actual-reroll wait", () => {
    const board = createEmptyBoard().map((row) => row.map(() => true));
    const session = analyze(
      createSession(game({ board, remainingPieces: [instance(0)] })),
    );
    expect(session.analysis!.plans[0].candidate.evaluation.phase).toBe(
      "gameover",
    );
    expect(nextPreview(session).action).toBeUndefined();
    expect(getAvailableActions(createSession().game)).toMatchObject({
      ok: true,
      phase: "await-next-pieces",
    });
  });
});
