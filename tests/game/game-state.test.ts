import { describe, expect, it } from "vitest";
import {
  enterNextPieces,
  resolveReroll,
  validateGameState,
} from "../../src/domain/game/game-state";
import {
  getAvailableActions,
  applyAction,
} from "../../src/domain/game/actions";
import {
  game,
  instance,
  piece,
  placement,
  success,
  freezeDeep,
} from "./fixtures";

describe("GameState ingress", () => {
  it("accepts three detached icons and rejects a fourth at every game ingress", () => {
    const input = game({
      hiddenItems: [
        { row: 0, col: 0, type: "reroll" },
        { row: 0, col: 1, type: "single-cell" },
        { row: 15, col: 9, type: "reroll" },
      ],
    });
    const valid = success(validateGameState(freezeDeep(input))).state;
    valid.hiddenItems[0].row = 2;
    expect(input.hiddenItems[0].row).toBe(0);
    const invalid = freezeDeep({
      ...input,
      hiddenItems: [
        ...input.hiddenItems,
        { row: 1, col: 0, type: "single-cell" },
      ],
    });
    for (const output of [
      validateGameState(invalid),
      getAvailableActions(invalid),
      applyAction(invalid, placement(input.remainingPieces[0], 2, 2)),
    ]) {
      expect(output).toMatchObject({
        ok: false,
        error: {
          code: "INVALID_HIDDEN_ITEMS",
          message: expect.stringContaining("최대 3개"),
        },
      });
    }
    expect(invalid.hiddenItems).toHaveLength(4);
  });
  it("accepts duplicate kinds in three distinct stable slots and detaches all fields", () => {
    const input = freezeDeep(
      game({ hiddenItems: [{ row: 4, col: 2, type: "reroll" }] }),
    );
    const output = success(validateGameState(input)).state;
    expect(output).toEqual(input);
    output.board[0][0] = true;
    output.remainingPieces[0].piece.shape[0][0] = false;
    output.remainingPieces[0].instanceId = "changed";
    output.hiddenItems[0].row = 5;
    output.abilities.reroll = 1;
    expect(input.board[0][0]).toBe(false);
    expect(input.remainingPieces[0].piece.shape).toEqual([[true]]);
    expect(output.remainingPieces[1].piece.shape).toEqual([[true]]);
    expect(input.hiddenItems[0].row).toBe(4);
    expect(input.abilities.reroll).toBe(0);
  });
  it.each([
    null,
    {},
    [],
    { ...game(), pendingReroll: undefined },
    { ...game(), pendingReroll: { instanceId: "missing", pieceIndex: 0 } },
  ])("rejects invalid state/pending ingress %#", (input) => {
    expect(validateGameState(input)).toMatchObject({
      ok: false,
      error: { code: expect.any(String) },
    });
  });
  it.each([
    { reroll: -1, singleCell: 0 },
    { reroll: 0.5, singleCell: 0 },
    { reroll: 4, singleCell: 4 },
    { reroll: NaN, singleCell: 0 },
    { reroll: Infinity, singleCell: 0 },
    { reroll: "1", singleCell: 0 },
    null,
  ])("rejects invalid ability counts %#", (abilities) => {
    expect(validateGameState({ ...game(), abilities })).toMatchObject({
      ok: false,
      error: { code: "INVALID_ABILITIES" },
    });
  });
  it.each(
    [
      [instance(0), instance(0)],
      [instance(0), instance(1, "DOT", "set1-0")],
      [{ ...instance(0), pieceIndex: 3 }],
      [{ ...instance(0), pieceIndex: 0.5 }],
      [{ ...instance(0), instanceId: " " }],
      [null],
      Array(1),
      [instance(0), instance(1), instance(2), instance(0, "DOT", "extra")],
    ].map((remainingPieces) => ({ remainingPieces })),
  )(
    "rejects duplicate/malformed/sparse instances %#",
    ({ remainingPieces }) => {
      expect(validateGameState({ ...game(), remainingPieces })).toMatchObject({
        ok: false,
        error: { code: "INVALID_PIECE_INSTANCE" },
      });
    },
  );
  it.each(
    [
      [{ row: -1, col: 0, type: "reroll" }],
      [{ row: 16, col: 0, type: "reroll" }],
      [{ row: 0, col: 10, type: "reroll" }],
      [{ row: 0.1, col: 0, type: "reroll" }],
      [{ row: 0, col: 0, type: "unknown" }],
      [
        { row: 0, col: 0, type: "reroll" },
        { row: 0, col: 0, type: "single-cell" },
      ],
      [null],
      Array(1),
      null,
    ].map((hiddenItems) => ({ hiddenItems })),
  )("rejects invalid/duplicate item coordinates %#", ({ hiddenItems }) => {
    expect(validateGameState({ ...game(), hiddenItems })).toMatchObject({
      ok: false,
      error: { code: "INVALID_HIDDEN_ITEMS" },
    });
  });
});

describe("actual action availability and lifecycle", () => {
  it("enumerates every duplicate-slot placement and all single/reroll actions", () => {
    const state = freezeDeep(game({ abilities: { reroll: 1, singleCell: 1 } }));
    const result = success(getAvailableActions(state));
    expect(result.phase).toBe("playing");
    expect(
      result.actions.filter((action) => action.type === "place-piece"),
    ).toHaveLength(480);
    expect(
      result.actions.filter((action) => action.type === "single-cell"),
    ).toHaveLength(160);
    expect(
      result.actions.filter((action) => action.type === "reroll"),
    ).toHaveLength(3);
    const first = result.actions[0];
    const second = result.actions[1];
    if (first.type !== "place-piece" || second.type !== "place-piece")
      throw new Error("Expected placements");
    first.variant[0][0] = false;
    expect(second.variant).toEqual([[true]]);
    expect(state.remainingPieces[0].piece.shape).toEqual([[true]]);
  });
  it("allows the unblocked instance when another piece is blocked", () => {
    const state = game({
      remainingPieces: [instance(0, "MIEUM"), instance(2)],
    });
    state.board.forEach((row) => row.fill(true));
    state.board[5][5] = false;
    const result = success(getAvailableActions(state));
    expect(result).toMatchObject({
      phase: "playing",
      actions: [{ type: "place-piece", pieceIndex: 2, row: 5, col: 5 }],
    });
  });
  it.each([
    [{ reroll: 0, singleCell: 0 }, "gameover", 0],
    [{ reroll: 0, singleCell: 1 }, "playing", 1],
    [{ reroll: 1, singleCell: 0 }, "playing", 1],
  ])(
    "uses real special actions on an isolated empty cell %#",
    (abilities, phase, count) => {
      const state = game({
        abilities,
        remainingPieces: [instance(1, "MIEUM")],
      });
      state.board.forEach((row) => row.fill(true));
      state.board[8][3] = false;
      const result = success(getAvailableActions(state));
      expect(result.phase).toBe(phase);
      expect(result.actions).toHaveLength(count);
    },
  );
  it("does not treat unused single-cell counts as a legal action on a full board", () => {
    const state = game({ abilities: { reroll: 0, singleCell: 7 } });
    state.board.forEach((row) => row.fill(true));
    expect(success(getAvailableActions(state))).toEqual({
      ok: true,
      actions: [],
      phase: "gameover",
    });
    state.abilities = { reroll: 1, singleCell: 6 };
    expect(success(getAvailableActions(state)).actions).toHaveLength(3);
  });
  it("waits for three new pieces after zero remaining, preserving board/rewards", () => {
    const original = game({
      remainingPieces: [instance(2)],
      hiddenItems: [{ row: 9, col: 4, type: "reroll" }],
      abilities: { reroll: 3, singleCell: 2 },
    });
    const depleted = success(
      applyAction(original, placement(original.remainingPieces[0], 0, 0)),
    ).state;
    expect(success(getAvailableActions(depleted))).toEqual({
      ok: true,
      phase: "await-next-pieces",
      actions: [],
    });
    expect(
      applyAction(depleted, { type: "single-cell", row: 0, col: 1 }),
    ).toMatchObject({ ok: false, error: { code: "AWAITING_NEXT_PIECES" } });
    const entered = success(
      enterNextPieces(freezeDeep(depleted), [
        instance(0, "DOT", "set2-0"),
        instance(1, "LINE_3", "set2-1"),
        instance(2, "L_3", "set2-2"),
      ]),
    ).state;
    expect(entered.board).toEqual(depleted.board);
    expect(entered.hiddenItems).toEqual(depleted.hiddenItems);
    expect(entered.abilities).toEqual(depleted.abilities);
    expect(entered.remainingPieces.map((target) => target.instanceId)).toEqual([
      "set2-0",
      "set2-1",
      "set2-2",
    ]);
    expect(success(getAvailableActions(entered)).phase).toBe("playing");
    entered.board[0][0] = false;
    expect(depleted.board[0][0]).toBe(true);
  });
  it.each(
    [[], [instance(0)], [instance(0), instance(1)]].map((pieces) => ({
      pieces,
    })),
  )("requires exactly three new instances %#", ({ pieces }) => {
    expect(
      enterNextPieces(game({ remainingPieces: [] }), pieces),
    ).toMatchObject({ ok: false, error: { code: "INVALID_LIFECYCLE" } });
  });
  it("rejects premature next-set entry and reroll result entry", () => {
    expect(
      enterNextPieces(game(), [instance(0), instance(1), instance(2)]),
    ).toMatchObject({ ok: false, error: { code: "INVALID_LIFECYCLE" } });
    expect(resolveReroll(game(), piece("LINE_3"))).toMatchObject({
      ok: false,
      error: { code: "INVALID_LIFECYCLE" },
    });
  });
  it("propagates malformed state errors through every public transition/action boundary", () => {
    expect(getAvailableActions({})).toMatchObject({ ok: false });
    expect(applyAction({}, null)).toMatchObject({ ok: false });
    expect(enterNextPieces({}, [])).toMatchObject({ ok: false });
    expect(resolveReroll({}, piece())).toMatchObject({ ok: false });
  });
});
