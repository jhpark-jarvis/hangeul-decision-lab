import { describe, expect, it } from "vitest";
import {
  applyAction,
  getAvailableActions,
} from "../../src/domain/game/actions";
import {
  enterNextPieces,
  resolveReroll,
} from "../../src/domain/game/game-state";
import {
  game,
  instance,
  piece,
  placement,
  success,
  freezeDeep,
} from "./fixtures";

describe("central piece and reward transition", () => {
  it("consumes only the chosen duplicate instance, retaining original slot indices", () => {
    const input = freezeDeep(game());
    const first = success(
      applyAction(input, placement(input.remainingPieces[1], 5, 5)),
    );
    expect(first.info).toEqual({
      clearedRows: [],
      acquiredItems: [],
      retainedItems: [],
      spentAbility: null,
      consumedPiece: { instanceId: "set1-1", pieceIndex: 1 },
    });
    expect(
      first.state.remainingPieces.map((entry) => entry.pieceIndex),
    ).toEqual([0, 2]);
    const last = success(
      applyAction(first.state, placement(first.state.remainingPieces[1], 5, 6)),
    );
    expect(last.state.remainingPieces.map((entry) => entry.pieceIndex)).toEqual(
      [0],
    );
    expect(input.board[5][5]).toBe(false);
    expect(
      applyAction(first.state, placement(input.remainingPieces[1], 6, 5)),
    ).toMatchObject({ ok: false, error: { code: "MISSING_PIECE" } });
  });
  it("covering an item alone neither acquires nor removes it", () => {
    const input = freezeDeep(
      game({ hiddenItems: [{ row: 4, col: 4, type: "reroll" }] }),
    );
    const output = success(
      applyAction(input, placement(input.remainingPieces[0], 4, 4)),
    );
    expect(output.state.board[4][4]).toBe(true);
    expect(output.state.hiddenItems).toEqual(input.hiddenItems);
    expect(output.state.abilities).toEqual({ reroll: 0, singleCell: 0 });
    expect(output.info.acquiredItems).toEqual([]);
  });
  it("clears three rows simultaneously, acquiring/retaining by coordinates without gravity", () => {
    const input = game({
      remainingPieces: [instance(2, "LINE_3")],
      abilities: { reroll: 4, singleCell: 1 },
      hiddenItems: [
        { row: 5, col: 4, type: "reroll" },
        { row: 3, col: 8, type: "reroll" },
        { row: 3, col: 1, type: "single-cell" },
        { row: 6, col: 3, type: "single-cell" },
      ],
    });
    for (const row of [3, 4, 5]) {
      input.board[row].fill(true);
      input.board[row][4] = false;
    }
    input.board[6][3] = true;
    const action = {
      ...placement(input.remainingPieces[0], 3, 4),
      variant: [[true], [true], [true]],
    };
    const output = success(applyAction(freezeDeep(input), freezeDeep(action)));
    expect(output.info.clearedRows).toEqual([3, 4, 5]);
    expect(output.info.acquiredItems).toEqual([
      input.hiddenItems[2],
      input.hiddenItems[1],
    ]);
    expect(output.info.retainedItems).toEqual([input.hiddenItems[0]]);
    expect(output.state.hiddenItems).toEqual([
      input.hiddenItems[0],
      input.hiddenItems[3],
    ]);
    expect(output.state.abilities).toEqual({ reroll: 5, singleCell: 2 });
    expect(output.state.board.flat().filter(Boolean)).toHaveLength(1);
    expect(output.state.board[6][3]).toBe(true);
    expect(output.state.remainingPieces).toEqual([]);
    expect(input.board[3][0]).toBe(true);
  });
  it("accepts a normalized padded legal variant and rejects a fabricated shape", () => {
    const input = game({ remainingPieces: [instance(0, "LINE_3")] });
    const base = placement(input.remainingPieces[0], 15, 7);
    expect(
      success(
        applyAction(input, {
          ...base,
          variant: [
            [false, false, false, false],
            [false, true, true, true],
          ],
        }),
      ).state.board[15].slice(7),
    ).toEqual([true, true, true]);
    expect(
      applyAction(input, { ...base, variant: [[true, true]] }),
    ).toMatchObject({ ok: false, error: { code: "INVALID_VARIANT" } });
  });
  it("only horizontal completion clears rows through the action boundary", () => {
    const input = game({ hiddenItems: [{ row: 8, col: 0, type: "reroll" }] });
    for (let row = 0; row < 15; row++) input.board[row][0] = true;
    const output = success(
      applyAction(input, placement(input.remainingPieces[0], 15, 0)),
    );
    expect(output.info.clearedRows).toEqual([]);
    expect(output.info.acquiredItems).toEqual([]);
    expect(output.state.board.every((row) => row[0])).toBe(true);
  });
  it.each([
    [{ type: "unknown" }, "INVALID_ACTION"],
    [null, "INVALID_ACTION"],
    [
      { type: "place-piece", instanceId: "missing", pieceIndex: 0 },
      "MISSING_PIECE",
    ],
    [
      { type: "place-piece", instanceId: "set1-0", pieceIndex: 1 },
      "MISSING_PIECE",
    ],
    [
      {
        type: "place-piece",
        instanceId: "set1-0",
        pieceIndex: 0,
        pieceId: "LINE_3",
        variant: [[true]],
        row: 0,
        col: 0,
      },
      "MISSING_PIECE",
    ],
    [
      {
        type: "place-piece",
        instanceId: "set1-0",
        pieceIndex: 0,
        pieceId: "DOT",
        variant: [],
        row: 0,
        col: 0,
      },
      "INVALID_SHAPE",
    ],
    [
      {
        type: "place-piece",
        instanceId: "set1-0",
        pieceIndex: 0,
        pieceId: "DOT",
        variant: [[true]],
        row: "0",
        col: 0,
      },
      "INVALID_COORDINATE",
    ],
    [
      {
        type: "place-piece",
        instanceId: "set1-0",
        pieceIndex: 0,
        pieceId: "DOT",
        variant: [[true]],
        row: 0.5,
        col: 0,
      },
      "INVALID_COORDINATE",
    ],
    [
      {
        type: "place-piece",
        instanceId: "set1-0",
        pieceIndex: 0,
        pieceId: "DOT",
        variant: [[true]],
        row: 16,
        col: 0,
      },
      "OUT_OF_BOUNDS",
    ],
  ])("fails explicitly without modifying input %#", (action, code) => {
    const input = freezeDeep(game());
    const snapshot = structuredClone(input);
    expect(applyAction(input, action)).toMatchObject({
      ok: false,
      error: { code },
    });
    expect(input).toEqual(snapshot);
    expect(
      success(applyAction(input, placement(input.remainingPieces[0], 0, 0)))
        .state.board[0][0],
    ).toBe(true);
  });
  it("rejects collision while retaining input and permits a corrected retry", () => {
    const input = game();
    input.board[0][0] = true;
    freezeDeep(input);
    expect(
      applyAction(input, placement(input.remainingPieces[0], 0, 0)),
    ).toMatchObject({ ok: false, error: { code: "COLLISION" } });
    expect(
      success(
        applyAction(input, placement(input.remainingPieces[0], 0, 1)),
      ).state.board[0].slice(0, 2),
    ).toEqual([true, true]);
  });
});

describe("single-cell spend and immediate reward", () => {
  it("is legal with ordinary placements available and does not consume a piece", () => {
    const input = freezeDeep(game({ abilities: { reroll: 0, singleCell: 2 } }));
    const output = success(
      applyAction(input, { type: "single-cell", row: 3, col: 3 }),
    );
    expect(output.state.remainingPieces).toEqual(input.remainingPieces);
    expect(output.state.board[3][3]).toBe(true);
    expect(output.state.abilities.singleCell).toBe(1);
    expect(output.info).toMatchObject({
      spentAbility: "single-cell",
      consumedPiece: null,
    });
  });
  it("spends before acquisition, regaining capacity and retaining later coordinates", () => {
    const input = game({
      abilities: { reroll: 6, singleCell: 1 },
      hiddenItems: [
        { row: 2, col: 9, type: "reroll" },
        { row: 2, col: 0, type: "single-cell" },
      ],
    });
    input.board[2].fill(true);
    input.board[2][9] = false;
    const output = success(
      applyAction(freezeDeep(input), { type: "single-cell", row: 2, col: 9 }),
    );
    expect(output.state.abilities).toEqual({ reroll: 6, singleCell: 1 });
    expect(output.info.clearedRows).toEqual([2]);
    expect(output.info.acquiredItems).toEqual([input.hiddenItems[1]]);
    expect(output.info.retainedItems).toEqual([input.hiddenItems[0]]);
    expect(output.state.hiddenItems).toEqual([input.hiddenItems[0]]);
    expect(output.state.board[2].some(Boolean)).toBe(false);
  });
  it.each([
    [
      { reroll: 0, singleCell: 0 },
      { type: "single-cell", row: 0, col: 0 },
      "NO_ABILITY",
    ],
    [
      { reroll: 0, singleCell: 1 },
      { type: "single-cell", row: 0, col: 0 },
      "COLLISION",
    ],
    [
      { reroll: 0, singleCell: 1 },
      { type: "single-cell", row: -1, col: 0 },
      "OUT_OF_BOUNDS",
    ],
    [
      { reroll: 0, singleCell: 1 },
      { type: "single-cell", row: NaN, col: 0 },
      "INVALID_COORDINATE",
    ],
    [
      { reroll: 0, singleCell: 1 },
      { type: "single-cell", row: 0, col: "0" },
      "INVALID_COORDINATE",
    ],
  ])(
    "rejects failed use without spending ability %#",
    (abilities, action, code) => {
      const input = game({ abilities });
      input.board[0][0] = true;
      freezeDeep(input);
      expect(applyAction(input, action)).toMatchObject({
        ok: false,
        error: { code },
      });
      expect(input.abilities).toEqual(abilities);
    },
  );
});

describe("reroll actual result ingress", () => {
  it("can reacquire reroll after accepting the actual result and clearing a reward row", () => {
    const input = game({
      remainingPieces: [instance(2)],
      abilities: { reroll: 1, singleCell: 0 },
      hiddenItems: [{ row: 7, col: 3, type: "reroll" }],
    });
    input.board[7].fill(true);
    input.board[7].fill(false, 7);
    const waiting = success(
      applyAction(freezeDeep(input), {
        type: "reroll",
        instanceId: "set1-2",
        pieceIndex: 2,
      }),
    ).state;
    expect(waiting.abilities.reroll).toBe(0);
    const actual = success(resolveReroll(waiting, piece("LINE_3"))).state;
    const placed = success(
      applyAction(actual, placement(actual.remainingPieces[0], 7, 7)),
    );
    expect(placed.info.clearedRows).toEqual([7]);
    expect(placed.state.abilities).toEqual({ reroll: 1, singleCell: 0 });
    expect(placed.state.hiddenItems).toEqual([]);
  });
  it("spends once, pauses without sampling, keeps target slot, then accepts actual different kind", () => {
    const input = freezeDeep(
      game({
        abilities: { reroll: 2, singleCell: 1 },
        hiddenItems: [{ row: 4, col: 1, type: "reroll" }],
      }),
    );
    const rolled = success(
      applyAction(input, {
        type: "reroll",
        instanceId: "set1-2",
        pieceIndex: 2,
      }),
    );
    expect(rolled.state.board).toEqual(input.board);
    expect(rolled.state.hiddenItems).toEqual(input.hiddenItems);
    expect(rolled.state.remainingPieces).toEqual(input.remainingPieces);
    expect(rolled.state.abilities).toEqual({ reroll: 1, singleCell: 1 });
    expect(rolled.state.pendingReroll).toEqual({
      instanceId: "set1-2",
      pieceIndex: 2,
    });
    expect(rolled.info).toEqual({
      spentAbility: "reroll",
      consumedPiece: null,
      clearedRows: [],
      acquiredItems: [],
      retainedItems: [],
    });
    expect(success(getAvailableActions(rolled.state))).toEqual({
      ok: true,
      phase: "await-reroll-result",
      actions: [],
    });
    freezeDeep(rolled.state);
    expect(
      applyAction(rolled.state, placement(input.remainingPieces[0], 0, 0)),
    ).toMatchObject({ ok: false, error: { code: "AWAITING_REROLL" } });
    expect(
      applyAction(rolled.state, {
        type: "reroll",
        instanceId: "set1-2",
        pieceIndex: 2,
      }),
    ).toMatchObject({ ok: false, error: { code: "AWAITING_REROLL" } });
    expect(enterNextPieces(rolled.state, input.remainingPieces)).toMatchObject({
      ok: false,
      error: { code: "INVALID_LIFECYCLE" },
    });
    expect(resolveReroll(rolled.state, piece())).toMatchObject({
      ok: false,
      error: { code: "INVALID_LIFECYCLE" },
    });
    expect(
      resolveReroll(rolled.state, { id: "bad", name: "bad", shape: [] }),
    ).toMatchObject({ ok: false, error: { code: "INVALID_SHAPE" } });
    const replacement = freezeDeep(piece("LINE_3"));
    const entered = success(resolveReroll(rolled.state, replacement)).state;
    expect(entered.pendingReroll).toBeNull();
    expect(
      entered.remainingPieces.map((target) => [
        target.instanceId,
        target.pieceIndex,
        target.piece.id,
      ]),
    ).toEqual([
      ["set1-0", 0, "DOT"],
      ["set1-1", 1, "DOT"],
      ["set1-2", 2, "LINE_3"],
    ]);
    expect(entered.abilities).toEqual({ reroll: 1, singleCell: 1 });
    expect(success(getAvailableActions(entered)).phase).toBe("playing");
    entered.remainingPieces[2].piece.shape[0][0] = false;
    expect(replacement.shape[0][0]).toBe(true);
    expect(rolled.state.remainingPieces[2].piece.id).toBe("DOT");
  });
  it("fails without spending when ability or remaining target is absent", () => {
    const input = freezeDeep(game());
    expect(
      applyAction(input, {
        type: "reroll",
        instanceId: "set1-0",
        pieceIndex: 0,
      }),
    ).toMatchObject({ ok: false, error: { code: "NO_ABILITY" } });
    expect(
      applyAction(input, {
        type: "reroll",
        instanceId: "missing",
        pieceIndex: 0,
      }),
    ).toMatchObject({ ok: false, error: { code: "MISSING_PIECE" } });
  });
});

describe("synthetic shared-domain journey", () => {
  it("replays enumerated actions through rewards, special use, depletion and next-set input", () => {
    const initial = game({
      hiddenItems: [
        { row: 0, col: 2, type: "single-cell" },
        { row: 1, col: 1, type: "reroll" },
      ],
    });
    initial.board[0].fill(true);
    initial.board[0][9] = false;
    initial.board[1].fill(true);
    initial.board[1][9] = false;
    freezeDeep(initial);
    const actions = success(getAvailableActions(initial)).actions;
    const fill = actions.find(
      (action) =>
        action.type === "place-piece" &&
        action.pieceIndex === 1 &&
        action.row === 0 &&
        action.col === 9,
    );
    if (!fill) throw new Error("Missing fixture action");
    const first = success(applyAction(initial, fill));
    expect(first.info.clearedRows).toEqual([0]);
    expect(first.state.abilities).toEqual({ reroll: 0, singleCell: 1 });
    const single = success(getAvailableActions(first.state)).actions.find(
      (action) =>
        action.type === "single-cell" && action.row === 1 && action.col === 9,
    );
    if (!single) throw new Error("Missing single-cell action");
    const second = success(applyAction(first.state, single));
    expect(second.info.clearedRows).toEqual([1]);
    expect(second.state.abilities).toEqual({ reroll: 1, singleCell: 0 });
    expect(second.state.hiddenItems).toEqual([]);
    const rerolled = success(
      applyAction(second.state, {
        type: "reroll",
        instanceId: "set1-2",
        pieceIndex: 2,
      }),
    );
    expect(success(getAvailableActions(rerolled.state)).phase).toBe(
      "await-reroll-result",
    );
    const resolved = success(
      resolveReroll(rerolled.state, piece("LINE_3")),
    ).state;
    const third = success(
      applyAction(resolved, placement(resolved.remainingPieces[1], 0, 0)),
    ).state;
    const fourth = success(
      applyAction(third, placement(third.remainingPieces[0], 1, 0)),
    ).state;
    expect(success(getAvailableActions(fourth)).phase).toBe(
      "await-next-pieces",
    );
    expect(fourth.board[0].slice(0, 4)).toEqual([true, true, true, false]);
    expect(fourth.board[1].slice(0, 2)).toEqual([true, false]);
    expect(fourth.abilities).toEqual({ reroll: 0, singleCell: 0 });
    const next = success(
      enterNextPieces(fourth, [
        instance(0, "DOT", "next-0"),
        instance(1, "DOT", "next-1"),
        instance(2, "DOT", "next-2"),
      ]),
    ).state;
    expect(success(getAvailableActions(next)).phase).toBe("playing");
    expect(next.board).toEqual(fourth.board);
    expect(initial.remainingPieces).toHaveLength(3);
    expect(initial.hiddenItems).toHaveLength(2);
  });
  it("every enumerated action is applicable and detached on a constrained board", () => {
    const input = game({
      remainingPieces: [instance(2, "LINE_3")],
      abilities: { reroll: 1, singleCell: 1 },
    });
    for (const row of input.board) row.fill(true);
    input.board[0].fill(false);
    freezeDeep(input);
    const original = structuredClone(input);
    const actions = success(getAvailableActions(input)).actions;
    expect(actions).toHaveLength(19); // 8 horizontal LINE_3 + 10 single-cell + 1 reroll
    for (const action of actions) {
      const output = success(applyAction(input, freezeDeep(action)));
      expect(output.state).not.toBe(input);
      expect(input).toEqual(original);
    }
  });
});
