import { describe, expect, it } from "vitest";
import { getAvailableActions } from "../../src/domain/game/actions";
import { encodeDecision } from "../../src/research/imitation";
import {
  actionOutcome,
  extendOutcome,
} from "../../src/research/action-outcome";
import {
  freezeDeep,
  game,
  instance,
  placement,
  success,
} from "../game/fixtures";

describe("action outcome inputs", () => {
  it("clears rows without gravity and counts capacity-limited acquisition in coordinate order", () => {
    const s = game({
      remainingPieces: [instance(0)],
      abilities: { singleCell: 1, reroll: 6 },
      hiddenItems: [
        { row: 5, col: 8, type: "reroll" },
        { row: 5, col: 2, type: "single-cell" },
      ],
    });
    s.board[5].fill(true);
    s.board[5][9] = false;
    s.board[2][3] = true;
    const original = structuredClone(s);
    const v = actionOutcome(freezeDeep(s), {
      type: "single-cell",
      row: 5,
      col: 9,
    });
    expect(v.slice(0, 6)).toEqual([1 / 16, 1 / 3, 0, 1 / 3, 1 / 160, 0]);
    expect(v[6 + 2]).toBe(1 / 10);
    expect(v[6 + 5]).toBe(0);
    expect(s).toEqual(original);
  });
  it("counts simultaneous row clears rather than columns", () => {
    const s = game({ remainingPieces: [instance(0, "LINE_5")] });
    for (const row of [3, 4]) {
      s.board[row].fill(true);
      s.board[row][2] = false;
    }
    const v = actionOutcome(s, placement(s.remainingPieces[0], 2, 2));
    expect(v[0]).toBe(2 / 16);
    expect(v[4]).toBe(3 / 160);
    expect(v[6 + 2]).toBe(1 / 10);
    expect(v[6 + 3]).toBe(0);
    expect(v[6 + 4]).toBe(0);
  });
  it("reroll remains pending and leaves the known board unchanged", () => {
    const s = game({ abilities: { singleCell: 0, reroll: 1 } });
    s.board[15][4] = true;
    expect(
      actionOutcome(freezeDeep(s), {
        type: "reroll",
        instanceId: "set1-1",
        pieceIndex: 1,
      }),
    ).toEqual([0, 0, 0, 0, 1 / 160, 1, ...Array(15).fill(0), 1 / 10]);
    expect(() =>
      actionOutcome(s, { type: "single-cell", row: 0, col: 0 }),
    ).toThrow();
  });
  it("preserves all keys, original features and label; rejects drift and exhausted budget before transitions", () => {
    const s = game({ remainingPieces: [instance(0)] });
    const actions = success(getAvailableActions(s)).actions;
    const e = encodeDecision(s, actions, actions[100]);
    if (!e.ok) throw new Error(e.error);
    const original = {
      ...e.value,
      state: s,
      labels: { A512: e.value.label, A1024: 0 },
    };
    const r = extendOutcome(freezeDeep(original), actions.length);
    expect(r.transitions).toBe(actions.length);
    expect(r.candidateFeatures.map((f) => f.slice(0, 52))).toEqual(
      e.value.candidateFeatures,
    );
    expect(
      r.candidateFeatures.every(
        (f) => f.length === 74 && f.every((n) => n >= 0 && n <= 1),
      ),
    ).toBe(true);
    expect(() => extendOutcome(original, actions.length - 1)).toThrow("budget");
    expect(() =>
      extendOutcome(
        { ...original, candidateKeys: [...original.candidateKeys].reverse() },
        actions.length,
      ),
    ).toThrow("mismatch");
  });
});
