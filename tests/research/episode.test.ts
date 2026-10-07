import { describe, expect, it } from "vitest";
import { createEmptyBoard } from "../../src/domain/board/board";
import type { GameAction } from "../../src/domain/game/types";
import {
  createSyntheticFixture,
  indexedUint32,
  validateFixture,
} from "../../src/research/fixture";
import { runEpisode, verifyReplay } from "../../src/research/episode";
import type {
  EpisodeFixture,
  EpisodePolicy,
  ResearchResult,
} from "../../src/research/types";
import { game, instance, freezeDeep, placement } from "../game/fixtures";

function value<T>(result: ResearchResult<T>): T {
  if (!result.ok) throw new Error(result.error);
  return result.value;
}
function scripted(maxActions = 5): EpisodeFixture {
  return {
    ...value(createSyntheticFixture(42, "sparse", maxActions)),
    provenance: "synthetic-scripted",
    initialState: game(),
    nextPieceSets: [["LINE_3", "LINE_5", "MIEUM"]],
  };
}
const first: EpisodePolicy = ({ legalActions }) => ({
  action: legalActions[0] ?? null,
});

describe("synthetic episode boundary and domain reuse", () => {
  it("clears two rows together, retains overflow in row order and leaves upper cells fixed", () => {
    const fixture = scripted(1);
    fixture.initialState.remainingPieces[0] = instance(0, "LINE_3");
    for (const row of [0, 1]) {
      fixture.initialState.board[row].fill(true);
      fixture.initialState.board[row][9] = false;
    }
    fixture.initialState.board[2][0] = true;
    fixture.initialState.abilities = { singleCell: 1, reroll: 5 };
    fixture.initialState.hiddenItems = [
      { row: 1, col: 9, type: "reroll" },
      { row: 0, col: 9, type: "single-cell" },
    ];
    const result = value(
      runEpisode(fixture, ({ state }) => ({
        action: {
          ...placement(state.remainingPieces[0], 0, 9),
          variant: [[true], [true], [true]],
        },
      })),
    );
    expect(result.totals).toMatchObject({
      actions: 1,
      clearedRows: 2,
      acquiredItems: 1,
    });
    expect(result.events[0]).toMatchObject({ info: { clearedRows: [0, 1] } });
    expect(result.finalState.abilities).toEqual({ singleCell: 2, reroll: 5 });
    expect(result.finalState.hiddenItems).toEqual([
      { row: 1, col: 9, type: "reroll" },
    ]);
    expect(result.finalState.board[2][0]).toBe(true);
    expect(result.finalState.board[2][9]).toBe(true);
    expect(verifyReplay(fixture, result).ok).toBe(true);
  });
  it("rejects nonfinite or circular diagnostics before applying an action", () => {
    const fixture = scripted(1);
    const circular: { self?: unknown } = {};
    circular.self = circular;
    for (const search of [{ visitedNodes: NaN }, circular]) {
      const result = value(
        runEpisode(fixture, ({ legalActions }) => ({
          action: legalActions[0],
          search:
            search as unknown as import("../../src/domain/solver/types").SearchInfo,
        })),
      );
      expect(result.ending).toMatchObject({
        kind: "error",
        detail: "Invalid policy diagnostics",
      });
      expect(result.events).toEqual([]);
      expect(result.totals.actions).toBe(0);
      expect(result.finalState).toEqual(fixture.initialState);
    }
  });
  it("plays across a three-piece boundary and replays without mutating a frozen fixture", () => {
    const fixture = freezeDeep(scripted());
    const result = value(runEpisode(fixture, first));
    expect(result.ending.kind).toBe("horizon");
    expect(result.totals).toMatchObject({
      actions: 5,
      placements: 5,
      completedSets: 1,
      suppliedSets: 1,
    });
    expect(result.events[3]).toMatchObject({
      type: "next-pieces",
      suppliedIds: ["LINE_3", "LINE_5", "MIEUM"],
    });
    expect(verifyReplay(fixture, result)).toEqual({ ok: true, value: true });
    expect(runEpisode(fixture, first)).toEqual({ ok: true, value: result });
    expect(fixture.initialState.board).toEqual(createEmptyBoard());
  });
  it("counts clears, spent abilities, capacity retention and no gravity by hand", () => {
    const fixture = scripted(2);
    fixture.initialState.board[0].fill(true);
    fixture.initialState.board[1].fill(true);
    fixture.initialState.board[0][9] = false;
    fixture.initialState.board[1][9] = false;
    fixture.initialState.board[2][0] = true;
    fixture.initialState.abilities = { singleCell: 1, reroll: 6 };
    fixture.initialState.hiddenItems = [
      { row: 0, col: 9, type: "reroll" },
      { row: 1, col: 9, type: "reroll" },
    ];
    const result = value(
      runEpisode(fixture, ({ state, actionIndex }) => ({
        action:
          actionIndex === 0
            ? placement(state.remainingPieces[0], 0, 9)
            : { type: "single-cell", row: 1, col: 9 },
      })),
    );
    expect(result.totals).toMatchObject({
      clearedRows: 2,
      acquiredItems: 1,
      singleCellUses: 1,
      placements: 1,
    });
    expect(result.finalState.hiddenItems).toEqual([
      { row: 0, col: 9, type: "reroll" },
    ]);
    expect(result.finalState.abilities).toEqual({ singleCell: 0, reroll: 7 });
    expect(result.finalState.board[0].some(Boolean)).toBe(false);
    expect(result.finalState.board[1].some(Boolean)).toBe(false);
    expect(result.finalState.board[2][0]).toBe(true);
    expect(verifyReplay(fixture, result).ok).toBe(true);
  });
  it("resolves reroll externally, excludes the current kind and preserves target identity", () => {
    const fixture = scripted(2);
    fixture.initialState.abilities.reroll = 1;
    fixture.rerollTickets[0] = [0, 0, 0];
    const result = value(
      runEpisode(fixture, ({ state, legalActions, actionIndex }) => ({
        action:
          actionIndex === 0
            ? {
                type: "reroll",
                instanceId: state.remainingPieces[2].instanceId,
                pieceIndex: 2,
              }
            : legalActions.find(
                (a) => a.type === "place-piece" && a.pieceIndex === 2,
              )!,
      })),
    );
    expect(result.events[1]).toMatchObject({
      type: "reroll-result",
      actionIndex: 1,
      suppliedIds: ["LINE_3"],
    });
    expect(result.events[2]).toMatchObject({
      action: { instanceId: "set1-2", pieceIndex: 2, pieceId: "LINE_3" },
    });
    expect(result.totals).toMatchObject({
      actions: 2,
      placements: 1,
      rerollUses: 1,
    });
    expect(result.finalState.pendingReroll).toBeNull();
    expect(verifyReplay(fixture, result).ok).toBe(true);
  });
  it("treats initial pending as an environment event and a pending horizon as truncation", () => {
    const fixture = scripted(1);
    fixture.initialState.pendingReroll = {
      instanceId: "set1-0",
      pieceIndex: 0,
    };
    const result = value(runEpisode(fixture, first));
    expect(result.events[0].type).toBe("reroll-result");
    expect(result.totals.rerollUses).toBe(0);
    fixture.initialState.pendingReroll = null;
    fixture.initialState.abilities.reroll = 1;
    const capped = value(
      runEpisode(fixture, ({ state }) => ({
        action: {
          type: "reroll",
          instanceId: state.remainingPieces[0].instanceId,
          pieceIndex: 0,
        },
      })),
    );
    expect(capped.ending.kind).toBe("horizon");
    expect(capped.finalState.pendingReroll).not.toBeNull();
    expect(capped.events).toHaveLength(1);
  });
  it("distinguishes actual gameover, legal ability rescue, abstention and exhausted tapes", () => {
    const fixture = scripted();
    fixture.initialState.board.forEach((row) => row.fill(true));
    expect(
      value(
        runEpisode(fixture, () => {
          throw new Error("must not call policy");
        }),
      ).ending.kind,
    ).toBe("gameover");
    fixture.initialState.abilities.reroll = 1;
    expect(value(runEpisode(fixture, first))).toMatchObject({
      ending: { kind: "gameover" },
      totals: { rerollUses: 1 },
    });
    fixture.initialState = game();
    expect(
      value(runEpisode(fixture, () => ({ action: null }))).ending.kind,
    ).toBe("policy-abstention");
    fixture.initialState.remainingPieces = [];
    fixture.nextPieceSets = [];
    expect(value(runEpisode(fixture, first)).ending).toMatchObject({
      kind: "tape-exhausted",
    });
    fixture.initialState = game({
      pendingReroll: { instanceId: "set1-0", pieceIndex: 0 },
    });
    fixture.rerollTickets = [];
    expect(value(runEpisode(fixture, first)).ending).toMatchObject({
      kind: "tape-exhausted",
    });
  });
  it("does not expose tapes and rejects policy mutation, malformed cells, exceptions and illegal moves", () => {
    const fixture = scripted();
    let keys: string[] = [];
    value(
      runEpisode(fixture, (observation) => {
        keys = Object.keys(observation).sort();
        return { action: null };
      }),
    );
    expect(keys).toEqual(["actionIndex", "legalActions", "state"]);
    for (const mutate of [
      (state: typeof fixture.initialState) => {
        state.board[0][0] = true;
      },
      (state: typeof fixture.initialState) => {
        state.board[0][0] = "bad" as unknown as boolean;
      },
    ]) {
      expect(
        value(
          runEpisode(fixture, ({ state }) => {
            mutate(state);
            return { action: null };
          }),
        ).ending.detail,
      ).toBe("Policy mutated observation");
    }
    expect(
      value(
        runEpisode(fixture, () => {
          throw new Error("private native error");
        }),
      ).ending.detail,
    ).toBe("Policy threw an exception");
    const invalid = placement(instance(0), -1, 0);
    expect(
      value(runEpisode(fixture, () => ({ action: invalid }))).ending.kind,
    ).toBe("error");
    expect(
      value(
        runEpisode(fixture, () => ({
          action: { type: "unknown" } as unknown as GameAction,
        })),
      ).ending.kind,
    ).toBe("error");
  });
  it("detects replay tampering in action, state, result, environment and fixture", () => {
    const fixture = scripted();
    const result = value(runEpisode(fixture, first));
    for (const mutate of [
      (r: typeof result) => {
        r.totals.clearedRows++;
      },
      (r: typeof result) => {
        r.events[0].after += "bad";
      },
      (r: typeof result) => {
        r.events.splice(3, 1);
      },
      (r: typeof result) => {
        const e = r.events[0];
        if (e.type === "action" && e.action.type === "place-piece")
          e.action.row = -1;
      },
    ]) {
      const corrupted = structuredClone(result);
      mutate(corrupted);
      expect(verifyReplay(fixture, corrupted).ok).toBe(false);
    }
    const changed = structuredClone(fixture);
    changed.nextPieceSets[0][0] = "DOT";
    expect(verifyReplay(changed, result).ok).toBe(false);
  });
});

describe("versioned synthetic fixture streams", () => {
  it("supports seed zero, independent board/piece streams and deterministic generation", () => {
    expect(indexedUint32(0, 0, 0)).toBe(2462723854);
    const fixture = value(createSyntheticFixture(0, "sparse", 5));
    expect(createSyntheticFixture(0, "sparse", 5)).toEqual({
      ok: true,
      value: fixture,
    });
    const pressure = value(createSyntheticFixture(0, "pressure", 5));
    expect(pressure.nextPieceSets).toEqual(fixture.nextPieceSets);
    expect(pressure.rerollTickets).toEqual(fixture.rerollTickets);
    expect(
      pressure.initialState.remainingPieces.map((p) => p.piece.id),
    ).toEqual(fixture.initialState.remainingPieces.map((p) => p.piece.id));
    expect(pressure.initialState.board).not.toEqual(fixture.initialState.board);
    expect(fixture.initialState.board.every((row) => row.includes(false))).toBe(
      true,
    );
    expect(createSyntheticFixture(1, "sparse", 5)).not.toEqual({
      ok: true,
      value: fixture,
    });
  });
  it("rejects unknown versions, malformed tapes, noncanonical shapes and unsupported limits", () => {
    for (const bad of [
      null,
      {},
      { ...scripted(), seed: -1 },
      { ...scripted(), seed: NaN },
      { ...scripted(), profile: "actual-game" },
      { ...scripted(), catalogVersion: "old" },
      { ...scripted(), maxActions: 1001 },
      { ...scripted(), nextPieceSets: [["DOT"]] },
      { ...scripted(), nextPieceSets: [["DOT", "DOT", "unknown"]] },
      { ...scripted(), rerollTickets: [[0, 0, -1]] },
    ])
      expect(validateFixture(bad).ok).toBe(false);
    const wrong = scripted();
    wrong.initialState.remainingPieces[0].piece.shape = [[true, true]];
    expect(validateFixture(wrong).ok).toBe(false);
    expect(createSyntheticFixture(0x100000000, "sparse").ok).toBe(false);
    expect(createSyntheticFixture(0, "sparse", 0).ok).toBe(false);
    expect(createSyntheticFixture(0, "invalid" as "sparse").ok).toBe(false);
  });
});
