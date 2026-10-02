import { afterEach, describe, it, expect, vi } from "vitest";
import {
  scheduleNextTurn,
  NEXT_TURN_REFRESH,
  readyNextTurn,
  nextTurnSignature,
} from "../../src/features/recognition/next-turn-refresh";
import {
  createSession,
  loadNextPieces,
} from "../../src/features/puzzle/session";
import { recognizeManualState } from "../../src/features/recognition/mock";
const clock = {
  schedule: (f: () => void, d: number) => setTimeout(f, d),
  cancel: (id: ReturnType<typeof setTimeout>) => clearTimeout(id),
};
afterEach(() => vi.useRealTimers());
describe("bounded next-turn event", () => {
  it("waits for delayed new frames, carries previous stable signature and stops once", () => {
    vi.useFakeTimers();
    const calls: [boolean, string | null][] = [];
    const finish = vi.fn();
    scheduleNextTurn(
      (final, previous) => {
        calls.push([final, previous]);
        const signature = calls.length < 3 ? null : "fresh";
        return { done: previous === "fresh", signature };
      },
      clock,
      finish,
    );
    vi.runAllTimers();
    expect(calls).toEqual([
      [false, null],
      [false, null],
      [false, null],
      [false, "fresh"],
    ]);
    expect(finish).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);
  });
  it("stops at the configured bound and marks only the last read final", () => {
    vi.useFakeTimers();
    const finals: boolean[] = [];
    const finish = vi.fn();
    scheduleNextTurn(
      (final) => {
        finals.push(final);
        return { done: false, signature: null };
      },
      clock,
      finish,
    );
    vi.runAllTimers();
    expect(finals).toHaveLength(NEXT_TURN_REFRESH.maxAttempts);
    expect(finals.filter(Boolean)).toHaveLength(1);
    expect(finals.at(-1)).toBe(true);
    expect(finish).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);
  });
  it("cancels before the first frame and during retry without future reads/finish", () => {
    vi.useFakeTimers();
    const read = vi.fn(() => ({ done: false, signature: null })),
      finish = vi.fn();
    const cancel = scheduleNextTurn(read, clock, finish);
    cancel();
    vi.runAllTimers();
    expect(read).not.toHaveBeenCalled();
    const cancelAgain = scheduleNextTurn(read, clock, finish);
    vi.advanceTimersByTime(NEXT_TURN_REFRESH.delayMs);
    cancelAgain();
    cancelAgain();
    vi.runAllTimers();
    expect(read).toHaveBeenCalledTimes(1);
    expect(finish).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);
  });
  it("requires fresh full set, known counts and sufficient matching board; timestamps do not affect stability", () => {
    const game = loadNextPieces(createSession(), [
      "DOT",
      "LINE_3",
      "LINE_5",
    ]).game;
    const result = recognizeManualState(game);
    expect(readyNextTurn(result, game)).toBe(true);
    const changed = structuredClone(result);
    changed.source.timestamp += 1000;
    expect(nextTurnSignature(result)).toBe(nextTurnSignature(changed));
    changed.pieces[2].empty = true;
    expect(readyNextTurn(changed, game)).toBe(false);
    changed.pieces[2] = result.pieces[2];
    changed.abilities.reroll.value = null;
    expect(readyNextTurn(changed, game)).toBe(false);
    changed.abilities = result.abilities;
    changed.board[0][0].occupied = true;
    expect(readyNextTurn(changed, game)).toBe(false);
    game.pendingReroll = {
      instanceId: game.remainingPieces[0].instanceId,
      pieceIndex: 0,
    };
    expect(readyNextTurn(result, game)).toBe(false);
  });
});
