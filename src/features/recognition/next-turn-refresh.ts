import type { GameState } from "../../domain/game/types";
import type { RecognitionResult } from "./types";
import { continueTurn } from "./continuity";
export const NEXT_TURN_REFRESH = { delayMs: 500, maxAttempts: 8 } as const;
export type RefreshRead = { done: boolean; signature: string | null };
export type RefreshClock<T> = {
  schedule: (callback: () => void, delay: number) => T;
  cancel: (id: T) => void;
};
/** One event, bounded retries. The caller owns frame release and review guards. */
export function scheduleNextTurn<T>(
  read: (final: boolean, previous: string | null) => RefreshRead,
  clock: RefreshClock<T>,
  finish: () => void,
) {
  let active = true,
    attempt = 0,
    previous: string | null = null;
  let timer: T;
  const tick = () => {
    if (!active) return;
    const final = ++attempt >= NEXT_TURN_REFRESH.maxAttempts;
    const next = read(final, previous);
    previous = next.signature;
    if (next.done || final) {
      active = false;
      finish();
      return;
    }
    timer = clock.schedule(tick, NEXT_TURN_REFRESH.delayMs);
  };
  timer = clock.schedule(tick, NEXT_TURN_REFRESH.delayMs);
  return () => {
    if (active) {
      active = false;
      clock.cancel(timer);
    }
  };
}
export function readyNextTurn(
  result: RecognitionResult,
  game: GameState,
): boolean {
  return (
    continueTurn(result, game).matched &&
    result.pieces.every(
      (p) =>
        p.status === "recognized" && p.empty === false && p.pieceId !== null,
    ) &&
    Object.values(result.abilities).every(
      (a) => a.status === "recognized" && a.value !== null,
    )
  );
}
export function nextTurnSignature(result: RecognitionResult): string {
  return JSON.stringify([result.board, result.pieces, result.abilities]);
}
