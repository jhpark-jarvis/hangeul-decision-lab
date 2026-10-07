import { applyAction, getAvailableActions } from "../domain/game/actions";
import {
  enterNextPieces,
  resolveReroll,
  validateGameState,
} from "../domain/game/game-state";
import type { GameState } from "../domain/game/types";
import { getInitialCatalog } from "../domain/pieces/catalog";
import { suppliedPieces, validateFixture } from "./fixture";
import type { EpisodePolicy, EpisodeResult, ResearchResult } from "./types";

/** Canonical logical state key, not a cryptographic digest or a latency measurement. */
export function stateKey(state: GameState): string {
  return JSON.stringify({
    board: state.board.map((row) => row.map(Number).join("")).join(""),
    pieces: [...state.remainingPieces]
      .sort((a, b) => a.pieceIndex - b.pieceIndex)
      .map((p) => [
        p.instanceId,
        p.pieceIndex,
        p.piece.id,
        p.piece.shape,
        p.piece.name,
      ]),
    items: [...state.hiddenItems]
      .sort((a, b) => a.row - b.row || a.col - b.col)
      .map((i) => [i.row, i.col, i.type]),
    abilities: [state.abilities.singleCell, state.abilities.reroll],
    pending: state.pendingReroll,
  });
}

export function runEpisode(
  input: unknown,
  policy: EpisodePolicy,
): ResearchResult<EpisodeResult> {
  const validated = validateFixture(input);
  if (!validated.ok) return validated;
  const fixture = validated.value;
  let state = fixture.initialState;
  let setIndex = 0;
  const result: EpisodeResult = {
    schemaVersion: 1,
    fixtureId: fixture.id,
    events: [],
    finalState: state,
    ending: { kind: "horizon", detail: "Action horizon reached" },
    totals: {
      actions: 0,
      placements: 0,
      clearedRows: 0,
      acquiredItems: 0,
      singleCellUses: 0,
      rerollUses: 0,
      completedSets: 0,
      suppliedSets: 0,
    },
  };
  function finish(
    kind: EpisodeResult["ending"]["kind"],
    detail: string,
  ): ResearchResult<EpisodeResult> {
    result.ending = { kind, detail };
    result.finalState = state;
    return { ok: true, value: result };
  }
  while (true) {
    const available = getAvailableActions(state);
    if (!available.ok) return finish("error", available.error.message);
    if (available.phase === "gameover")
      return finish("gameover", "No domain-legal action");
    if (result.totals.actions >= fixture.maxActions)
      return finish("horizon", "Action horizon reached");
    if (available.phase === "await-next-pieces") {
      const ids = fixture.nextPieceSets[setIndex];
      if (!ids) return finish("tape-exhausted", "Next-piece tape exhausted");
      const before = stateKey(state);
      const supplied = enterNextPieces(
        state,
        suppliedPieces(fixture.id, ++setIndex, ids),
      );
      if (!supplied.ok) return finish("error", supplied.error.message);
      state = supplied.state;
      result.totals.suppliedSets++;
      result.events.push({
        type: "next-pieces",
        actionIndex: result.totals.actions,
        before,
        after: stateKey(state),
        suppliedIds: [...ids],
      });
      continue;
    }
    if (available.phase === "await-reroll-result") {
      const target = state.remainingPieces.find(
        (p) => p.instanceId === state.pendingReroll!.instanceId,
      )!;
      const index = Math.max(0, result.totals.actions - 1);
      const ticket = fixture.rerollTickets[index]?.[target.pieceIndex];
      if (ticket === undefined)
        return finish("tape-exhausted", "Reroll tape exhausted");
      const choices = getInitialCatalog().filter(
        (p) => p.id !== target.piece.id,
      );
      const piece = choices[Math.floor((ticket / 2 ** 32) * choices.length)];
      const before = stateKey(state);
      const supplied = resolveReroll(state, piece);
      if (!supplied.ok) return finish("error", supplied.error.message);
      state = supplied.state;
      result.events.push({
        type: "reroll-result",
        actionIndex: result.totals.actions,
        before,
        after: stateKey(state),
        suppliedIds: [piece.id],
      });
      continue;
    }
    const observationState = validateGameState(state);
    if (!observationState.ok)
      return finish("error", observationState.error.message);
    const before = stateKey(state);
    let decision;
    try {
      decision = policy({
        state: observationState.state,
        legalActions: available.actions,
        actionIndex: result.totals.actions,
      });
    } catch {
      return finish("error", "Policy threw an exception");
    }
    const checkedObservation = validateGameState(observationState.state);
    if (!checkedObservation.ok || stateKey(checkedObservation.state) !== before)
      return finish("error", "Policy mutated observation");
    if (!decision || typeof decision !== "object")
      return finish("error", "Invalid policy decision");
    if (decision.action === null)
      return finish(
        "policy-abstention",
        "Legal actions remain; policy returned no action",
      );
    let search: typeof decision.search;
    try {
      if (decision.search)
        search = JSON.parse(
          JSON.stringify(decision.search, (_key, value) => {
            if (
              (typeof value === "number" && !Number.isFinite(value)) ||
              typeof value === "function"
            )
              throw new Error("Invalid search metadata");
            return value;
          }),
        );
    } catch {
      return finish("error", "Invalid policy diagnostics");
    }
    const applied = applyAction(state, decision.action);
    if (!applied.ok) return finish("error", applied.error.message);
    state = applied.state;
    const info = applied.info;
    const selected = decision.action;
    const action =
      selected.type === "place-piece"
        ? {
            type: selected.type,
            instanceId: selected.instanceId,
            pieceIndex: selected.pieceIndex,
            pieceId: selected.pieceId,
            variant: selected.variant.map((row) => [...row]),
            row: selected.row,
            col: selected.col,
          }
        : selected.type === "reroll"
          ? {
              type: selected.type,
              instanceId: selected.instanceId,
              pieceIndex: selected.pieceIndex,
            }
          : { type: selected.type, row: selected.row, col: selected.col };
    result.events.push({
      type: "action",
      index: result.totals.actions,
      before,
      after: stateKey(state),
      action,
      info,
      ...(search ? { search } : {}),
    });
    result.totals.actions++;
    result.totals.placements += info.consumedPiece ? 1 : 0;
    result.totals.clearedRows += info.clearedRows.length;
    result.totals.acquiredItems += info.acquiredItems.length;
    result.totals.singleCellUses += info.spentAbility === "single-cell" ? 1 : 0;
    result.totals.rerollUses += info.spentAbility === "reroll" ? 1 : 0;
    if (info.consumedPiece && !state.remainingPieces.length)
      result.totals.completedSets++;
  }
}

export function verifyReplay(
  fixture: unknown,
  recorded: EpisodeResult,
): ResearchResult<true> {
  if (
    !recorded ||
    recorded.schemaVersion !== 1 ||
    !Array.isArray(recorded.events) ||
    recorded.ending?.kind === "error"
  )
    return { ok: false, error: "Invalid/non-replayable episode" };
  const actions = recorded.events.filter((event) => event.type === "action");
  let index = 0;
  const replayed = runEpisode(fixture, () => {
    const event = actions[index++];
    return event
      ? { action: event.action, search: event.search }
      : { action: null };
  });
  if (!replayed.ok) return replayed;
  return JSON.stringify(replayed.value) === JSON.stringify(recorded)
    ? { ok: true, value: true }
    : { ok: false, error: "Replay action/environment/state/result mismatch" };
}
