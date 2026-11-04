import { applyAction, getAvailableActions } from "../domain/game/actions";
import type { GameState } from "../domain/game/types";
import { actionOutcome } from "./action-outcome";
import { actionKey } from "./imitation";
import { aliasItemIdentity } from "./alias-item-identity";

export type StoredAliasCase = {
  state: GameState;
  candidateKeys: string[];
  members: number[];
  expectedSurvivors: number[];
  outcomes: number[][];
  label: number;
};

function requireValue(condition: boolean, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

/** Exactly two existing transition calls per member; no search or new targets. */
export function replayStoredAlias(
  input: StoredAliasCase,
  reserve: (member: number, kind: "direct" | "outcome") => void,
) {
  const before = JSON.stringify(input);
  const root = getAvailableActions(input.state);
  requireValue(root.ok, "Invalid original state");
  requireValue(root.phase === "playing", "Original phase");
  requireValue(
    JSON.stringify(root.actions.map(actionKey)) ===
      JSON.stringify(input.candidateKeys),
    "Original ordered legal mask mismatch",
  );
  const rows = input.members.map((member, i) => {
    const action = root.actions[member];
    requireValue(
      action?.type === "single-cell",
      "Existing point candidate only",
    );
    reserve(member, "direct");
    const next = applyAction(input.state, action);
    requireValue(next.ok, "Direct transition failed");
    requireValue(
      next.info.clearedRows.length === 0 &&
        next.info.acquiredItems.length === 0 &&
        next.state.abilities.singleCell === 0 &&
        next.state.abilities.reroll === 0 &&
        next.state.pendingReroll === null &&
        JSON.stringify(next.state.remainingPieces) ===
          JSON.stringify(input.state.remainingPieces),
      "Nonclearing/no-acquisition class mismatch",
    );
    requireValue(
      aliasItemIdentity(next.state.hiddenItems) ===
        aliasItemIdentity(input.state.hiddenItems),
      "Hidden item identities changed",
    );
    const available = getAvailableActions(next.state);
    requireValue(available.ok, "Next mask failed");
    requireValue(
      available.actions.every((a) => a.type === "place-piece"),
      "Unexpected remaining ability",
    );
    const ordinaryPlacements = available.actions.length;
    requireValue(
      ordinaryPlacements === input.expectedSurvivors[i] &&
        available.phase === (ordinaryPlacements > 0 ? "playing" : "gameover"),
      "Stored mask survivor prediction mismatch",
    );
    reserve(member, "outcome");
    const actual = actionOutcome(input.state, action).map(Math.fround);
    requireValue(
      actual.length === 22 &&
        JSON.stringify(actual) ===
          JSON.stringify(input.outcomes[i].map(Math.fround)) &&
        JSON.stringify(actual) ===
          JSON.stringify(input.outcomes[0].map(Math.fround)),
      "Stored FP32 outcome mismatch",
    );
    requireValue(JSON.stringify(input) === before, "Original input mutated");
    return {
      member,
      target: member === input.label,
      earliest: i === 0,
      nextPhase: available.phase,
      ordinaryPlacements,
      outcomeExact: true,
      inputUnchanged: true,
    };
  });
  return rows;
}
