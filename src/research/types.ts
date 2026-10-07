import type {
  GameAction,
  GameState,
  TransitionInfo,
} from "../domain/game/types";
import type { SearchInfo } from "../domain/solver/types";

export const SYNTHETIC_PROFILE = "equal-pieces-initial-items-only-v1";
export const RNG_VERSION = "indexed-mix32-v1";
export const RULES_VERSION = "domain-row-clear-no-gravity-retained-items-v1";
export type EpisodeFixture = {
  schemaVersion: 1;
  id: string;
  provenance: "synthetic-generated" | "synthetic-scripted";
  catalogVersion: string;
  rulesVersion: typeof RULES_VERSION;
  profile: typeof SYNTHETIC_PROFILE;
  rngVersion: typeof RNG_VERSION;
  seed: number;
  maxActions: number;
  initialState: GameState;
  nextPieceSets: [string, string, string][];
  rerollTickets: [number, number, number][];
};
export type PolicyObservation = {
  state: GameState;
  legalActions: GameAction[];
  actionIndex: number;
};
export type PolicyDecision = { action: GameAction | null; search?: SearchInfo };
export type EpisodePolicy = (observation: PolicyObservation) => PolicyDecision;
export type EpisodeEnding = {
  kind:
    "gameover" | "horizon" | "policy-abstention" | "tape-exhausted" | "error";
  detail: string;
};
export type EpisodeEvent =
  | {
      type: "action";
      index: number;
      before: string;
      after: string;
      action: GameAction;
      info: TransitionInfo;
      search?: SearchInfo;
    }
  | {
      type: "next-pieces" | "reroll-result";
      actionIndex: number;
      before: string;
      after: string;
      suppliedIds: string[];
    };
export type EpisodeResult = {
  schemaVersion: 1;
  fixtureId: string;
  events: EpisodeEvent[];
  finalState: GameState;
  ending: EpisodeEnding;
  totals: {
    actions: number;
    placements: number;
    clearedRows: number;
    acquiredItems: number;
    singleCellUses: number;
    rerollUses: number;
    completedSets: number;
    suppliedSets: number;
  };
};
export type ResearchResult<T> =
  { ok: true; value: T } | { ok: false; error: string };
