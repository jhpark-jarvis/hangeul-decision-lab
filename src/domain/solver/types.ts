import type {
  GameAction,
  AbilityCounts,
  GameState,
  GamePhase,
  GameError,
} from "../game/types";
import type { Rotation } from "../pieces/types";
import type { MobilityScore, MobilityError } from "./mobility";

export type SolverAction = Extract<GameAction, { type: "place-piece" }> & {
  rotation: Rotation;
  flipped: boolean;
};
export type SolverConfig = {
  maxNodes: number;
  maxAlternatives: number;
  useMemoization: boolean;
};
export const DEFAULT_SOLVER_CONFIG: Readonly<SolverConfig> = {
  maxNodes: 512,
  maxAlternatives: 3,
  useMemoization: true,
};
export type SolverEvaluation = {
  survivable: boolean;
  allCurrentPiecesPlaced: boolean;
  phase: GamePhase;
  clearedRows: number;
  acquiredItems: number;
  mobility: MobilityScore;
  remainingAbilities: AbilityCounts;
  isolatedEmptyCells: number;
  unfillableGaps: number;
  emptyCells: number;
};
export type PathRewards = { clearedRows: number; acquiredItems: number };
export type SolverCandidate = {
  actions: SolverAction[];
  evaluation: SolverEvaluation;
  finalState: GameState;
};
export type SolverError = {
  code:
    | GameError["code"]
    | MobilityError["code"]
    | "INVALID_SOLVER_CONFIG"
    | "INVALID_REWARDS";
  message: string;
};
export type SolverResult<T> =
  ({ ok: true } & T) | { ok: false; error: SolverError };
export type SearchInfo = {
  scope: "ordinary-pieces";
  searchComplete: boolean;
  optimalWithinScope: boolean;
  specialAbilitiesSearched: false;
  alternativesMayOmitEquivalentPaths: boolean;
  stopReason:
    "exhausted" | "node-budget" | "await-next-pieces" | "await-reroll-result";
  visitedNodes: number;
  memoPrunedNodes: number;
  memoEntries: number;
  discoveredCompletePaths: number;
  evaluatedCandidates: number;
  maxNodes: number;
};
export type TurnSolution = SolverCandidate & {
  alternatives: SolverCandidate[];
  search: SearchInfo;
};
