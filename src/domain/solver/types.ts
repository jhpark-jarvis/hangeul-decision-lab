import type {
  GameAction,
  AbilityCounts,
  GameState,
  GamePhase,
  GameError,
} from "../game/types";
import type { Rotation } from "../pieces/types";
import type { MobilityScore, MobilityError } from "./mobility";

export type OrdinarySolverAction = Extract<
  GameAction,
  { type: "place-piece" }
> & {
  rotation: Rotation;
  flipped: boolean;
};
export type RerollReason =
  | "blocked-piece"
  | "initial-ordinary-probe-no-complete-path"
  | "zero-catalog-mobility";
export type SolverAction =
  | OrdinarySolverAction
  | Extract<GameAction, { type: "single-cell" }>
  | (Extract<GameAction, { type: "reroll" }> & { reason: RerollReason });
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
export const ORDINARY_PROBE_BUDGET_DIVISOR = 4;
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
export type SolverCandidate<Action extends SolverAction = SolverAction> = {
  actions: Action[];
  evaluation: SolverEvaluation;
  finalState: GameState;
  usedAbilities: AbilityCounts;
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
  scope: "ordinary-pieces" | "pieces-and-abilities";
  searchComplete: boolean;
  optimalWithinScope: boolean;
  specialAbilitiesSearched: boolean;
  alternativesMayOmitEquivalentPaths: boolean;
  stopReason:
    "exhausted" | "node-budget" | "await-next-pieces" | "await-reroll-result";
  visitedNodes: number;
  memoPrunedNodes: number;
  memoEntries: number;
  discoveredCompletePaths: number;
  evaluatedCandidates: number;
  maxNodes: number;
  abilitySearch?: {
    singleCellCandidates: "all-legal-cells";
    rerollCandidatesRestricted: true;
    excludedRerollTargets: number;
    rerollFutureEvaluated: false;
    maxActionDepth: number;
    ordinaryProbe: {
      visitedNodes: number;
      searchComplete: boolean;
      discoveredCompletePaths: number;
      noCompletePathProven: boolean;
    };
  };
};
export type TurnSolution<Action extends SolverAction = SolverAction> =
  SolverCandidate<Action> & {
    alternatives: SolverCandidate<Action>[];
    search: SearchInfo;
  };
