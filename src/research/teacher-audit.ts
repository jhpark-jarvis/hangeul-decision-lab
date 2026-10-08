import type { GameAction, GameState, PieceIndex } from "../domain/game/types";
import { getAvailableActions, applyAction } from "../domain/game/actions";
import { validateGameState } from "../domain/game/game-state";
import { getInitialCatalog } from "../domain/pieces/catalog";
import { solveTurn } from "../domain/solver/solver";
import { evaluateState, compareEvaluations } from "../domain/solver/evaluator";
import type {
  SolverCandidate,
  SolverConfig,
  TurnSolution,
} from "../domain/solver/types";
import {
  actionKey,
  encodeObservation,
  type ModelExample,
  type DatasetPack,
  MODEL_SCHEMA,
  SPLITS,
} from "./imitation";
import { stateKey } from "./episode";

export type AuditProtocol = {
  schemaVersion: 1;
  id: string;
  trainSeeds: number[];
  devSeeds: number[];
  families: ("sparse" | "pressure")[];
  actionIndex: 0;
  solver: SolverConfig;
  variants: ["baseline", "reverse-array", "cycle-slots"];
};
export type AuditRow = {
  split: "train" | "dev";
  seed: number;
  family: "sparse" | "pressure";
  exampleId: string;
  probes: {
    variant: AuditProtocol["variants"][number];
    state: GameState;
    result: TurnSolution;
    physicalFirst: string;
    physicalPath: string[];
    latencyMs: number;
    returnedPaths: number;
    distinctReturnedBestTieRoots: number;
  }[];
  cycleChangedFirst: boolean;
  cycleEvaluationComparison: number;
};
export function validateAuditProtocol(
  input: unknown,
  pack: DatasetPack,
): AuditProtocol {
  const p = input as AuditProtocol;
  if (
    !p ||
    p.schemaVersion !== 1 ||
    typeof p.id !== "string" ||
    !/^[a-z0-9-]+$/.test(p.id) ||
    p.actionIndex !== 0 ||
    JSON.stringify(p.variants) !==
      JSON.stringify(["baseline", "reverse-array", "cycle-slots"]) ||
    !Array.isArray(p.families) ||
    !p.families.length ||
    new Set(p.families).size !== p.families.length ||
    p.families.some((f) => !pack.spec.families.includes(f)) ||
    !p.solver ||
    JSON.stringify(p.solver) !== JSON.stringify(pack.spec.teacher)
  )
    throw new Error("Invalid audit protocol");
  const seen = new Set<number>();
  for (const split of ["train", "dev"] as const) {
    const group = p[`${split}Seeds`];
    if (
      !Array.isArray(group) ||
      !group.length ||
      group.some(
        (seed) =>
          !Number.isInteger(seed) ||
          !pack.spec[`${split}Seeds`].includes(seed) ||
          seen.has(seed),
      )
    )
      throw new Error("Audit seeds must be disjoint train/dev subsets");
    for (const seed of group) {
      if (seen.has(seed)) throw new Error("Duplicate audit seed");
      seen.add(seed);
    }
  }
  if (seen.size * p.families.length > 16 || p.solver.maxNodes > 512)
    throw new Error("Audit budget exceeded");
  return structuredClone(p);
}
/** Remove only the display slot; retain physical instance identity. */
export function physicalActionKey(action: GameAction): string {
  if (action.type === "place-piece")
    return JSON.stringify([
      action.type,
      action.instanceId,
      action.pieceId,
      action.row,
      action.col,
      action.variant,
    ]);
  if (action.type === "single-cell")
    return JSON.stringify([action.type, action.row, action.col]);
  return JSON.stringify([action.type, action.instanceId]);
}
export function transformAuditState(
  input: GameState,
  variant: "baseline" | "reverse-array" | "cycle-slots",
): GameState {
  const valid = validateGameState(input);
  if (!valid.ok) throw new Error(valid.error.message);
  const state = structuredClone(valid.state);
  if (variant === "reverse-array") state.remainingPieces.reverse();
  if (variant === "cycle-slots") {
    state.remainingPieces.forEach((p) => {
      p.pieceIndex = ((p.pieceIndex + 1) % 3) as PieceIndex;
    });
    if (state.pendingReroll)
      state.pendingReroll.pieceIndex = ((state.pendingReroll.pieceIndex + 1) %
        3) as PieceIndex;
  }
  return state;
}
const legalKeys = (state: GameState) => {
  const result = getAvailableActions(state);
  if (!result.ok) throw new Error(result.error.message);
  return result.actions.map(physicalActionKey).sort();
};
export function verifyCandidatePath(
  input: GameState,
  candidate: SolverCandidate,
): void {
  let state = structuredClone(input),
    clearedRows = 0,
    acquiredItems = 0,
    singleCell = 0,
    reroll = 0;
  for (const action of candidate.actions) {
    const transition = applyAction(state, action);
    if (!transition.ok) throw new Error(transition.error.message);
    state = transition.state;
    clearedRows += transition.info.clearedRows.length;
    acquiredItems += transition.info.acquiredItems.length;
    singleCell += Number(action.type === "single-cell");
    reroll += Number(action.type === "reroll");
  }
  const evaluated = evaluateState(state, getInitialCatalog(), {
    clearedRows,
    acquiredItems,
  });
  if (
    !evaluated.ok ||
    JSON.stringify(evaluated.evaluation) !==
      JSON.stringify(candidate.evaluation) ||
    stateKey(state) !== stateKey(candidate.finalState) ||
    candidate.usedAbilities.singleCell !== singleCell ||
    candidate.usedAbilities.reroll !== reroll
  )
    throw new Error("Returned path/evaluation mismatch");
}
export function describeExamples(examples: ModelExample[]) {
  const count = examples.length;
  let normalizedRank = 0,
    first = 0,
    minSlot = 0,
    placements = 0;
  const actionTypes = { "place-piece": 0, "single-cell": 0, reroll: 0 };
  for (const e of examples) {
    if (
      !Number.isInteger(e.label) ||
      e.label < 0 ||
      e.label >= e.candidateKeys.length
    )
      throw new Error("Invalid label");
    const type = JSON.parse(
      e.candidateKeys[e.label],
    )[0] as keyof typeof actionTypes;
    if (!(type in actionTypes)) throw new Error("Unknown action type");
    actionTypes[type]++;
    first += Number(e.label === 0);
    normalizedRank +=
      e.candidateKeys.length <= 1 ? 0 : e.label / (e.candidateKeys.length - 1);
    if (type === "place-piece") {
      placements++;
      const slot = JSON.parse(e.candidateKeys[e.label])[2];
      minSlot += Number(
        slot === Math.min(...e.state.remainingPieces.map((p) => p.pieceIndex)),
      );
    }
  }
  return {
    count,
    firstLegalLabels: first,
    firstLegalFraction: count ? first / count : null,
    meanNormalizedLabelRank: count ? normalizedRank / count : null,
    actionTypes,
    placementLabels: placements,
    minimumRemainingSlotLabels: minSlot,
    minimumRemainingSlotFraction: placements ? minSlot / placements : null,
  };
}
export function describeDataset(pack: DatasetPack) {
  if (
    pack.status !== "PASS" ||
    JSON.stringify(pack.schema) !== JSON.stringify(MODEL_SCHEMA)
  )
    throw new Error("Dataset schema/status mismatch");
  const observationOwners = new Map<string, Set<string>>();
  const splitDescriptions = Object.fromEntries(
    SPLITS.map((split) => {
      const examples = pack.examples[split];
      const keys = new Set<string>();
      for (const e of examples) {
        const obs = encodeObservation(e.state);
        if (
          !obs.ok ||
          JSON.stringify(obs.value) !== JSON.stringify(e.observation)
        )
          throw new Error("Model observation mismatch");
        const key = JSON.stringify(obs.value);
        keys.add(key);
        if (!observationOwners.has(key)) observationOwners.set(key, new Set());
        observationOwners.get(key)!.add(split);
      }
      return [
        split,
        {
          all: describeExamples(examples),
          complete: describeExamples(
            examples.filter((e) => e.teacher.search.searchComplete),
          ),
          incomplete: describeExamples(
            examples.filter((e) => !e.teacher.search.searchComplete),
          ),
          families: Object.fromEntries(
            pack.spec.families.map((family) => [
              family,
              describeExamples(
                examples.filter((e) =>
                  pack.episodes.some(
                    (ep) =>
                      ep.fixture.id === e.fixtureId &&
                      ep.fixture.id.includes(family),
                  ),
                ),
              ),
            ]),
          ),
          uniqueModelObservations: keys.size,
          repeatedModelObservations: examples.length - keys.size,
          acquiredItems: pack.episodes
            .filter((e) => e.split === split)
            .reduce((a, e) => a + e.result.totals.acquiredItems, 0),
        },
      ];
    }),
  );
  const overlap = Object.fromEntries(
    [
      ["train", "dev"],
      ["train", "test"],
      ["dev", "test"],
    ].map(([a, b]) => [
      `${a}-${b}`,
      [...observationOwners.values()].filter(
        (owners) => owners.has(a) && owners.has(b),
      ).length,
    ]),
  );
  return { splits: splitDescriptions, exactModelObservationOverlap: overlap };
}
export function runTeacherAudit(
  pack: DatasetPack,
  input: unknown,
  clock: () => number,
  progress?: (n: number, row: AuditRow) => void,
) {
  const protocol = validateAuditProtocol(input, pack);
  const descriptions = describeDataset(pack);
  const rows: AuditRow[] = [];
  for (const split of ["train", "dev"] as const)
    for (const seed of protocol[`${split}Seeds`])
      for (const family of protocol.families) {
        const fixture = pack.episodes.find(
          (e) =>
            e.split === split &&
            e.fixture.seed === seed &&
            e.fixture.id.includes(family),
        );
        const example =
          fixture &&
          pack.examples[split].find(
            (e) => e.fixtureId === fixture.fixture.id && e.actionIndex === 0,
          );
        if (!example) throw new Error("Missing preselected initial example");
        const before = JSON.stringify(example);
        const probes = protocol.variants.map((variant) => {
          const state = transformAuditState(example.state, variant);
          if (
            JSON.stringify(legalKeys(state)) !==
            JSON.stringify(legalKeys(example.state))
          )
            throw new Error("Physical legal list changed");
          const start = clock();
          const solved = solveTurn(state, getInitialCatalog(), protocol.solver);
          const latencyMs = clock() - start;
          if (!solved.ok || !Number.isFinite(latencyMs) || latencyMs < 0)
            throw new Error("Invalid probe/clock");
          const result = solved.result;
          if (!result.actions.length) throw new Error("Probe abstained");
          const returned = [result, ...result.alternatives];
          returned.forEach((c) => verifyCandidatePath(state, c));
          const tiedRoots = new Set(
            returned
              .filter(
                (c) =>
                  c.actions.length &&
                  compareEvaluations(c.evaluation, result.evaluation) === 0,
              )
              .map((c) => physicalActionKey(c.actions[0])),
          );
          return {
            variant,
            state,
            result,
            physicalFirst: physicalActionKey(result.actions[0]),
            physicalPath: result.actions.map(physicalActionKey),
            latencyMs,
            returnedPaths: returned.length,
            distinctReturnedBestTieRoots: tiedRoots.size,
          };
        });
        const [baseline, reverse, cycle] = probes;
        if (
          actionKey(baseline.result.actions[0]) !==
            example.candidateKeys[example.label] ||
          JSON.stringify(baseline.result.search) !==
            JSON.stringify(example.teacher.search)
        )
          throw new Error("Stored teacher does not match current solver");
        if (
          JSON.stringify(baseline.physicalPath) !==
            JSON.stringify(reverse.physicalPath) ||
          JSON.stringify(baseline.result.evaluation) !==
            JSON.stringify(reverse.result.evaluation) ||
          JSON.stringify(baseline.result.search) !==
            JSON.stringify(reverse.result.search)
        )
          throw new Error("Array-order control changed solver result");
        if (JSON.stringify(example) !== before)
          throw new Error("Audit mutated input");
        rows.push({
          split,
          seed,
          family,
          exampleId: example.id,
          probes,
          cycleChangedFirst: cycle.physicalFirst !== baseline.physicalFirst,
          cycleEvaluationComparison: compareEvaluations(
            cycle.result.evaluation,
            baseline.result.evaluation,
          ),
        });
        progress?.(rows.length, structuredClone(rows[rows.length - 1]));
      }
  return {
    status: "PASS" as const,
    protocol,
    descriptions,
    rows,
    summary: {
      states: rows.length,
      probes: rows.length * 3,
      arrayControlsUnchanged: rows.length,
      cycleChangedFirst: rows.filter((r) => r.cycleChangedFirst).length,
      changedFirstEqualEvaluation: rows.filter(
        (r) => r.cycleChangedFirst && r.cycleEvaluationComparison === 0,
      ).length,
      cycleBetter: rows.filter((r) => r.cycleEvaluationComparison > 0).length,
      cycleWorse: rows.filter((r) => r.cycleEvaluationComparison < 0).length,
      cycleEqual: rows.filter((r) => r.cycleEvaluationComparison === 0).length,
      baselineIncomplete: rows.filter(
        (r) => !r.probes[0].result.search.searchComplete,
      ).length,
      baselineMultipleReturnedTieRoots: rows.filter(
        (r) => r.probes[0].distinctReturnedBestTieRoots > 1,
      ).length,
      returnedPathsVerified: rows.reduce(
        (sum, r) => sum + r.probes.reduce((n, p) => n + p.returnedPaths, 0),
        0,
      ),
    },
  };
}
