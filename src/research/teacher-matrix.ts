import { applyAction, getAvailableActions } from "../domain/game/actions";
import type { GameState } from "../domain/game/types";
import { getInitialCatalog } from "../domain/pieces/catalog";
import { solveTurn } from "../domain/solver/solver";
import { compareEvaluations } from "../domain/solver/evaluator";
import type { TurnSolution } from "../domain/solver/types";
import { createSyntheticFixture } from "./fixture";
import {
  physicalActionKey,
  transformAuditState,
  verifyCandidatePath,
} from "./teacher-audit";
import { validateQualityProtocol } from "./teacher-quality";

export const MATRIX_EXECUTION = {
  version: "initial-state-matrix-v1",
  actionIndex: 0,
  scope: "solveTurn-current-state",
  useMemoization: true,
  maxAlternatives: 3,
  order: "rotate-six-conditions-by-state-index-plus-repetition",
  warmup: "none",
  heldOut: "NOT_RUN",
  training: "NOT_RUN",
} as const;

export function freezeMatrixProtocol(input: unknown) {
  return {
    proposal: validateQualityProtocol(input),
    execution: structuredClone(MATRIX_EXECUTION),
  };
}
type Variant = "baseline" | "reverse-array" | "cycle-slots";
export type MatrixProbe = {
  split: "train" | "dev";
  seed: number;
  family: "sparse" | "pressure";
  budget: number;
  variant: Variant;
  repetition: number;
  sequence: number;
  state: GameState;
  result: TurnSolution;
  physicalFirst: string | null;
  physicalPath: string[];
  latencyMs: number;
  returnedPaths: number;
  coverage: ReturnType<typeof pathCoverage>;
};

/** Counts only the selected current-turn path. Retention counts icon-transition events. */
export function pathCoverage(state: GameState, result: TurnSolution) {
  const counts = {
    clearedRows: 0,
    acquiredItems: 0,
    retainedItemEvents: 0,
    placements: 0,
    singleCellSpends: 0,
    rerollSpends: 0,
    pendingReroll: false,
  };
  for (const action of result.actions) {
    const t = applyAction(state, action);
    if (!t.ok) throw new Error(t.error.message);
    state = t.state;
    counts.clearedRows += t.info.clearedRows.length;
    counts.acquiredItems += t.info.acquiredItems.length;
    counts.retainedItemEvents += t.info.retainedItems.length;
    counts.placements += Number(action.type === "place-piece");
    counts.singleCellSpends += Number(t.info.spentAbility === "single-cell");
    counts.rerollSpends += Number(t.info.spentAbility === "reroll");
  }
  counts.pendingReroll = state.pendingReroll !== null;
  return counts;
}
const physicalLegal = (state: GameState) => {
  const legal = getAvailableActions(state);
  if (!legal.ok) throw new Error(legal.error.message);
  return JSON.stringify(legal.actions.map(physicalActionKey).sort());
};
const signature = (p: MatrixProbe) =>
  JSON.stringify({
    path: p.physicalPath,
    evaluation: p.result.evaluation,
    search: p.result.search,
    used: p.result.usedAbilities,
    final: {
      board: p.result.finalState.board,
      items: p.result.finalState.hiddenItems,
      abilities: p.result.finalState.abilities,
      remaining: p.result.finalState.remainingPieces
        .map((x) => [x.instanceId, x.piece.id])
        .sort(),
      pending: p.result.finalState.pendingReroll?.instanceId ?? null,
    },
    alternatives: p.result.alternatives.map((c) => ({
      path: c.actions.map(physicalActionKey),
      evaluation: c.evaluation,
      used: c.usedAbilities,
    })),
  });
export function summarizeMatrix(rows: MatrixProbe[]) {
  // Repetitions check determinism and timing; never count them as independent states.
  const unique = rows.filter((p) => p.repetition === 0);
  const groups = [];
  for (const split of ["train", "dev"] as const)
    for (const family of ["sparse", "pressure"] as const)
      for (const budget of [128, 512])
        for (const variant of [
          "baseline",
          "reverse-array",
          "cycle-slots",
        ] as const) {
          const match = (p: MatrixProbe) =>
            p.split === split &&
            p.family === family &&
            p.budget === budget &&
            p.variant === variant;
          const sample = unique.filter(match),
            timings = rows
              .filter(match)
              .map((p) => p.latencyMs)
              .sort((a, b) => a - b);
          if (!sample.length) continue;
          const totals = sample.reduce(
            (a, p) => {
              for (const k of [
                "clearedRows",
                "acquiredItems",
                "retainedItemEvents",
                "placements",
                "singleCellSpends",
                "rerollSpends",
              ] as const)
                a[k] += p.coverage[k];
              return a;
            },
            {
              clearedRows: 0,
              acquiredItems: 0,
              retainedItemEvents: 0,
              placements: 0,
              singleCellSpends: 0,
              rerollSpends: 0,
            },
          );
          groups.push({
            split,
            family,
            budget,
            variant,
            states: sample.length,
            incomplete: sample.filter((p) => !p.result.search.searchComplete)
              .length,
            pendingRerollPaths: sample.filter((p) => p.coverage.pendingReroll)
              .length,
            acquisitionPaths: sample.filter((p) => p.coverage.acquiredItems > 0)
              .length,
            retentionPaths: sample.filter(
              (p) => p.coverage.retainedItemEvents > 0,
            ).length,
            singleSpendPaths: sample.filter(
              (p) => p.coverage.singleCellSpends > 0,
            ).length,
            rerollSpendPaths: sample.filter((p) => p.coverage.rerollSpends > 0)
              .length,
            totals,
            timing: {
              samples: timings.length,
              meanMs: timings.reduce((a, b) => a + b, 0) / timings.length,
              p95Ms: timings[Math.ceil(timings.length * 0.95) - 1],
              minMs: timings[0],
              maxMs: timings.at(-1)!,
            },
          });
        }
  const comparisons = [128, 512].map((budget) => {
    const baselines = unique.filter(
      (p) => p.budget === budget && p.variant === "baseline",
    );
    const pairs = baselines.map((b) => {
      const c = unique.find(
        (p) =>
          p.seed === b.seed &&
          p.family === b.family &&
          p.budget === budget &&
          p.variant === "cycle-slots",
      )!;
      return {
        changed: c.physicalFirst !== b.physicalFirst,
        comparison: compareEvaluations(
          c.result.evaluation,
          b.result.evaluation,
        ),
      };
    });
    return {
      budget,
      states: pairs.length,
      changedFirst: pairs.filter((p) => p.changed).length,
      better: pairs.filter((p) => p.comparison > 0).length,
      worse: pairs.filter((p) => p.comparison < 0).length,
      equal: pairs.filter((p) => p.comparison === 0).length,
      changedFirstEqualEvaluation: pairs.filter(
        (p) => p.changed && p.comparison === 0,
      ).length,
    };
  });
  const budgetComparisons = (
    ["baseline", "reverse-array", "cycle-slots"] as const
  ).map((variant) => {
    const pairs = unique
      .filter((p) => p.budget === 128 && p.variant === variant)
      .map((low) => {
        const high = unique.find(
          (p) =>
            p.budget === 512 &&
            p.variant === variant &&
            p.seed === low.seed &&
            p.family === low.family,
        )!;
        return {
          changed: high.physicalFirst !== low.physicalFirst,
          comparison: compareEvaluations(
            high.result.evaluation,
            low.result.evaluation,
          ),
        };
      });
    return {
      variant,
      states: pairs.length,
      changedFirst: pairs.filter((p) => p.changed).length,
      better: pairs.filter((p) => p.comparison > 0).length,
      worse: pairs.filter((p) => p.comparison < 0).length,
      equal: pairs.filter((p) => p.comparison === 0).length,
    };
  });
  return {
    states: unique.length / 6,
    probes: rows.length,
    uniqueConditions: unique.length,
    repeatedConditionsChecked: unique.length,
    arrayControlsChecked: unique.length / 3,
    returnedPathsVerified: rows.reduce((a, p) => a + p.returnedPaths, 0),
    generatedReservedTestStates: 0,
    trainingRuns: 0,
    comparisons,
    budgetComparisons,
    groups,
  };
}

export function runTeacherMatrix(
  input: unknown,
  clock: () => number,
  progress?: (row: MatrixProbe) => void,
) {
  const protocol = freezeMatrixProtocol(input),
    p = protocol.proposal,
    rows: MatrixProbe[] = [];
  let stateIndex = 0;
  for (const split of ["train", "dev"] as const)
    for (const seed of p.splits[split])
      for (const family of p.families) {
        const fixture = createSyntheticFixture(seed, family, 1);
        if (!fixture.ok) throw new Error(fixture.error);
        const inputState = fixture.value.initialState,
          before = JSON.stringify(fixture.value),
          legal = physicalLegal(inputState);
        const conditions = p.nodeBudgets.flatMap((budget) =>
          p.slotVariants.map((variant) => ({ budget, variant })),
        );
        const stateRows: MatrixProbe[] = [];
        for (let repetition = 0; repetition < p.repetitions; repetition++) {
          const offset = (stateIndex + repetition) % conditions.length;
          for (let i = 0; i < conditions.length; i++) {
            const { budget, variant } =
              conditions[(i + offset) % conditions.length];
            const state = transformAuditState(inputState, variant),
              stateBefore = JSON.stringify(state);
            if (physicalLegal(state) !== legal)
              throw new Error("Physical legal list changed");
            const start = clock(),
              solved = solveTurn(state, getInitialCatalog(), {
                maxNodes: budget,
                maxAlternatives: 3,
                useMemoization: true,
              }),
              end = clock();
            if (!solved.ok) throw new Error(solved.error.message);
            const latencyMs = end - start;
            if (
              ![start, end, latencyMs].every(Number.isFinite) ||
              latencyMs < 0
            )
              throw new Error("Invalid probe clock");
            const result = solved.result;
            if (
              result.search.visitedNodes > budget ||
              result.search.maxNodes !== budget
            )
              throw new Error("Node budget mismatch");
            [result, ...result.alternatives].forEach((c) =>
              verifyCandidatePath(state, c),
            );
            if (
              JSON.stringify(state) !== stateBefore ||
              JSON.stringify(fixture.value) !== before
            )
              throw new Error("Input mutated");
            const row: MatrixProbe = {
              split,
              seed,
              family,
              budget,
              variant,
              repetition,
              sequence: rows.length,
              state,
              result,
              physicalFirst: result.actions.length
                ? physicalActionKey(result.actions[0])
                : null,
              physicalPath: result.actions.map(physicalActionKey),
              latencyMs,
              returnedPaths: 1 + result.alternatives.length,
              coverage: pathCoverage(state, result),
            };
            stateRows.push(row);
            rows.push(row);
            progress?.(structuredClone(row));
          }
        }
        for (const budget of p.nodeBudgets) {
          const first = stateRows.find(
            (r) =>
              r.budget === budget &&
              r.variant === "baseline" &&
              r.repetition === 0,
          )!;
          for (const row of stateRows.filter((r) => r.budget === budget)) {
            if (
              row.variant === "reverse-array" &&
              signature(row) !== signature(first)
            )
              throw new Error("Array-order control changed solver result");
            const original = stateRows.find(
              (r) =>
                r.budget === budget &&
                r.variant === row.variant &&
                r.repetition === 0,
            )!;
            if (signature(row) !== signature(original))
              throw new Error("Repeated probe changed solver result");
          }
        }
        stateIndex++;
      }
  return {
    status: "PASS" as const,
    protocol,
    rows,
    summary: summarizeMatrix(rows),
  };
}
