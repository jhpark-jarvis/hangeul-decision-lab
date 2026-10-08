import { applyAction, getAvailableActions } from "../domain/game/actions";
import { validateGameState } from "../domain/game/game-state";
import type { GameState, PieceIndex } from "../domain/game/types";
import { getInitialCatalog } from "../domain/pieces/catalog";
import { compareEvaluations } from "../domain/solver/evaluator";
import { solveTurn } from "../domain/solver/solver";
import type { SolverCandidate, TurnSolution } from "../domain/solver/types";
import { stateKey } from "./episode";
import { createSyntheticFixture } from "./fixture";
import { itemCoverageFixtures } from "./item-coverage";
import { physicalActionKey, verifyCandidatePath } from "./teacher-audit";

export const PERMUTATION_EXECUTION = {
  schemaVersion: 1,
  id: "permutation-teacher-20261008-v1",
  status: "FROZEN_DIAGNOSTIC",
  scope: "returned-current-turn-paths-only",
  trainSeeds: [4000, 4001, 4002, 4003],
  devSeeds: [5000, 5001],
  families: ["sparse", "pressure"],
  actionIndex: 0,
  scriptedCases: 8,
  states: 20,
  permutations: ["012", "021", "102", "120", "201", "210"],
  conditions: {
    baseline: { maxNodes: 512 },
    largeSingle: { maxNodes: 3072 },
    permutationUnion: { maxNodesPerCall: 512, callsPerState: 6 },
  },
  perCall: { maxAlternatives: 3, useMemoization: true, sharedMemo: false },
  controls: ["permutationUnionAfterCycle", "baselineAfterReverseArray"],
  repetitions: 2,
  measuredSolveCalls: 600,
  measuredNodeCap: 409600,
  warmupSolveCalls: 8,
  warmupNodeCap: 6656,
  warmupSource: "first-scripted-state",
  executionOrder: "rotate-five-conditions-by-state-index-plus-repetition",
  timeCriterion: "OBSERVATION_ONLY_NOT_EQUAL_WALL_TIME",
  heldOut: "NOT_RUN",
  training: "NOT_RUN",
} as const;
export type DiagnosticProtocol = typeof PERMUTATION_EXECUTION & {
  inputs: { id: string; stateSha256: string }[];
};
export type DiagnosticFixture = {
  id: string;
  stratum: "generated" | "scripted";
  split: "train" | "dev" | null;
  seed: number | null;
  family: "sparse" | "pressure" | null;
  state: GameState;
};
const ordinal = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);
const stable = (value: unknown): string =>
  JSON.stringify(value, (_key, x) =>
    x && typeof x === "object" && !Array.isArray(x)
      ? Object.fromEntries(Object.entries(x).sort(([a], [b]) => ordinal(a, b)))
      : x,
  );
export function validatePermutationProtocol(
  input: unknown,
): DiagnosticProtocol {
  const p = input as DiagnosticProtocol;
  if (
    !p ||
    typeof p !== "object" ||
    Object.keys(p).length !== Object.keys(PERMUTATION_EXECUTION).length + 1 ||
    Object.entries(PERMUTATION_EXECUTION).some(
      ([key, value]) =>
        stable((p as unknown as Record<string, unknown>)[key]) !==
        stable(value),
    ) ||
    !Array.isArray(p.inputs) ||
    p.inputs.length !== 20 ||
    p.inputs.some(
      (x) =>
        !x ||
        Object.keys(x).length !== 2 ||
        typeof x.id !== "string" ||
        !/^[a-f0-9]{64}$/.test(x.stateSha256),
    ) ||
    new Set(p.inputs.map((x) => x.id)).size !== 20
  )
    throw new Error("Invalid permutation diagnostic protocol");
  return structuredClone(p);
}

/** No tapes leave the generator: only the old train/dev initial states are retained. */
export function diagnosticFixtures(): DiagnosticFixture[] {
  const fixtures: DiagnosticFixture[] = [];
  for (const split of ["train", "dev"] as const)
    for (const seed of PERMUTATION_EXECUTION[`${split}Seeds`])
      for (const family of PERMUTATION_EXECUTION.families) {
        const f = createSyntheticFixture(seed, family, 1);
        if (!f.ok) throw new Error(f.error);
        fixtures.push({
          id: `${split}-${seed}-${family}`,
          stratum: "generated",
          split,
          seed,
          family,
          state: f.value.initialState,
        });
      }
  fixtures.push(
    ...itemCoverageFixtures().map((f) => ({
      id: f.id,
      stratum: "scripted" as const,
      split: null,
      seed: null,
      family: null,
      state: f.state,
    })),
  );
  return fixtures;
}
function permutationMap(permutation: string): PieceIndex[] {
  if (!/^[012]{3}$/.test(permutation) || new Set(permutation).size !== 3)
    throw new Error("Invalid slot bijection");
  return [...permutation].map(Number) as PieceIndex[];
}
export function permuteSlots(input: unknown, permutation: string): GameState {
  const map = permutationMap(permutation),
    valid = validateGameState(input);
  if (!valid.ok) throw new Error(valid.error.message);
  const state = structuredClone(valid.state);
  for (const piece of state.remainingPieces)
    piece.pieceIndex = map[piece.pieceIndex];
  if (state.pendingReroll)
    state.pendingReroll.pieceIndex = map[state.pendingReroll.pieceIndex];
  return state;
}
export function restoreCandidate(
  candidate: SolverCandidate,
  permutation: string,
): SolverCandidate {
  const map = permutationMap(permutation),
    inverse = map.map((_, i) => map.indexOf(i as PieceIndex)) as PieceIndex[];
  const restored = candidateOnly(candidate);
  for (const action of restored.actions)
    if (action.type !== "single-cell")
      action.pieceIndex = inverse[action.pieceIndex];
  for (const piece of restored.finalState.remainingPieces)
    piece.pieceIndex = inverse[piece.pieceIndex];
  if (restored.finalState.pendingReroll)
    restored.finalState.pendingReroll.pieceIndex =
      inverse[restored.finalState.pendingReroll.pieceIndex];
  return restored;
}
/** Semantic comparisons collapse identical piece kinds; physical legality never does. */
export function semanticPath(
  input: GameState,
  candidate: SolverCandidate,
): string[] {
  let state = input;
  return candidate.actions.map((action) => {
    const piece =
      action.type === "single-cell"
        ? null
        : state.remainingPieces.find(
            (p) =>
              p.instanceId === action.instanceId &&
              p.pieceIndex === action.pieceIndex,
          );
    if (action.type !== "single-cell" && !piece)
      throw new Error("Missing semantic target");
    const key =
      action.type === "place-piece"
        ? JSON.stringify([
            action.type,
            piece!.piece.id,
            action.variant,
            action.row,
            action.col,
          ])
        : action.type === "reroll"
          ? JSON.stringify([action.type, piece!.piece.id])
          : JSON.stringify([action.type, action.row, action.col]);
    const t = applyAction(state, action);
    if (!t.ok) throw new Error(t.error.message);
    state = t.state;
    return key;
  });
}
export function semanticStateKey(state: GameState): string {
  const pending = state.pendingReroll
    ? state.remainingPieces.find(
        (p) =>
          p.instanceId === state.pendingReroll!.instanceId &&
          p.pieceIndex === state.pendingReroll!.pieceIndex,
      )
    : null;
  if (state.pendingReroll && !pending)
    throw new Error("Missing pending target");
  return stable({
    board: state.board,
    pieces: state.remainingPieces
      .map((p) => JSON.stringify([p.piece.id, p.piece.shape]))
      .sort(ordinal),
    items: [...state.hiddenItems].sort(
      (a, b) => a.row - b.row || a.col - b.col,
    ),
    abilities: state.abilities,
    pending: pending ? [pending.piece.id, pending.piece.shape] : null,
  });
}
export type PathSource = {
  callIndex: number;
  candidateIndex: number;
  permutation: string;
};
type ReturnedPath = { candidate: SolverCandidate; source: PathSource };
const candidateOnly = (c: SolverCandidate): SolverCandidate =>
  structuredClone({
    actions: c.actions,
    evaluation: c.evaluation,
    finalState: c.finalState,
    usedAbilities: c.usedAbilities,
  });
export function selectReturnedPath(
  input: GameState,
  paths: ReturnedPath[],
): ReturnedPath {
  if (!paths.length) throw new Error("No returned paths");
  const ranked = paths.map((p) => ({
    ...p,
    semantic: JSON.stringify(semanticPath(input, p.candidate)),
    physical: JSON.stringify(p.candidate.actions.map(physicalActionKey)),
  }));
  ranked.sort(
    (a, b) =>
      -compareEvaluations(a.candidate.evaluation, b.candidate.evaluation) ||
      ordinal(a.semantic, b.semantic) ||
      ordinal(a.physical, b.physical),
  );
  return {
    candidate: candidateOnly(ranked[0].candidate),
    source: { ...ranked[0].source },
  };
}
const CONDITIONS = [
  "baseline",
  "largeSingle",
  "permutationUnion",
  "permutationUnionAfterCycle",
  "baselineAfterReverseArray",
] as const;
export type Condition = (typeof CONDITIONS)[number];
export type DiagnosticCall = {
  fixtureId: string;
  condition: Condition;
  repetition: number;
  phase: "warmup" | "measured";
  sequence: number;
  permutation: string;
  maxNodes: number;
  state: GameState;
  result: TurnSolution;
  restored: TurnSolution;
  timing: { transformMs: number; solveMs: number; validationMs: number };
};
type Coverage = {
  clearedRows: number;
  acquiredItems: number;
  retainedItemEvents: number;
  placements: number;
  singleCellSpends: number;
  rerollSpends: number;
  pendingReroll: boolean;
};
function coverage(input: GameState, candidate: SolverCandidate): Coverage {
  let state = input;
  const out: Coverage = {
    clearedRows: 0,
    acquiredItems: 0,
    retainedItemEvents: 0,
    placements: 0,
    singleCellSpends: 0,
    rerollSpends: 0,
    pendingReroll: false,
  };
  for (const action of candidate.actions) {
    const t = applyAction(state, action);
    if (!t.ok) throw new Error(t.error.message);
    state = t.state;
    out.clearedRows += t.info.clearedRows.length;
    out.acquiredItems += t.info.acquiredItems.length;
    out.retainedItemEvents += t.info.retainedItems.length;
    out.placements += Number(action.type === "place-piece");
    out.singleCellSpends += Number(t.info.spentAbility === "single-cell");
    out.rerollSpends += Number(t.info.spentAbility === "reroll");
  }
  out.pendingReroll = state.pendingReroll !== null;
  return out;
}
export type DiagnosticRow = {
  fixtureId: string;
  stratum: DiagnosticFixture["stratum"];
  split: DiagnosticFixture["split"];
  condition: Condition;
  repetition: number;
  calls: DiagnosticCall[];
  selected: ReturnedPath;
  semanticPath: string[];
  physicalPath: string[];
  coverage: Coverage;
  uniqueTransformedStates: number;
  latencyMs: number;
  loggingMs: number;
};
const candidateSignature = (c: SolverCandidate) => ({
  actions: c.actions.map(physicalActionKey),
  evaluation: c.evaluation,
  used: c.usedAbilities,
  final: stateKey(c.finalState),
});
const rowSignature = (r: DiagnosticRow) =>
  stable({
    selected: candidateSignature(r.selected.candidate),
    source: r.selected.source,
    calls: r.calls.map((c) => ({
      permutation: c.permutation,
      search: c.restored.search,
      paths: [c.restored, ...c.restored.alternatives].map(candidateSignature),
    })),
    coverage: r.coverage,
  });
const controlSignature = (r: DiagnosticRow) =>
  stable({
    search: r.calls[0].restored.search,
    paths: [r.calls[0].restored, ...r.calls[0].restored.alternatives].map(
      candidateSignature,
    ),
  });
function timing(samples: number[]) {
  const sorted = [...samples].sort((a, b) => a - b);
  return {
    samples: sorted.length,
    meanMs: sorted.reduce((a, b) => a + b, 0) / sorted.length,
    p95Ms: sorted[Math.ceil(sorted.length * 0.95) - 1],
    minMs: sorted[0],
    maxMs: sorted.at(-1)!,
  };
}
export function summarizePermutation(rows: DiagnosticRow[]) {
  const unique = rows.filter((r) => r.repetition === 0);
  const comparisons = (["generated", "scripted"] as const).flatMap((stratum) =>
    (["baseline", "largeSingle"] as const).map((other) => {
      const pairs = unique
        .filter(
          (r) => r.stratum === stratum && r.condition === "permutationUnion",
        )
        .map((c) => {
          const b = unique.find(
            (r) => r.fixtureId === c.fixtureId && r.condition === other,
          )!;
          return {
            comparison: compareEvaluations(
              c.selected.candidate.evaluation,
              b.selected.candidate.evaluation,
            ),
            physicalChanged:
              stable(c.physicalPath[0] ?? null) !==
              stable(b.physicalPath[0] ?? null),
            semanticChanged:
              stable(c.semanticPath[0] ?? null) !==
              stable(b.semanticPath[0] ?? null),
          };
        });
      return {
        stratum,
        comparison: `permutationUnion-vs-${other}`,
        states: pairs.length,
        better: pairs.filter((p) => p.comparison > 0).length,
        worse: pairs.filter((p) => p.comparison < 0).length,
        equal: pairs.filter((p) => p.comparison === 0).length,
        physicalFirstChanged: pairs.filter((p) => p.physicalChanged).length,
        semanticFirstChanged: pairs.filter((p) => p.semanticChanged).length,
      };
    }),
  );
  const groups = (["generated", "scripted"] as const).flatMap((stratum) =>
    CONDITIONS.slice(0, 3).map((condition) => {
      const sample = unique.filter(
        (r) => r.stratum === stratum && r.condition === condition,
      );
      return {
        stratum,
        condition,
        states: sample.length,
        calls: sample.reduce((n, r) => n + r.calls.length, 0),
        incompleteCalls: sample.reduce(
          (n, r) =>
            n + r.calls.filter((c) => !c.restored.search.searchComplete).length,
          0,
        ),
        conditionsWithIncompleteCall: sample.filter((r) =>
          r.calls.some((c) => !c.restored.search.searchComplete),
        ).length,
        selectedSourceCallIncomplete: sample.filter(
          (r) =>
            !r.calls[r.selected.source.callIndex].restored.search
              .searchComplete,
        ).length,
        acquisitionPaths: sample.filter((r) => r.coverage.acquiredItems > 0)
          .length,
        retentionPaths: sample.filter((r) => r.coverage.retainedItemEvents > 0)
          .length,
        singleSpendPaths: sample.filter((r) => r.coverage.singleCellSpends > 0)
          .length,
        pendingRerollPaths: sample.filter((r) => r.coverage.pendingReroll)
          .length,
        abstentions: sample.filter(
          (r) => r.selected.candidate.actions.length === 0,
        ).length,
        visitedNodes: sample.reduce(
          (n, r) =>
            n + r.calls.reduce((m, c) => m + c.restored.search.visitedNodes, 0),
          0,
        ),
        ordinaryProbeNodes: sample.reduce(
          (n, r) =>
            n +
            r.calls.reduce(
              (m, c) =>
                m +
                (c.restored.search.abilitySearch?.ordinaryProbe.visitedNodes ??
                  0),
              0,
            ),
          0,
        ),
        totals: sample.reduce(
          (out, r) => {
            for (const k of [
              "clearedRows",
              "acquiredItems",
              "retainedItemEvents",
              "placements",
              "singleCellSpends",
              "rerollSpends",
            ] as const)
              out[k] += r.coverage[k];
            return out;
          },
          {
            clearedRows: 0,
            acquiredItems: 0,
            retainedItemEvents: 0,
            placements: 0,
            singleCellSpends: 0,
            rerollSpends: 0,
          },
        ),
        timing: timing(
          rows
            .filter((r) => r.stratum === stratum && r.condition === condition)
            .map((r) => r.latencyMs),
        ),
      };
    }),
  );
  const cyclePairs = unique
    .filter((r) => r.condition === "permutationUnion")
    .map((c) => {
      const cycle = unique.find(
        (r) =>
          r.fixtureId === c.fixtureId &&
          r.condition === "permutationUnionAfterCycle",
      )!;
      return {
        physicalFirstChanged: c.physicalPath[0] !== cycle.physicalPath[0],
        semanticFirstSame: c.semanticPath[0] === cycle.semanticPath[0],
        evaluationSame:
          compareEvaluations(
            c.selected.candidate.evaluation,
            cycle.selected.candidate.evaluation,
          ) === 0,
        finalSemanticSame:
          semanticStateKey(c.selected.candidate.finalState) ===
          semanticStateKey(cycle.selected.candidate.finalState),
      };
    });
  const natural = comparisons.find(
    (c) =>
      c.stratum === "generated" &&
      c.comparison === "permutationUnion-vs-largeSingle",
  )!;
  return {
    states: unique.length / 5,
    uniqueConditions: unique.length,
    measuredCalls: rows.reduce((n, r) => n + r.calls.length, 0),
    returnedPathsVerified: rows.reduce(
      (n, r) =>
        n + r.calls.reduce((m, c) => m + 1 + c.restored.alternatives.length, 0),
      0,
    ),
    repeatedConditionsChecked: unique.length,
    arrayControlsChecked: unique.length / 5,
    cycleControls: {
      states: cyclePairs.length,
      semanticFirstSame: cyclePairs.filter((p) => p.semanticFirstSame).length,
      evaluationSame: cyclePairs.filter((p) => p.evaluationSame).length,
      finalSemanticSame: cyclePairs.filter((p) => p.finalSemanticSame).length,
      physicalFirstChanged: cyclePairs.filter((p) => p.physicalFirstChanged)
        .length,
    },
    candidateForUserReview: natural.worse === 0 && natural.better > 0,
    trainingRuns: 0,
    generatedTestStates: 0,
    comparisons,
    groups,
  };
}

export function runPermutationDiagnostic(
  input: unknown,
  clock: () => number,
  checkInput: (
    fixture: DiagnosticFixture,
    canonicalKey: string,
    expectedHash: string,
  ) => void,
  progress?: (
    event:
      | { type: "call"; call: DiagnosticCall }
      | { type: "condition"; row: DiagnosticRow },
  ) => void,
) {
  const protocol = validatePermutationProtocol(input),
    fixtures = diagnosticFixtures();
  if (typeof checkInput !== "function")
    throw new Error("Historical input hash checker required");
  fixtures.forEach((f, i) => {
    if (protocol.inputs[i].id !== f.id)
      throw new Error("Input manifest order mismatch");
    checkInput(
      structuredClone(f),
      stateKey(f.state),
      protocol.inputs[i].stateSha256,
    );
  });
  let previousTime = -Infinity,
    sequence = 0;
  const now = () => {
    const t = clock();
    if (!Number.isFinite(t) || t < previousTime)
      throw new Error("Invalid diagnostic clock");
    previousTime = t;
    return t;
  };
  const legalKey = (state: GameState, permutation: string) => {
    const legal = getAvailableActions(state),
      inverse = permutationMap(permutation);
    if (!legal.ok) throw new Error(legal.error.message);
    return stable(
      legal.actions
        .map((a) => {
          const action = structuredClone(a);
          if (action.type !== "single-cell")
            action.pieceIndex = inverse.indexOf(
              action.pieceIndex,
            ) as PieceIndex;
          return JSON.stringify(action);
        })
        .sort(ordinal),
    );
  };
  const execute = (
    fixture: DiagnosticFixture,
    condition: Condition,
    repetition: number,
    phase: DiagnosticCall["phase"],
  ): DiagnosticRow => {
    const start = now(),
      root = structuredClone(fixture.state),
      before = stable(root),
      legal = legalKey(root, "012");
    const union = condition.startsWith("permutationUnion"),
      budget = condition === "largeSingle" ? 3072 : 512;
    const calls: DiagnosticCall[] = [],
      paths: ReturnedPath[] = [],
      transformedStates = new Set<string>();
    let loggingMs = 0;
    for (const permutation of union ? protocol.permutations : ["012"]) {
      const effective =
        condition === "permutationUnionAfterCycle"
          ? [1, 2, 0].map((i) => permutation[i]).join("")
          : permutation;
      const t0 = now(),
        state = permuteSlots(root, effective);
      if (condition === "baselineAfterReverseArray")
        state.remainingPieces.reverse();
      transformedStates.add(stateKey(state));
      const stateBefore = stable(state);
      if (legalKey(state, effective) !== legal)
        throw new Error("Exact legal bijection mismatch");
      const t1 = now(),
        solved = solveTurn(state, getInitialCatalog(), {
          maxNodes: budget,
          maxAlternatives: 3,
          useMemoization: true,
        }),
        t2 = now();
      if (!solved.ok) throw new Error(solved.error.message);
      if (stable(state) !== stateBefore || stable(root) !== before)
        throw new Error("Diagnostic input mutated");
      const result = solved.result;
      if (
        !Number.isInteger(result.search.visitedNodes) ||
        result.search.visitedNodes < 0 ||
        result.search.visitedNodes > budget ||
        result.search.maxNodes !== budget
      )
        throw new Error("Node budget mismatch");
      const restored: TurnSolution = {
        ...restoreCandidate(result, effective),
        search: structuredClone(result.search),
        alternatives: result.alternatives.map((c) =>
          restoreCandidate(c, effective),
        ),
      };
      [result, ...result.alternatives].forEach((c) =>
        verifyCandidatePath(state, c),
      );
      [restored, ...restored.alternatives].forEach((c, candidateIndex) => {
        verifyCandidatePath(root, c);
        paths.push({
          candidate: c,
          source: {
            callIndex: calls.length,
            candidateIndex,
            permutation: effective,
          },
        });
      });
      const t3 = now(),
        call: DiagnosticCall = {
          fixtureId: fixture.id,
          condition,
          repetition,
          phase,
          sequence: sequence++,
          permutation: effective,
          maxNodes: budget,
          state,
          result,
          restored,
          timing: {
            transformMs: t1 - t0,
            solveMs: t2 - t1,
            validationMs: t3 - t2,
          },
        };
      calls.push(call);
      const logStart = now();
      progress?.({ type: "call", call: structuredClone(call) });
      loggingMs += now() - logStart;
    }
    // A/B retain the actual existing teacher first choice; only C uses the new deterministic tie rule.
    const selected = union
      ? selectReturnedPath(root, paths)
      : {
          candidate: candidateOnly(calls[0].restored),
          source: { callIndex: 0, candidateIndex: 0, permutation: "012" },
        };
    const row: DiagnosticRow = {
      fixtureId: fixture.id,
      stratum: fixture.stratum,
      split: fixture.split,
      condition,
      repetition,
      calls,
      selected,
      semanticPath: semanticPath(root, selected.candidate),
      physicalPath: selected.candidate.actions.map(physicalActionKey),
      coverage: coverage(root, selected.candidate),
      uniqueTransformedStates: transformedStates.size,
      latencyMs: now() - start - loggingMs,
      loggingMs,
    };
    if (row.latencyMs < 0 || !Number.isFinite(row.latencyMs))
      throw new Error("Invalid condition clock");
    progress?.({ type: "condition", row: structuredClone(row) });
    return row;
  };
  const warmup = CONDITIONS.slice(0, 3).map((condition) =>
    execute(fixtures[12], condition, 0, "warmup"),
  );
  const rows: DiagnosticRow[] = [];
  for (const [stateIndex, fixture] of fixtures.entries()) {
    const stateRows: DiagnosticRow[] = [];
    for (let repetition = 0; repetition < 2; repetition++) {
      const offset = (stateIndex + repetition) % 5;
      for (let i = 0; i < 5; i++) {
        const condition = CONDITIONS[(i + offset) % 5],
          row = execute(fixture, condition, repetition, "measured");
        rows.push(row);
        stateRows.push(row);
      }
    }
    for (const condition of CONDITIONS) {
      const pair = stateRows.filter((r) => r.condition === condition);
      if (rowSignature(pair[0]) !== rowSignature(pair[1]))
        throw new Error("Repeated condition mismatch");
    }
    for (let repetition = 0; repetition < 2; repetition++) {
      const get = (condition: Condition) =>
        stateRows.find(
          (r) => r.condition === condition && r.repetition === repetition,
        )!;
      const a = get("baseline"),
        c = get("permutationUnion"),
        cycle = get("permutationUnionAfterCycle");
      if (
        controlSignature(a) !==
        controlSignature(get("baselineAfterReverseArray"))
      )
        throw new Error("Array-order control mismatch");
      if (
        compareEvaluations(
          c.selected.candidate.evaluation,
          a.selected.candidate.evaluation,
        ) < 0
      )
        throw new Error("Union worse than its identity baseline");
      if (
        c.semanticPath[0] !== cycle.semanticPath[0] ||
        compareEvaluations(
          c.selected.candidate.evaluation,
          cycle.selected.candidate.evaluation,
        ) !== 0 ||
        semanticStateKey(c.selected.candidate.finalState) !==
          semanticStateKey(cycle.selected.candidate.finalState)
      )
        throw new Error("Cycle semantic control mismatch");
    }
  }
  const summary = summarizePermutation(rows);
  if (summary.measuredCalls !== protocol.measuredSolveCalls)
    throw new Error("Measured call count mismatch");
  return { status: "PASS" as const, protocol, fixtures, warmup, rows, summary };
}
