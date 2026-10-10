import { applyAction, getAvailableActions } from "../domain/game/actions";
import { validateGameState } from "../domain/game/game-state";
import type {
  GameAction,
  GameState,
  PieceIndex,
  PieceInstance,
} from "../domain/game/types";
import { getUniqueVariants, serializeShape } from "../domain/pieces/transforms";
import { getInitialCatalog } from "../domain/pieces/catalog";
import { solveTurn } from "../domain/solver/solver";
import { compareEvaluations } from "../domain/solver/evaluator";
import type { SolverCandidate, TurnSolution } from "../domain/solver/types";
import { stateKey } from "./episode";
import {
  diagnosticFixtures,
  permuteSlots,
  restoreCandidate,
  semanticPath,
  semanticStateKey,
  type DiagnosticFixture,
} from "./permutation-teacher";
import { physicalActionKey, verifyCandidatePath } from "./teacher-audit";

export const CANONICAL_EXECUTION = {
  schemaVersion: 1,
  id: "canonical-teacher-20261010-v1",
  status: "FROZEN_DIAGNOSTIC",
  scope: "canonical-single-search-no-teacher-adoption",
  inputSource: "P041 fixed 20 initial states",
  states: 20,
  strata: { generated: 12, scripted: 8 },
  sortKey: [
    "variantOrbitMinimum",
    "normalizedShapeKey",
    "pieceId",
    "pendingRank",
  ],
  ordinal: "JavaScript UTF-16 code-unit < >; no localeCompare",
  duplicatePhysicalTie:
    "original slot ascending; instanceId is never an ordering key",
  map: "n occupied to 0..n-1; missing source slots ascending to remaining destinations",
  inverseTargets: [
    "actions",
    "finalState.remainingPieces",
    "finalState.pendingReroll",
  ],
  selection: "keep solveTurn best; restore every alternative separately",
  perCall: {
    maxNodes: 512,
    maxAlternatives: 3,
    useMemoization: true,
    sharedMemo: false,
  },
  conditions: {
    baseline: { callsPerState: 1 },
    canonicalAllSlots: {
      permutations: ["012", "021", "102", "120", "201", "210"],
      callsPerState: 6,
    },
    canonicalReverseArray: { callsPerState: 1 },
    canonicalRenameInstances: { callsPerState: 1 },
  },
  repetitions: 2,
  measuredSolveCalls: 360,
  measuredNodeCap: 184320,
  warmupSolveCalls: 2,
  warmupNodeCap: 1024,
  warmupSource: "first-scripted-state A/D once",
  executionOrder: "rotate-nine-conditions-by-state-index-plus-repetition",
  repeatedConditions: 180,
  invariancePairsWithoutRepeat: 140,
  invarianceChecksWithRepeat: 280,
  timeCriterion: "OBSERVATION_ONLY_NOT_EQUAL_WALL_TIME",
  training: "NOT_AUTHORIZED",
  testGenerationAndEvaluation: "NOT_AUTHORIZED",
  pilotGate: {
    generatedBetterAtLeast: 1,
    generatedWorse: 0,
    requiredInvariance:
      "semantic full path/evaluation/final state; physical changes recorded separately",
    doesNotApproveLabelOrTraining: true,
  },
} as const;
export type CanonicalProtocol = typeof CANONICAL_EXECUTION & {
  inputs: { id: string; stateSha256: string }[];
};
export const ordinal = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);
const stable = (value: unknown): string =>
  JSON.stringify(value, (_key, x) =>
    x && typeof x === "object" && !Array.isArray(x)
      ? Object.fromEntries(Object.entries(x).sort(([a], [b]) => ordinal(a, b)))
      : x,
  );
export function validateCanonicalProtocol(input: unknown): CanonicalProtocol {
  const p = input as CanonicalProtocol;
  if (
    !p ||
    typeof p !== "object" ||
    Object.keys(p).length !== Object.keys(CANONICAL_EXECUTION).length + 1 ||
    Object.entries(CANONICAL_EXECUTION).some(
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
    throw new Error("Invalid canonical diagnostic protocol");
  return structuredClone(p);
}
export type CanonicalKey = [string, string, string, number];
export function canonicalPieceKey(
  piece: PieceInstance,
  pending: GameState["pendingReroll"],
): CanonicalKey {
  const variants = getUniqueVariants(piece.piece.shape),
    current = serializeShape(piece.piece.shape);
  if (!variants.ok || !current.ok) throw new Error("Invalid canonical shape");
  const keys = variants.variants
    .map((v) => {
      const key = serializeShape(v.shape);
      if (!key.ok) throw new Error(key.error.message);
      return key.key;
    })
    .sort(ordinal);
  return [
    keys[0],
    current.key,
    piece.piece.id,
    pending?.instanceId === piece.instanceId &&
    pending.pieceIndex === piece.pieceIndex
      ? 0
      : 1,
  ];
}
export function canonicalizeState(input: unknown) {
  const valid = validateGameState(input);
  if (!valid.ok) throw new Error(valid.error.message);
  const state = structuredClone(valid.state);
  const keys = state.remainingPieces.map((piece) => ({
    sourceSlot: piece.pieceIndex,
    key: canonicalPieceKey(piece, state.pendingReroll),
  }));
  const keyed = state.remainingPieces.map((piece, i) => ({
    piece,
    key: keys[i].key,
  }));
  keyed.sort((a, b) => {
    for (let i = 0; i < 3; i++) {
      const diff = ordinal(a.key[i] as string, b.key[i] as string);
      if (diff) return diff;
    }
    return a.key[3] - b.key[3] || a.piece.pieceIndex - b.piece.pieceIndex;
  });
  const map: PieceIndex[] = [0, 1, 2];
  keyed.forEach(({ piece }, dest) => {
    map[piece.pieceIndex] = dest as PieceIndex;
  });
  const occupied = new Set(keyed.map((x) => x.piece.pieceIndex));
  ([0, 1, 2] as PieceIndex[])
    .filter((slot) => !occupied.has(slot))
    .forEach((slot, i) => {
      map[slot] = (keyed.length + i) as PieceIndex;
    });
  const permutation = map.join("");
  const mapped = permuteSlots(state, permutation);
  mapped.remainingPieces.sort((a, b) => a.pieceIndex - b.pieceIndex);
  return { state: mapped, permutation, keys };
}
export const CANONICAL_CONDITIONS = [
  "baseline",
  "canonical-012",
  "canonical-021",
  "canonical-102",
  "canonical-120",
  "canonical-201",
  "canonical-210",
  "canonical-reverse-array",
  "canonical-rename-instances",
] as const;
export type CanonicalCondition = (typeof CANONICAL_CONDITIONS)[number];
export function prepareCanonicalInput(
  input: GameState,
  condition: CanonicalCondition,
) {
  if (!CANONICAL_CONDITIONS.includes(condition))
    throw new Error("Invalid canonical condition");
  const slotControl = /^canonical-[012]{3}$/.test(condition)
    ? condition.slice(-3)
    : "012";
  let state = permuteSlots(input, slotControl);
  const inverseNames: Record<string, string> = {};
  if (condition === "canonical-reverse-array") state.remainingPieces.reverse();
  if (condition === "canonical-rename-instances") {
    const forward = new Map(
      state.remainingPieces.map((p) => [
        p.instanceId,
        `canonical-control-${2 - p.pieceIndex}`,
      ]),
    );
    for (const p of state.remainingPieces) {
      const name = forward.get(p.instanceId)!;
      inverseNames[name] = p.instanceId;
      p.instanceId = name;
    }
    if (state.pendingReroll)
      state.pendingReroll.instanceId = forward.get(
        state.pendingReroll.instanceId,
      )!;
  }
  const canonical =
    condition === "baseline"
      ? { state, permutation: "012", keys: [] }
      : canonicalizeState(state);
  state = canonical.state;
  const permutation = [...slotControl]
    .map((slot) => canonical.permutation[Number(slot)])
    .join("");
  return {
    state,
    permutation,
    slotControl,
    canonicalPermutation: canonical.permutation,
    keys: canonical.keys,
    inverseNames,
  };
}
type Mapping = ReturnType<typeof prepareCanonicalInput>;
function restoreAction<T extends GameAction>(input: T, mapping: Mapping): T {
  const action = structuredClone(input);
  if (action.type !== "single-cell") {
    action.pieceIndex = mapping.permutation.indexOf(
      String(action.pieceIndex),
    ) as PieceIndex;
    action.instanceId =
      mapping.inverseNames[action.instanceId] ?? action.instanceId;
  }
  return action;
}
export function restoreCanonicalCandidate(
  candidate: SolverCandidate,
  mapping: Mapping,
): SolverCandidate {
  const result = restoreCandidate(candidate, mapping.permutation);
  result.actions = candidate.actions.map((action) =>
    restoreAction(action, mapping),
  );
  for (const piece of result.finalState.remainingPieces)
    piece.instanceId =
      mapping.inverseNames[piece.instanceId] ?? piece.instanceId;
  if (result.finalState.pendingReroll)
    result.finalState.pendingReroll.instanceId =
      mapping.inverseNames[result.finalState.pendingReroll.instanceId] ??
      result.finalState.pendingReroll.instanceId;
  return result;
}
function legalKeys(state: GameState, mapping?: Mapping) {
  const legal = getAvailableActions(state);
  if (!legal.ok) throw new Error(legal.error.message);
  return legal.actions
    .map((action) => stable(mapping ? restoreAction(action, mapping) : action))
    .sort(ordinal);
}
export function assertCanonicalLegality(original: GameState, mapping: Mapping) {
  if (stable(legalKeys(original)) !== stable(legalKeys(mapping.state, mapping)))
    throw new Error("Canonical legal-action bijection changed");
}
export function pathCoverage(input: GameState, candidate: SolverCandidate) {
  let state = input,
    clearedRows = 0,
    acquiredItems = 0,
    retainedItemPlacements = 0,
    singleCell = 0,
    reroll = 0;
  for (const action of candidate.actions) {
    const next = applyAction(state, action);
    if (!next.ok) throw new Error(next.error.message);
    clearedRows += next.info.clearedRows.length;
    acquiredItems += next.info.acquiredItems.length;
    retainedItemPlacements += next.info.retainedItems.length;
    singleCell += Number(action.type === "single-cell");
    reroll += Number(action.type === "reroll");
    state = next.state;
  }
  return {
    clearedRows,
    acquiredItems,
    retainedItemPlacements,
    singleCell,
    reroll,
    pendingReroll: Boolean(state.pendingReroll),
  };
}
export type CanonicalCall = {
  fixtureId: string;
  condition: CanonicalCondition;
  repetition: number;
  phase: "warmup" | "measured";
  sequence: number;
  maxNodes: 512;
  mapping: Mapping;
  state: GameState;
  result: TurnSolution;
  restored: TurnSolution;
  timing: { transformMs: number; solveMs: number; validationMs: number };
};
export type CanonicalRow = {
  fixtureId: string;
  stratum: DiagnosticFixture["stratum"];
  split: DiagnosticFixture["split"];
  condition: CanonicalCondition;
  repetition: number;
  calls: CanonicalCall[];
  semanticPath: string[];
  physicalPath: string[];
  semanticFinal: string;
  coverage: ReturnType<typeof pathCoverage>;
  latencyMs: number;
  loggingMs: number;
};
function semanticSignature(row: CanonicalRow) {
  return stable([
    row.semanticPath,
    row.calls[0].restored.evaluation,
    row.semanticFinal,
  ]);
}
function repeatSignature(row: CanonicalRow) {
  return stable([
    row.calls.map((input) =>
      Object.fromEntries(
        Object.entries(input).filter(
          ([key]) => !["timing", "sequence", "repetition"].includes(key),
        ),
      ),
    ),
    row.semanticPath,
    row.physicalPath,
    row.coverage,
  ]);
}
function stats(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  return {
    samples: sorted.length,
    minMs: sorted[0],
    medianMs: sorted[Math.floor(sorted.length / 2)],
    p95Ms: sorted[Math.ceil(sorted.length * 0.95) - 1],
    maxMs: sorted.at(-1),
  };
}
export function summarizeCanonical(rows: CanonicalRow[]) {
  const first = rows.filter((row) => row.repetition === 0),
    comparisons = (["generated", "scripted"] as const).map((stratum) => {
      const baseline = first.filter(
        (row) => row.stratum === stratum && row.condition === "baseline",
      );
      let better = 0,
        worse = 0,
        equal = 0;
      const differences = baseline.map((a) => {
        const d = first.find(
          (row) =>
            row.fixtureId === a.fixtureId && row.condition === "canonical-012",
        )!;
        const comparison = compareEvaluations(
          d.calls[0].restored.evaluation,
          a.calls[0].restored.evaluation,
        );
        if (comparison > 0) better++;
        else if (comparison < 0) worse++;
        else equal++;
        return {
          fixtureId: a.fixtureId,
          comparison,
          physicalPathChanged:
            stable(a.physicalPath) !== stable(d.physicalPath),
          semanticPathChanged:
            stable(a.semanticPath) !== stable(d.semanticPath),
        };
      });
      return {
        stratum,
        states: baseline.length,
        better,
        worse,
        equal,
        differences,
      };
    });
  const generated = comparisons[0];
  const invariance = rows
    .filter(
      (row) =>
        row.condition !== "baseline" && row.condition !== "canonical-012",
    )
    .map((row) => {
      const identity = rows.find(
        (x) =>
          x.fixtureId === row.fixtureId &&
          x.repetition === row.repetition &&
          x.condition === "canonical-012",
      )!;
      return {
        fixtureId: row.fixtureId,
        condition: row.condition,
        repetition: row.repetition,
        semanticEqual: semanticSignature(row) === semanticSignature(identity),
        physicalPathChanged:
          stable(row.physicalPath) !== stable(identity.physicalPath),
      };
    });
  const repeated = first.map((row) => ({
    fixtureId: row.fixtureId,
    condition: row.condition,
    equal:
      repeatSignature(row) ===
      repeatSignature(
        rows.find(
          (x) =>
            x.fixtureId === row.fixtureId &&
            x.condition === row.condition &&
            x.repetition === 1,
        )!,
      ),
  }));
  const invariancePassed = invariance.every((x) => x.semanticEqual),
    repeatPassed = repeated.every((x) => x.equal);
  return {
    measuredCalls: rows.reduce((n, row) => n + row.calls.length, 0),
    nominalNodeCap: rows.length * 512,
    returnedPathsVerified: rows.reduce(
      (n, row) => n + 1 + row.calls[0].restored.alternatives.length,
      0,
    ),
    comparisons,
    invariance: {
      uniquePairs: first.filter(
        (row) =>
          row.condition !== "baseline" && row.condition !== "canonical-012",
      ).length,
      checks: invariance.length,
      passed: invariancePassed,
      physicalChanges: invariance.filter((x) => x.physicalPathChanged).length,
      details: invariance,
    },
    repeated: {
      conditions: repeated.length,
      passed: repeatPassed,
      details: repeated,
    },
    candidateForUserReview:
      generated.better >= 1 &&
      generated.worse === 0 &&
      invariancePassed &&
      repeatPassed,
    conditions: CANONICAL_CONDITIONS.map((condition) => {
      const selected = rows.filter((row) => row.condition === condition),
        unique = selected.filter((row) => row.repetition === 0);
      return {
        condition,
        latency: stats(selected.map((row) => row.latencyMs)),
        logging: stats(selected.map((row) => row.loggingMs)),
        strata: (["generated", "scripted"] as const).map((stratum) => {
          const group = unique.filter((row) => row.stratum === stratum);
          return {
            stratum,
            states: group.length,
            incomplete: group.filter(
              (row) => !row.calls[0].restored.search.searchComplete,
            ).length,
            coverage: group.map((row) => ({
              fixtureId: row.fixtureId,
              ...row.coverage,
            })),
          };
        }),
      };
    }),
  };
}
export function runCanonicalDiagnostic(
  input: unknown,
  clock: () => number,
  checkInput: (
    fixture: DiagnosticFixture,
    key: string,
    expectedHash: string,
  ) => void,
  progress?: (
    event:
      | { type: "call"; call: CanonicalCall }
      | { type: "condition"; row: CanonicalRow },
  ) => void,
) {
  const protocol = validateCanonicalProtocol(input),
    fixtures = diagnosticFixtures();
  if (typeof checkInput !== "function")
    throw new Error("Historical input hash verifier required");
  fixtures.forEach((fixture, i) => {
    if (fixture.id !== protocol.inputs[i].id)
      throw new Error("Historical input order mismatch");
    checkInput(
      structuredClone(fixture),
      stateKey(fixture.state),
      protocol.inputs[i].stateSha256,
    );
  });
  let prior = -Infinity,
    sequence = 0;
  const now = () => {
    const t = clock();
    if (!Number.isFinite(t) || t < prior)
      throw new Error("Clock must be finite and monotonic");
    prior = t;
    return t;
  };
  function execute(
    fixture: DiagnosticFixture,
    condition: CanonicalCondition,
    repetition: number,
    phase: "warmup" | "measured",
  ): CanonicalRow {
    const original = structuredClone(fixture.state),
      before = stable(original),
      start = now(),
      mapping = prepareCanonicalInput(original, condition);
    assertCanonicalLegality(original, mapping);
    const prepared = stable(mapping.state),
      solveStart = now();
    const solved = solveTurn(mapping.state, getInitialCatalog(), {
      maxNodes: 512,
      maxAlternatives: 3,
      useMemoization: true,
    });
    const solveEnd = now();
    if (!solved.ok) throw new Error(solved.error.message);
    const result = solved.result;
    if (
      stable(mapping.state) !== prepared ||
      stable(original) !== before ||
      stable(fixture.state) !== before
    )
      throw new Error("Canonical diagnostic input mutation");
    if (
      !Number.isInteger(result.search.visitedNodes) ||
      result.search.visitedNodes < 0 ||
      result.search.visitedNodes > 512 ||
      result.search.maxNodes !== 512
    )
      throw new Error("Canonical node cap exceeded");
    const restored: TurnSolution = {
      ...restoreCanonicalCandidate(result, mapping),
      search: structuredClone(result.search),
      alternatives: result.alternatives.map((c) =>
        restoreCanonicalCandidate(c, mapping),
      ),
    };
    for (const c of [result, ...result.alternatives])
      verifyCandidatePath(mapping.state, c);
    for (const c of [restored, ...restored.alternatives])
      verifyCandidatePath(original, c);
    const path = semanticPath(original, restored),
      physical = restored.actions.map(physicalActionKey),
      coverage = pathCoverage(original, restored),
      final = semanticStateKey(restored.finalState),
      end = now();
    const call: CanonicalCall = {
      fixtureId: fixture.id,
      condition,
      repetition,
      phase,
      sequence: ++sequence,
      maxNodes: 512,
      mapping: structuredClone(mapping),
      state: structuredClone(mapping.state),
      result: structuredClone(result),
      restored,
      timing: {
        transformMs: solveStart - start,
        solveMs: solveEnd - solveStart,
        validationMs: end - solveEnd,
      },
    };
    const logStart = now();
    progress?.({ type: "call", call: structuredClone(call) });
    const logEnd = now();
    const row: CanonicalRow = {
      fixtureId: fixture.id,
      stratum: fixture.stratum,
      split: fixture.split,
      condition,
      repetition,
      calls: [call],
      semanticPath: path,
      physicalPath: physical,
      semanticFinal: final,
      coverage,
      latencyMs: end - start,
      loggingMs: logEnd - logStart,
    };
    progress?.({ type: "condition", row: structuredClone(row) });
    return row;
  }
  const warmup = [
      execute(fixtures[12], "baseline", -1, "warmup"),
      execute(fixtures[12], "canonical-012", -1, "warmup"),
    ],
    rows: CanonicalRow[] = [];
  fixtures.forEach((fixture, i) => {
    for (let repetition = 0; repetition < 2; repetition++) {
      const offset = (i + repetition) % 9,
        order = [
          ...CANONICAL_CONDITIONS.slice(offset),
          ...CANONICAL_CONDITIONS.slice(0, offset),
        ];
      for (const condition of order)
        rows.push(execute(fixture, condition, repetition, "measured"));
    }
  });
  const summary = summarizeCanonical(rows);
  return {
    protocol,
    fixtures: fixtures.map((input) =>
      Object.fromEntries(
        Object.entries(input).filter(([key]) => key !== "state"),
      ),
    ),
    warmup,
    rows,
    summary,
    status:
      summary.invariance.passed && summary.repeated.passed ? "PASS" : "FAIL",
  };
}
