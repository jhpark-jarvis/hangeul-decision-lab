import { applyAction } from "../domain/game/actions";
import { getInitialCatalog } from "../domain/pieces/catalog";
import { getAbilityCandidates } from "../domain/solver/ability-candidates";
import { compareEvaluations } from "../domain/solver/evaluator";
import type {
  SolverCandidate,
  SolverEvaluation,
  TurnSolution,
} from "../domain/solver/types";
import { stateKey } from "./episode";
import {
  assertCanonicalLegality,
  prepareCanonicalInput,
  validateCanonicalProtocol,
  type CanonicalRow,
} from "./canonical-teacher";
import {
  diagnosticFixtures,
  permuteSlots,
  validatePermutationProtocol,
  type DiagnosticRow,
} from "./permutation-teacher";
import { physicalActionKey, verifyCandidatePath } from "./teacher-audit";

export const ROOT_CAUSE_EXECUTION = {
  schemaVersion: 1,
  id: "teacher-root-cause-20261011-v1",
  status: "FROZEN_DIAGNOSTIC",
  states: 20,
  lossIds: [
    "train-4001-pressure",
    "train-4002-sparse",
    "dev-5000-sparse",
    "dev-5000-pressure",
  ],
  baselineBudget: 1024,
  canonicalLossBudget: 3072,
  repetitions: 2,
  untracedCalls: 48,
  tracedCalls: 12,
  totalSolveCalls: 60,
  nominalNodeCap: 81920,
  warmupCalls: 0,
  timeCriterion: "OBSERVATION_ONLY_TRACE_OVERHEAD_NOT_POLICY_LATENCY",
  portfolio:
    "historical-A512-best-plus-D512-best; keep-A-on-evaluation-ties; diagnostic-only",
  noNewSeeds: true,
  noTeacherLabelOrTrainingChanges: true,
} as const;
export type RootCauseProtocol = typeof ROOT_CAUSE_EXECUTION & {
  inputs: { id: string; stateSha256: string }[];
  reports: { canonicalSha256: string; permutationSha256: string };
};
export function validateRootCauseProtocol(input: unknown): RootCauseProtocol {
  const p = input as RootCauseProtocol;
  if (
    !p ||
    Object.keys(p).length !== Object.keys(ROOT_CAUSE_EXECUTION).length + 2 ||
    Object.entries(ROOT_CAUSE_EXECUTION).some(
      ([k, v]) =>
        JSON.stringify((p as unknown as Record<string, unknown>)[k]) !==
        JSON.stringify(v),
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
    new Set(p.inputs.map((x) => x.id)).size !== 20 ||
    !p.reports ||
    Object.keys(p.reports).length !== 2 ||
    Object.values(p.reports).some(
      (x) => typeof x !== "string" || !/^[a-f0-9]{64}$/.test(x),
    )
  )
    throw new Error("Invalid root-cause protocol");
  return structuredClone(p);
}
export function logicalTurnSignature(turn: TurnSolution): string {
  const candidate = (c: SolverCandidate) => ({
    actions: c.actions.map(physicalActionKey),
    evaluation: c.evaluation,
    finalState: stateKey(c.finalState),
    usedAbilities: c.usedAbilities,
  });
  return JSON.stringify({
    best: candidate(turn),
    alternatives: turn.alternatives.map(candidate),
    search: turn.search,
  });
}
const axisNames = [
  "survivable",
  "allCurrentPiecesPlaced",
  "playablePieceTypes",
  "totalPlacements",
  "clearedRows",
  "acquiredItems",
  "remainingAbilityTotal",
  "isolatedEmptyCells",
  "unfillableGaps",
  "emptyCells",
] as const;
const axes = (e: SolverEvaluation) => [
  Number(e.survivable),
  Number(e.allCurrentPiecesPlaced),
  e.mobility.playablePieceTypes,
  e.mobility.totalPlacements,
  e.clearedRows,
  e.acquiredItems,
  e.remainingAbilities.reroll + e.remainingAbilities.singleCell,
  -e.isolatedEmptyCells,
  -e.unfillableGaps,
  e.emptyCells,
];
export function firstEvaluationDifference(
  a: SolverEvaluation,
  b: SolverEvaluation,
) {
  const left = axes(a),
    right = axes(b),
    index = left.findIndex((x, i) => x !== right[i]);
  return index < 0
    ? null
    : {
        axis: axisNames[index],
        left: left[index],
        right: right[index],
        comparison: left[index] > right[index] ? 1 : -1,
      };
}
/** Research candidate only; stable incumbent prevents changes on evaluation ties. */
export function incumbentPortfolio(a: TurnSolution, d: TurnSolution) {
  return compareEvaluations(d.evaluation, a.evaluation) > 0
    ? { source: "D512" as const, result: structuredClone(d) }
    : { source: "A512" as const, result: structuredClone(a) };
}
export type HistoricalCase = {
  fixtureId: string;
  stratum: "generated" | "scripted";
  state: ReturnType<typeof diagnosticFixtures>[number]["state"];
  a: TurnSolution;
  d: TurnSolution;
  b3072: TurnSolution;
  map: string;
  dMatchesHistoricalPermutation: boolean;
  difference: ReturnType<typeof firstEvaluationDifference>;
  root: {
    aFirst: string | null;
    dFirst: string | null;
    aFirstOrdinalUnderA: number;
    aFirstOrdinalUnderD: number;
    originalPieceIds: string[];
    canonicalPieceIds: string[];
  };
};
export function analyzeHistoricalReports(
  canonical: unknown,
  permutation: unknown,
) {
  const c = canonical as {
      status: string;
      protocol: unknown;
      rows: CanonicalRow[];
    },
    p = permutation as {
      status: string;
      protocol: unknown;
      rows: DiagnosticRow[];
    };
  if (
    !c ||
    !p ||
    c.status !== "PASS" ||
    p.status !== "PASS" ||
    !Array.isArray(c.rows) ||
    c.rows.length !== 360 ||
    !Array.isArray(p.rows) ||
    p.rows.length !== 200
  )
    throw new Error("Historical diagnostic report shape/status mismatch");
  const cp = validateCanonicalProtocol(c.protocol),
    pp = validatePermutationProtocol(p.protocol);
  if (JSON.stringify(cp.inputs) !== JSON.stringify(pp.inputs))
    throw new Error("Historical input manifest mismatch");
  let returnedPathsVerified = 0,
    sourceCallsVerified = 0;
  const cases: HistoricalCase[] = diagnosticFixtures().map((f) => {
    const a = p.rows.find(
        (r) =>
          r.fixtureId === f.id &&
          r.repetition === 0 &&
          r.condition === "baseline",
      ),
      b = p.rows.find(
        (r) =>
          r.fixtureId === f.id &&
          r.repetition === 0 &&
          r.condition === "largeSingle",
      ),
      union = p.rows.find(
        (r) =>
          r.fixtureId === f.id &&
          r.repetition === 0 &&
          r.condition === "permutationUnion",
      ),
      d = c.rows.find(
        (r) =>
          r.fixtureId === f.id &&
          r.repetition === 0 &&
          r.condition === "canonical-012",
      );
    if (!a || !b || !union || !d)
      throw new Error(`Missing historical condition ${f.id}`);
    const mapping = prepareCanonicalInput(f.state, "canonical-012");
    assertCanonicalLegality(f.state, mapping);
    for (const call of [...a.calls, ...b.calls, ...union.calls, ...d.calls]) {
      if (
        stateKey(call.state) !==
        stateKey(
          permuteSlots(
            f.state,
            "mapping" in call ? call.mapping.permutation : call.permutation,
          ),
        )
      )
        throw new Error("Historical transformed state mismatch");
      for (const candidate of [call.result, ...call.result.alternatives])
        verifyCandidatePath(call.state, candidate);
      for (const candidate of [call.restored, ...call.restored.alternatives])
        verifyCandidatePath(f.state, candidate);
      returnedPathsVerified += 1 + call.restored.alternatives.length;
      sourceCallsVerified++;
    }
    if (
      logicalTurnSignature(a.calls[0].restored) !==
      logicalTurnSignature(
        c.rows.find(
          (r) =>
            r.fixtureId === f.id &&
            r.repetition === 0 &&
            r.condition === "baseline",
        )!.calls[0].restored,
      )
    )
      throw new Error("Historical A drift");
    const equivalent = union.calls.find(
      (call) => call.permutation === mapping.permutation,
    );
    if (
      !equivalent ||
      logicalTurnSignature(equivalent.restored) !==
        logicalTurnSignature(d.calls[0].restored)
    )
      throw new Error("Canonical differs from equivalent historical slot call");
    const rootOrdinal = (
      state: HistoricalCase["state"],
      result: TurnSolution,
      key: string | null,
    ) => {
      const list = getAbilityCandidates(
        state,
        getInitialCatalog(),
        result.search.abilitySearch?.ordinaryProbe.discoveredCompletePaths ===
          0,
      );
      if (!list.ok) throw new Error(list.error.message);
      return key === null
        ? -1
        : list.actions.map(physicalActionKey).indexOf(key);
    };
    const aKey = a.calls[0].restored.actions[0]
        ? physicalActionKey(a.calls[0].restored.actions[0])
        : null,
      dKey = d.calls[0].restored.actions[0]
        ? physicalActionKey(d.calls[0].restored.actions[0])
        : null;
    // The old A path remains physically legal under canonical slots.
    const witness = structuredClone(a.calls[0].restored);
    witness.actions = witness.actions.map((action) =>
      action.type === "single-cell"
        ? action
        : {
            ...action,
            pieceIndex: Number(mapping.permutation[action.pieceIndex]) as
              0 | 1 | 2,
          },
    );
    witness.finalState = permuteSlots(witness.finalState, mapping.permutation);
    verifyCandidatePath(mapping.state, witness);
    return {
      fixtureId: f.id,
      stratum: f.stratum,
      state: structuredClone(f.state),
      a: structuredClone(a.calls[0].restored),
      d: structuredClone(d.calls[0].restored),
      b3072: structuredClone(b.calls[0].restored),
      map: mapping.permutation,
      dMatchesHistoricalPermutation: true,
      difference: firstEvaluationDifference(
        d.calls[0].restored.evaluation,
        a.calls[0].restored.evaluation,
      ),
      root: {
        aFirst: aKey,
        dFirst: dKey,
        aFirstOrdinalUnderA: rootOrdinal(f.state, a.calls[0].result, aKey),
        aFirstOrdinalUnderD: rootOrdinal(
          mapping.state,
          d.calls[0].result,
          aKey,
        ),
        originalPieceIds: [...f.state.remainingPieces]
          .sort((a, b) => a.pieceIndex - b.pieceIndex)
          .map((p) => p.piece.id),
        canonicalPieceIds: mapping.state.remainingPieces.map((p) => p.piece.id),
      },
    };
  });
  const losses = cases.filter(
    (c) => compareEvaluations(c.d.evaluation, c.a.evaluation) < 0,
  );
  if (
    JSON.stringify(losses.map((c) => c.fixtureId)) !==
    JSON.stringify(ROOT_CAUSE_EXECUTION.lossIds)
  )
    throw new Error("Frozen loss set drift");
  return { cases, losses, sourceCallsVerified, returnedPathsVerified };
}
export type TraceEvent = {
  type: "visit" | "consider";
  scope: "ordinary-pieces" | "pieces-and-abilities";
  node: number;
  path: string[];
  evaluation?: SolverEvaluation;
};
/** Fail closed when the compiled solver's exact observation points change. */
export function buildPassiveTrace(originalCode: string): string {
  function replaceExactly(text: string, needle: string, replacement: string) {
    if (text.split(needle).length !== 2)
      throw new Error("Passive trace insertion drift");
    return text.replace(needle, replacement);
  }
  let code =
    "import {physicalActionKey as __physicalKey} " +
    ["fr", "om"].join("") +
    " " +
    JSON.stringify("../../research/teacher-audit.mjs") +
    ";\nlet __events=[];\n" +
    originalCode;
  code = replaceExactly(
    code,
    "search.visitedNodes++;",
    "search.visitedNodes++; __events.push({type:'visit',scope:search.scope,node:search.visitedNodes,path:actions.map(__physicalKey)});",
  );
  code = replaceExactly(
    code,
    "search.evaluatedCandidates++;",
    "search.evaluatedCandidates++; __events.push({type:'consider',scope:search.scope,node:search.visitedNodes,path:actions.map(__physicalKey),evaluation:structuredClone(evaluated.evaluation)});",
  );
  return (
    code +
    "\nexport function tracedTurn(...args){__events=[]; const result=solveTurn(...args); const events=__events; __events=[]; return {result,events};}\n"
  );
}
export function traceWitness(trace: TraceEvent[], target: SolverCandidate) {
  const keys = target.actions.map(physicalActionKey),
    same = (a: string[], b: string[]) =>
      JSON.stringify(a) === JSON.stringify(b);
  const firstVisit = (path: string[], scope: TraceEvent["scope"]) =>
    trace.find(
      (e) => e.type === "visit" && e.scope === scope && same(e.path, path),
    )?.node ?? null;
  return {
    targetLength: keys.length,
    prefixes: keys.map((_, i) => ({
      length: i + 1,
      ordinaryNode: firstVisit(keys.slice(0, i + 1), "ordinary-pieces"),
      fullNode: firstVisit(keys.slice(0, i + 1), "pieces-and-abilities"),
    })),
    evaluations: trace
      .filter((e) => e.type === "consider" && same(e.path, keys))
      .map((e) => ({ scope: e.scope, node: e.node, evaluation: e.evaluation })),
    rootVisited:
      keys.length === 0 ||
      trace.some((e) => e.type === "visit" && same(e.path, keys.slice(0, 1))),
  };
}
export function assertTraceAccounting(
  trace: TraceEvent[],
  result: TurnSolution,
) {
  const visits = trace.filter((e) => e.type === "visit"),
    ordinary = visits.filter((e) => e.scope === "ordinary-pieces");
  if (
    visits.length !== result.search.visitedNodes ||
    ordinary.length !==
      (result.search.abilitySearch?.ordinaryProbe.visitedNodes ?? 0) ||
    trace.filter(
      (e) => e.type === "consider" && e.scope === "pieces-and-abilities",
    ).length !== result.search.evaluatedCandidates
  )
    throw new Error("Passive trace accounting mismatch");
}
export function replayPath(
  state: HistoricalCase["state"],
  path: SolverCandidate["actions"],
) {
  let current = structuredClone(state);
  for (const action of path) {
    const next = applyAction(current, action);
    if (!next.ok) throw new Error(next.error.message);
    current = next.state;
  }
  return current;
}
