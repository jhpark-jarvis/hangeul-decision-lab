import { getAvailableActions } from "../domain/game/actions";
import type { GameState } from "../domain/game/types";
import { getInitialCatalog } from "../domain/pieces/catalog";
import { solveTurn } from "../domain/solver/solver";
import type { TurnSolution } from "../domain/solver/types";
import { runEpisode, stateKey, verifyReplay } from "./episode";
import { createSyntheticFixture } from "./fixture";
import {
  actionKey,
  encodeDecision,
  MODEL_SCHEMA,
  type ModelExample,
} from "./imitation";
import { semanticStateKey } from "./permutation-teacher";
import { verifyCandidatePath } from "./teacher-audit";
import type { EpisodeFixture, EpisodePolicy, EpisodeResult } from "./types";

export const TEACHERS = ["A512", "A1024"] as const;
export type Teacher = (typeof TEACHERS)[number];
export type PairedSplit = "train" | "dev" | "test";
export type Pair = Omit<ModelExample, "label" | "teacher"> & {
  labels: Record<Teacher, number>;
  teachers: Record<Teacher, { result: TurnSolution; latencyMs: number }>;
};
export type RegistryEntry = {
  split: PairedSplit;
  id: string;
  forward: string;
  semantic: string;
  labels: Record<Teacher, number>;
};
export type PairedProtocol = {
  schemaVersion: number;
  id: string;
  status: string;
  sourceProtocolSha256: string;
  dataset: {
    trainSeeds: number[];
    devSeeds: number[];
    reservedTestSeeds: number[];
    families: ("sparse" | "pressure")[];
    maxActions: number;
    [key: string]: unknown;
  };
  teachers: {
    A512: {
      maxNodes: number;
      maxAlternatives: number;
      useMemoization: boolean;
    };
    A1024: {
      maxNodes: number;
      maxAlternatives: number;
      useMemoization: boolean;
    };
    [key: string]: unknown;
  };
  training: { [key: string]: unknown };
  evaluation: { [key: string]: unknown };
  cost: { [key: string]: unknown };
  boundaries: { [key: string]: unknown };
  execution: string;
};
/** Exact frozen experiment; this does not relax the original pilot's limits. */
export function validatePairedProtocol(input: unknown): PairedProtocol {
  const p = input as PairedProtocol;
  const equal = (a: unknown, b: unknown) =>
    JSON.stringify(a) === JSON.stringify(b);
  if (
    !p ||
    Object.keys(p).length !== 11 ||
    p.schemaVersion !== 1 ||
    p.id !== "paired-teacher-cnn-20261011-v1" ||
    p.status !== "FROZEN_PAIRED_EXPERIMENT" ||
    p.execution !== "FROZEN_BEFORE_EXECUTION" ||
    p.sourceProtocolSha256 !==
      "dd43c9a946d8e7ca251a2a5012bb13bd722357c22286d58569c7ab81c3c8242d" ||
    !equal(p.dataset, {
      trainSeeds: Array.from({ length: 32 }, (_, i) => 8000 + i),
      devSeeds: Array.from({ length: 8 }, (_, i) => 9000 + i),
      reservedTestSeeds: [6000, 6001, 6002, 6003],
      families: ["sparse", "pressure"],
      maxActions: 6,
      collectionPolicy: "A512",
      episodeCap: 88,
      observationCap: 528,
      sharedObservations: true,
      pairedSkip: true,
      crossSplitForwardAndSemanticDuplicate: "STOP",
      withinSplitDuplicate: "KEEP_FIRST_RECORD_CONFLICTS",
      excludedSeeds: "prior3000-test; diagnostics4000/5000",
      scriptedTraining: false,
      testGate: "BOTH_DEV_CHECKPOINTS_AND_PROTOCOL_SOURCE_FREEZE",
    }) ||
    !equal(p.teachers, {
      A512: { maxNodes: 512, maxAlternatives: 3, useMemoization: true },
      A1024: { maxNodes: 1024, maxAlternatives: 3, useMemoization: true },
      sharedMemo: false,
      label: "best-first-exact-domain-legal-index",
      ties: "existing-solver-best-order",
    }) ||
    !equal(p.training, {
      seed: 20261008,
      epochs: 12,
      batchSize: 8,
      threads: 2,
      convChannels: 8,
      embedding: 32,
      poolRows: 4,
      poolCols: 5,
      learningRate: 0.001,
      weightDecay: 0.0001,
      runs: 2,
      loss: "single-hard-label-cross-entropy",
      initialization: "same-weights",
      sampleOrder: "same-paired-example-indices",
      checkpointSelection: "own-teacher-highest-dev-top1-first-tie",
      additionalTuning: false,
    }) ||
    !equal(p.evaluation, {
      crossTeacherTop1: true,
      domainLegalAndNaN: true,
      coverageBySplitFamily: true,
      rolloutPolicies: [
        "A512",
        "A1024",
        "first-legal",
        "CNN-A512",
        "CNN-A1024",
      ],
      testEpisodes: 8,
      maxActions: 6,
      decisionCap: 240,
      futureTapeOwner: "environment-only",
      postTestReselection: false,
    }) ||
    !equal(p.cost, {
      collectionNodeCap: 270336,
      labelNodeCap: 811008,
      rolloutNodeCap: 73728,
      totalNominalDfsNodeCap: 1155072,
      wallTime: "OBSERVATION_ONLY",
      memory: "OBSERVATION_ONLY",
    }) ||
    !equal(p.boundaries, {
      existingDataModelPreserved: true,
      newRuntime: false,
      dependencyInstall: false,
      externalService: false,
      dataTransfer: false,
      appModel: false,
      reinforcementLearning: false,
      push: false,
    })
  )
    throw new Error("Invalid frozen paired protocol");
  return structuredClone(p);
}
export function pairDecision(
  state: GameState,
  legal: Parameters<typeof encodeDecision>[1],
  results: Pair["teachers"],
  identity: Pick<Pair, "id" | "fixtureId" | "seed" | "split" | "actionIndex">,
): Pair | null {
  for (const name of TEACHERS) {
    const t = results[name];
    if (
      !Number.isFinite(t.latencyMs) ||
      t.latencyMs < 0 ||
      t.result.search.maxNodes !== (name === "A512" ? 512 : 1024)
    )
      throw new Error("Invalid teacher diagnostics");
    for (const c of [t.result, ...t.result.alternatives])
      verifyCandidatePath(state, c);
  }
  const choices = TEACHERS.map((n) => results[n].result.actions[0]);
  if (choices.some((c) => !c)) return null;
  const a = encodeDecision(state, legal, choices[0]),
    b = encodeDecision(state, legal, choices[1]);
  if (!a.ok || !b.ok) throw new Error("Invalid paired domain encoding");
  return {
    ...identity,
    state: structuredClone(state),
    stateKey: stateKey(state),
    observation: a.value.observation,
    candidateKeys: a.value.candidateKeys,
    candidateFeatures: a.value.candidateFeatures,
    labels: { A512: a.value.label, A1024: b.value.label },
    teachers: structuredClone(results),
  };
}
export function registerPair(
  pair: Pair,
  registry: RegistryEntry[],
  fingerprint: (text: string) => string,
) {
  const entry: RegistryEntry = {
    split: pair.split,
    id: pair.id,
    forward: fingerprint(
      JSON.stringify([pair.observation, pair.candidateFeatures]),
    ),
    semantic: fingerprint(semanticStateKey(pair.state)),
    labels: { ...pair.labels },
  };
  const matching = registry.filter(
    (e) => e.forward === entry.forward || e.semantic === entry.semantic,
  );
  if (matching.some((e) => e.split !== pair.split))
    throw new Error("Forward/semantic state leaked across splits");
  if (matching.length)
    return {
      keep: false,
      entry,
      priorId: matching[0].id,
      labelConflict: TEACHERS.some(
        (n) => matching[0].labels[n] !== pair.labels[n],
      ),
    };
  registry.push(entry);
  return { keep: true, entry, priorId: null, labelConflict: false };
}
export type PairPack = {
  schemaVersion: 1;
  status: "PASS";
  schema: typeof MODEL_SCHEMA;
  stage: "train-dev" | "test";
  protocolId: string;
  examples: Record<PairedSplit, Pair[]>;
  registry: RegistryEntry[];
  episodes: {
    split: PairedSplit;
    fixture: EpisodeFixture;
    result: EpisodeResult;
  }[];
  skips: { id: string; split: PairedSplit; reason: string }[];
  duplicates: ReturnType<typeof registerPair>[];
  accounting: {
    solveCalls: number;
    nominalNodes: number;
    visitedNodes: number;
    observations: number;
  };
};
export function collectPairs(
  protocol: unknown,
  stage: PairPack["stage"],
  registry: RegistryEntry[],
  fingerprint: (text: string) => string,
  clock: () => number,
  callbacks: {
    pair?: (p: Pair) => void;
    episode?: (e: PairPack["episodes"][number], count: number) => void;
  } = {},
): PairPack {
  const p = validatePairedProtocol(protocol),
    catalog = getInitialCatalog();
  const pack: PairPack = {
    schemaVersion: 1,
    status: "PASS",
    schema: structuredClone(MODEL_SCHEMA),
    stage,
    protocolId: p.id,
    examples: { train: [], dev: [], test: [] },
    registry: structuredClone(registry),
    episodes: [],
    skips: [],
    duplicates: [],
    accounting: {
      solveCalls: 0,
      nominalNodes: 0,
      visitedNodes: 0,
      observations: 0,
    },
  };
  const splits: PairedSplit[] =
    stage === "train-dev" ? ["train", "dev"] : ["test"];
  for (const split of splits)
    for (const seed of split === "test"
      ? p.dataset.reservedTestSeeds
      : p.dataset[split === "train" ? "trainSeeds" : "devSeeds"])
      for (const family of p.dataset.families) {
        const built = createSyntheticFixture(
          seed,
          family,
          p.dataset.maxActions,
        );
        if (!built.ok) throw new Error(built.error);
        const fixture = built.value;
        let thrown: unknown;
        const played = runEpisode(
          fixture,
          ({ state, legalActions, actionIndex }) => {
            try {
              const results = {} as Pair["teachers"];
              for (const name of TEACHERS) {
                const t0 = clock(),
                  solved = solveTurn(state, catalog, p.teachers[name]);
                if (!solved.ok) throw new Error(solved.error.message);
                results[name] = {
                  result: solved.result,
                  latencyMs: clock() - t0,
                };
                pack.accounting.solveCalls++;
                pack.accounting.nominalNodes += p.teachers[name].maxNodes;
                pack.accounting.visitedNodes +=
                  solved.result.search.visitedNodes;
              }
              pack.accounting.observations++;
              const id = `${split}:${fixture.id}:${actionIndex}`,
                pair = pairDecision(state, legalActions, results, {
                  id,
                  fixtureId: fixture.id,
                  seed,
                  split,
                  actionIndex,
                });
              if (!pair) {
                pack.skips.push({
                  id,
                  split,
                  reason: "paired-teacher-abstention",
                });
                return {
                  action: results.A512.result.actions[0] ?? null,
                  search: results.A512.result.search,
                };
              }
              callbacks.pair?.(pair);
              const dedup = registerPair(pair, pack.registry, fingerprint);
              if (dedup.keep) pack.examples[split].push(pair);
              else pack.duplicates.push(dedup);
              return {
                action: results.A512.result.actions[0],
                search: results.A512.result.search,
              };
            } catch (e) {
              thrown = e;
              throw e;
            }
          },
        );
        if (thrown) throw thrown;
        if (
          !played.ok ||
          played.value.ending.kind === "error" ||
          !verifyReplay(fixture, played.value).ok
        )
          throw new Error("Paired collection/replay failed");
        const e = { split, fixture, result: played.value };
        pack.episodes.push(e);
        callbacks.episode?.(e, pack.episodes.length);
      }
  for (const split of splits)
    if (!pack.examples[split].length) throw new Error("Empty paired split");
  if (
    pack.accounting.observations > (stage === "test" ? 48 : 480) ||
    pack.accounting.nominalNodes > (stage === "test" ? 73728 : 737280)
  )
    throw new Error("Paired collection cap exceeded");
  return pack;
}
/** Domain replay pauses at the next unknown action; the environment alone owns tapes. */
export function pauseEpisode(
  fixture: EpisodeFixture,
  history: ReturnType<EpisodePolicy>[],
) {
  let pending: Parameters<EpisodePolicy>[0] | null = null;
  const played = runEpisode(fixture, (observation) => {
    const known = history[observation.actionIndex];
    if (known) return structuredClone(known);
    if (pending) throw new Error("Multiple pause observations");
    pending = structuredClone(observation);
    return { action: null };
  });
  if (!played.ok || played.value.ending.kind === "error")
    throw new Error("Paused domain episode failed");
  return {
    result: played.value,
    pending: pending as Parameters<EpisodePolicy>[0] | null,
  };
}
export function verifyPairPack(pack: PairPack) {
  if (pack.status !== "PASS") throw new Error("Invalid paired pack status");
  for (const e of pack.episodes)
    if (!verifyReplay(e.fixture, e.result).ok)
      throw new Error("Paired episode replay mismatch");
  for (const split of ["train", "dev", "test"] as const)
    for (const pair of pack.examples[split]) {
      if (pair.split !== split || pair.stateKey !== stateKey(pair.state))
        throw new Error("Pair identity mismatch");
      const legal = getAvailableActions(pair.state);
      if (!legal.ok) throw new Error(legal.error.message);
      const recoded = pairDecision(
        pair.state,
        legal.actions,
        pair.teachers,
        pair,
      );
      if (
        !recoded ||
        JSON.stringify([
          recoded.observation,
          recoded.candidateKeys,
          recoded.candidateFeatures,
          recoded.labels,
        ]) !==
          JSON.stringify([
            pair.observation,
            pair.candidateKeys,
            pair.candidateFeatures,
            pair.labels,
          ])
      )
        throw new Error("Paired encoding/label mismatch");
      for (const n of TEACHERS)
        if (
          pair.candidateKeys[pair.labels[n]] !==
          actionKey(pair.teachers[n].result.actions[0])
        )
          throw new Error("Paired label key mismatch");
    }
}
