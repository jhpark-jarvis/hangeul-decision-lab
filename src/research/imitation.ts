import { getAvailableActions } from "../domain/game/actions";
import { validateGameState } from "../domain/game/game-state";
import type { GameAction, GameState } from "../domain/game/types";
import {
  getInitialCatalog,
  EVENT_CATALOG_VERSION,
} from "../domain/pieces/catalog";
import { solveTurn } from "../domain/solver/solver";
import type { SearchInfo, SolverConfig } from "../domain/solver/types";
import { runEpisode, stateKey, verifyReplay } from "./episode";
import { createSyntheticFixture } from "./fixture";
import type { EpisodeFixture, EpisodeResult, ResearchResult } from "./types";

export type Split = "train" | "dev" | "test";
export const SPLITS: Split[] = ["train", "dev", "test"];
const catalog = getInitialCatalog();
export const MODEL_SCHEMA = {
  version: "legal-candidate-cnn-v1",
  catalogVersion: EVENT_CATALOG_VERSION,
  pieceIds: catalog.map((p) => p.id),
  planeChannels: ["occupied", "single-cell-item", "reroll-item"],
  rows: 16,
  cols: 10,
  shapeRows: 5,
  shapeCols: 5,
  metadataSize: 140,
  candidateSize: 52,
  maxCandidates: 4096,
};
export type ModelObservation = { planes: number[][][]; metadata: number[] };
export type DatasetSpec = {
  trainSeeds: number[];
  devSeeds: number[];
  testSeeds: number[];
  families: ("sparse" | "pressure")[];
  maxActions: number;
  teacher: SolverConfig;
};
export type ModelExample = {
  id: string;
  fixtureId: string;
  seed: number;
  split: Split;
  actionIndex: number;
  state: GameState;
  stateKey: string;
  observation: ModelObservation;
  candidateKeys: string[];
  candidateFeatures: number[][];
  label: number;
  teacher: { search: SearchInfo; latencyMs: number };
};
export type DatasetPack = {
  schema: typeof MODEL_SCHEMA;
  spec: DatasetSpec;
  examples: Record<Split, ModelExample[]>;
  episodes: {
    split: Split;
    fixture: EpisodeFixture;
    result: EpisodeResult;
    replayValid: boolean;
  }[];
  skips: {
    split: Split;
    fixtureId: string;
    actionIndex: number;
    reason: "teacher-abstention";
  }[];
  status: "PASS" | "FAIL";
};
function oneHot(size: number, index: number) {
  return Array.from({ length: size }, (_, i) => Number(i === index));
}
function shapeVector(shape: boolean[][]): number[] {
  if (shape.length > 5 || shape.some((row) => row.length > 5))
    throw new Error("Shape exceeds schema");
  return Array.from({ length: 25 }, (_, i) =>
    Number(shape[Math.floor(i / 5)]?.[i % 5] ?? false),
  );
}
export function actionKey(action: GameAction): string {
  switch (action.type) {
    case "place-piece":
      return JSON.stringify([
        action.type,
        action.instanceId,
        action.pieceIndex,
        action.pieceId,
        action.row,
        action.col,
        action.variant,
      ]);
    case "single-cell":
      return JSON.stringify([action.type, action.row, action.col]);
    case "reroll":
      return JSON.stringify([
        action.type,
        action.instanceId,
        action.pieceIndex,
      ]);
  }
}
export function encodeObservation(
  input: unknown,
): ResearchResult<ModelObservation> {
  const validated = validateGameState(input);
  if (!validated.ok) return { ok: false, error: validated.error.message };
  const state = validated.state;
  for (const p of state.remainingPieces) {
    const reference = catalog.find((c) => c.id === p.piece.id);
    if (
      !reference ||
      JSON.stringify(reference.shape) !== JSON.stringify(p.piece.shape)
    )
      return { ok: false, error: "Noncanonical model piece" };
  }
  const planes = [
    state.board.map((row) => row.map(Number)),
    ...["single-cell", "reroll"].map((type) =>
      Array.from({ length: 16 }, (_, row) =>
        Array.from({ length: 10 }, (_, col) =>
          Number(
            state.hiddenItems.some(
              (item) =>
                item.type === type && item.row === row && item.col === col,
            ),
          ),
        ),
      ),
    ),
  ];
  const metadata: number[] = [];
  for (const slot of [0, 1, 2]) {
    const piece = state.remainingPieces.find(
      (p) => p.pieceIndex === slot,
    )?.piece;
    metadata.push(
      Number(Boolean(piece)),
      ...oneHot(19, piece ? MODEL_SCHEMA.pieceIds.indexOf(piece.id) : -1),
      ...(piece ? shapeVector(piece.shape) : Array(25).fill(0)),
    );
  }
  metadata.push(
    state.abilities.singleCell / 7,
    state.abilities.reroll / 7,
    ...oneHot(3, state.pendingReroll?.pieceIndex ?? -1),
  );
  return { ok: true, value: { planes, metadata } };
}
function candidateVector(action: GameAction, state: GameState): number[] {
  const type = ["place-piece", "single-cell", "reroll"].indexOf(action.type);
  const target =
    action.type === "single-cell"
      ? undefined
      : state.remainingPieces.find(
          (p) =>
            p.instanceId === action.instanceId &&
            p.pieceIndex === action.pieceIndex,
        );
  return [
    ...oneHot(3, type),
    ...oneHot(19, target ? MODEL_SCHEMA.pieceIds.indexOf(target.piece.id) : -1),
    ...oneHot(3, action.type === "single-cell" ? -1 : action.pieceIndex),
    action.type === "reroll" ? 0 : action.row / 15,
    action.type === "reroll" ? 0 : action.col / 9,
    ...shapeVector(
      action.type === "place-piece"
        ? action.variant
        : action.type === "single-cell"
          ? [[true]]
          : target!.piece.shape,
    ),
  ];
}
export function encodeDecision(
  state: GameState,
  legalActions: GameAction[],
  chosen: GameAction,
): ResearchResult<
  Pick<
    ModelExample,
    "observation" | "candidateKeys" | "candidateFeatures" | "label"
  >
> {
  const observation = encodeObservation(state);
  if (!observation.ok) return observation;
  const available = getAvailableActions(state);
  if (!available.ok) return { ok: false, error: available.error.message };
  const keys = legalActions.map(actionKey);
  if (
    !keys.length ||
    keys.length > MODEL_SCHEMA.maxCandidates ||
    new Set(keys).size !== keys.length ||
    JSON.stringify(keys) !== JSON.stringify(available.actions.map(actionKey))
  )
    return {
      ok: false,
      error: "Candidates must equal the full ordered domain legal list",
    };
  const label = keys.indexOf(actionKey(chosen));
  if (label < 0)
    return { ok: false, error: "Teacher action is not domain-legal" };
  return {
    ok: true,
    value: {
      observation: observation.value,
      candidateKeys: keys,
      candidateFeatures: legalActions.map((action) =>
        candidateVector(action, state),
      ),
      label,
    },
  };
}
export function validateDatasetSpec(
  input: unknown,
): ResearchResult<DatasetSpec> {
  if (!input || typeof input !== "object")
    return { ok: false, error: "Dataset spec required" };
  const s = input as DatasetSpec;
  const groups = [s.trainSeeds, s.devSeeds, s.testSeeds];
  if (
    groups.some(
      (g) =>
        !Array.isArray(g) ||
        !g.length ||
        g.some((n) => !Number.isInteger(n) || n < 0 || n > 0xffffffff),
    ) ||
    groups.flat().length > 32 ||
    new Set(groups.flat()).size !== groups.flat().length ||
    !Array.isArray(s.families) ||
    !s.families.length ||
    new Set(s.families).size !== s.families.length ||
    s.families.some((f) => !["sparse", "pressure"].includes(f)) ||
    !Number.isInteger(s.maxActions) ||
    s.maxActions < 1 ||
    s.maxActions > 12 ||
    groups.flat().length * s.families.length * s.maxActions > 512 ||
    !s.teacher ||
    !Number.isInteger(s.teacher.maxNodes) ||
    s.teacher.maxNodes < 1 ||
    s.teacher.maxNodes > 512 ||
    !Number.isInteger(s.teacher.maxAlternatives) ||
    s.teacher.maxAlternatives < 1 ||
    s.teacher.maxAlternatives > 3 ||
    typeof s.teacher.useMemoization !== "boolean"
  )
    return {
      ok: false,
      error: "Invalid, overlapping or oversized dataset protocol",
    };
  return { ok: true, value: structuredClone(s) };
}
export function buildImitationDataset(
  input: unknown,
  clock: () => number,
  onEpisode?: (completed: number) => void,
): ResearchResult<DatasetPack> {
  const validated = validateDatasetSpec(input);
  if (!validated.ok) return validated;
  const spec = validated.value;
  const pack: DatasetPack = {
    schema: structuredClone(MODEL_SCHEMA),
    spec,
    examples: { train: [], dev: [], test: [] },
    episodes: [],
    skips: [],
    status: "PASS",
  };
  for (const split of SPLITS)
    for (const seed of spec[`${split}Seeds`])
      for (const family of spec.families) {
        const fixture = createSyntheticFixture(seed, family, spec.maxActions);
        if (!fixture.ok) return fixture;
        const played = runEpisode(
          fixture.value,
          ({ state, legalActions, actionIndex }) => {
            const start = clock();
            const solved = solveTurn(state, catalog, spec.teacher);
            const latencyMs = clock() - start;
            if (!solved.ok || !Number.isFinite(latencyMs) || latencyMs < 0)
              throw new Error("Invalid teacher result/clock");
            const chosen = solved.result.actions[0];
            if (!chosen) {
              pack.skips.push({
                split,
                fixtureId: fixture.value.id,
                actionIndex,
                reason: "teacher-abstention",
              });
              return { action: null, search: solved.result.search };
            }
            const encoded = encodeDecision(state, legalActions, chosen);
            if (!encoded.ok) throw new Error(encoded.error);
            pack.examples[split].push({
              id: `${split}:${fixture.value.id}:${actionIndex}`,
              fixtureId: fixture.value.id,
              seed,
              split,
              actionIndex,
              state: structuredClone(state),
              stateKey: stateKey(state),
              ...encoded.value,
              teacher: { search: solved.result.search, latencyMs },
            });
            return { action: chosen, search: solved.result.search };
          },
        );
        if (!played.ok) return played;
        const replayValid = verifyReplay(fixture.value, played.value).ok;
        if (!replayValid || played.value.ending.kind === "error")
          pack.status = "FAIL";
        pack.episodes.push({
          split,
          fixture: fixture.value,
          result: played.value,
          replayValid,
        });
        onEpisode?.(pack.episodes.length);
      }
  if (SPLITS.some((split) => !pack.examples[split].length))
    pack.status = "FAIL";
  return { ok: true, value: pack };
}
