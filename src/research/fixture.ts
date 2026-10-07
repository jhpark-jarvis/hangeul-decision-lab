import {
  createEmptyBoard,
  BOARD_HEIGHT,
  BOARD_WIDTH,
} from "../domain/board/board";
import { validateGameState } from "../domain/game/game-state";
import type { GameState, PieceIndex } from "../domain/game/types";
import {
  EVENT_CATALOG_VERSION,
  getInitialCatalog,
} from "../domain/pieces/catalog";
import {
  RNG_VERSION,
  RULES_VERSION,
  SYNTHETIC_PROFILE,
  type EpisodeFixture,
  type ResearchResult,
} from "./types";

/** Indexed streams: supply consumption never advances the board or reroll stream. */
export function indexedUint32(
  seed: number,
  stream: number,
  index: number,
): number {
  let value = (seed ^ stream ^ Math.imul(index + 1, 0x9e3779b9)) >>> 0;
  value = Math.imul(value ^ (value >>> 16), 0x85ebca6b);
  value = Math.imul(value ^ (value >>> 13), 0xc2b2ae35);
  return (value ^ (value >>> 16)) >>> 0;
}
export function suppliedPieces(id: string, setIndex: number, ids: string[]) {
  const catalog = getInitialCatalog();
  return ids.map((pieceId, slot) => ({
    instanceId: `${id}:set-${setIndex}:slot-${slot}`,
    pieceIndex: slot as PieceIndex,
    piece: catalog.find((piece) => piece.id === pieceId)!,
  }));
}
const uint32 = (value: unknown): value is number =>
  typeof value === "number" &&
  Number.isInteger(value) &&
  value >= 0 &&
  value <= 0xffffffff;

export function validateFixture(
  input: unknown,
): ResearchResult<EpisodeFixture> {
  if (!input || typeof input !== "object")
    return { ok: false, error: "Fixture required" };
  const f = input as EpisodeFixture;
  if (
    f.schemaVersion !== 1 ||
    typeof f.id !== "string" ||
    !/^[a-z0-9-]+$/.test(f.id) ||
    !["synthetic-generated", "synthetic-scripted"].includes(f.provenance) ||
    f.catalogVersion !== EVENT_CATALOG_VERSION ||
    f.rulesVersion !== RULES_VERSION ||
    f.profile !== SYNTHETIC_PROFILE ||
    f.rngVersion !== RNG_VERSION ||
    !uint32(f.seed) ||
    !Number.isSafeInteger(f.maxActions) ||
    f.maxActions < 1 ||
    f.maxActions > 1000
  ) {
    return {
      ok: false,
      error: "Invalid fixture identity/profile/version/seed/horizon",
    };
  }
  const state = validateGameState(f.initialState);
  if (!state.ok) return { ok: false, error: state.error.message };
  const catalog = getInitialCatalog();
  if (
    state.state.remainingPieces.some(
      ({ piece }) =>
        JSON.stringify(piece) !==
        JSON.stringify(catalog.find((entry) => entry.id === piece.id)),
    )
  ) {
    return {
      ok: false,
      error: "Initial pieces must match the versioned catalog",
    };
  }
  if (
    !Array.isArray(f.nextPieceSets) ||
    f.nextPieceSets.length > 1001 ||
    f.nextPieceSets.some(
      (set) =>
        !Array.isArray(set) ||
        set.length !== 3 ||
        set.some((id) => !catalog.some((piece) => piece.id === id)),
    ) ||
    !Array.isArray(f.rerollTickets) ||
    f.rerollTickets.length > 1000 ||
    f.rerollTickets.some(
      (row) => !Array.isArray(row) || row.length !== 3 || !row.every(uint32),
    )
  ) {
    return { ok: false, error: "Invalid synthetic supply tape" };
  }
  return {
    ok: true,
    value: {
      schemaVersion: 1,
      id: f.id,
      provenance: f.provenance,
      catalogVersion: f.catalogVersion,
      rulesVersion: f.rulesVersion,
      profile: f.profile,
      rngVersion: f.rngVersion,
      seed: f.seed,
      maxActions: f.maxActions,
      initialState: state.state,
      nextPieceSets: f.nextPieceSets.map((row) => [...row]),
      rerollTickets: f.rerollTickets.map((row) => [...row]),
    },
  };
}

export function createSyntheticFixture(
  seed: number,
  family: "sparse" | "pressure",
  maxActions = 12,
): ResearchResult<EpisodeFixture> {
  if (
    !uint32(seed) ||
    !["sparse", "pressure"].includes(family) ||
    !Number.isSafeInteger(maxActions) ||
    maxActions < 1 ||
    maxActions > 1000
  ) {
    return { ok: false, error: "Invalid generator seed/family/horizon" };
  }
  const catalog = getInitialCatalog();
  const id = `${family}-${seed}`;
  const board = createEmptyBoard();
  for (let row = 0; row < BOARD_HEIGHT; row++) {
    for (let col = 0; col < BOARD_WIDTH; col++) {
      const draw =
        indexedUint32(seed, 0xc2b2ae35, row * BOARD_WIDTH + col) / 2 ** 32;
      board[row][col] = draw < (family === "sparse" ? 0.08 : 0.65);
    }
    board[row][indexedUint32(seed, 0x27d4eb2f, row) % BOARD_WIDTH] = false;
  }
  const nextPieceSets = Array.from(
    { length: maxActions + 1 },
    (_, setIndex) =>
      [0, 1, 2].map(
        (slot) =>
          catalog[
            Math.floor(
              (indexedUint32(seed, 0x9e3779b9, setIndex * 3 + slot) / 2 ** 32) *
                catalog.length,
            )
          ].id,
      ) as [string, string, string],
  );
  const initialState: GameState = {
    board,
    remainingPieces: suppliedPieces(id, 0, nextPieceSets[0]),
    hiddenItems: [
      { row: 15, col: 0, type: "single-cell" },
      { row: 14, col: 9, type: "reroll" },
    ],
    abilities: { singleCell: 1, reroll: 1 },
    pendingReroll: null,
  };
  return validateFixture({
    schemaVersion: 1,
    id,
    provenance: "synthetic-generated",
    catalogVersion: EVENT_CATALOG_VERSION,
    rulesVersion: RULES_VERSION,
    profile: SYNTHETIC_PROFILE,
    rngVersion: RNG_VERSION,
    seed,
    maxActions,
    initialState,
    nextPieceSets: nextPieceSets.slice(1),
    rerollTickets: Array.from({ length: maxActions }, (_, index) =>
      [0, 1, 2].map((slot) =>
        indexedUint32(seed, 0x85ebca6b, index * 3 + slot),
      ),
    ),
  });
}
