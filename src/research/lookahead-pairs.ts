import { getInitialCatalog } from "../domain/pieces/catalog";
import { applyAction, getAvailableActions } from "../domain/game/actions";
import type { GameAction, GameState } from "../domain/game/types";
import { compareEvaluations, evaluateState } from "../domain/solver/evaluator";
import type { PathRewards } from "../domain/solver/types";
import { actionKey, encodeDecision } from "./imitation";
import { actionOutcome } from "./action-outcome";
import { encodeFootprints } from "./spatial-footprint";
import { conditionalReference } from "./conditional-reference";

export type PairDescriptor = {
  familyIndex: number;
  firstRow: number;
  secondRow: number;
  column: number;
  phase: number;
  arm: string;
};
export type PairReservation = {
  kind: "root" | "edge" | "outcome";
  rootIndex: number;
  depth: number;
  key: string | null;
};
function requireValue(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

/** Only the approved owner materializes these states, after a state reservation. */
export function buildLookaheadPairCase(
  d: PairDescriptor,
  reserve: (r: PairReservation) => void,
  saveInitial: (s: GameState) => void,
) {
  requireValue(
    [0, 2, 4, 6].includes(d.firstRow) &&
      [0, 2, 5, 7].includes(d.column) &&
      d.secondRow === d.firstRow + 7 &&
      d.phase === (3 * d.firstRow + d.column) % 10 &&
      ["early-aligned", "late-aligned"].includes(d.arm),
    "Fixed descriptor",
  );
  const { column: c } = d;
  const board = Array.from({ length: 16 }, (_, r) =>
    Array.from(
      { length: 10 },
      (_, col) =>
        col !== (3 * r + d.phase) % 10 && col !== (3 * r + d.phase + 1) % 10,
    ),
  );
  for (const [which, r] of [d.firstRow, d.secondRow].entries()) {
    board[r] = Array.from(
      { length: 10 },
      (_, col) => col !== c + (which === 0 ? 1 : 2),
    );
    const aligned = (which === 0) === (d.arm === "early-aligned");
    board[r + 1] = Array.from(
      { length: 10 },
      (_, col) => col !== (aligned ? c : c + 1) && col !== c + 2,
    );
    board[r + 2] = Array.from(
      { length: 10 },
      (_, col) => col < c || col > c + 2,
    );
  }
  const catalog = getInitialCatalog(),
    piece = catalog.find((p) => p.id === "MIEUM");
  requireValue(
    piece && piece.shape.flat().filter(Boolean).length === 8,
    "Frozen MIEUM",
  );
  const state: GameState = {
    board,
    remainingPieces: ([0, 1] as const).map((i) => ({
      instanceId: `counterfactual-piece-${i}`,
      pieceIndex: i,
      piece: structuredClone(piece),
    })),
    hiddenItems: [],
    abilities: { singleCell: 1, reroll: 0 },
    pendingReroll: null,
  };
  saveInitial(state);
  const before = JSON.stringify(state),
    legal = getAvailableActions(state);
  requireValue(
    legal.ok &&
      legal.phase === "playing" &&
      legal.actions.length === 32 &&
      legal.actions.every((a) => a.type === "single-cell"),
    "Initial32 point-only actions",
  );
  requireValue(
    board.every((row) => row.some((v) => !v)) &&
      board.flat().filter((v) => !v).length === 32,
    "No completed rows /32 holes",
  );
  const keys = legal.actions.map(actionKey);
  const reference = conditionalReference(
    state,
    { clearedRows: 0, acquiredItems: 0 } as PathRewards,
    {
      actions(s: GameState) {
        const result = getAvailableActions(s);
        if (!result.ok) throw new Error(result.error.message);
        requireValue(
          result.phase === "playing" ||
            result.phase === "gameover" ||
            result.phase === "await-next-pieces",
          "No future lifecycle",
        );
        return result.actions;
      },
      advance(s: GameState, a: GameAction, rewards: PathRewards) {
        const t = applyAction(s, a);
        if (!t.ok) throw new Error(t.error.message);
        return {
          state: t.state,
          rewards: {
            clearedRows: rewards.clearedRows + t.info.clearedRows.length,
            acquiredItems: rewards.acquiredItems + t.info.acquiredItems.length,
          },
        };
      },
      evaluate(s: GameState, rewards: PathRewards) {
        const result = evaluateState(s, catalog, rewards);
        if (!result.ok) throw new Error(result.error.message);
        return result.evaluation;
      },
      compare: compareEvaluations,
      reserve(r) {
        reserve({
          kind: r.kind,
          rootIndex: r.rootIndex,
          depth: r.depth,
          key: r.action ? actionKey(r.action) : null,
        });
      },
    },
    64,
    3,
  );
  const label = reference.qualityIndices[0],
    encoded = encodeDecision(state, legal.actions, legal.actions[label]);
  requireValue(encoded.ok, "Original encoding");
  const candidateFeatures = legal.actions.map((a, i) => {
    reserve({ kind: "outcome", rootIndex: i, depth: 1, key: keys[i] });
    return [...encoded.value.candidateFeatures[i], ...actionOutcome(state, a)];
  });
  requireValue(
    candidateFeatures.every(
      (f) =>
        f.length === 74 &&
        f.every((x) => Number.isFinite(x) && x >= 0 && x <= 1),
    ),
    "Feature74 schema",
  );
  requireValue(JSON.stringify(state) === before, "Initial state mutated");
  return {
    descriptor: d,
    state,
    observation: encoded.value.observation,
    candidateKeys: keys,
    candidateFeatures,
    footprints: encodeFootprints(keys, 32).footprints,
    label,
    reference: {
      ...reference,
      leaves: reference.leaves.map((l) => ({
        ...l,
        path: l.path.map(actionKey),
      })),
    },
  };
}
