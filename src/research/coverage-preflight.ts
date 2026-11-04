import { getAvailableActions } from "../domain/game/actions";
import { getInitialCatalog } from "../domain/pieces/catalog";
import type { GameState } from "../domain/game/types";
import { actionKey, encodeDecision } from "./imitation";
import { actionOutcome } from "./action-outcome";
import { encodeFootprints } from "./spatial-footprint";
import { pointReference } from "./point-simulation";
import { uniqueFitLabel } from "./controlled-point-fit";

export type CoverageSplit = "control" | "diverse" | "dev" | "final";
export type CoverageReserve = (
  kind: "reference" | "outcome",
  split: CoverageSplit,
  row: number,
  index: number,
) => void;
/** Exactly one authorized build; the owner supplies the shared two-build ledger. */
export function buildCoverage(reserve: CoverageReserve, check: () => void) {
  const line = getInitialCatalog().find((p) => p.id === "LINE_3")!;
  let nodes = 0,
    transitions = 0;
  const splits = Object.fromEntries(
    (["control", "diverse", "dev", "final"] as const).map((split) => [
      split,
      Array.from({ length: 16 }, (_, row) => {
        check();
        const a =
            split === "control"
              ? 0
              : split === "diverse"
                ? 1 + (row % 3)
                : split === "dev"
                  ? 5
                  : 7,
          phase = split === "dev" ? 1 : split === "final" ? 2 : 0;
        const board = Array.from({ length: 16 }, (_, r) => {
          const order = (r - row + 16) % 16,
            k = order <= a ? 2 : order > 15 - a ? 4 : 3;
          const offsets =
            r === row
              ? [0]
              : k === 2
                ? [0, 1]
                : k === 3
                  ? [0, 1, 4]
                  : [0, 1, 4, 5];
          const holes = new Set(offsets.map((n) => (3 * r + phase + n) % 10));
          return Array.from({ length: 10 }, (_, c) => !holes.has(c));
        });
        const state: GameState = {
          board,
          remainingPieces: [
            {
              instanceId: "coverage-line",
              pieceIndex: 0,
              piece: structuredClone(line),
            },
          ],
          hiddenItems: [],
          abilities: { singleCell: 1, reroll: 0 },
          pendingReroll: null,
        };
        const before = JSON.stringify(state),
          legal = getAvailableActions(state);
        if (
          !legal.ok ||
          legal.phase !== "playing" ||
          legal.actions.length !== 46 ||
          legal.actions.some((x) => x.type !== "single-cell")
        )
          throw new Error("Coverage requires 46 point-only legal actions");
        reserve("reference", split, row, 0);
        const reference = pointReference(state, 256);
        nodes += reference.visitedNodes;
        if (nodes > 16384) throw new Error("Per-build reference cap");
        const keys = legal.actions.map(actionKey),
          label = uniqueFitLabel(reference, keys),
          target = legal.actions[label];
        if (
          target.type !== "single-cell" ||
          target.row !== row ||
          target.col !== (3 * row + phase) % 10
        )
          throw new Error("Coverage reference target mismatch");
        const encoded = encodeDecision(state, legal.actions, target);
        if (!encoded.ok) throw new Error(encoded.error);
        const candidateFeatures = legal.actions.map((action, index) => {
          reserve("outcome", split, row, index);
          transitions++;
          if (transitions > 2944) throw new Error("Per-build outcome cap");
          const f = [
            ...encoded.value.candidateFeatures[index],
            ...actionOutcome(state, action),
          ];
          if (
            f.length !== 74 ||
            f.some((v) => !Number.isFinite(v) || v < 0 || v > 1) ||
            f[52] !== Number(index === label) / 16
          )
            throw new Error("Coverage feature schema/shortcut mismatch");
          return f;
        });
        if (JSON.stringify(state) !== before)
          throw new Error("Coverage input mutated");
        check();
        return {
          id: `coverage-${split}-${row}`,
          split,
          row,
          a,
          phase,
          observation: encoded.value.observation,
          candidateKeys: keys,
          candidateFeatures,
          footprints: encodeFootprints(keys, 46).footprints,
          label,
          reference: {
            complete: true,
            unique: true,
            visitedNodes: reference.visitedNodes,
          },
        };
      }),
    ]),
  );
  if (transitions !== 2944) throw new Error("Coverage transition accounting");
  return {
    schemaVersion: 1,
    id: "coverage-data-v1",
    counts: {
      examples: 64,
      candidates: 2944,
      referenceNodes: nodes,
      referenceNodeCap: 16384,
      outcomeTransitions: transitions,
      dfsCalls: 0,
    },
    splits,
  };
}
