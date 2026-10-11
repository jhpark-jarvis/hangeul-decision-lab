import { applyAction, getAvailableActions } from "../domain/game/actions";
import type { GameAction, GameState } from "../domain/game/types";
import { actionKey, encodeDecision, MODEL_SCHEMA } from "./imitation";

export const OUTCOME_SCHEMA = {
  ...MODEL_SCHEMA,
  version: "legal-candidate-outcome-v2",
  candidateSize: 74,
};

/** One known action only. A reroll stops pending; no future is resolved. */
export function actionOutcome(state: GameState, action: GameAction): number[] {
  const next = applyAction(state, action);
  if (!next.ok) throw new Error(next.error.message);
  const { info } = next;
  const rows = next.state.board.map((row) => row.filter(Boolean).length);
  const features = [
    info.clearedRows.length / 16,
    info.acquiredItems.filter((item) => item.type === "single-cell").length / 3,
    info.acquiredItems.filter((item) => item.type === "reroll").length / 3,
    info.retainedItems.length / 3,
    rows.reduce((a, b) => a + b, 0) / 160,
    Number(Boolean(next.state.pendingReroll)),
    ...rows.map((n) => n / 10),
  ];
  if (
    features.length !== 22 ||
    features.some((n) => !Number.isFinite(n) || n < 0 || n > 1)
  )
    throw new Error("Outcome feature range/schema");
  return features;
}

export type OutcomeSource = {
  state: GameState;
  observation: { planes: number[][][]; metadata: number[] };
  candidateKeys: string[];
  candidateFeatures: number[][];
  labels: { A512: number; A1024: number };
};

/** Check the old ordered mask and label before making exactly one transition per candidate. */
export function extendOutcome(
  example: OutcomeSource,
  transitionBudget: number,
) {
  const legal = getAvailableActions(example.state);
  if (!legal.ok) throw new Error(legal.error.message);
  if (
    !Number.isInteger(transitionBudget) ||
    legal.actions.length > transitionBudget
  )
    throw new Error("Outcome transition budget");
  const label = example.labels.A512;
  if (!Number.isInteger(label) || label < 0 || label >= legal.actions.length)
    throw new Error("Outcome source label");
  const encoded = encodeDecision(
    example.state,
    legal.actions,
    legal.actions[label],
  );
  if (!encoded.ok) throw new Error(encoded.error);
  for (const key of [
    "observation",
    "candidateKeys",
    "candidateFeatures",
  ] as const)
    if (JSON.stringify(encoded.value[key]) !== JSON.stringify(example[key]))
      throw new Error(`Outcome source mismatch: ${key}`);
  if (
    encoded.value.label !== label ||
    legal.actions.some((a, i) => actionKey(a) !== example.candidateKeys[i])
  )
    throw new Error("Outcome source order/label");
  return {
    candidateFeatures: legal.actions.map((action, i) => [
      ...example.candidateFeatures[i],
      ...actionOutcome(example.state, action),
    ]),
    transitions: legal.actions.length,
  };
}
