import { getAvailableActions } from "../domain/game/actions";
import { actionKey, encodeDecision } from "./imitation";
import { actionOutcome } from "./action-outcome";
import { encodeFootprints } from "./spatial-footprint";
import { pointFixtures, pointReference } from "./point-simulation";
import type { PointReference } from "./point-simulation";

export function fitSubsetIndices() {
  return Array.from({ length: 16 }, (_, r) => 10 * r + ((3 * r) % 10));
}
export function uniqueFitLabel(reference: PointReference, keys: string[]) {
  if (!reference.complete || reference.optimalFirstActions?.length !== 1)
    throw new Error("Fit reference must be complete and unique");
  const target = reference.optimalFirstActions[0];
  if (
    new Set(keys).size !== keys.length ||
    keys.filter((k) => k === target).length !== 1
  )
    throw new Error("Fit reference/order mismatch");
  return keys.indexOf(target);
}
/** Called only once for the approved isolated fit materialization, never from app. */
export function buildControlledPointFit(checkBudget: () => void) {
  const fixtures = pointFixtures();
  let nodes = 0,
    transitions = 0;
  const examples = fitSubsetIndices().map((index, row) => {
    checkBudget();
    const fixture = fixtures[index],
      before = JSON.stringify(fixture.state);
    const reference = pointReference(fixture.state, 256);
    nodes += reference.visitedNodes;
    if (nodes > 4096) throw new Error("Fit reference node cap");
    const legal = getAvailableActions(fixture.state);
    if (
      !legal.ok ||
      legal.phase !== "playing" ||
      legal.actions.length !== 31 ||
      legal.actions.some((a) => a.type !== "single-cell")
    )
      throw new Error("Fit requires 31 point-only legal candidates");
    const keys = legal.actions.map(actionKey),
      label = uniqueFitLabel(reference, keys);
    const target = legal.actions[label];
    if (
      target.type !== "single-cell" ||
      target.row !== row ||
      target.col !== (3 * row) % 10
    )
      throw new Error("Fit target differs from controlled geometry");
    const encoded = encodeDecision(fixture.state, legal.actions, target);
    if (!encoded.ok) throw new Error(encoded.error);
    const candidateFeatures = legal.actions.map((action, i) => {
      if (++transitions > 496) throw new Error("Fit outcome transition cap");
      return [
        ...encoded.value.candidateFeatures[i],
        ...actionOutcome(fixture.state, action),
      ];
    });
    if (
      candidateFeatures.some(
        (f, i) =>
          f.length !== 74 ||
          f.some((v) => !Number.isFinite(v) || v < 0 || v > 1) ||
          f[52] !== (i === label ? 1 / 16 : 0),
      )
    )
      throw new Error("Fit outcome shortcut/schema mismatch");
    const footprints = encodeFootprints(keys, 31).footprints;
    if (JSON.stringify(fixture.state) !== before)
      throw new Error("Fit state mutated");
    checkBudget();
    return {
      id: `controlled-fit-${row}`,
      observation: encoded.value.observation,
      candidateKeys: keys,
      candidateFeatures,
      footprints,
      label,
      reference: {
        complete: reference.complete,
        unique: true,
        visitedNodes: reference.visitedNodes,
      },
    };
  });
  if (transitions !== 496) throw new Error("Fit transition accounting");
  return {
    schemaVersion: 1,
    id: "controlled-point-fit-data-v1",
    split: "fit-only",
    counts: {
      examples: 16,
      candidates: 496,
      referenceNodes: nodes,
      referenceNodeCap: 4096,
      outcomeTransitions: transitions,
      dfsCalls: 0,
    },
    examples,
  };
}
