import { getAvailableActions } from "../domain/game/actions";
import { actionKey, encodeDecision } from "./imitation";
import { actionOutcome } from "./action-outcome";
import { encodeFootprints } from "./spatial-footprint";
import { pointFixtures, pointReference } from "./point-simulation";
import { fitSubsetIndices, uniqueFitLabel } from "./controlled-point-fit";

/** Isolated evaluation materialization; never app data or training data. */
export function buildFrozenPointTransfer(checkBudget: () => void) {
  const fixtures = pointFixtures(),
    fit = new Set(fitSubsetIndices());
  let nodes = 0,
    transitions = 0;
  const examples = fixtures.flatMap((fixture, fixtureIndex) => {
    if (fit.has(fixtureIndex)) return [];
    checkBudget();
    const before = JSON.stringify(fixture.state);
    const reference = pointReference(fixture.state, 256);
    nodes += reference.visitedNodes;
    if (nodes > 36864) throw new Error("Transfer reference node cap");
    const legal = getAvailableActions(fixture.state);
    if (
      !legal.ok ||
      legal.actions.length !== 31 ||
      legal.actions.some((a) => a.type !== "single-cell")
    )
      throw new Error("Transfer full legal point mask");
    const keys = legal.actions.map(actionKey),
      label = uniqueFitLabel(reference, keys);
    const target = legal.actions[label];
    if (
      target.type !== "single-cell" ||
      target.row !== fixture.target.row ||
      target.col !== fixture.target.col
    )
      throw new Error("Transfer unique target geometry");
    const encoded = encodeDecision(fixture.state, legal.actions, target);
    if (!encoded.ok) throw new Error(encoded.error);
    const candidateFeatures = legal.actions.map((action, i) => {
      if (++transitions > 4464) throw new Error("Transfer transition cap");
      const features = [
        ...encoded.value.candidateFeatures[i],
        ...actionOutcome(fixture.state, action),
      ];
      if (
        features.length !== 74 ||
        features.some((v) => !Number.isFinite(v) || v < 0 || v > 1) ||
        features[52] !== (i === label ? 1 / 16 : 0)
      )
        throw new Error("Transfer feature schema/shortcut");
      return features;
    });
    if (JSON.stringify(fixture.state) !== before)
      throw new Error("Transfer input mutated");
    checkBudget();
    return [
      {
        id: `controlled-transfer-${fixtureIndex}`,
        fixtureIndex,
        observation: encoded.value.observation,
        candidateKeys: keys,
        candidateFeatures,
        footprints: encodeFootprints(keys, 31).footprints,
        label,
        reference: {
          complete: reference.complete,
          unique: true,
          visitedNodes: reference.visitedNodes,
        },
      },
    ];
  });
  if (examples.length !== 144 || transitions !== 4464)
    throw new Error("Transfer exact partition/counts");
  return {
    schemaVersion: 1,
    id: "frozen-point-transfer-data-v1",
    split: "controlled-transfer-only",
    counts: {
      examples: 144,
      candidates: 4464,
      referenceNodes: nodes,
      referenceNodeCap: 36864,
      outcomeTransitions: transitions,
      dfsCalls: 0,
    },
    examples,
  };
}
