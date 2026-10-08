import { describe, expect, it } from "vitest";
import { getAvailableActions } from "../../src/domain/game/actions";
import {
  actionKey,
  encodeObservation,
  encodeDecision,
  validateDatasetSpec,
  buildImitationDataset,
  MODEL_SCHEMA,
  type DatasetSpec,
} from "../../src/research/imitation";
import { game, instance } from "../game/fixtures";
const spec = (): DatasetSpec => ({
  trainSeeds: [1],
  devSeeds: [2],
  testSeeds: [3],
  families: ["sparse"],
  maxActions: 1,
  teacher: { maxNodes: 8, maxAlternatives: 1, useMemoization: true },
});
const legal = (state: ReturnType<typeof game>) => {
  const r = getAvailableActions(state);
  if (!r.ok) throw new Error(r.error.message);
  return r.actions;
};
describe("CNN imitation contract", () => {
  it("encodes exact board/item planes, slot/ability fields and no future/seed/key/label", () => {
    const state = game({
      abilities: { singleCell: 2, reroll: 3 },
      hiddenItems: [{ type: "reroll", row: 2, col: 3 }],
    });
    state.board[1][2] = true;
    const result = encodeObservation(state);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(Object.keys(result.value)).toEqual(["planes", "metadata"]);
    expect(
      result.value.planes.map((p) => p.flat().reduce((a, b) => a + b, 0)),
    ).toEqual([1, 0, 1]);
    expect(result.value.planes[0][1][2]).toBe(1);
    expect(result.value.planes[2][2][3]).toBe(1);
    expect(result.value.metadata).toHaveLength(MODEL_SCHEMA.metadataSize);
    expect(result.value.metadata.slice(135, 137)).toEqual([2 / 7, 3 / 7]);
    expect(encodeObservation({} as unknown).ok).toBe(false);
    const wrong = game();
    wrong.remainingPieces[0].piece = {
      ...wrong.remainingPieces[0].piece,
      shape: [[true, true]],
    };
    expect(encodeObservation(wrong).ok).toBe(false);
  });
  it("roundtrips all legal action types, stable shape/instance keys and rejects a pruned mask", () => {
    const state = game({ abilities: { singleCell: 1, reroll: 1 } }),
      actions = legal(state);
    for (const type of ["place-piece", "single-cell", "reroll"]) {
      const action = actions.find((a) => a.type === type)!;
      const result = encodeDecision(state, actions, action);
      expect(result.ok).toBe(true);
      if (!result.ok) continue;
      expect(result.value.candidateKeys[result.value.label]).toBe(
        actionKey(action),
      );
      expect(
        result.value.candidateFeatures.every(
          (v) => v.length === 52 && v.every(Number.isFinite),
        ),
      ).toBe(true);
      const annotated = { ...action, rotation: 90 };
      expect(actionKey(annotated)).toBe(actionKey(action));
      expect(encodeDecision(state, [action], action).ok).toBe(false);
    }
    expect(
      encodeDecision(state, actions, { type: "single-cell", row: -1, col: 0 })
        .ok,
    ).toBe(false);
    expect(encodeDecision(state, [...actions, actions[0]], actions[0]).ok).toBe(
      false,
    );
    const a = actions.find((a) => a.type === "place-piece")!;
    if (a.type !== "place-piece") return;
    expect(actionKey({ ...a, instanceId: "other" })).not.toBe(actionKey(a));
    const renamed = structuredClone(state);
    renamed.remainingPieces.forEach((p) => {
      p.instanceId = `another-${p.pieceIndex}`;
    });
    expect(encodeObservation(renamed)).toEqual(encodeObservation(state));
    const original = encodeDecision(state, actions, actions[0]);
    const renamedActions = legal(renamed);
    const changed = encodeDecision(renamed, renamedActions, renamedActions[0]);
    if (!original.ok || !changed.ok) throw new Error("Invalid fixture");
    expect(changed.value.candidateFeatures).toEqual(
      original.value.candidateFeatures,
    );
    expect(changed.value.candidateKeys).not.toEqual(
      original.value.candidateKeys,
    );
  });
  it("keeps item plane independent from occupancy and slot presence distinct from empty padding", () => {
    const state = game({
      remainingPieces: [instance(2, "DOT")],
      abilities: { singleCell: 0, reroll: 0 },
      hiddenItems: [{ row: 0, col: 0, type: "single-cell" }],
    });
    state.board[0][0] = true;
    const r = encodeObservation(state);
    if (!r.ok) throw new Error(r.error);
    expect(r.value.planes[0][0][0]).toBe(1);
    expect(r.value.planes[1][0][0]).toBe(1);
    expect(r.value.metadata.slice(0, 90).every((v) => v === 0)).toBe(true);
    expect(r.value.metadata[90]).toBe(1);
  });
  it("rejects overlapping seeds, invalid inputs, nondevelopment limits and incomplete splits", () => {
    for (const bad of [
      null,
      {},
      { ...spec(), devSeeds: [1] },
      { ...spec(), trainSeeds: [] },
      { ...spec(), testSeeds: [NaN] },
      { ...spec(), families: ["actual-game"] },
      { ...spec(), maxActions: 100 },
      { ...spec(), teacher: { ...spec().teacher, maxNodes: 8192 } },
    ])
      expect(validateDatasetSpec(bad).ok).toBe(false);
  });
  it("generates separate splits through actual DFS, preserves complete diagnostics and replays every episode", () => {
    const result = buildImitationDataset(spec(), () => 1);
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.value.status).toBe("PASS");
    expect(result.value.episodes).toHaveLength(3);
    expect(result.value.episodes.every((e) => e.replayValid)).toBe(true);
    for (const split of ["train", "dev", "test"] as const) {
      const example = result.value.examples[split][0];
      expect(example.split).toBe(split);
      expect(example.teacher.search.searchComplete).toBe(false);
      expect(example.teacher.latencyMs).toBe(0);
      const event = result.value.episodes.find((e) => e.split === split)!.result
        .events[0];
      expect(event.type).toBe("action");
      if (event.type !== "action") throw new Error("Missing action event");
      expect(example.candidateKeys[example.label]).toBe(
        actionKey(event.action),
      );
    }
    const error = buildImitationDataset(spec(), () => NaN);
    expect(error.ok && error.value.status).toBe("FAIL");
  });
  it("preserves diagnostics for teacher abstention without inventing a label", () => {
    const input = spec();
    input.teacher.maxNodes = 1;
    const built = buildImitationDataset(input, () => 1);
    expect(built.ok).toBe(true);
    if (!built.ok) return;
    expect(built.value.status).toBe("FAIL");
    expect(built.value.skips).toHaveLength(3);
    expect(Object.values(built.value.examples).flat()).toHaveLength(0);
    for (const skip of built.value.skips) {
      expect(skip.teacher.search.searchComplete).toBe(false);
      expect(skip.teacher.search.maxNodes).toBe(1);
      expect(skip.teacher.latencyMs).toBe(0);
    }
  });
});
