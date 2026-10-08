import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import assert from "node:assert/strict";
import { compileResearch } from "./runtime.mjs";
const root = resolve(fileURLToPath(new URL("../../", import.meta.url)));
const [directory, predictionsPath, outputPath] = process.argv.slice(2);
if (!directory)
  throw new Error("Use dataset-directory [predictions.json] [new-report.json]");
const read = (name) => readFileSync(resolve(directory, name), "utf8");
const body = read("dataset.json"),
  pack = JSON.parse(body),
  manifest = JSON.parse(read("manifest.json")),
  protocol = JSON.parse(read("protocol.json"));
const sha = (s) => createHash("sha256").update(s).digest("hex");
assert.equal(sha(body), manifest.dataSha256);
assert.equal(sha(JSON.stringify(protocol)), manifest.protocolSha256);
assert.equal(pack.status, "PASS");
assert.deepEqual(pack.spec, protocol.dataset);
const runtime = compileResearch(root, "src/research/imitation.ts");
const { encodeDecision, MODEL_SCHEMA, actionKey, SPLITS } = await import(
  runtime.url
);
const { getAvailableActions, applyAction } = await import(
  new URL("../domain/game/actions.mjs", runtime.url)
);
const { verifyReplay, stateKey } = await import(
  new URL("./episode.mjs", runtime.url)
);
assert.deepEqual(pack.schema, MODEL_SCHEMA);
const ids = new Set(),
  seeds = new Set();
let checked = 0;
for (const split of SPLITS) {
  for (const seed of pack.spec[`${split}Seeds`]) {
    assert(!seeds.has(seed));
    seeds.add(seed);
  }
  const episodes = pack.episodes.filter((e) => e.split === split);
  assert.equal(
    episodes.length,
    pack.spec[`${split}Seeds`].length * pack.spec.families.length,
  );
  let actionCount = 0;
  for (const episode of episodes) {
    assert.equal(episode.replayValid, true);
    assert.equal(verifyReplay(episode.fixture, episode.result).ok, true);
    actionCount += episode.result.events.filter(
      (e) => e.type === "action",
    ).length;
  }
  assert.equal(actionCount, pack.examples[split].length);
  for (const example of pack.examples[split]) {
    assert(!ids.has(example.id));
    ids.add(example.id);
    assert.equal(example.split, split);
    assert(pack.spec[`${split}Seeds`].includes(example.seed));
    assert.equal(example.stateKey, stateKey(example.state));
    const available = getAvailableActions(example.state);
    assert.equal(available.ok, true);
    const encoded = encodeDecision(
      example.state,
      available.actions,
      available.actions[example.label],
    );
    assert.equal(encoded.ok, true);
    for (const field of [
      "observation",
      "candidateKeys",
      "candidateFeatures",
      "label",
    ])
      assert.deepEqual(example[field], encoded.value[field]);
    const episode = episodes.find((e) => e.fixture.id === example.fixtureId);
    assert(episode);
    const event = episode.result.events.filter((e) => e.type === "action")[
      example.actionIndex
    ];
    assert(event);
    assert.equal(example.candidateKeys[example.label], actionKey(event.action));
    assert.deepEqual(example.teacher.search, event.search);
    checked++;
  }
}
let predicted = 0;
if (predictionsPath) {
  const predictions = JSON.parse(
    readFileSync(resolve(predictionsPath), "utf8"),
  );
  assert.equal(predictions.dataSha256, manifest.dataSha256);
  assert.equal(predictions.records.length, pack.examples.test.length);
  const seen = new Set();
  for (const prediction of predictions.records) {
    assert(!seen.has(prediction.id));
    seen.add(prediction.id);
    const example = pack.examples.test.find((e) => e.id === prediction.id);
    assert(example);
    assert(Number.isInteger(prediction.index));
    assert.equal(example.candidateKeys[prediction.index], prediction.key);
    const available = getAvailableActions(example.state);
    assert.equal(available.ok, true);
    const action = available.actions.find(
      (a) => actionKey(a) === prediction.key,
    );
    assert(action);
    assert.equal(applyAction(example.state, action).ok, true);
    predicted++;
  }
}
const report = {
  status: "PASS",
  dataSha256: manifest.dataSha256,
  episodesReplayed: pack.episodes.length,
  examplesChecked: checked,
  testPredictionsApplied: predicted,
  illegalPredictions: 0,
};
if (outputPath)
  writeFileSync(resolve(outputPath), JSON.stringify(report, null, 2) + "\n", {
    flag: "wx",
  });
console.log(JSON.stringify(report, null, 2));
