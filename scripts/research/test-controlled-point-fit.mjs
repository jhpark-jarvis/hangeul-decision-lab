import assert from "node:assert/strict";
import test from "node:test";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  root,
  approvedContract,
  auditRows,
  auditLedger,
  reserveFitOutput,
} from "./controlled-point-fit-contract.mjs";
import { parseReplayArgs } from "./replay-controlled-point-fit.mjs";

test("execution snapshot preserves the historical proposal and fixed contract", () => {
  const c = approvedContract();
  assert.equal(c.contract.forwardBudget.totalTopLevelMax, 3248);
  assert.equal(c.contract.fit.steps, 200);
  assert.equal(c.contract.fit.runs, 1);
});
test("ledger validates reservation order including every step and final repeat", () => {
  const rows = [];
  for (const [phase, n] of [
    ["initial", 16],
    ["train", 3200],
    ["final1", 16],
    ["final2", 16],
  ])
    for (let i = 0; i < n; i++)
      rows.push({
        call: rows.length + 1,
        phase,
        phaseCall: i + 1,
        step: phase === "train" ? Math.floor(i / 16) + 1 : null,
        example: i % 16,
      });
  auditLedger(rows);
  for (const mutate of [
    (r) => r.pop(),
    (r) => r.push(r[0]),
    (r) => r[16].step++,
    (r) => (r[3232].phase = "final1"),
    (r) => (r[1].example = 0),
  ]) {
    const c = structuredClone(rows);
    mutate(c);
    assert.throws(() => auditLedger(c));
  }
});
const examples = Array.from({ length: 16 }, (_, i) => ({
  id: `controlled-fit-${i}`,
  label: 1,
}));
const rows = examples.map((e) => ({
  id: e.id,
  label: 1,
  index: 0,
  correct: false,
  nll: Math.fround(Math.log(31)),
  rank: 2,
  margin: 0,
  logits: Array(31).fill(0),
}));
test("saved float32 tie scores preserve first-index choice and rank", () => {
  const s = auditRows(rows, examples);
  assert.equal(s.correct, 0);
  assert.equal(s.maximumTargetRank, 2);
});
test("saved score, target, metric, candidate count and prediction drift fail", () => {
  for (const mutate of [
    (r) => (r[0].logits[1] = NaN),
    (r) => r[0].logits.pop(),
    (r) => (r[0].index = 1),
    (r) => (r[0].rank = 1),
    (r) => (r[0].nll = 0),
    (r) => (r[0].margin = 1),
    (r) => (r[0].id = "other"),
  ]) {
    const c = structuredClone(rows);
    mutate(c);
    assert.throws(() => auditRows(c, examples));
  }
});
test("replay CLI never accepts training, inference or override options", () => {
  assert.deepEqual(parseReplayArgs(["--recorded", "a", "--output", "b"]), {
    recorded: "a",
    output: "b",
  });
  for (const args of [
    [],
    ["--train", "1", "--output", "b"],
    ["--recorded", "a", "--output", "b", "--forward"],
    ["--recorded", "a", "--recorded", "b"],
  ])
    assert.throws(() => parseReplayArgs(args));
});
test("exclusive output rejects escape, overwrite and input overlap", () => {
  const dir = resolve(root, ".research-output");
  mkdirSync(dir, { recursive: true });
  const base = mkdtempSync(resolve(dir, "controlled-fit-test-"));
  for (const path of ["../escape", "outside", ".research-output"])
    assert.throws(() => reserveFitOutput(path, base));
  const out = reserveFitOutput(".research-output/original", base);
  writeFileSync(resolve(out, "report.json"), "original\n");
  assert.throws(() => reserveFitOutput(".research-output/original", base));
  assert.throws(() =>
    reserveFitOutput(".research-output/original/child", base, out),
  );
  assert.equal(readFileSync(resolve(out, "report.json"), "utf8"), "original\n");
});
