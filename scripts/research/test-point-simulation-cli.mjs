import test from "node:test";
import assert from "node:assert/strict";
import {
  parsePointSimulationArgs,
  reservePointOutput,
} from "./point-simulation.mjs";
import {
  mkdirSync,
  mkdtempSync,
  writeFileSync,
  readFileSync,
  existsSync,
  symlinkSync,
} from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
const root = fileURLToPath(new URL("../../", import.meta.url));
test("simulation CLI only accepts an explicit new output, not training or budget overrides", () => {
  assert.equal(
    parsePointSimulationArgs(["--output", ".research-output/example"]),
    ".research-output/example",
  );
  for (const a of [
    [],
    ["--output"],
    ["--train", "12"],
    ["--output", "x", "--nodes", "2048"],
    ["--output", "x", "--output", "y"],
  ])
    assert.throws(() => parsePointSimulationArgs(a));
});
test("owned output rejects escape/reuse without altering an existing report", () => {
  const base = resolve(root, ".research-output");
  mkdirSync(base, { recursive: true });
  const task = mkdtempSync(resolve(base, "point-cli-test-"));
  for (const p of ["outside", "../escape", ".research-output"])
    assert.throws(() => reservePointOutput(p, task));
  const out = reservePointOutput(".research-output/owned", task);
  writeFileSync(resolve(out, "report.json"), "original\n");
  assert.throws(() => reservePointOutput(".research-output/owned", task));
  assert.equal(readFileSync(resolve(out, "report.json"), "utf8"), "original\n");
  assert(!existsSync(resolve(task, "outside")));
});
test("a redirected parent cannot write outside the task-owned research directory", () => {
  const base = resolve(root, ".research-output");
  mkdirSync(base, { recursive: true });
  const task = mkdtempSync(resolve(base, "point-cli-link-test-")),
    outside = resolve(task, "outside"),
    research = resolve(task, ".research-output");
  mkdirSync(outside);
  mkdirSync(research);
  symlinkSync(outside, resolve(research, "redirect"), "junction");
  assert.throws(() =>
    reservePointOutput(".research-output/redirect/new", task),
  );
  assert(!existsSync(resolve(outside, "new")));
});
