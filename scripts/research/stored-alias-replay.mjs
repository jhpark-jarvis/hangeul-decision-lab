import assert from "node:assert/strict";
import { appendFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  root,
  json,
  hash,
  bytes,
  write,
  reserveFitOutput,
} from "./controlled-point-fit-contract.mjs";
import { compileResearch } from "./runtime.mjs";
import {
  configPath,
  validateSelection,
  createLedger,
  auditReplayRows,
  parseStoredAliasArgs,
} from "./stored-alias-replay-contract.mjs";

const git = (args) =>
  execFileSync("git", args, {
    cwd: root,
    encoding: "utf8",
    windowsHide: true,
  }).trim();
const digest = (value) =>
  createHash("sha256").update(JSON.stringify(value)).digest("hex");

export async function runStoredAliasReplay(args) {
  assert.equal(
    git(["status", "--porcelain"]),
    "",
    "Commit implementation before domain replay",
  );
  const c = json(resolve(root, configPath));
  assert.equal(c.id, "stored-alias-replay-v2");
  assert.equal(c.status, "FROZEN_BEFORE_EXECUTION");
  assert.deepEqual(c.budget, {
    states: 2,
    candidates: 9,
    repetitions: 2,
    direct: 18,
    outcome: 18,
    totalTransitions: 36,
    referenceTeacherModelLoadForwardTrainingNewDataset: 0,
  });
  assert.deepEqual(c.resources, {
    seconds: 60,
    sampledRssBytes: 2147483648,
    instantaneousPeakMeasured: false,
  });
  const out = reserveFitOutput(args.output),
    started = performance.now();
  let maxSampledRss = 0;
  const check = () => {
    maxSampledRss = Math.max(maxSampledRss, process.memoryUsage().rss);
    assert(
      (performance.now() - started) / 1000 <= c.resources.seconds &&
        maxSampledRss <= c.resources.sampledRssBytes,
      "Cooperative time/RSS cap",
    );
  };
  const checkHashes = () => {
    for (const [p, h] of Object.entries(c.hashes))
      assert.equal(hash(resolve(root, p)), h, p);
  };
  try {
    write(resolve(out, "start.json"), {
      atUtc: new Date().toISOString(),
      executor: "Codex",
      pid: process.pid,
      node: process.version,
      sourceHead: git(["rev-parse", "HEAD"]),
      configSha256: hash(resolve(root, configPath)),
      contractId: c.id,
      retry: 0,
    });
    checkHashes();
    check();
    const selection = validateSelection(json(resolve(root, c.selection)));
    const dataset = json(resolve(root, c.sourceDataset)),
      augmented = json(resolve(root, c.augmentedDataset));
    assert.deepEqual(dataset.examples.test, []);
    assert.equal(
      selection.sourceDataSha256,
      hash(resolve(root, c.sourceDataset)),
    );
    assert.equal(
      selection.augmentedSha256,
      hash(resolve(root, c.augmentedDataset)),
    );
    const inputs = selection.cases.map((s) => {
      const e = dataset.examples.train[s.sourceIndex],
        a = augmented.examples.train[s.sourceIndex];
      assert.equal(e.id, s.id);
      assert.equal(a.id, e.id);
      assert.equal(e.fixtureId, s.fixtureId);
      assert.equal(digest(e.state), s.stateSha256);
      assert.equal(e.labels.A512, s.label);
      assert.equal(e.teachers.A512.result.evaluation.survivable, false);
      return {
        state: e.state,
        candidateKeys: e.candidateKeys,
        members: s.members22,
        expectedSurvivors: s.storedMaskSurvivors,
        outcomes: s.members22.map((m) => {
          const f = a.candidateFeatures[m];
          assert.equal(f.length, 74);
          assert.deepEqual(f.slice(0, 52), e.candidateFeatures[m]);
          return f.slice(52);
        }),
        label: s.label,
      };
    });
    check();
    const runtime = compileResearch(
      root,
      "src/research/stored-alias-replay.ts",
    );
    for (const [p, h] of Object.entries(runtime.hashes))
      assert.equal(c.hashes[p], h, "Unfrozen runtime source " + p);
    const { replayStoredAlias } = await import(runtime.url);
    const ledgerPath = resolve(out, "ledger.jsonl");
    writeFileSync(ledgerPath, "", { flag: "wx" });
    const ledger = createLedger(
      selection,
      (row) => appendFileSync(ledgerPath, JSON.stringify(row) + "\n"),
      check,
    );
    const passes = [0, 1].map((pass) =>
      inputs.map((input, caseIndex) =>
        replayStoredAlias(input, (member, kind) =>
          ledger.reserve({ pass, caseIndex, member, kind }),
        ),
      ),
    );
    const rows = ledger.finish(),
      summary = auditReplayRows(selection, passes, rows);
    checkHashes();
    check();
    write(resolve(out, "private-rows.json"), passes);
    write(resolve(out, "summary.json"), {
      schemaVersion: 1,
      status: "PASS",
      ...summary,
      newTeacherReferenceModelLoadForwardTrainingDataset: 0,
      wholeTurnSalvageability: "NOT_EVALUATED",
      teacherLabelError: "NOT_ESTABLISHED",
    });
    write(resolve(out, "report.json"), {
      status: "PASS",
      ...json(resolve(out, "start.json")),
      elapsedSeconds: (performance.now() - started) / 1000,
      maxSampledRss,
      instantaneousPeakMeasured: false,
      runtime,
      hashes: Object.fromEntries(
        ["private-rows.json", "summary.json", "ledger.jsonl"].map((p) => [
          p,
          hash(resolve(out, p)),
        ]),
      ),
      originalInputsAndSourcesUnchanged: true,
    });
    return json(resolve(out, "summary.json"));
  } catch (error) {
    write(resolve(out, "failure.json"), {
      status: "FAIL",
      error: String(error),
      automaticRetry: false,
    });
    throw error;
  }
}

/** Saved evidence replay only: no runtime compilation/import or domain call. */
export function auditRecordedAlias(recorded, base = root) {
  const c = json(resolve(base, configPath)),
    report = json(resolve(recorded, "report.json"));
  assert.equal(report.status, "PASS");
  assert.equal(c.id, "stored-alias-replay-v2");
  assert.equal(report.contractId, c.id);
  assert.equal(report.configSha256, hash(resolve(base, configPath)));
  assert.deepEqual(Object.keys(report.hashes).sort(), [
    "ledger.jsonl",
    "private-rows.json",
    "summary.json",
  ]);
  assert(
    report.elapsedSeconds >= 0 && report.elapsedSeconds <= c.resources.seconds,
  );
  assert(
    report.maxSampledRss > 0 &&
      report.maxSampledRss <= c.resources.sampledRssBytes,
  );
  assert.equal(report.instantaneousPeakMeasured, false);
  assert.equal(report.originalInputsAndSourcesUnchanged, true);
  for (const [p, h] of Object.entries(report.hashes))
    assert.equal(hash(resolve(recorded, p)), h);
  assert.equal(hash(resolve(base, c.selection)), c.hashes[c.selection]);
  const selection = json(resolve(base, c.selection)),
    passes = json(resolve(recorded, "private-rows.json"));
  const ledger = bytes(resolve(recorded, "ledger.jsonl"))
    .toString("utf8")
    .trim()
    .split("\n")
    .map((s) => JSON.parse(s));
  const result = auditReplayRows(selection, passes, ledger),
    saved = json(resolve(recorded, "summary.json"));
  assert.deepEqual(saved, {
    schemaVersion: 1,
    status: "PASS",
    ...result,
    newTeacherReferenceModelLoadForwardTrainingDataset: 0,
    wholeTurnSalvageability: "NOT_EVALUATED",
    teacherLabelError: "NOT_ESTABLISHED",
  });
  return saved;
}

if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
)
  console.log(
    JSON.stringify(
      await runStoredAliasReplay(parseStoredAliasArgs(process.argv.slice(2))),
    ),
  );
