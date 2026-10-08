import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { format } from "prettier";
const [modelDirectory, reproductionDirectory, output] = process.argv.slice(2);
if (!modelDirectory || !reproductionDirectory || !output)
  throw new Error(
    "Use model-directory reproduction-directory new-summary.json",
  );
const read = (dir, name) => readFileSync(resolve(dir, name));
const reportBody = read(modelDirectory, "report.json"),
  domainBody = read(modelDirectory, "domain-validation.json"),
  repeatBody = read(reproductionDirectory, "reproduction.json");
const report = JSON.parse(reportBody),
  domain = JSON.parse(domainBody),
  repeated = JSON.parse(repeatBody);
assert.equal(report.status, "PASS");
assert.equal(domain.status, "PASS");
assert.equal(repeated.status, "PASS");
assert.equal(domain.dataSha256, report.dataSha256);
assert.equal(repeated.dataSha256, report.dataSha256);
assert.equal(domain.testPredictionsApplied, report.test.examples);
assert.equal(domain.illegalPredictions, 0);
assert.equal(repeated.testScored, false);
assert.equal(repeated.exactWeightsEqual, true);
assert.equal(repeated.exactTrainDevHistoryEqual, true);
const hash = (data) => createHash("sha256").update(data).digest("hex");
assert.equal(
  hash(read(modelDirectory, "checkpoint.pt")),
  report.checkpointSha256,
);
const summary = {
  schemaVersion: 1,
  id: report.protocol.id,
  kind: "offline-synthetic-dfs-imitation-feasibility",
  status: "PASS",
  protocol: report.protocol,
  environment: {
    python: report.environment.python.split(" ")[0],
    torch: report.environment.torch,
    threads: report.environment.threads,
    interopThreads: report.environment.interopThreads,
    cudaBuild: report.environment.cudaBuild,
  },
  parameters: report.parameters,
  counts: report.counts,
  selectedEpoch: report.selectedEpoch,
  selectedDev: report.history.find((e) => e.epoch === report.selectedEpoch).dev,
  trainLoss: {
    first: report.history[0].trainLoss,
    last: report.history.at(-1).trainLoss,
  },
  test: report.test,
  teacherSubgroups: report.teacherSubgroups,
  timings: report.timings,
  memory: report.memory,
  domain,
  reproduction: {
    exactWeightsEqual: true,
    exactTrainDevHistoryEqual: true,
    testScored: false,
  },
  provenance: {
    dataSha256: report.dataSha256,
    checkpointSha256: report.checkpointSha256,
    trainingReportSha256: hash(reportBody),
    domainReportSha256: hash(domainBody),
    reproductionReportSha256: hash(repeatBody),
    pilotSourceHashes: report.sourceHashes,
    reproductionSourceHashes: repeated.sourceHashes,
  },
  interpretation:
    "Test top1 is below the first-legal-action baseline; no adoption or survival benefit established.",
  limits: report.limits,
  regenerate:
    "node scripts/research/summarize-cnn.mjs <model-directory> <reproduction-directory> <new-summary.json>",
};
writeFileSync(
  resolve(output),
  await format(JSON.stringify(summary, null, 2) + "\n", { parser: "json" }),
  {
    flag: "wx",
  },
);
console.log(
  JSON.stringify(
    {
      status: summary.status,
      test: summary.test,
      selectedEpoch: summary.selectedEpoch,
    },
    null,
    2,
  ),
);
