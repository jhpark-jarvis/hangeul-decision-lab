import assert from "node:assert/strict";
import { resolve } from "node:path";
import {
  root,
  json,
  hash,
  bytes,
  write,
  reserveFitOutput,
} from "./outcome-mlp-contract.mjs";
import {
  validateCoverageProposal,
  proposalPath,
} from "./validate-coverage-proposal.mjs";
import {
  boardGroup,
  auditCoverageSplits,
  coveragePreflightGate,
} from "./audit-coverage-inputs.mjs";
export { root, json, hash, bytes, write, reserveFitOutput };
export const configPath = "research/experiments/coverage-preflight-v1.json";
export function approvedCoverage(base = root) {
  const p = json(resolve(base, proposalPath));
  validateCoverageProposal(p, base);
  const c = json(resolve(base, configPath));
  assert.equal(c.id, "coverage-preflight-v1");
  assert.equal(c.approval, "USER_APPROVED");
  assert.equal(c.proposalSha256, hash(resolve(base, proposalPath)));
  assert.deepEqual(c.contract, p);
  return c;
}
export function auditCoverageData(d) {
  assert.equal(d.schemaVersion, 1);
  assert.equal(d.id, "coverage-data-v1");
  assert.deepEqual(Object.keys(d.splits).sort(), [
    "control",
    "dev",
    "diverse",
    "final",
  ]);
  let nodes = 0;
  const all = [];
  for (const [split, rows] of Object.entries(d.splits)) {
    assert.equal(rows.length, 16);
    rows.forEach((e, row) => {
      const a =
          split === "control"
            ? 0
            : split === "diverse"
              ? 1 + (row % 3)
              : split === "dev"
                ? 5
                : 7,
        phase = split === "dev" ? 1 : split === "final" ? 2 : 0;
      assert.equal(e.id, `coverage-${split}-${row}`);
      assert.equal(e.split, split);
      assert.equal(e.row, row);
      assert.equal(e.a, a);
      assert.equal(e.phase, phase);
      assert.equal(e.observation.planes.length, 3);
      assert.equal(e.observation.metadata.length, 140);
      assert(
        e.observation.metadata.every(
          (v) => Number.isFinite(v) && v >= 0 && v <= 1,
        ),
      );
      const b = e.observation.planes[0];
      assert.equal(b.length, 16);
      b.forEach((r, i) => {
        assert.equal(r.length, 10);
        const order = (i - row + 16) % 16,
          k = order <= a ? 2 : order > 15 - a ? 4 : 3,
          offsets =
            i === row
              ? [0]
              : k === 2
                ? [0, 1]
                : k === 3
                  ? [0, 1, 4]
                  : [0, 1, 4, 5],
          holes = new Set(offsets.map((n) => (3 * i + phase + n) % 10));
        r.forEach((v, j) => assert.equal(v, Number(!holes.has(j))));
      });
      for (const plane of e.observation.planes.slice(1)) {
        assert.equal(plane.length, 16);
        plane.forEach((r) => assert.deepEqual(r, Array(10).fill(0)));
      }
      const keys = [];
      b.forEach((r, i) =>
        r.forEach((v, j) => {
          if (!v) keys.push(["single-cell", i, j]);
        }),
      );
      assert.equal(keys.length, 46);
      assert.deepEqual(e.candidateKeys.map(JSON.parse), keys);
      assert(Number.isInteger(e.label) && e.label >= 0 && e.label < 46);
      assert.deepEqual(keys[e.label], [
        "single-cell",
        row,
        (3 * row + phase) % 10,
      ]);
      assert.equal(e.candidateFeatures.length, 46);
      assert.equal(e.footprints.length, 46);
      e.candidateFeatures.forEach((f, j) => {
        assert.equal(f.length, 74);
        assert(f.every((v) => Number.isFinite(v) && v >= 0 && v <= 1));
        assert.deepEqual(f.slice(0, 3), [0, 1, 0]);
        assert.equal(f[52], Number(j === e.label) / 16);
        const [, r, c] = keys[j];
        assert.deepEqual(f.slice(25, 27), [r / 15, c / 9]);
        assert.equal(
          e.footprints[j],
          Array.from({ length: 160 }, (_, n) => Number(n === 10 * r + c)).join(
            "",
          ),
        );
      });
      assert.deepEqual(
        [e.reference.complete, e.reference.unique],
        [true, true],
      );
      assert(
        Number.isInteger(e.reference.visitedNodes) &&
          e.reference.visitedNodes > 0 &&
          e.reference.visitedNodes <= 256,
      );
      nodes += e.reference.visitedNodes;
      all.push({
        boardGroup: boardGroup(b),
        label: e.label,
        candidates: e.candidateFeatures.map((f) => f.slice(52)),
      });
    });
  }
  assert.equal(new Set(all.map((e) => e.boardGroup)).size, 64);
  assert.deepEqual(d.counts, {
    examples: 64,
    candidates: 2944,
    referenceNodes: nodes,
    referenceNodeCap: 16384,
    outcomeTransitions: 2944,
    dfsCalls: 0,
  });
  const convert = (rows) =>
    rows.map((e) => ({
      boardGroup: boardGroup(e.observation.planes[0]),
      label: e.label,
      candidates: e.candidateFeatures.map((f) => f.slice(52)),
    }));
  const audit = auditCoverageSplits(
      convert([...d.splits.control, ...d.splits.diverse]),
      { dev: convert(d.splits.dev), final: convert(d.splits.final) },
    ),
    gate = coveragePreflightGate(audit);
  assert.equal(gate.status, "PASS", "Coverage data leakage/novelty gate; STOP");
  return { audit, gate };
}
export function auditCoverageLedger(rows) {
  assert.equal(rows.length, 6016);
  let n = 0;
  for (let build = 1; build <= 2; build++)
    for (const split of ["control", "diverse", "dev", "final"])
      for (let row = 0; row < 16; row++)
        for (let index = -1; index < 46; index++) {
          assert.deepEqual(rows[n], {
            call: n + 1,
            build,
            kind: index === -1 ? "reference" : "outcome",
            split,
            row,
            index: index === -1 ? 0 : index,
            reservedNodes: index === -1 ? 256 : 0,
          });
          n++;
        }
}
export function auditCoverageStored(recorded, base = root) {
  approvedCoverage(base);
  const path = (p) => resolve(recorded, p),
    r = json(path("report.json")),
    f = json(path("execution.json"));
  assert.equal(r.status, "PASS");
  assert(r.elapsedSeconds <= 120 && r.maxSampledRss <= 2147483648);
  assert.equal(f.configSha256, hash(resolve(base, configPath)));
  for (const [p, h] of Object.entries(r.hashes))
    assert.equal(hash(path(p)), h, p);
  assert.deepEqual(Object.keys(r.hashes).sort(), [
    "build1.json",
    "build2.json",
    "execution.json",
    "ledger.jsonl",
    "start.json",
  ]);
  assert.equal(hash(path("build1.json")), hash(path("build2.json")));
  auditCoverageLedger(
    bytes(path("ledger.jsonl")).toString().trim().split("\n").map(JSON.parse),
  );
  const d = json(path("build1.json")),
    audit = auditCoverageData(d);
  return {
    schemaVersion: 1,
    id: "coverage-preflight-v1",
    status: "PASS",
    ...audit,
    construction: {
      builds: 2,
      referenceCalls: 128,
      referenceNodeCap: 32768,
      actualReferenceNodes: 2 * d.counts.referenceNodes,
      outcomeTransitions: 5888,
      dfsCalls: 0,
    },
    dataSha256: hash(path("build1.json")),
    twoBuildsByteExact: true,
    timing: {
      ownerWallSeconds: r.elapsedSeconds,
      maxSampledRss: r.maxSampledRss,
      instantaneousPeakMeasured: false,
    },
    additionalPythonModelLoadForwardTrainingDomainDataset: 0,
    interpretation:
      "Prospective synthetic coverage data; feature52 remains an immediate-clear shortcut; no learned accuracy measured here.",
    appUsesDfs: true,
  };
}
