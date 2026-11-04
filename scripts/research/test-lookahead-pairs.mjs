import assert from "node:assert/strict";
import test from "node:test";
import { compileResearch } from "./runtime.mjs";
import {
  root,
  createPairLedger,
  auditPairLedger,
  compareStoredEvaluations,
} from "./lookahead-pairs-contract.mjs";
import { replayLookaheadPairs } from "./replay-lookahead-pairs.mjs";
const { conditionalReference } = await import(
  compileResearch(root, "src/research/conditional-reference.ts").url
);

const scalarApi = (reservations, extra = {}) => ({
  actions: (s) => (s === 0 ? [1, 2] : s === 1 ? [3] : []),
  advance: (s, a, r) => ({ state: a, rewards: r + a }),
  evaluate: (s, r) => r,
  compare: (a, b) => a - b,
  reserve: (r) => reservations.push(r),
  ...extra,
});
test("conditional reference groups complete terminal rewards by root", () => {
  const rows = [],
    r = conditionalReference(0, 0, scalarApi(rows), 64, 3);
  assert.deepEqual(r.best, [4, 2]);
  assert.deepEqual(r.qualityIndices, [0]);
  assert.equal(r.nodes, 4);
  assert.equal(r.maxDepth, 2);
  assert.equal(rows.length, 4);
});
test("reference preserves all equivalent roots without weighted score selection", () => {
  const r = conditionalReference(
    0,
    0,
    scalarApi([], { evaluate: () => 7 }),
    64,
    3,
  );
  assert.deepEqual(r.qualityIndices, [0, 1]);
});
test("reference reserves before advance and withholds labels at node exhaustion", () => {
  const reservations = [];
  let advanced = 0;
  const api = scalarApi(reservations, {
    advance: (s, a, r) => {
      assert.equal(reservations.at(-1).action, a);
      advanced++;
      return { state: a, rewards: r + a };
    },
  });
  assert.throws(
    () => conditionalReference(0, 0, api, 2, 3),
    (e) => {
      assert.equal(e.partial.complete, false);
      assert.equal(e.partial.nodes, 2);
      assert.equal("qualityIndices" in e.partial, false);
      return true;
    },
  );
  assert.equal(advanced, 1);
});
test("playing depth-limit leaves are not terminal successes", () => {
  assert.throws(
    () => conditionalReference(0, 0, scalarApi([]), 64, 1),
    (e) => e.partial.complete === false && /depth cap/.test(e.message),
  );
});
function fillLedger(l) {
  for (let build = 0; build < 2; build++)
    for (let index = 0; index < 32; index++) {
      l.state(build, index);
      l.reserve({
        build,
        index,
        kind: "root",
        rootIndex: -1,
        depth: 0,
        key: null,
      });
      for (let j = 0; j < 32; j++)
        l.reserve({
          build,
          index,
          kind: "edge",
          rootIndex: j,
          depth: 1,
          key: `key${j}`,
        });
      for (let j = 0; j < 32; j++)
        l.reserve({
          build,
          index,
          kind: "outcome",
          rootIndex: j,
          depth: 1,
          key: `key${j}`,
        });
    }
}
test("global ledger separates root nodes, edges, outcomes and state counts", () => {
  const l = createPairLedger(() => {});
  fillLedger(l);
  assert.deepEqual(l.finish(), {
    states: 64,
    roots: 64,
    nodes: 2112,
    referenceApply: 2048,
    outcomeApply: 2048,
    totalApply: 4096,
  });
  assert.deepEqual(auditPairLedger(l.rows), l.counts);
  assert.throws(() => l.state(2, 0));
});
test("ledger rejects out-of-order root coverage and premature outcomes", () => {
  const l = createPairLedger(() => {});
  l.state(0, 0);
  l.reserve({
    build: 0,
    index: 0,
    kind: "root",
    rootIndex: -1,
    depth: 0,
    key: null,
  });
  assert.throws(() =>
    l.reserve({
      build: 0,
      index: 0,
      kind: "edge",
      rootIndex: 1,
      depth: 1,
      key: "x",
    }),
  );
  assert.throws(() =>
    l.reserve({
      build: 0,
      index: 0,
      kind: "outcome",
      rootIndex: 0,
      depth: 1,
      key: "x",
    }),
  );
  assert.throws(() => l.finish());
});
test("ledger refuses transition65 and leaves failed durable reservations ungranted", () => {
  const l = createPairLedger(() => {});
  l.state(0, 0);
  l.reserve({
    build: 0,
    index: 0,
    kind: "root",
    rootIndex: -1,
    depth: 0,
    key: null,
  });
  for (let i = 0; i < 63; i++)
    l.reserve({
      build: 0,
      index: 0,
      kind: "edge",
      rootIndex: 0,
      depth: i === 0 ? 1 : 2,
      key: "x",
    });
  assert.throws(() =>
    l.reserve({
      build: 0,
      index: 0,
      kind: "edge",
      rootIndex: 0,
      depth: 1,
      key: "x",
    }),
  );
  assert.equal(l.counts.referenceApply, 63);
  const bad = createPairLedger(() => {
    throw new Error("disk");
  });
  assert.throws(() => bad.state(0, 0), /disk/);
  assert.equal(bad.counts.states, 0);
});
test("saved ledger detects event renumbering and resource failure before write", () => {
  const l = createPairLedger(() => {});
  fillLedger(l);
  const changed = structuredClone(l.rows);
  changed[1].event = 10;
  assert.throws(() => auditPairLedger(changed));
  const b = createPairLedger(
    () => assert.fail("no persistence"),
    () => {
      throw new Error("RSS cap");
    },
  );
  assert.throws(() => b.state(0, 0), /RSS cap/);
});
function evaluation(extra = {}) {
  return {
    survivable: true,
    allCurrentPiecesPlaced: true,
    mobility: { playablePieceTypes: 3, totalPlacements: 10 },
    clearedRows: 3,
    acquiredItems: 0,
    remainingAbilities: { singleCell: 0, reroll: 0 },
    isolatedEmptyCells: 1,
    unfillableGaps: 0,
    emptyCells: 44,
    ...extra,
  };
}
test("saved comparator preserves survival before mobility and total placements before clears", () => {
  assert.equal(
    compareStoredEvaluations(
      evaluation(),
      evaluation({
        survivable: false,
        mobility: { playablePieceTypes: 19, totalPlacements: 999 },
      }),
    ),
    1,
  );
  assert.equal(
    compareStoredEvaluations(
      evaluation({ clearedRows: 100 }),
      evaluation({ mobility: { playablePieceTypes: 3, totalPlacements: 11 } }),
    ),
    -1,
  );
  assert.equal(compareStoredEvaluations(evaluation(), evaluation()), 0);
});
test("saved CLI refuses execution or tuning flags before any I/O", () => {
  for (const args of [
    [],
    ["--execute", "yes", "--output", "x"],
    ["--recorded", "x", "--output", "y", "--repeat", "3"],
  ])
    assert.throws(() => replayLookaheadPairs(args));
});
