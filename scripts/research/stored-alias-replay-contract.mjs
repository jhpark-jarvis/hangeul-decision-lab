import assert from "node:assert/strict";
export const configPath = "research/experiments/stored-alias-replay-v2.json";
export const schedule = (cases) =>
  [0, 1].flatMap((pass) =>
    cases.flatMap((c, caseIndex) =>
      c.members22.flatMap((member) =>
        ["direct", "outcome"].map((kind) => ({
          pass,
          caseIndex,
          member,
          kind,
        })),
      ),
    ),
  );

export function validateSelection(selection) {
  assert.equal(selection.schemaVersion, 1);
  assert.equal(selection.cases.length, 2);
  assert.equal(new Set(selection.cases.map((c) => c.fixtureId)).size, 2);
  for (const [i, c] of selection.cases.entries()) {
    assert.equal(c.split, "train");
    assert(Number.isInteger(c.sourceIndex) && c.sourceIndex >= 0);
    assert.match(c.stateSha256, /^[a-f0-9]{64}$/);
    assert.equal(c.members22.length, i === 0 ? 4 : 5);
    assert.deepEqual(
      c.members22,
      [...new Set(c.members22)].sort((a, b) => a - b),
    );
    assert(c.members22.every((m) => Number.isInteger(m) && m >= 0));
    assert(c.members22.includes(c.label));
    assert.equal(c.storedTeacherTerminalSurvivable, false);
    assert.equal(c.storedTeacherAllPiecesPlaced, false);
    assert.equal(c.teacherSearchComplete, true);
    assert.equal(c.eligibleLiveAlias, true);
    assert.equal(c.ordinaryPlacements, 1);
    assert.equal(c.storedMaskSurvivors.length, c.members22.length);
    assert(c.storedMaskSurvivors.every((n) => n === 0 || n === 1));
    assert.deepEqual([...new Set(c.storedMaskSurvivors)].sort(), [0, 1]);
  }
  assert.equal(schedule(selection.cases).length, 36);
  return selection;
}

/** Reservation is persisted before either transition call, including failed calls. */
export function createLedger(selection, persist, check = () => {}) {
  const expected = schedule(validateSelection(selection).cases),
    rows = [];
  return {
    rows,
    reserve(entry) {
      check();
      assert(rows.length < 36, "Transition cap");
      assert.deepEqual(entry, expected[rows.length], "Transition order");
      const row = { call: rows.length + 1, ...entry };
      persist(row);
      rows.push(row);
    },
    finish() {
      assert.equal(rows.length, 36);
      return rows;
    },
  };
}

export function auditReplayRows(selection, passes, ledger) {
  validateSelection(selection);
  assert.equal(passes.length, 2);
  assert.deepEqual(passes[0], passes[1], "Repeated rows exact");
  assert.deepEqual(
    ledger,
    schedule(selection.cases).map((e, i) => ({ call: i + 1, ...e })),
  );
  const aggregate = {
    states: 2,
    distinctCandidates: 9,
    repetitions: 2,
    directCalls: 18,
    outcomeCalls: 18,
    totalTransitions: 36,
    nextPlayingPerPass: 0,
    nextGameoverPerPass: 0,
    mixedPhaseStates: 0,
    hardLabelNextGameover: 0,
    earliestAliasNextGameover: 0,
    storedTeacherTerminalFailure: 2,
    repeatedRowsExact: true,
  };
  assert.equal(passes[0].length, 2);
  for (const [i, c] of selection.cases.entries()) {
    const rows = passes[0][i];
    assert.equal(rows.length, c.members22.length);
    for (const [j, r] of rows.entries()) {
      assert.deepEqual(r, {
        member: c.members22[j],
        target: c.members22[j] === c.label,
        earliest: j === 0,
        nextPhase: c.storedMaskSurvivors[j] ? "playing" : "gameover",
        ordinaryPlacements: c.storedMaskSurvivors[j],
        outcomeExact: true,
        inputUnchanged: true,
      });
      if (r.nextPhase === "playing") aggregate.nextPlayingPerPass++;
      else {
        aggregate.nextGameoverPerPass++;
        if (r.target) aggregate.hardLabelNextGameover++;
        if (r.earliest) aggregate.earliestAliasNextGameover++;
      }
    }
    if (new Set(rows.map((r) => r.nextPhase)).size === 2)
      aggregate.mixedPhaseStates++;
  }
  return aggregate;
}

export function parseStoredAliasArgs(args) {
  assert.equal(args.length, 2);
  assert.equal(args[0], "--output");
  assert(args[1] && !args[1].startsWith("--"));
  return { output: args[1] };
}
