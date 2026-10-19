import assert from "node:assert/strict";
import { runBoundedChild } from "./bounded-child.mjs";

const rows = [];
async function run(name, script, extra = {}) {
  let stdout = "",
    stderr = "";
  const result = await runBoundedChild({
    command: process.execPath,
    args: ["-e", script],
    cwd: process.cwd(),
    timeoutMs: 5000,
    onStdout: (b) => {
      stdout += b;
    },
    onStderr: (b) => {
      stderr += b;
    },
    ...extra,
  });
  assert.throws(() => process.kill(result.pid, 0), { code: "ESRCH" });
  rows.push({ name, ...result, stdout, stderr });
  return result;
}
const success = await run("success", "console.log('ok')");
assert.equal(success.code, 0);
assert.equal(success.timedOut, false);
const failed = await run(
  "nonzero-no-retry",
  "console.error('expected');process.exit(7)",
);
assert.equal(failed.code, 7);
const timeout = await run(
  "timeout-owned-child-exited",
  "setTimeout(()=>{},60000)",
  { timeoutMs: 200 },
);
assert.equal(timeout.timedOut, true);
assert.notEqual(timeout.code, 0);
const cancelled = await run(
  "cancel-owned-child-exited",
  "setTimeout(()=>{},60000)",
  {
    onSpawn: (child) => setTimeout(() => child.kill(), 200),
  },
);
assert.equal(cancelled.timedOut, false);
assert.notEqual(cancelled.code, 0);
let spawned = 0;
for (const script of ["process.exit(9)", "console.log('must-not-run')"]) {
  const result = await run("sequence-stop", script, {
    onSpawn: () => {
      spawned++;
    },
  });
  if (result.code !== 0) break;
}
assert.equal(spawned, 1);
assert.throws(() =>
  runBoundedChild({ command: process.execPath, args: [], timeoutMs: 0 }),
);
process.stdout.write(
  JSON.stringify({
    status: "PASS",
    ownedChildren: rows.length,
    rows,
    newModelForwardTraining: 0,
  }) + "\n",
);
