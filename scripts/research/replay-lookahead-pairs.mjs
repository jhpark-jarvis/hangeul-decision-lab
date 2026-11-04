import assert from "node:assert/strict";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { root, auditPairsStored } from "./lookahead-pairs-contract.mjs";
import { reserveFitOutput, write } from "./controlled-point-fit-contract.mjs";
export function replayLookaheadPairs(args) {
  assert.equal(args.length, 4);
  assert.equal(args[0], "--recorded");
  assert.equal(args[2], "--output");
  assert(args[1] && args[3]);
  const recorded = resolve(root, args[1]),
    out = reserveFitOutput(args[3], root, recorded),
    summary = auditPairsStored(recorded);
  write(resolve(out, "summary.json"), summary);
  return summary;
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
)
  console.log(JSON.stringify(replayLookaheadPairs(process.argv.slice(2))));
