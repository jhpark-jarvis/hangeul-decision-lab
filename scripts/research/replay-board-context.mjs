import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  root,
  write,
  reserveFitOutput,
  parseBoardArgs,
  auditBoardStored,
} from "./board-context-contract.mjs";
export function replayBoardContext(args) {
  const a = parseBoardArgs("replay", args),
    recorded = resolve(root, a.recorded),
    out = reserveFitOutput(a.output, root, recorded),
    summary = auditBoardStored(recorded);
  write(resolve(out, "summary.json"), summary);
  return summary;
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
)
  console.log(JSON.stringify(replayBoardContext(process.argv.slice(2))));
