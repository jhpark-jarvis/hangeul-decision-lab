import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  root,
  write,
  reserveFitOutput,
} from "./controlled-point-fit-contract.mjs";
import { auditRecordedAlias } from "./stored-alias-replay.mjs";

export function replayStoredAliasEvidence(args) {
  if (
    args.length !== 4 ||
    args[0] !== "--recorded" ||
    args[2] !== "--output" ||
    !args[1] ||
    !args[3]
  )
    throw new Error(
      "Use --recorded <private archive> --output <new ignored output>",
    );
  const recorded = resolve(root, args[1]);
  const out = reserveFitOutput(args[3], root, recorded);
  const summary = auditRecordedAlias(recorded);
  write(resolve(out, "summary.json"), summary);
  return summary;
}

if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
)
  console.log(JSON.stringify(replayStoredAliasEvidence(process.argv.slice(2))));
