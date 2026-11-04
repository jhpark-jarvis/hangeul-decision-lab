import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  root,
  write,
  reserveFitOutput,
  auditCoverageStored,
} from "./coverage-preflight-contract.mjs";
import { parseTransferArgs } from "./outcome-mlp-contract.mjs";
export function replayCoverage(args) {
  const recorded = resolve(root, args.recorded),
    out = reserveFitOutput(args.output, root, recorded);
  const summary = auditCoverageStored(recorded);
  write(resolve(out, "summary.json"), summary);
  return summary;
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
)
  console.log(
    JSON.stringify(
      replayCoverage(parseTransferArgs("replay", process.argv.slice(2))),
    ),
  );
