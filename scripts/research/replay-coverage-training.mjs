import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  root,
  write,
  reserveFitOutput,
  auditCoverageTrainingStored,
  parseCoverageTrainingArgs,
} from "./coverage-training-contract.mjs";
export function replayCoverageTraining(args) {
  const recorded = resolve(root, args.recorded),
    out = reserveFitOutput(args.output, root, recorded),
    summary = auditCoverageTrainingStored(recorded);
  write(resolve(out, "summary.json"), summary);
  return summary;
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
)
  console.log(
    JSON.stringify(
      replayCoverageTraining(
        parseCoverageTrainingArgs("replay", process.argv.slice(2)),
      ),
    ),
  );
