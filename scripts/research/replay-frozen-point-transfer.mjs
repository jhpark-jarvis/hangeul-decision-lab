import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  root,
  parseTransferArgs,
  reserveFitOutput,
  auditTransferStored,
  write,
} from "./frozen-point-transfer-contract.mjs";
export function replayTransfer(args) {
  const recorded = resolve(root, args.recorded),
    output = reserveFitOutput(args.output, root, recorded);
  try {
    const summary = auditTransferStored(recorded);
    write(resolve(output, "summary.json"), summary);
    return summary;
  } catch (error) {
    write(resolve(output, "failure.json"), {
      status: "FAIL",
      error: String(error),
      automaticRetry: false,
    });
    throw error;
  }
}
if (
  process.argv[1] &&
  resolve(process.argv[1]) === fileURLToPath(import.meta.url)
)
  console.log(
    JSON.stringify(
      replayTransfer(parseTransferArgs("replay", process.argv.slice(2))),
    ),
  );
