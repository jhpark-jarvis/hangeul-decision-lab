import assert from "node:assert/strict";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  root,
  auditStored,
  write,
  reserveFitOutput,
} from "./controlled-point-fit-contract.mjs";

export function parseReplayArgs(args) {
  assert.equal(
    args.length,
    4,
    "Use --recorded <existing-run> --output <new-directory> only",
  );
  const pairs = [args.slice(0, 2), args.slice(2, 4)];
  assert.deepEqual(pairs.map((p) => p[0]).sort(), ["--output", "--recorded"]);
  assert(
    pairs.every(
      (p) => typeof p[1] === "string" && p[1] && !p[1].startsWith("--"),
    ),
  );
  return Object.fromEntries(pairs.map(([k, v]) => [k.slice(2), v]));
}
export function replay(args) {
  const output = reserveFitOutput(
    args.output,
    root,
    resolve(root, args.recorded),
  );
  try {
    const result = auditStored(resolve(root, args.recorded));
    write(resolve(output, "summary.json"), result.summary);
    write(resolve(output, "report.json"), result);
    return result;
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
  console.log(JSON.stringify(replay(parseReplayArgs(process.argv.slice(2)))));
