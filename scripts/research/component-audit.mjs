import { spawn } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve, relative, isAbsolute } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL("../../", import.meta.url)));
const args = process.argv.slice(2),
  options = {};
if (args.length === 1 && args[0] === "--help") {
  console.log(
    "node scripts/research/component-audit.mjs --python <existing-cpu-python> --source <paired> --experiment <P056> --original <outcome> --output <new-ignored-directory>",
  );
  process.exit(0);
}
for (let i = 0; i < args.length; i += 2) {
  if (
    ![
      "--python",
      "--source",
      "--experiment",
      "--original",
      "--output",
    ].includes(args[i]) ||
    !args[i + 1] ||
    options[args[i]]
  )
    throw new Error("Unknown/duplicate/missing argument; no tuning options");
  options[args[i]] = args[i + 1];
}
if (Object.keys(options).length !== 5)
  throw new Error("python/source/experiment/original/output required");
const output = resolve(root, options["--output"]),
  inside = relative(resolve(root, ".research-output"), output);
if (
  !inside ||
  inside.startsWith("..") ||
  isAbsolute(inside) ||
  existsSync(output)
)
  throw new Error("New ignored output required");
const supervisor = output + "-supervisor";
mkdirSync(supervisor, { recursive: false });
const log = resolve(supervisor, "child.log"),
  started = Date.now();
const child = spawn(
  resolve(options["--python"]),
  [
    resolve(root, "research/cnn/component_audit.py"),
    "--mode",
    "infer",
    ...["--source", "--experiment", "--original", "--output"].flatMap((key) => [
      key,
      resolve(root, options[key]),
    ]),
  ],
  { cwd: root, windowsHide: true, stdio: ["ignore", "pipe", "pipe"] },
);
let timedOut = false,
  cancelled = false;
const stop = () => {
  cancelled = true;
  child.kill();
};
const timer = setTimeout(() => {
  timedOut = true;
  child.kill();
}, 600000);
process.once("SIGINT", stop);
process.once("SIGTERM", stop);
writeFileSync(
  resolve(supervisor, "started.json"),
  JSON.stringify({
    pid: child.pid,
    atUtc: new Date().toISOString(),
    limitMs: 600000,
  }) + "\n",
  { flag: "wx" },
);
child.stdout.on("data", (b) => {
  appendFileSync(log, b);
  process.stdout.write(b);
});
child.stderr.on("data", (b) => {
  appendFileSync(log, b);
  process.stderr.write(b);
});
try {
  const result = await new Promise((success, reject) => {
    child.once("error", reject);
    child.once("close", (code, signal) =>
      success({
        code,
        signal,
        timedOut,
        cancelled,
        elapsedMs: Date.now() - started,
      }),
    );
  });
  const pass =
    result.code === 0 &&
    !result.timedOut &&
    !result.cancelled &&
    result.elapsedMs <= 600000;
  writeFileSync(
    resolve(supervisor, "report.json"),
    JSON.stringify(
      {
        status: pass ? "PASS" : "FAIL",
        ...result,
        preservation: pass
          ? "Check separate original-file audit"
          : "NOT_VERIFIED; separate original-file audit required",
      },
      null,
      2,
    ) + "\n",
    { flag: "wx" },
  );
  if (!pass)
    throw new Error(
      "Owned diagnostic child failed/cancelled/timed out; STOP, no extra forward retry",
    );
} finally {
  clearTimeout(timer);
  process.removeListener("SIGINT", stop);
  process.removeListener("SIGTERM", stop);
  if (child.exitCode === null && child.signalCode === null) child.kill();
}
