import assert from "node:assert/strict";
import { spawn } from "node:child_process";

// Owns only the child it creates; callers retain the overall-run timer/cancel state.
export function runBoundedChild({
  command,
  args,
  cwd,
  timeoutMs,
  onSpawn = () => {},
  onStdout = () => {},
  onStderr = () => {},
}) {
  assert(Number.isFinite(timeoutMs) && timeoutMs > 0);
  return new Promise((resolve, reject) => {
    const started = performance.now();
    let timedOut = false;
    const child = spawn(command, args, {
      cwd,
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill();
    }, timeoutMs);
    child.stdout.on("data", onStdout);
    child.stderr.on("data", onStderr);
    child.once("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.once("close", (code, signal) => {
      clearTimeout(timer);
      resolve({
        code,
        signal,
        timedOut,
        pid: child.pid,
        elapsedMs: performance.now() - started,
      });
    });
    onSpawn(child);
  });
}
