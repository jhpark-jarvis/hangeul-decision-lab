import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import ts from "typescript";

/** Evidence tools only: load pure TS modules without adding a runtime package. */
const modules = new Map();
export async function source(file) {
  file = resolve(file);
  if (modules.has(file)) return modules.get(file);
  let code = ts.transpileModule(readFileSync(file, "utf8"), {
    compilerOptions: {
      module: ts.ModuleKind.ESNext,
      target: ts.ScriptTarget.ES2022,
    },
  }).outputText;
  for (const match of [...code.matchAll(/from "(\.[^"]+)"/g)]) {
    const url = await source(resolve(dirname(file), match[1] + ".ts"));
    code = code.replace(match[0], `from "${url}"`);
  }
  const url =
    "data:text/javascript;base64," + Buffer.from(code).toString("base64");
  modules.set(file, url);
  return url;
}
