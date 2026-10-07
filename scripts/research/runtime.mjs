import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { resolve, dirname, relative } from "node:path";
import { pathToFileURL } from "node:url";
import { createHash } from "node:crypto";
import ts from "typescript";

/** Existing compiler only. Project-local pure sources are emitted as native ESM. */
export function compileResearch(root, entry) {
  const modules = new Map();
  const hashes = {};
  const base = resolve(root, "src");
  const outputRoot = resolve(root, ".research-output/runtime");
  function visit(file) {
    file = resolve(file);
    if (!file.startsWith(base + "/") && !file.startsWith(base + "\\"))
      throw new Error("Runtime source leaves src");
    if (!file.endsWith(".ts"))
      throw new Error("Only project TypeScript source is supported");
    if (modules.has(file)) return modules.get(file);
    const name = relative(root, file).replaceAll("\\", "/");
    const output = resolve(outputRoot, name.replace(/\.ts$/, ".mjs"));
    modules.set(file, output);
    const raw = readFileSync(file, "utf8");
    const source = ts.createSourceFile(file, raw, ts.ScriptTarget.ES2022, true);
    function check(node) {
      if (
        (ts.isImportDeclaration(node) || ts.isExportDeclaration(node)) &&
        node.moduleSpecifier
      ) {
        if (
          !ts.isStringLiteral(node.moduleSpecifier) ||
          !node.moduleSpecifier.text.startsWith(".") ||
          (ts.isImportDeclaration(node) && !node.importClause)
        )
          throw new Error(
            "Only static relative imports with bindings are supported",
          );
      }
      if (
        ts.isCallExpression(node) &&
        (node.expression.kind === ts.SyntaxKind.ImportKeyword ||
          (ts.isIdentifier(node.expression) &&
            node.expression.text === "require"))
      )
        throw new Error("Dynamic runtime imports are unsupported");
      ts.forEachChild(node, check);
    }
    check(source);
    hashes[name] = createHash("sha256").update(raw).digest("hex");
    let code = ts.transpileModule(raw, {
      compilerOptions: {
        module: ts.ModuleKind.ESNext,
        target: ts.ScriptTarget.ES2022,
      },
    }).outputText;
    code = code.replace(
      /(from\s+["'])([^"']+)(["'])/g,
      (_all, prefix, specifier, suffix) => {
        if (!specifier.startsWith("."))
          throw new Error("Research runtime imports must be project-relative");
        visit(resolve(dirname(file), specifier + ".ts"));
        return `${prefix}${specifier}.mjs${suffix}`;
      },
    );
    mkdirSync(dirname(output), { recursive: true });
    writeFileSync(output, code);
    return output;
  }
  return {
    url: pathToFileURL(visit(resolve(root, entry))).href,
    hashes,
    cache: ".research-output/runtime",
    compilerVersion: ts.version,
  };
}
