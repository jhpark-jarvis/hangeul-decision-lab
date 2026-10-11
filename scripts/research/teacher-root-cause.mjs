import {
  readFileSync,
  writeFileSync,
  appendFileSync,
  existsSync,
  mkdirSync,
} from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { performance } from "node:perf_hooks";
import { format } from "prettier";
import { compileResearch } from "./runtime.mjs";

const root = resolve(fileURLToPath(new URL("../../", import.meta.url))),
  args = process.argv.slice(2),
  seen = new Set();
const options = {
  "--protocol": "research/experiments/teacher-root-cause-v1.json",
  "--canonical-report":
    ".research-output/canonical/canonical-teacher-20261010-v1/report.json",
  "--permutation-report":
    ".research-output/permutation/permutation-teacher-20261008-v1/report.json",
};
let output;
for (let i = 0; i < args.length; i += 2) {
  if (
    !args[i + 1] ||
    seen.has(args[i]) ||
    !["--output", ...Object.keys(options)].includes(args[i])
  )
    throw new Error(
      "Use --output new-directory [--protocol file] [--canonical-report file] [--permutation-report file]",
    );
  seen.add(args[i]);
  if (args[i] === "--output") output = resolve(args[i + 1]);
  else options[args[i]] = args[i + 1];
}
if (!output) throw new Error("New output directory required");
if (existsSync(output))
  throw new Error("Existing output will not be overwritten");
const runtime = compileResearch(root, "src/research/teacher-root-cause.ts"),
  api = await import(runtime.url);
const protocol = api.validateRootCauseProtocol(
  JSON.parse(readFileSync(resolve(root, options["--protocol"]), "utf8")),
);
const sha = (data) => createHash("sha256").update(data).digest("hex"),
  json = (p) => JSON.parse(readFileSync(p, "utf8"));
const metadata = {
  startedAtUtc: new Date().toISOString(),
  executor: "local-cli",
  node: process.version,
  compiler: runtime.compilerVersion,
  revision: execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: root,
    encoding: "utf8",
  }).trim(),
  workingTree: execFileSync("git", ["status", "--short"], {
    cwd: root,
    encoding: "utf8",
  }).trim(),
  memoryBefore: process.memoryUsage(),
};
const identity = {
  protocolSha256: sha(JSON.stringify(protocol)),
  sourceHashes: runtime.hashes,
  runnerSha256: sha(readFileSync(fileURLToPath(import.meta.url))),
  lockSha256: sha(readFileSync(resolve(root, "pnpm-lock.yaml"))),
};
const write = (name, value) =>
  writeFileSync(resolve(output, name), JSON.stringify(value, null, 2) + "\n", {
    flag: "wx",
  });
mkdirSync(output, { recursive: true });
write("protocol.json", protocol);
write("started.json", { identity, metadata });
writeFileSync(resolve(output, "calls.jsonl"), "", { flag: "wx" });
let completedCalls = 0;
const start = performance.now();
try {
  const canonicalPath = resolve(root, options["--canonical-report"]),
    permutationPath = resolve(root, options["--permutation-report"]);
  if (
    sha(readFileSync(canonicalPath)) !== protocol.reports.canonicalSha256 ||
    sha(readFileSync(permutationPath)) !== protocol.reports.permutationSha256
  )
    throw new Error("Historical report hash mismatch");
  const canonical = json(canonicalPath),
    permutation = json(permutationPath);
  for (const report of [canonical, permutation])
    for (const [file, digest] of Object.entries(report.identity.sourceHashes))
      if (sha(readFileSync(resolve(root, file))) !== digest)
        throw new Error(`Historical source drift: ${file}`);
  const episode = await import(new URL("./episode.mjs", runtime.url)),
    catalogApi = await import(
      new URL("../domain/pieces/catalog.mjs", runtime.url)
    ),
    solver = await import(new URL("../domain/solver/solver.mjs", runtime.url)),
    canon = await import(new URL("./canonical-teacher.mjs", runtime.url)),
    verify = await import(new URL("./teacher-audit.mjs", runtime.url)),
    evaluate = await import(
      new URL("../domain/solver/evaluator.mjs", runtime.url)
    );
  const history = api.analyzeHistoricalReports(canonical, permutation),
    catalog = catalogApi.getInitialCatalog();
  history.cases.forEach((c, i) => {
    if (
      c.fixtureId !== protocol.inputs[i].id ||
      sha(episode.stateKey(c.state)) !== protocol.inputs[i].stateSha256
    )
      throw new Error("Historical input hash/order mismatch");
  });
  write("historical-analysis.json", history);
  const solverFile = fileURLToPath(
    new URL("../domain/solver/solver.mjs", runtime.url),
  );
  const originalCode = readFileSync(solverFile, "utf8");
  const traceCode = api.buildPassiveTrace(originalCode);
  const traceFile = resolve(
    root,
    ".research-output/runtime/src/domain/solver/solver-root-cause-trace.mjs",
  );
  writeFileSync(traceFile, traceCode);
  writeFileSync(resolve(output, "trace-code.mjs"), traceCode, { flag: "wx" });
  identity.trace = {
    originalCompiledSha256: sha(originalCode),
    instrumentedSha256: sha(traceCode),
    insertionCounts: { visit: 1, consider: 1 },
    originalSourceUnchanged: true,
  };
  write("trace-identity.json", identity.trace);
  const tracer = await import(pathToFileURL(traceFile).href),
    rows = [],
    traced = [];
  function execute(
    c,
    condition,
    budget,
    repetition,
    isTraced = false,
    expected,
  ) {
    const before = JSON.stringify(c.state),
      t0 = performance.now(),
      mapping = canon.prepareCanonicalInput(
        c.state,
        condition === "A1024" || condition === "A512"
          ? "baseline"
          : "canonical-012",
      );
    canon.assertCanonicalLegality(c.state, mapping);
    const stateBefore = JSON.stringify(mapping.state),
      config = { maxNodes: budget, maxAlternatives: 3, useMemoization: true };
    const call = isTraced
      ? tracer.tracedTurn(mapping.state, catalog, config)
      : {
          result: solver.solveTurn(mapping.state, catalog, config),
          events: null,
        };
    if (!call.result.ok) throw new Error(call.result.error.message);
    const result = call.result.result,
      restored = {
        ...canon.restoreCanonicalCandidate(result, mapping),
        search: structuredClone(result.search),
        alternatives: result.alternatives.map((x) =>
          canon.restoreCanonicalCandidate(x, mapping),
        ),
      };
    if (
      before !== JSON.stringify(c.state) ||
      stateBefore !== JSON.stringify(mapping.state)
    )
      throw new Error("Input mutation");
    if (
      result.search.maxNodes !== budget ||
      result.search.visitedNodes > budget
    )
      throw new Error("Node budget drift");
    for (const candidate of [result, ...result.alternatives])
      verify.verifyCandidatePath(mapping.state, candidate);
    for (const candidate of [restored, ...restored.alternatives])
      verify.verifyCandidatePath(c.state, candidate);
    if (
      expected &&
      api.logicalTurnSignature(restored) !== api.logicalTurnSignature(expected)
    )
      throw new Error(
        `Trace/result parity mismatch ${c.fixtureId}/${condition}`,
      );
    if (isTraced) api.assertTraceAccounting(call.events, result);
    const row = {
      fixtureId: c.fixtureId,
      stratum: c.stratum,
      condition,
      budget,
      repetition,
      sequence: ++completedCalls,
      isTraced,
      mapping,
      result,
      restored,
      trace: call.events,
      latencyWithReplayAndTraceMs: performance.now() - t0,
    };
    appendFileSync(resolve(output, "calls.jsonl"), JSON.stringify(row) + "\n");
    console.log(
      `Verified ${row.sequence}/60 ${c.fixtureId} / ${condition}${isTraced ? " trace" : ""}`,
    );
    return row;
  }
  for (const [i, c] of history.cases.entries())
    for (let rep = 0; rep < 2; rep++) {
      const conditions = protocol.lossIds.includes(c.fixtureId)
        ? ["A1024", "D3072"]
        : ["A1024"];
      if ((i + rep) % 2) conditions.reverse();
      for (const condition of conditions)
        rows.push(
          execute(c, condition, condition === "A1024" ? 1024 : 3072, rep),
        );
    }
  for (const c of history.losses) {
    traced.push(
      execute(c, "A512", 512, 0, true, c.a),
      execute(c, "D512", 512, 0, true, c.d),
    );
    const high = rows.find(
      (r) =>
        r.fixtureId === c.fixtureId &&
        r.condition === "D3072" &&
        r.repetition === 0,
    );
    traced.push(execute(c, "D3072", 3072, 0, true, high.restored));
  }
  for (const row of rows.filter((r) => r.repetition === 0)) {
    const other = rows.find(
      (r) =>
        r.fixtureId === row.fixtureId &&
        r.condition === row.condition &&
        r.repetition === 1,
    );
    if (
      api.logicalTurnSignature(row.restored) !==
      api.logicalTurnSignature(other.restored)
    )
      throw new Error("Repeated result drift");
  }
  if (
    completedCalls !== 60 ||
    [...rows, ...traced].reduce((n, r) => n + r.budget, 0) !== 81920
  )
    throw new Error("Frozen call accounting drift");
  const comparisons = history.cases.map((c) => {
    const a1024 = rows.find(
        (r) =>
          r.fixtureId === c.fixtureId &&
          r.condition === "A1024" &&
          r.repetition === 0,
      ),
      portfolio = api.incumbentPortfolio(c.a, c.d);
    verify.verifyCandidatePath(c.state, portfolio.result);
    const compare = evaluate.compareEvaluations;
    if (compare(portfolio.result.evaluation, c.a.evaluation) < 0)
      throw new Error("Incumbent portfolio regression");
    return {
      fixtureId: c.fixtureId,
      stratum: c.stratum,
      portfolioSource: portfolio.source,
      portfolioVsA512: compare(portfolio.result.evaluation, c.a.evaluation),
      a1024VsA512: compare(a1024.restored.evaluation, c.a.evaluation),
      portfolioVsA1024: compare(
        portfolio.result.evaluation,
        a1024.restored.evaluation,
      ),
      a1024Difference: api.firstEvaluationDifference(
        a1024.restored.evaluation,
        c.a.evaluation,
      ),
      a1024Search: a1024.restored.search,
    };
  });
  const losses = history.losses.map((c) => {
    const low = traced.find(
        (r) => r.fixtureId === c.fixtureId && r.condition === "D512",
      ),
      high = traced.find(
        (r) => r.fixtureId === c.fixtureId && r.condition === "D3072",
      ),
      a = traced.find(
        (r) => r.fixtureId === c.fixtureId && r.condition === "A512",
      );
    return {
      fixtureId: c.fixtureId,
      map: c.map,
      firstDifference: c.difference,
      root: c.root,
      d512Search: c.d.search,
      a512Witness: api.traceWitness(a.trace, c.a),
      d512Witness: api.traceWitness(low.trace, c.a),
      d3072Witness: api.traceWitness(high.trace, c.a),
      d3072VsA512: evaluate.compareEvaluations(
        high.restored.evaluation,
        c.a.evaluation,
      ),
      d3072VsD512: evaluate.compareEvaluations(
        high.restored.evaluation,
        c.d.evaluation,
      ),
      d3072VsB3072: evaluate.compareEvaluations(
        high.restored.evaluation,
        c.b3072.evaluation,
      ),
      d3072Search: high.restored.search,
    };
  });
  const aggregate = (stratum, key) => {
    const group = comparisons.filter((c) => c.stratum === stratum);
    return {
      states: group.length,
      better: group.filter((c) => c[key] > 0).length,
      worse: group.filter((c) => c[key] < 0).length,
      equal: group.filter((c) => c[key] === 0).length,
    };
  };
  const summary = {
    status: "PASS",
    id: protocol.id,
    historical: {
      sourceCallsVerified: history.sourceCallsVerified,
      returnedPathsVerified: history.returnedPathsVerified,
      canonicalEquivalentPermutationMatches: history.cases.filter(
        (c) => c.dMatchesHistoricalPermutation,
      ).length,
    },
    measuredCalls: 60,
    nominalNodeCap: 81920,
    repeatedConditions: 24,
    traceParityCalls: 12,
    returnedPathsVerified: [...rows, ...traced].reduce(
      (n, r) => n + 1 + r.restored.alternatives.length,
      0,
    ),
    comparisons,
    losses,
    strata: ["generated", "scripted"].map((stratum) => ({
      stratum,
      portfolioVsA512: aggregate(stratum, "portfolioVsA512"),
      a1024VsA512: aggregate(stratum, "a1024VsA512"),
      portfolioVsA1024: aggregate(stratum, "portfolioVsA1024"),
    })),
    limits: [
      "Observed fixed20 pilot, not held-out teacher adoption or future turn optimality",
      "Larger budgets also change ordinary-probe reservation and may change restricted reroll gating; do not assume a nested search tree",
      "Portfolio protects current evaluation relative to retained A512 only; hard-label/future value equivalence is not claimed",
      "Trace overhead is included; timing is observation only, not equal-wall-time/app acceptance",
      "No new seeds/test/data/model/teacher labels/training/product changes or external data transmission",
    ],
  };
  write("report.json", {
    status: "PASS",
    protocol,
    identity,
    metadata: {
      ...metadata,
      finishedAtUtc: new Date().toISOString(),
      durationMs: performance.now() - start,
      memoryAfter: process.memoryUsage(),
    },
    history,
    rows,
    traced,
    summary,
  });
  writeFileSync(
    resolve(output, "summary.json"),
    await format(
      JSON.stringify(
        { schemaVersion: 1, protocol, identity, summary },
        null,
        2,
      ) + "\n",
      { parser: "json" },
    ),
    { flag: "wx" },
  );
  console.log(
    JSON.stringify(
      {
        status: "PASS",
        strata: summary.strata,
        losses: summary.losses.map((x) => ({
          id: x.fixtureId,
          axis: x.firstDifference.axis,
          aRootInD: x.d512Witness.rootVisited,
          highVsA: x.d3072VsA512,
        })),
      },
      null,
      2,
    ),
  );
} catch (error) {
  write("failure.json", {
    status: "FAIL",
    error: error.message,
    completedCalls,
    identity,
    metadata: { ...metadata, failedAtUtc: new Date().toISOString() },
  });
  throw error;
}
