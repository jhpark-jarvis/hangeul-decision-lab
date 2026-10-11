import {
  readFileSync,
  writeFileSync,
  appendFileSync,
  existsSync,
  mkdirSync,
} from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { spawn, execFileSync } from "node:child_process";
import { createInterface } from "node:readline";
import { performance } from "node:perf_hooks";
import { compileResearch } from "./runtime.mjs";
const root = resolve(fileURLToPath(new URL("../../", import.meta.url)));
const args = process.argv.slice(2),
  options = {},
  seen = new Set();
for (let i = 0; i < args.length; i += 2) {
  if (
    !["--output", "--python", "--protocol"].includes(args[i]) ||
    !args[i + 1] ||
    seen.has(args[i])
  )
    throw new Error(
      "Use --output new-directory --python existing-python [--protocol file]",
    );
  seen.add(args[i]);
  options[args[i]] = resolve(args[i + 1]);
}
if (!options["--output"] || !options["--python"])
  throw new Error("Output and existing Python required");
const output = options["--output"],
  python = options["--python"];
if (existsSync(output))
  throw new Error("Existing output will not be overwritten");
if (!existsSync(python))
  throw new Error("Existing Python not found; no install is performed");
const hash = (body) => createHash("sha256").update(body).digest("hex"),
  fileHash = (p) => hash(readFileSync(p)),
  json = (p) => JSON.parse(readFileSync(p, "utf8"));
const runtime = compileResearch(root, "src/research/paired-teacher.ts"),
  api = await import(runtime.url);
const protocolBytes = readFileSync(
    options["--protocol"] ?? resolve(root, "research/cnn/paired-protocol.json"),
  ),
  protocol = api.validatePairedProtocol(JSON.parse(protocolBytes));
if (
  fileHash(resolve(root, "research/cnn/protocol.json")) !==
  protocol.sourceProtocolSha256
)
  throw new Error("Original CNN protocol drift");
const environment = JSON.parse(
  execFileSync(
    python,
    [
      "-c",
      "import sys,json,torch; print(json.dumps({'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda}))",
    ],
    { encoding: "utf8", windowsHide: true },
  ),
);
if (
  !environment.python.startsWith("3.13.13") ||
  environment.torch !== "2.14.1+cpu" ||
  environment.cuda !== null
)
  throw new Error("Approved CPU environment required");
const sourceHashes = {
  ...runtime.hashes,
  ...Object.fromEntries(
    [
      "research/cnn/paired.py",
      "research/cnn/model.py",
      "research/cnn/train.py",
      "research/cnn/data.py",
      "research/cnn/paired-protocol.json",
      "research/cnn/protocol.json",
      "research/cnn/requirements-lock.txt",
      "scripts/research/paired-cnn.mjs",
      "package.json",
      "pnpm-lock.yaml",
    ].map((p) => [p, fileHash(resolve(root, p))]),
  ),
};
const metadata = {
  startedAtUtc: new Date().toISOString(),
  executor: "local-paired-cli",
  node: process.version,
  compiler: runtime.compilerVersion,
  environment,
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
mkdirSync(output, { recursive: true });
const write = (folder, name, value) =>
  writeFileSync(resolve(folder, name), JSON.stringify(value, null, 2) + "\n", {
    flag: "wx",
  });
writeFileSync(resolve(output, "protocol.json"), protocolBytes, { flag: "wx" });
write(output, "started.json", {
  protocolSha256: hash(protocolBytes),
  sourceHashes,
  metadata,
});
writeFileSync(resolve(output, "progress.jsonl"), "", { flag: "wx" });
let stage = "train-dev",
  worker;
const progress = (value) => {
  appendFileSync(
    resolve(output, "progress.jsonl"),
    JSON.stringify({ at: new Date().toISOString(), ...value }) + "\n",
  );
};
function sourceCheck() {
  for (const [p, d] of Object.entries(sourceHashes))
    if (fileHash(resolve(root, p)) !== d)
      throw new Error("Frozen source changed: " + p);
}
function collect(stage, registry, freezeSha256) {
  const folder = resolve(output, stage);
  mkdirSync(folder);
  writeFileSync(resolve(folder, "pairs.jsonl"), "", { flag: "wx" });
  writeFileSync(resolve(folder, "episodes.jsonl"), "", { flag: "wx" });
  const pack = api.collectPairs(
    protocol,
    stage,
    registry,
    hash,
    () => performance.now(),
    {
      pair: (p) =>
        appendFileSync(
          resolve(folder, "pairs.jsonl"),
          JSON.stringify(p) + "\n",
        ),
      episode: (e, n) => {
        appendFileSync(
          resolve(folder, "episodes.jsonl"),
          JSON.stringify(e) + "\n",
        );
        progress({ stage, episode: n, fixtureId: e.fixture.id });
        console.log(`Verified ${stage} episode ${n}: ${e.fixture.id}`);
      },
    },
  );
  api.verifyPairPack(pack);
  const body = JSON.stringify(pack);
  writeFileSync(resolve(folder, "dataset.json"), body, { flag: "wx" });
  writeFileSync(resolve(folder, "protocol.json"), protocolBytes, {
    flag: "wx",
  });
  const manifest = {
    status: "PASS",
    stage,
    dataSha256: hash(body),
    protocolSha256: hash(protocolBytes),
    sourceHashes,
    freezeSha256: freezeSha256 ?? null,
    accounting: pack.accounting,
    counts: Object.fromEntries(
      ["train", "dev", "test"].map((s) => [s, pack.examples[s].length]),
    ),
    duplicates: pack.duplicates.length,
    skips: pack.skips.length,
  };
  write(folder, "manifest.json", manifest);
  return { pack, manifest, folder };
}
async function child(arguments_, logName) {
  const log = resolve(output, logName);
  writeFileSync(log, "", { flag: "wx" });
  await new Promise((accept, reject) => {
    const proc = spawn(
      python,
      [resolve(root, "research/cnn/paired.py"), ...arguments_],
      { cwd: root, windowsHide: true, stdio: ["ignore", "pipe", "pipe"] },
    );
    progress({ stage, childPid: proc.pid });
    proc.stdout.on("data", (b) => {
      appendFileSync(log, b);
      process.stdout.write(b);
    });
    proc.stderr.on("data", (b) => {
      appendFileSync(log, b);
      process.stderr.write(b);
    });
    proc.on("error", reject);
    proc.on("close", (code) =>
      code === 0 ? accept() : reject(new Error("Python child failed: " + code)),
    );
  });
}
async function startWorker(freezePath) {
  const proc = spawn(
    python,
    [resolve(root, "research/cnn/paired.py"), "worker", "--freeze", freezePath],
    { cwd: root, windowsHide: true, stdio: ["pipe", "pipe", "pipe"] },
  );
  const log = resolve(output, "worker-stderr.log");
  writeFileSync(log, "", { flag: "wx" });
  proc.stderr.on("data", (b) => appendFileSync(log, b));
  const reader = createInterface({ input: proc.stdout }),
    queue = [],
    waiting = [];
  let failed = null,
    sequence = 0;
  reader.on("line", (line) => {
    try {
      const v = JSON.parse(line);
      if (waiting.length) waiting.shift().accept(v);
      else queue.push(v);
    } catch (e) {
      failed = e;
      while (waiting.length) waiting.shift().reject(e);
    }
  });
  proc.on("error", (e) => {
    failed = e;
    while (waiting.length) waiting.shift().reject(e);
  });
  proc.on("close", (code) => {
    failed ??= new Error("Inference worker closed: " + code);
    while (waiting.length) waiting.shift().reject(failed);
  });
  const receive = () =>
    failed
      ? Promise.reject(failed)
      : queue.length
        ? Promise.resolve(queue.shift())
        : new Promise((accept, reject) => waiting.push({ accept, reject }));
  if ((await receive()).ready !== true)
    throw new Error("Inference worker readiness failed");
  progress({ stage, childPid: proc.pid });
  return {
    proc,
    reader,
    async predict(model, examples) {
      const id = ++sequence;
      proc.stdin.write(JSON.stringify({ id, model, examples }) + "\n");
      const r = await receive();
      if (
        r.id !== id ||
        !Array.isArray(r.indices) ||
        r.indices.length !== examples.length
      )
        throw new Error("Inference response mismatch");
      return r;
    },
    async close() {
      if (proc.exitCode !== null) return;
      const closed = new Promise((accept) => proc.once("close", accept));
      proc.stdin.end(JSON.stringify({ stop: true }) + "\n");
      const timer = setTimeout(() => proc.kill(), 5000);
      await closed;
      clearTimeout(timer);
      reader.close();
      if (proc.exitCode !== 0) throw new Error("Worker did not exit cleanly");
    },
  };
}
try {
  const episode = await import(new URL("./episode.mjs", runtime.url)),
    imitation = await import(new URL("./imitation.mjs", runtime.url)),
    solver = await import(new URL("../domain/solver/solver.mjs", runtime.url)),
    catalog = await import(
      new URL("../domain/pieces/catalog.mjs", runtime.url)
    ),
    verify = await import(new URL("./teacher-audit.mjs", runtime.url)),
    evaluate = await import(
      new URL("../domain/solver/evaluator.mjs", runtime.url)
    );
  const train = collect("train-dev", []);
  stage = "training";
  progress({ stage });
  for (const name of api.TEACHERS)
    await child(
      [
        "train",
        "--dataset",
        train.folder,
        "--teacher",
        name,
        "--output",
        resolve(output, "models", name),
      ],
      `training-${name}.log`,
    );
  const trainingReports = Object.fromEntries(
    api.TEACHERS.map((n) => [
      n,
      json(resolve(output, "models", n, "report.json")),
    ]),
  );
  for (const name of api.TEACHERS) {
    const r = trainingReports[name];
    if (
      r.status !== "PASS" ||
      r.testLoaded ||
      r.testScored ||
      r.history.length !== 12 ||
      r.dataSha256 !== train.manifest.dataSha256 ||
      r.protocolSha256 !== hash(protocolBytes) ||
      r.checkpointSha256 !==
        fileHash(resolve(output, "models", name, "checkpoint.pt"))
    )
      throw new Error("Training identity/freeze mismatch");
  }
  if (
    JSON.stringify(trainingReports.A512.audit) !==
    JSON.stringify(trainingReports.A1024.audit)
  )
    throw new Error("Initialization/sample order mismatch");
  sourceCheck();
  const freeze = {
    schemaVersion: 1,
    status: "FROZEN_FOR_TEST",
    frozenAtUtc: new Date().toISOString(),
    protocolSha256: hash(protocolBytes),
    sourceHashes,
    trainDevDataSha256: train.manifest.dataSha256,
    trainingAudit: trainingReports.A512.audit,
    models: Object.fromEntries(
      api.TEACHERS.map((n) => [
        n,
        {
          checkpoint: `models/${n}/checkpoint.pt`,
          sha256: trainingReports[n].checkpointSha256,
          reportSha256: fileHash(resolve(output, "models", n, "report.json")),
          selectedEpoch: trainingReports[n].selectedEpoch,
        },
      ]),
    ),
  };
  write(output, "freeze.json", freeze);
  stage = "test";
  progress({ stage, freezeSha256: fileHash(resolve(output, "freeze.json")) });
  // No test fixture is created before both reports, checkpoints, and freeze persist.
  const test = collect(
    "test",
    train.pack.registry,
    fileHash(resolve(output, "freeze.json")),
  );
  stage = "test-evaluation";
  await child(
    [
      "evaluate",
      "--dataset",
      test.folder,
      "--freeze",
      resolve(output, "freeze.json"),
      "--output",
      resolve(output, "evaluation"),
    ],
    "evaluation.log",
  );
  const predictions = json(resolve(output, "evaluation", "predictions.json"));
  for (const n of api.TEACHERS)
    for (const [i, pair] of test.pack.examples.test.entries()) {
      const r = predictions[n][i];
      if (
        !r ||
        r.id !== pair.id ||
        !Number.isSafeInteger(r.index) ||
        r.index < 0 ||
        r.index >= pair.candidateKeys.length ||
        r.key !== pair.candidateKeys[r.index]
      )
        throw new Error("Prediction failed domain key validation");
    }
  stage = "rollout";
  worker = await startWorker(resolve(output, "freeze.json"));
  const rollout = [],
    rolloutPredictions = [];
  let rolloutNominalNodes = 0,
    rolloutVisitedNodes = 0;
  const fixtures = test.pack.episodes.map((e) => e.fixture);
  for (const policy of protocol.evaluation.rolloutPolicies) {
    if (!policy.startsWith("CNN-"))
      for (const f of fixtures) {
        const r = episode.runEpisode(f, ({ state, legalActions }) => {
          if (policy === "first-legal")
            return { action: legalActions[0] ?? null };
          const s = solver.solveTurn(
            state,
            catalog.getInitialCatalog(),
            protocol.teachers[policy],
          );
          if (!s.ok) throw new Error(s.error.message);
          rolloutNominalNodes += protocol.teachers[policy].maxNodes;
          rolloutVisitedNodes += s.result.search.visitedNodes;
          for (const c of [s.result, ...s.result.alternatives])
            verify.verifyCandidatePath(state, c);
          return {
            action: s.result.actions[0] ?? null,
            search: s.result.search,
          };
        });
        if (
          !r.ok ||
          !episode.verifyReplay(f, r.value).ok ||
          r.value.ending.kind === "error"
        )
          throw new Error("Baseline rollout failed");
        rollout.push({ policy, fixtureId: f.id, result: r.value });
        console.log(`Verified rollout ${policy}: ${f.id}`);
      }
    else {
      const histories = fixtures.map(() => []),
        done = new Set();
      for (let round = 0; round <= protocol.dataset.maxActions; round++) {
        const pending = [];
        for (const [i, f] of fixtures.entries())
          if (!done.has(i)) {
            const paused = api.pauseEpisode(f, histories[i]);
            if (!paused.pending) {
              if (!episode.verifyReplay(f, paused.result).ok)
                throw new Error("CNN rollout replay failed");
              rollout.push({ policy, fixtureId: f.id, result: paused.result });
              done.add(i);
              continue;
            }
            const o = paused.pending,
              encoded = imitation.encodeDecision(
                o.state,
                o.legalActions,
                o.legalActions[0],
              );
            if (!encoded.ok) throw new Error(encoded.error);
            pending.push({ i, o, encoded: encoded.value });
          }
        if (!pending.length) break;
        const response = await worker.predict(
          policy.slice(4),
          pending.map((x) => ({
            observation: x.encoded.observation,
            candidateFeatures: x.encoded.candidateFeatures,
          })),
        );
        for (const [j, p] of pending.entries()) {
          const index = response.indices[j];
          if (
            !Number.isSafeInteger(index) ||
            index < 0 ||
            index >= p.o.legalActions.length
          )
            throw new Error("Illegal rollout model index");
          histories[p.i].push({ action: p.o.legalActions[index] });
          rolloutPredictions.push({
            policy,
            fixtureId: fixtures[p.i].id,
            actionIndex: p.o.actionIndex,
            stateKey: episode.stateKey(p.o.state),
            index,
            key: p.encoded.candidateKeys[index],
            inferenceBatchMs: response.inferenceMs,
          });
        }
      }
      if (done.size !== 8)
        throw new Error("CNN rollout horizon accounting mismatch");
      console.log(`Verified ${policy} eight domain rollouts`);
    }
  }
  await worker.close();
  worker = null;
  const policyDecisions = rollout.reduce(
    (n, r) => n + r.result.totals.actions,
    0,
  );
  if (
    rollout.length !== 40 ||
    policyDecisions > 240 ||
    rolloutNominalNodes > 73728 ||
    train.pack.accounting.nominalNodes +
      test.pack.accounting.nominalNodes +
      rolloutNominalNodes >
      1155072
  )
    throw new Error("Experiment cost/call cap exceeded");
  sourceCheck();
  const pairs = [
    ...train.pack.examples.train,
    ...train.pack.examples.dev,
    ...test.pack.examples.test,
  ];
  const teacherComparison = Object.fromEntries(
    ["train", "dev", "test"].map((split) => {
      const group = pairs.filter((p) => p.split === split),
        comparisons = group.map((p) =>
          evaluate.compareEvaluations(
            p.teachers.A1024.result.evaluation,
            p.teachers.A512.result.evaluation,
          ),
        );
      return [
        split,
        {
          examples: group.length,
          labelsChanged: group.filter((p) => p.labels.A512 !== p.labels.A1024)
            .length,
          better: comparisons.filter((x) => x > 0).length,
          worse: comparisons.filter((x) => x < 0).length,
          equal: comparisons.filter((x) => x === 0).length,
          incomplete: Object.fromEntries(
            api.TEACHERS.map((n) => [
              n,
              group.filter((p) => !p.teachers[n].result.search.searchComplete)
                .length,
            ]),
          ),
        },
      ];
    }),
  );
  const policySummary = protocol.evaluation.rolloutPolicies.map((policy) => {
    const rows = rollout.filter((r) => r.policy === policy),
      keys = Object.keys(rows[0].result.totals);
    return {
      policy,
      episodes: rows.length,
      totals: Object.fromEntries(
        keys.map((k) => [k, rows.reduce((n, r) => n + r.result.totals[k], 0)]),
      ),
      endings: rows.map((r) => ({
        fixtureId: r.fixtureId,
        ...r.result.ending,
      })),
      retainedItems: rows.reduce(
        (n, r) =>
          n +
          r.result.events
            .filter((e) => e.type === "action")
            .reduce((m, e) => m + e.info.retainedItems.length, 0),
        0,
      ),
    };
  });
  write(output, "rollout.json", {
    status: "PASS",
    domainReplay: true,
    policies: rollout,
    predictions: rolloutPredictions,
    policyDecisions,
    rolloutNominalNodes,
    rolloutVisitedNodes,
  });
  const evalReport = json(resolve(output, "evaluation", "report.json"));
  const summary = {
    schemaVersion: 1,
    status: "PASS",
    id: protocol.id,
    protocol,
    sourceHashes,
    trainDevDataSha256: train.manifest.dataSha256,
    testDataSha256: test.manifest.dataSha256,
    freezeSha256: fileHash(resolve(output, "freeze.json")),
    training: Object.fromEntries(
      api.TEACHERS.map((n) => [
        n,
        {
          selectedEpoch: trainingReports[n].selectedEpoch,
          dev: trainingReports[n].history[trainingReports[n].selectedEpoch - 1]
            .dev,
          checkpointSha256: trainingReports[n].checkpointSha256,
          audit: trainingReports[n].audit,
          timings: trainingReports[n].timings,
        },
      ]),
    ),
    counts: {
      ...train.manifest.counts,
      test: test.manifest.counts.test,
      episodes: 88,
      duplicates: train.pack.duplicates.length + test.pack.duplicates.length,
      skips: train.pack.skips.length + test.pack.skips.length,
    },
    teacherComparison,
    crossTeacher: evalReport.crossTeacher,
    firstLegal: evalReport.firstLegal,
    policySummary,
    accounting: {
      collectionAndLabelNominalNodes:
        train.pack.accounting.nominalNodes + test.pack.accounting.nominalNodes,
      collectionA512Reused: true,
      rolloutNominalNodes,
      actualVisitedNodes:
        train.pack.accounting.visitedNodes +
        test.pack.accounting.visitedNodes +
        rolloutVisitedNodes,
      policyDecisions,
    },
    checks: {
      bothCheckpointsFrozenBeforeTest: true,
      sameInitializationAndSampleOrder: true,
      domainPredictionKeys: true,
      domainRolloutReplay: true,
      finitePredictions: true,
      postTestTuning: false,
    },
    limits: [
      "88 synthetic episodes, two shared generator families; not real-game distribution or independent action samples",
      "Bounded teachers and single hard labels; evaluation gains are not label quality or future value guarantees",
      "Six-action rollouts truncate play; not long-term survival or application acceptance",
      "Two fixed training runs only; no test-driven tuning or statistical superiority claim",
      "Original dataset/checkpoint and application solver remain unchanged",
    ],
  };
  write(output, "summary.json", summary);
  write(output, "report.json", {
    ...summary,
    metadata: {
      ...metadata,
      finishedAtUtc: new Date().toISOString(),
      memoryAfter: process.memoryUsage(),
    },
    trainingReports,
  });
  progress({ stage: "COMPLETE" });
  console.log(
    JSON.stringify(
      {
        status: "PASS",
        counts: summary.counts,
        teacherComparison,
        crossTeacher: summary.crossTeacher,
        policySummary,
      },
      null,
      2,
    ),
  );
} catch (error) {
  if (worker) {
    worker.proc.kill();
    worker.reader.close();
  }
  write(output, "failure.json", {
    status: "FAIL",
    stage,
    error: error.message,
    at: new Date().toISOString(),
    sourceHashes,
    metadata,
  });
  throw error;
}
