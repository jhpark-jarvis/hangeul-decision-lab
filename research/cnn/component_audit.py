"""Frozen spatial hooks once; saved-component replay with zero model forwards."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
from data import require, sha256, vector
from diagnose import score
from within_kind import conditional_score, summarize as conditional_summary
from spatial_audit import prepare as stored_prepare, controlled_row, error_kind, pair_rows, summarize_pairs, compute_guard
from spatial import SPATIAL_SCHEMA, make_model, spatial_tensors, Budget
from paired import weights_hash
from train import write_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "research/experiments/frozen-components-v1.json"


@contextmanager
def training_guard():
    saved_optimizer, saved_backward, saved_train = torch.optim.Optimizer.__init__, torch.Tensor.backward, torch.nn.Module.train
    def forbidden(*args, **kwargs): raise ValueError("Training/optimizer/backward forbidden; STOP")
    def evaluation_only(module, mode=True):
        require(mode is False, "train(True) forbidden; STOP")
        return saved_train(module, False)
    try:
        torch.optim.Optimizer.__init__, torch.Tensor.backward, torch.nn.Module.train = forbidden, forbidden, evaluation_only
        yield
    finally:
        torch.optim.Optimizer.__init__, torch.Tensor.backward, torch.nn.Module.train = saved_optimizer, saved_backward, saved_train


@contextmanager
def bounded_model_calls(model, budget):
    original = torch.nn.Module.__call__
    allowed = {id(m) for m in model.modules()}
    def call(module, *args, **kwargs):
        require(id(module) in allowed, "Only frozen registered modules may forward; STOP")
        if module is model:
            require(budget["calls"] < budget["limit"], "Forward budget exhausted; STOP")
            budget["calls"] += 1
            if budget.get("ledger"):
                with budget["ledger"].open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps({"call": budget["calls"], "limit": budget["limit"]}) + "\n")
        return original(module, *args, **kwargs)
    try:
        torch.nn.Module.__call__ = call
        yield
    finally: torch.nn.Module.__call__ = original


def capture(model, inputs):
    common, chunks = [], []
    handles = [model.common.register_forward_hook(lambda module, args, out: common.append(out.detach().clone())),
               model.spatial.register_forward_hook(lambda module, args, out: chunks.append(out.detach().clone().squeeze(1)))]
    try:
        with torch.inference_mode(): full = model(*inputs)
    finally:
        for handle in handles: handle.remove()
    n = len(full)
    require(len(common) == 1 and len(chunks) == (n+31)//32 and
            [len(c) for c in chunks] == [min(32, n-i) for i in range(0, n, 32)], "Original chunk32 hooks")
    residual = torch.cat(chunks)
    values = (common[0], residual, full.detach().clone())
    require(all(t.shape == (n,) and t.dtype == torch.float32 and t.device.type == "cpu" and torch.isfinite(t).all().item() for t in values),
            "Finite float32 components")
    require(torch.equal(values[0] + values[1], values[2]), "Common plus residual must equal full exactly")
    return values


def validate_record(record, example, split, original):
    require(set(record) == {"id", "split", "common", "residual", "full"} and record["id"] == example["id"] and record["split"] == split,
            "Stored component id/split/order")
    values = []
    for key in ("common", "residual", "full"):
        raw = record[key]
        require(isinstance(raw, list) and len(raw) == len(example["candidateKeys"]) and all(type(v) in (int, float) for v in raw), "Full candidate component length")
        t = torch.tensor(raw, dtype=torch.float32)
        require(torch.isfinite(t).all().item() and t.tolist() == raw, "Exact finite float32 component storage")
        values.append(t)
    require(torch.equal(values[0]+values[1], values[2]) and values[2].tolist() == original["logits"], "Original spatial logits/sum identity")
    return values


def common_row(example, original_full, logits):
    result = score(logits, example["labels"])
    selected = result["index"]
    error = error_kind(example["candidateKeys"][example["labels"]["A512"]], example["candidateKeys"][selected])
    return {**original_full, "error": error, "globalCorrect": error == "exact-label", "selectedIndex": selected,
            "selectedKind": json.loads(example["candidateKeys"][selected])[0],
            "conditional": conditional_score(logits, example["candidateKeys"], example["labels"]["A512"]), "fullScore": result}


def residual_row(example, common, residual):
    target, winner = example["labels"]["A512"], int(common.argmax())
    reroll = [i for i, key in enumerate(example["candidateKeys"]) if json.loads(key)[0] == "reroll"]
    return {"id": example["id"], "teacherKind": json.loads(example["candidateKeys"][target])[0],
            "commonSpan": float(common.max()-common.min()), "residualSpan": float(residual.max()-residual.min()),
            "meanAbsResidual": float(residual.double().abs().mean()),
            "teacherMinusCommonWinnerResidual": float(residual[target]-residual[winner]),
            "rerollCandidates": len(reroll), "rerollResidualSpan": float(residual[reroll].max()-residual[reroll].min()) if reroll else None}


def residual_summary(rows):
    mean = lambda key: sum(r[key] for r in rows)/len(rows) if rows else None
    rerolls = [r["rerollResidualSpan"] for r in rows if r["rerollCandidates"] >= 2]
    return {"examples": len(rows), **{k: mean(k) for k in ("commonSpan", "residualSpan", "meanAbsResidual", "teacherMinusCommonWinnerResidual")},
            "atLeastTwoRerollExamples": len(rerolls), "maxWithinRerollResidualSpan": max(rerolls) if rerolls else None}


def comparison_summary(rows, left, right):
    result = summarize_pairs(pair_rows(rows[left], rows[right]))
    names = {"baseline": "left", "spatial": "right", "baselineCorrect": "leftCorrect", "spatialCorrect": "rightCorrect"}
    def rename(value):
        return {names.get(k, k): rename(v) for k, v in value.items()} if isinstance(value, dict) else value
    return {"leftModel": left, "rightModel": right, **rename(result)}


def summarize(rows, residuals):
    comparisons = {"baselineToCommon": ("baseline", "common"), "commonToFull": ("common", "full"), "baselineToFull": ("baseline", "full")}
    return {"models": {m: conditional_summary(rows[m]) for m in rows},
            "comparisons": {name: comparison_summary(rows, a, b) for name, (a, b) in comparisons.items()},
            "residual": {"overall": residual_summary(residuals), "teacherKind": {kind: residual_summary([r for r in residuals if r["teacherKind"] == kind])
                for kind in sorted({r["teacherKind"] for r in residuals})}}}


def prepare(source, experiment, original):
    config = json.loads(CONFIG.read_bytes())
    require(config["approval"] == "USER_APPROVED" and config["status"] == "FROZEN_COMPONENT_DIAGNOSTIC" and config["limits"] ==
            {"topLevelForwards": 856, "repeats": 2, "chunk": 32, "childSeconds": 600, "workingSetBytes": 2147483648,
             "trainingTeacherDomainTestRolloutGeometry": 0}, "Fixed component contract")
    _, pack, report, predicted, recorded, paths, hashes = stored_prepare(source, experiment)
    require(config["source"]["datasetSha256"] == hashes["dataset"] and
            config["source"]["p056Hashes"] == {p: hashes[p] for p in config["source"]["p056Hashes"]}, "Original data identity")
    execution = json.loads((Path(experiment)/"execution.json").read_bytes())
    original_path = Path(original)/"dataset.json"
    require(sha256(original_path.read_bytes()) == config["source"]["outcomeDatasetSha256"] == execution["originalHashes"]["dataset.json"], "Stored outcome74 hash")
    augmented, footprints = json.loads(original_path.read_bytes()), json.loads((Path(experiment)/"footprints.json").read_bytes())
    require(footprints["schema"] == SPATIAL_SCHEMA and footprints["candidates"] == 149597 and
            set(footprints["examples"]) == set(augmented["examples"]) == {"train", "dev"}, "Original component input schema")
    examples, candidates = {}, 0
    for split, count in (("train", 339), ("dev", 89)):
        require(len(pack["examples"][split]) == len(augmented["examples"][split]) == len(footprints["examples"][split]) == count, "Fixed component observations")
        require(all(len(predicted[m][split]) == len(recorded[m][split]) == count for m in ("baseline", "spatial")), "Original stored component row counts")
        examples[split] = []
        for e, a, f in zip(pack["examples"][split], augmented["examples"][split], footprints["examples"][split]):
            require(e["id"] == a["id"] == f["id"] and len(e["candidateKeys"]) == len(a["candidateFeatures"]) == len(f["footprints"]), "Original ordered component input")
            for old, new in zip(e["candidateFeatures"], a["candidateFeatures"]):
                vector(new, 74)
                require(old == new[:52], "Unchanged 52/74 prefix")
            examples[split].append({**e, "candidateFeatures": a["candidateFeatures"], "footprints": f["footprints"]})
            candidates += len(e["candidateKeys"])
    require(candidates == 149597, "Original full mask candidate count")
    protocol = json.loads((Path(source)/"protocol.json").read_bytes())
    families = {e["fixture"]["id"]: next(f for f in ("sparse", "pressure") if e["fixture"]["id"] == f+"-"+str(e["fixture"]["seed"])) for e in pack["episodes"]}
    paths["outcome74"], hashes["outcome74"] = original_path, config["source"]["outcomeDatasetSha256"]
    own_sources = {p: sha256((ROOT/p).read_bytes()) for p in ("research/cnn/component_audit.py", "research/experiments/frozen-components-v1.json",
                   "scripts/research/component-audit.mjs", "research/cnn/spatial_audit.py", "research/experiments/stored-spatial-audit-v1.json")}
    return config, examples, report, predicted, recorded, paths, hashes, own_sources, protocol, families


def process(mode, source, experiment, original, output, recorded_path=None):
    start = time.perf_counter()
    torch.set_num_threads(2)
    config, examples, prior, predicted, old_logits, paths, hashes, sources, protocol, families = prepare(source, experiment, original)
    torch.set_num_interop_threads(1)
    require(torch.version.cuda is None, "Existing CPU-only build required")
    monitor = Budget(600, 2147483648, output, start)
    monitor.sample("loaded")
    identity = {"sourceHashes": sources, "inputHashes": hashes}
    write_json(output/"started.json", {"atUtc": datetime.now(timezone.utc).isoformat(), "mode": mode, "identity": identity})
    budget = {"calls": 0, "limit": 856, "ledger": output/"forward-count.jsonl" if mode == "infer" else None}
    model, original_weights, replay = None, None, None
    if mode == "infer":
        model = make_model(protocol["training"], "spatial").eval()
        saved = torch.load(paths["spatial/checkpoint.pt"], map_location="cpu", weights_only=True)
        require(saved["epoch"] == config["source"]["selectedEpoch"] == 11 and saved["schema"] == SPATIAL_SCHEMA and
                saved["protocol"] == protocol and saved["dataSha256"] == hashes["outcome74"], "Frozen checkpoint identity")
        model.load_state_dict(saved["state_dict"], strict=True)
        original_weights = weights_hash(model)
        require(original_weights == prior["models"]["spatial"]["weightsSha256"], "Frozen logical weights")
    else:
        recorded_path = Path(recorded_path)
        saved = json.loads((recorded_path/"report.json").read_bytes())
        require(saved["status"] == "PASS" and saved["mode"] == "infer" and saved["identity"] == identity and
                saved["accounting"]["topLevelForwards"] == 856 and saved["componentsSha256"] == sha256((recorded_path/"components.jsonl").read_bytes()), "Saved component execution identity")
        replay = iter(json.loads(line) for line in (recorded_path/"components.jsonl").read_text(encoding="utf-8").splitlines())
    raw, summary = {}, {}
    for split in ("train", "dev"):
        rows, residuals = {m: [] for m in ("baseline", "common", "full")}, []
        for e, b, f, old_b, old_f in zip(examples[split], predicted["baseline"][split], predicted["spatial"][split], old_logits["baseline"][split], old_logits["spatial"][split]):
            if mode == "infer":
                inputs = spatial_tensors(e)
                with bounded_model_calls(model, budget):
                    values, repeat = capture(model, inputs), capture(model, inputs)
                require(all(torch.equal(a, b) for a, b in zip(values, repeat)), "Repeated component drift; STOP")
                record = {"id": e["id"], "split": split, **{k: t.tolist() for k, t in zip(("common", "residual", "full"), values)}}
            else: record = next(replay)
            common, residual, full = validate_record(record, e, split, old_f)
            baseline = controlled_row(e, b, old_b, families[e["fixtureId"]])
            original_full = controlled_row(e, f, {"id": e["id"], "logits": full.tolist()}, families[e["fixtureId"]])
            rows["baseline"].append(baseline)
            rows["full"].append(original_full)
            rows["common"].append(common_row(e, original_full, common))
            residuals.append(residual_row(e, common, residual))
            if mode == "infer":
                with (output/"components.jsonl").open("a", encoding="utf-8", newline="\n") as stream: stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
            monitor.sample("component-observation")
        raw[split], summary[split] = {"models": rows, "residuals": residuals}, summarize(rows, residuals)
        print(json.dumps({"split": split, "correct": {m: summary[split]["models"][m]["overall"]["globalHardCorrect"] for m in rows}}), flush=True)
    if replay is not None: require(next(replay, None) is None, "No extra component record")
    require(budget["calls"] == (856 if mode == "infer" else 0), "Exact component forward accounting")
    if model is not None: require(weights_hash(model) == original_weights, "Model weights changed")
    for p, h in hashes.items(): require(sha256(paths[p].read_bytes()) == h, "Original input changed")
    for p, h in sources.items(): require(sha256((ROOT/p).read_bytes()) == h, "Component source changed")
    result = {"schemaVersion": 1, "id": config["id"], "status": "PASS", "method": config["method"], "identity": identity,
              "summary": summary, "originalGate": prior["gate"], "limits": ["Common-only shares jointly trained frozen weights; not independent baseline or causal training effect",
                 "Known train/dev and hard labels unchanged; oracle is diagnostic, not policy", "No residual-only policy or scaling/tuning, training, teacher/domain/test/rollout"]}
    write_json(output/"raw.json", raw)
    write_json(output/"summary.json", result)
    components = output/"components.jsonl" if mode == "infer" else recorded_path/"components.jsonl"
    write_json(output/"report.json", {**result, "mode": mode, "atUtc": datetime.now(timezone.utc).isoformat(),
              "rawSha256": sha256((output/"raw.json").read_bytes()), "componentsSha256": sha256(components.read_bytes()),
              "accounting": {"topLevelForwards": budget["calls"], "storedComponentRecords": 428, "trainingTeacherDomainTestRolloutGeometry": 0},
              "budget": monitor.report(), "environment": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "threads": torch.get_num_threads()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("infer", "replay"), required=True)
    for name in ("source", "experiment", "original", "output"): parser.add_argument("--"+name, required=True)
    parser.add_argument("--recorded")
    args = parser.parse_args()
    require(bool(args.recorded) == (args.mode == "replay"), "Recorded components only for replay")
    output = Path(args.output).resolve()
    require(output.is_relative_to(ROOT/".research-output") and output != ROOT/".research-output", "New ignored output required")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(), "Commit component source/config before execution")
    output.mkdir(parents=True, exist_ok=False)
    try:
        with training_guard():
            if args.mode == "replay":
                with compute_guard(): process(args.mode, args.source, args.experiment, args.original, output, args.recorded)
            else: process(args.mode, args.source, args.experiment, args.original, output)
    except Exception as error:
        write_json(output/"failure.json", {"status": "FAIL", "error": str(error), "preservation": "NOT_VERIFIED; separate original-file audit required"})
        raise


if __name__ == "__main__": main()
