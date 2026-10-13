"""Frozen full-list inference; teacher-kind oracle statistics only after forward."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import torch
from data import require, integer, sha256
from model import CandidateCNN, configure, tensors
from paired import own_examples, weights_hash
from outcome import OutcomeCandidateCNN, load_inputs
from diagnose import score, action_description
from train import write_json, rss_bytes

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "research/experiments/within-kind-audit-v1.json"
KINDS = ("place-piece", "single-cell", "reroll")


def conditional_score(logits, keys, label):
    require(logits.ndim == 1 and logits.dtype == torch.float32 and logits.device.type == "cpu" and
            1 <= len(logits) <= 4096 and torch.isfinite(logits).all().item(), "Finite CPU float32 logits")
    require(len(keys) == len(logits) and len(set(keys)) == len(keys) and integer(label, 0, len(keys)-1),
            "Conditional candidate/label schema")
    kinds = [action_description(key)["kind"] for key in keys]
    teacher_kind = kinds[label]
    indices = [i for i, kind in enumerate(kinds) if kind == teacher_kind]
    values = logits.detach().tolist()
    ordered = sorted(indices, key=lambda i: (-values[i], i))
    rank = ordered.index(label) + 1
    log_probs = logits[indices].log_softmax(dim=0)
    nll = -float(log_probs[indices.index(label)])
    require(math.isfinite(nll), "Finite conditional NLL")
    centered = logits.double()-logits.double().max()
    denominator = torch.logsumexp(centered, dim=0)
    mass = {}
    for kind in KINDS:
        selected = [i for i, k in enumerate(kinds) if k == kind]
        probability = float((torch.logsumexp(centered[selected], dim=0)-denominator).exp()) if selected else 0.0
        mass[kind] = {"candidates": len(selected), "uniformMass": len(selected)/len(keys), "probabilityMass": probability}
    require(abs(sum(m["probabilityMass"] for m in mass.values())-1) <= 1e-9, "Kind mass normalization")
    return {"teacherKind": teacher_kind, "oracleIndex": ordered[0], "oracleCorrect": ordered[0] == label,
            "oracleTop5": rank <= 5, "oracleRank": rank, "oracleNll": nll,
            "teacherKindCandidates": len(indices), "kindMass": mass}


def aggregate(rows):
    count = len(rows)
    mean = lambda xs: sum(xs)/len(xs) if xs else None
    fixtures = {}
    for row in rows:
        fixtures.setdefault(row["fixtureId"], []).append(int(row["conditional"]["oracleCorrect"]))
    return {"examples": count, "fixtures": len(fixtures),
            "globalHardCorrect": sum(r["globalCorrect"] for r in rows),
            "teacherKindOracleCorrect": sum(r["conditional"]["oracleCorrect"] for r in rows),
            "teacherKindOracleTop1": mean([int(r["conditional"]["oracleCorrect"]) for r in rows]),
            "teacherKindOracleTop5": mean([int(r["conditional"]["oracleTop5"]) for r in rows]),
            "meanOracleRank": mean([r["conditional"]["oracleRank"] for r in rows]),
            "meanOracleNll": mean([r["conditional"]["oracleNll"] for r in rows]),
            "fixtureMacroOracleTop1": mean([mean(xs) for xs in fixtures.values()]),
            "oracleRecoveredHardErrors": sum(not r["globalCorrect"] and r["conditional"]["oracleCorrect"] for r in rows),
            "wrongKindExamples": sum(r["error"] == "wrong-kind" for r in rows),
            "wrongKindOracleRecoveries": sum(r["error"] == "wrong-kind" and r["conditional"]["oracleCorrect"] for r in rows),
            "kindMass": {kind: {"presentExamples": sum(r["conditional"]["kindMass"][kind]["candidates"] > 0 for r in rows),
                                "meanCandidates": mean([r["conditional"]["kindMass"][kind]["candidates"] for r in rows]),
                                "meanProbabilityMass": mean([r["conditional"]["kindMass"][kind]["probabilityMass"] for r in rows]),
                                "meanUniformMass": mean([r["conditional"]["kindMass"][kind]["uniformMass"] for r in rows])}
                         for kind in KINDS}}


def summarize(rows):
    selectors = {"teacherKind": lambda r: r["conditional"]["teacherKind"], "error": lambda r: r["error"],
                 "family": lambda r: r["family"], "searchComplete": lambda r: str(r["searchComplete"]).lower(),
                 "teacherFinalPhase": lambda r: r["teacherFinalPhase"], "legalKinds": lambda r: r["legalKinds"]}
    groups = {}
    for name, select in selectors.items():
        buckets = {}
        for row in rows: buckets.setdefault(select(row), []).append(row)
        groups[name] = {key: aggregate(xs) for key, xs in sorted(buckets.items())}
    return {"overall": aggregate(rows), "groups": groups}


def controlled_row(logits, example, expected, audit):
    actual = score(logits, example["labels"])
    require(all(actual[key] == expected[key] for key in actual), "Original frozen score drift; STOP")
    require(example["id"] == expected["id"] == audit["id"] and example["fixtureId"] == audit["fixtureId"] and
            actual["targets"]["A512"]["correct"] == audit["correct"], "Frozen observation join")
    conditional = conditional_score(logits, example["candidateKeys"], example["labels"]["A512"])
    require(conditional["teacherKind"] == audit["teacherKind"], "Frozen teacher kind")
    return {"id": example["id"], "fixtureId": example["fixtureId"], "family": audit["family"],
            "error": audit["error"], "searchComplete": audit["searchComplete"],
            "teacherFinalPhase": audit["teacherFinalPhase"], "legalKinds": audit["legalKinds"],
            "globalCorrect": audit["correct"], "conditional": conditional}


def bounded_forward(model, inputs, budget):
    require(budget["calls"] < budget["limit"], "Forward budget exhausted; STOP")
    budget["calls"] += 1
    if budget.get("ledger"):
        with Path(budget["ledger"]).open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"call": budget["calls"], "limit": budget["limit"]})+"\n")
    return model(*inputs)


def prepare(source, experiment, audit_path):
    config = json.loads(CONFIG.read_bytes())
    require(config["status"] == "FROZEN_READONLY_DIAGNOSTIC" and config["limits"]["modelForwards"] == 1712,
            "Fixed readonly forward contract")
    paths = {"dataset": Path(source)/"train-dev/dataset.json", "execution": Path(experiment)/"execution.json",
             "report": Path(experiment)/"report.json", "augmented": Path(experiment)/"dataset.json",
             "storedAuditRaw": Path(audit_path)}
    for key, path in paths.items():
        require(sha256(path.read_bytes()) == config["source"][key+"Sha256"], "Fixed input hash: " + key)
    pack, protocol, _, augmented, _ = load_inputs(source, paths["augmented"], paths["execution"])
    report = json.loads(paths["report"].read_bytes())
    require(report["status"] == "PASS" and report["executionSha256"] == config["source"]["executionSha256"], "Frozen report identity")
    expected = {}
    for mode in ("baseline", "outcome"):
        paths[mode+"Predictions"] = Path(experiment)/mode/"predictions.json"
        paths[mode+"Checkpoint"] = Path(experiment)/mode/"checkpoint.pt"
        require(sha256(paths[mode+"Predictions"].read_bytes()) == config["source"]["predictionSha256"][mode], "Frozen raw prediction hash")
        require(sha256(paths[mode+"Checkpoint"].read_bytes()) == report["models"][mode]["checkpointSha256"], "Frozen checkpoint hash")
        expected[mode] = json.loads(paths[mode+"Predictions"].read_bytes())
    audit = json.loads(paths["storedAuditRaw"].read_bytes())
    for mode in ("baseline", "outcome"):
        require(set(audit[mode]) == set(expected[mode]) == {"train", "dev"}, "Train/dev-only raw")
        for split in ("train", "dev"):
            require(len(pack["examples"][split]) == len(audit[mode][split]) == len(expected[mode][split]) ==
                    config["source"]["examples"][split], "Fixed split count")
    source_paths = ["research/cnn/within_kind.py", "research/experiments/within-kind-audit-v1.json"]
    source_hashes = {p: sha256((ROOT/p).read_bytes()) for p in source_paths}
    return config, pack, protocol, augmented, report, expected, audit, paths, source_hashes


def run(mode, source, experiment, audit_path, output, recorded=None):
    start, before = time.perf_counter(), rss_bytes()
    config, pack, protocol, augmented, report, expected, audit, paths, sources = prepare(source, experiment, audit_path)
    originals = {key: sha256(path.read_bytes()) for key, path in paths.items()}
    write_json(output/"started.json", {"atUtc": datetime.now(timezone.utc).isoformat(), "mode": mode,
                                      "configSha256": sha256(CONFIG.read_bytes()), "sourceHashes": sources, "inputHashes": originals})
    configure(protocol["training"])
    torch.set_num_interop_threads(1)
    require(torch.version.cuda is None, "Existing CPU-only build required")
    raw, summary, records = {}, {}, None
    if mode == "replay":
        recorded = Path(recorded)
        previous = json.loads((recorded/"report.json").read_bytes())
        require(previous["status"] == "PASS" and previous["identity"]["sourceHashes"] == sources and
                previous["identity"]["inputHashes"] == originals and
                previous["logitsSha256"] == sha256((recorded/"logits.jsonl").read_bytes()), "Replay frozen identity")
        records = iter([json.loads(line) for line in (recorded/"logits.jsonl").read_text(encoding="utf-8").splitlines()])
    budget = {"calls": 0, "limit": config["limits"]["modelForwards"], "ledger": output/"forward-count.jsonl" if mode == "infer" else None}
    for model_name in ("baseline", "outcome"):
        raw[model_name], summary[model_name] = {}, {}
        model = None
        if mode == "infer":
            configure(protocol["training"])
            common = CandidateCNN(protocol["training"])
            model = common if model_name == "baseline" else OutcomeCandidateCNN(protocol["training"], common)
            saved = torch.load(paths[model_name+"Checkpoint"], map_location="cpu", weights_only=True)
            require(saved["epoch"] == config["source"]["selectedEpochs"][model_name], "Frozen selected epoch")
            model.load_state_dict(saved["state_dict"], strict=True)
            model.eval()
            require(weights_hash(model) == report["models"][model_name]["weightsSha256"], "Frozen logical weights")
        for split in ("train", "dev"):
            examples = own_examples(pack, split, "A512") if model_name == "baseline" else augmented[split]
            rows = []
            for e, old, audit_row in zip(examples, expected[model_name][split], audit[model_name][split]):
                if mode == "infer":
                    with torch.inference_mode():
                        inputs = tensors(e)
                        logits = bounded_forward(model, inputs, budget)
                        repeat = bounded_forward(model, inputs, budget)
                    require(torch.equal(logits, repeat), "Repeated frozen logits drift; STOP")
                    record = {"model": model_name, "split": split, "id": e["id"], "logits": logits.detach().tolist()}
                    with (output/"logits.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")
                else:
                    record = next(records)
                    require(record["model"] == model_name and record["split"] == split and record["id"] == e["id"], "Replay ordered record")
                    logits = torch.tensor(record["logits"], dtype=torch.float32)
                rows.append(controlled_row(logits, e, old, audit_row))
            raw[model_name][split] = rows
            summary[model_name][split] = summarize(rows)
            print(f"{mode} {model_name}/{split}: global={summary[model_name][split]['overall']['globalHardCorrect']}, oracle={summary[model_name][split]['overall']['teacherKindOracleCorrect']}", flush=True)
    if records is not None: require(next(records, None) is None, "Extra replay record")
    require(budget["calls"] == (1712 if mode == "infer" else 0), "Exact forward accounting")
    for key, path in paths.items(): require(sha256(path.read_bytes()) == originals[key], "Original file changed: " + key)
    for path, digest in sources.items(): require(sha256((ROOT/path).read_bytes()) == digest, "Diagnostic source changed")
    result = {"schemaVersion": 1, "id": config["id"], "status": "PASS", "method": config["method"],
              "identity": {"inputHashes": originals, "sourceHashes": sources, "configSha256": sha256(CONFIG.read_bytes())},
              "summary": summary,
              "limits": ["Teacher action kind is ground truth used after inference: oracle diagnostic only",
                         "Same previously examined train/dev; no held-out generalization or policy claim",
                         "No new labels/loss/model/selection/training or test evaluation"]}
    write_json(output/"raw.json", raw)
    write_json(output/"summary.json", result)
    logits_path = output/"logits.jsonl" if mode == "infer" else recorded/"logits.jsonl"
    write_json(output/"report.json", {**result, "mode": mode, "atUtc": datetime.now(timezone.utc).isoformat(),
                                     "rawSha256": sha256((output/"raw.json").read_bytes()), "logitsSha256": sha256(logits_path.read_bytes()),
                                     "accounting": {"modelForwards": budget["calls"], "scoredObservations": 856,
                                                    "trainingTeacherDomainTest": 0, "selectionLabelsChanged": False},
                                     "timings": {"totalMs": (time.perf_counter()-start)*1000},
                                     "memory": {"beforeBytes": before, "afterBytes": rss_bytes(), "peak": False},
                                     "environment": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                                                     "threads": torch.get_num_threads(), "interopThreads": torch.get_num_interop_threads()}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("infer", "replay"), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--audit-raw", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--recorded")
    args = parser.parse_args()
    require(bool(args.recorded) == (args.mode == "replay"), "Recorded logits required only for replay")
    output = Path(args.output).resolve()
    require(output.is_relative_to(ROOT/".research-output") and output != ROOT/".research-output", "New ignored output required")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
            "Commit diagnostic source/config before execution")
    output.mkdir(parents=True, exist_ok=False)
    try:
        run(args.mode, args.source, args.experiment, args.audit_raw, output, args.recorded)
    except Exception as error:
        write_json(output/"failure.json", {"status": "FAIL", "error": str(error),
                                          "preservation": "NOT_VERIFIED; perform separate original-file audit"})
        raise


if __name__ == "__main__": main()
