"""Stored logits/score audit only: no model creation, checkpoint load or forward."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import torch
from data import require, integer, sha256
from diagnose import score, action_description
from within_kind import conditional_score, summarize as conditional_summary
from train import write_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "research/experiments/stored-spatial-audit-v1.json"
KINDS = ("place-piece", "single-cell", "reroll")
ERRORS = ("exact-label", "wrong-kind", "different-target", "different-shape", "different-position")


@contextmanager
def compute_guard():
    counter = {"forbiddenAttempts": 0}
    def forbidden(*args, **kwargs):
        counter["forbiddenAttempts"] += 1
        raise ValueError("Model creation/call, checkpoint load or optimizer forbidden; STOP")
    targets = [(torch.nn.Module, "__init__"), (torch.nn.Module, "__call__"),
               (torch, "load"), (torch.optim.Optimizer, "__init__")]
    saved = [(obj, name, getattr(obj, name)) for obj, name in targets]
    try:
        for obj, name in targets: setattr(obj, name, forbidden)
        yield counter
        require(counter["forbiddenAttempts"] == 0, "Forbidden compute attempted")
    finally:
        for obj, name, value in saved: setattr(obj, name, value)


def decode(key):
    a = json.loads(key)
    require(isinstance(a, list) and a and a[0] in KINDS and len(a) == (7 if a[0] == "place-piece" else 3),
            "Stored action syntax")
    if a[0] == "single-cell":
        require(integer(a[1], 0, 15) and integer(a[2], 0, 9), "Stored point syntax")
    else:
        require(isinstance(a[1], str) and bool(a[1]) and integer(a[2], 0, 2), "Stored target syntax")
        if a[0] == "place-piece":
            shape = a[6]
            require(isinstance(a[3], str) and bool(a[3]) and integer(a[4], 0, 15) and integer(a[5], 0, 9) and
                    isinstance(shape, list) and 1 <= len(shape) <= 5 and isinstance(shape[0], list) and
                    1 <= len(shape[0]) <= 5 and all(isinstance(r, list) and len(r) == len(shape[0]) and
                    all(type(v) is bool for v in r) for r in shape), "Stored shape syntax")
    return a


def error_kind(target_key, selected_key):
    # Stored JSON identity comparisons match the existing TS errorKind priority.
    target, selected = decode(target_key), decode(selected_key)
    if target == selected: return "exact-label"
    if target[0] != selected[0]: return "wrong-kind"
    if target[0] == "single-cell": return "different-position"
    if target[1:3] != selected[1:3]: return "different-target"
    if target[0] == "place-piece":
        return "different-shape" if target[3] != selected[3] or target[6] != selected[6] else "different-position"
    raise ValueError("Stored reroll identity mismatch")


def controlled_row(e, predicted, recorded, family):
    keys, labels = e["candidateKeys"], e["labels"]
    require(1 <= len(keys) <= 4096 and len(set(keys)) == len(keys) and len(e["candidateFeatures"]) == len(keys) and
            recorded["id"] == predicted["id"] == e["id"] and set(recorded) == {"id", "logits"}, "Stored observation/id/order")
    require(all(integer(labels[n], 0, len(keys)-1) for n in ("A512", "A1024")), "Stored label")
    actions = [decode(k) for k in keys]
    values = recorded["logits"]
    require(isinstance(values, list) and len(values) == len(keys) and
            all(type(v) in (int, float) for v in values), "Stored full-list logits")
    logits = torch.tensor(values, dtype=torch.float32)
    require(torch.isfinite(logits).all().item() and logits.tolist() == values, "Exact finite float32 storage")
    rebuilt = score(logits, labels)
    rebuilt.update(id=e["id"], fixtureId=e["fixtureId"], family=family,
        sameLabel=labels["A512"] == labels["A1024"],
        complete={n: e["teachers"][n]["result"]["search"]["searchComplete"] for n in ("A512", "A1024")},
        teacherActions={n: action_description(keys[labels[n]]) for n in ("A512", "A1024")},
        selected=action_description(keys[rebuilt["index"]]), firstLegalAction=action_description(keys[0]))
    require(rebuilt == predicted, "Original prediction score/description/metadata drift; STOP")
    conditional = conditional_score(logits, keys, labels["A512"])
    error = error_kind(keys[labels["A512"]], keys[rebuilt["index"]])
    require((error == "exact-label") == rebuilt["targets"]["A512"]["correct"], "Error correctness agreement")
    return {"id": e["id"], "fixtureId": e["fixtureId"], "family": family, "error": error,
            "teacherKind": conditional["teacherKind"], "selectedKind": actions[rebuilt["index"]][0],
            "selectedIndex": rebuilt["index"], "globalCorrect": error == "exact-label", "conditional": conditional,
            "searchComplete": e["teachers"]["A512"]["result"]["search"]["searchComplete"],
            "teacherFinalPhase": e["teachers"]["A512"]["result"]["evaluation"]["phase"],
            "legalKinds": "+".join(k for k in KINDS if any(a[0] == k for a in actions)), "fullScore": rebuilt}


def pair_rows(left, right):
    require(len(left) == len(right), "Paired count")
    rows = []
    for a, b in zip(left, right):
        require(all(a[k] == b[k] for k in ("id", "fixtureId", "family", "teacherKind", "searchComplete", "teacherFinalPhase", "legalKinds")),
                "Paired observation metadata/order")
        rows.append({**{k: a[k] for k in ("id", "fixtureId", "family", "teacherKind", "searchComplete", "teacherFinalPhase", "legalKinds")},
                     "baseline": a, "spatial": b})
    return rows


def paired_table(rows, oracle=False):
    value = lambda r, m: r[m]["conditional"]["oracleCorrect"] if oracle else r[m]["globalCorrect"]
    table = {"bothCorrect": 0, "improvement": 0, "regression": 0, "bothWrong": 0}
    for r in rows:
        a, b = value(r, "baseline"), value(r, "spatial")
        table["bothCorrect" if a and b else "regression" if a else "improvement" if b else "bothWrong"] += 1
    require(sum(table.values()) == len(rows), "Paired partition")
    return {"examples": len(rows), **table, "baselineCorrect": table["bothCorrect"] + table["regression"],
            "spatialCorrect": table["bothCorrect"] + table["improvement"], "netCorrectGain": table["improvement"] - table["regression"]}


def aggregate_pairs(rows):
    table = paired_table(rows)
    confusion = {m: {kind: {selected: sum(r["teacherKind"] == kind and r[m]["selectedKind"] == selected for r in rows)
                          for selected in KINDS} for kind in KINDS} for m in ("baseline", "spatial")}
    errors = {m: {error: sum(r[m]["error"] == error for r in rows) for error in ERRORS} for m in ("baseline", "spatial")}
    transition = {a: {b: sum(r["baseline"]["error"] == a and r["spatial"]["error"] == b for r in rows) for b in ERRORS} for a in ERRORS}
    same = sum(r["baseline"]["selectedIndex"] == r["spatial"]["selectedIndex"] for r in rows)
    return {"actualHardLabel": table, "teacherKindOracle": paired_table(rows, True), "sameSelectedIndex": same,
            "changedSelectedIndex": len(rows)-same, "changedButBothWrong": sum(not r["baseline"]["globalCorrect"] and
                 not r["spatial"]["globalCorrect"] and r["baseline"]["selectedIndex"] != r["spatial"]["selectedIndex"] for r in rows),
            "errors": errors, "errorTransition": transition, "kindConfusion": confusion}


def summarize_pairs(rows):
    selectors = {"teacherKind": lambda r: r["teacherKind"], "family": lambda r: r["family"],
                 "searchComplete": lambda r: str(r["searchComplete"]).lower(), "teacherFinalPhase": lambda r: r["teacherFinalPhase"],
                 "legalKinds": lambda r: r["legalKinds"]}
    return {"overall": aggregate_pairs(rows), "groups": {name: {key: aggregate_pairs([r for r in rows if select(r) == key])
            for key in sorted(set(select(r) for r in rows))} for name, select in selectors.items()}}


def prepare(source, experiment):
    config = json.loads(CONFIG.read_bytes())
    require(config["status"] == "FROZEN_READONLY_DIAGNOSTIC" and config["approval"] == "USER_APPROVED" and
            config["boundaries"]["forwardTrainingCheckpointLoad"] == config["boundaries"]["footprintTeacherDomainTestRollout"] == 0,
            "Readonly approved config")
    source, experiment = Path(source), Path(experiment)
    paths = {"dataset": source / "train-dev/dataset.json", "pairedFreeze": source / "freeze.json",
             **{p: experiment / p for p in config["source"]["experimentHashes"]}}
    expected = {"dataset": config["source"]["datasetSha256"], "pairedFreeze": config["source"]["pairedFreezeSha256"],
                **config["source"]["experimentHashes"]}
    require(all(sha256(paths[p].read_bytes()) == h for p, h in expected.items()), "Original stored input hash")
    execution, report, freeze = [json.loads((experiment / p).read_bytes()) for p in ("execution.json", "report.json", "freeze.json")]
    require(report["status"] == "PASS" and freeze["status"] == "FROZEN_DEV_ONLY" and
            report["executionSha256"] == expected["execution.json"] == freeze["executionSha256"] and
            freeze["reportSha256"] == expected["report.json"], "Original execution/report/freeze")
    for p, h in execution["sourceHashes"].items(): require(sha256((ROOT / p).read_bytes()) == h, "Original source changed: " + p)
    pack = json.loads(paths["dataset"].read_bytes())
    require(pack["status"] == "PASS" and pack["stage"] == "train-dev" and not pack["examples"]["test"], "Train/dev only")
    raw = {m: json.loads((experiment / m / "predictions.json").read_bytes()) for m in ("baseline", "spatial")}
    logits = {m: json.loads((experiment / m / "logits.json").read_bytes()) for m in ("baseline", "spatial")}
    for m in ("baseline", "spatial"):
        original = report["models"][m]
        require(original["selectedEpoch"] == config["source"]["selectedEpochs"][m] and
                original["checkpointSha256"] == freeze["checkpointSha256"][m] == expected[m+"/checkpoint.pt"] and
                original["predictionsSha256"] == expected[m+"/predictions.json"] and
                original["logitsSha256"] == expected[m+"/logits.json"] and
                set(raw[m]) == set(logits[m]) == {"train", "dev"}, "Frozen selected model files")
    return config, pack, report, raw, logits, paths, expected


def run(source, experiment, output):
    torch.set_num_threads(2)
    with compute_guard() as counter:
        config, pack, original, predictions, recorded, paths, inputs = prepare(source, experiment)
        own_sources = {p: sha256((ROOT / p).read_bytes()) for p in
                       ("research/cnn/spatial_audit.py", "research/experiments/stored-spatial-audit-v1.json")}
        write_json(output / "started.json", {"atUtc": datetime.now(timezone.utc).isoformat(), "sourceHashes": own_sources, "inputHashes": inputs})
        families = {e["fixture"]["id"]: next(f for f in ("sparse", "pressure") if e["fixture"]["id"] == f+"-"+str(e["fixture"]["seed"])) for e in pack["episodes"]}
        raw, summary = {"models": {m: {} for m in ("baseline", "spatial")}, "paired": {}}, {"models": {m: {} for m in ("baseline", "spatial")}, "paired": {}}
        for split, count in (("train", 339), ("dev", 89)):
            examples = pack["examples"][split]
            require(count == config["source"]["examples"][split] == len(examples) and len({e["id"] for e in examples}) == count, "Fixed observations")
            for m in ("baseline", "spatial"):
                require(len(predictions[m][split]) == len(recorded[m][split]) == count, "Stored row counts")
                rows = [controlled_row(e, p, l, families[e["fixtureId"]]) for e, p, l in zip(examples, predictions[m][split], recorded[m][split])]
                raw["models"][m][split], summary["models"][m][split] = rows, conditional_summary(rows)
                require(sum(r["globalCorrect"] for r in rows) == original["models"][m]["summary"][split]["targets"]["A512"]["overall"]["correct"], "Original hard accuracy")
            pairs = pair_rows(raw["models"]["baseline"][split], raw["models"]["spatial"][split])
            raw["paired"][split], summary["paired"][split] = pairs, summarize_pairs(pairs)
            print(json.dumps({"split": split, "actual": summary["paired"][split]["overall"]["actualHardLabel"]}), flush=True)
        require(counter["forbiddenAttempts"] == 0, "No forbidden compute")
        for p, h in inputs.items(): require(sha256(paths[p].read_bytes()) == h, "Input changed after audit")
        for p, h in own_sources.items(): require(sha256((ROOT / p).read_bytes()) == h, "Audit source changed")
        result = {"schemaVersion": 1, "id": config["id"], "status": "PASS", "method": config["method"],
                  "identity": {"sourceHashes": own_sources, "inputHashes": inputs}, "summary": summary,
                  "originalGate": original["gate"], "accounting": {"storedObservations": 428, "storedModelRows": 856,
                    "modelCreationCheckpointLoadForwardTraining": 0, "footprintTeacherDomainTestRollout": 0, **counter},
                  "limits": ["Previously examined train/dev; no held-out or policy claim", "Teacher-kind oracle is diagnostic only",
                             "Stored total logits cannot isolate jointly trained common/residual causal effects", "No label/feature/loss/model/gate changes"]}
        write_json(output / "raw.json", raw)
        write_json(output / "summary.json", result)
        write_json(output / "report.json", {**result, "atUtc": datetime.now(timezone.utc).isoformat(),
            "rawSha256": sha256((output / "raw.json").read_bytes()), "summarySha256": sha256((output / "summary.json").read_bytes()),
            "environment": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "threads": torch.get_num_threads()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "experiment", "output"): parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    require(output.is_relative_to(ROOT / ".research-output") and output != ROOT / ".research-output", "New ignored output required")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(), "Commit audit source/config before execution")
    output.mkdir(parents=True, exist_ok=False)
    try: run(args.source, args.experiment, output)
    except Exception as error:
        write_json(output / "failure.json", {"status": "FAIL", "error": str(error), "preservation": "NOT_VERIFIED; separate original-file audit required"})
        raise


if __name__ == "__main__": main()
