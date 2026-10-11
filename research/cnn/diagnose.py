"""Read-only train/dev diagnostics; no teacher calls, training or test access."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import sys
import torch
from data import require, sha256, integer
from model import tensors
from paired import NAMES, load_pack, load_frozen
from train import write_json


def score(logits, labels):
    require(logits.ndim == 1 and len(logits) > 0 and torch.isfinite(logits).all().item(),
            "Finite candidate logits required")
    values = logits.detach().tolist()
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    ranks = {i: r + 1 for r, i in enumerate(order)}
    log_probs = logits.log_softmax(dim=0)
    targets = {}
    for name in NAMES:
        label = labels[name]
        require(integer(label, 0, len(values) - 1), "Legal diagnostic label required")
        targets[name] = {"correct": order[0] == label, "top5": ranks[label] <= 5,
                         "rank": ranks[label], "nll": -float(log_probs[label]),
                         "firstLegal": label == 0}
    return {"index": order[0], "candidates": len(values), "targets": targets,
            "margin": values[order[0]] - values[order[1]] if len(values) > 1 else None,
            "probability": float(log_probs[order[0]].exp()), "uniformNll": math.log(len(values))}


def action_description(key):
    action = json.loads(key)
    require(isinstance(action, list) and action and action[0] in
            ("place-piece", "single-cell", "reroll"), "Diagnostic action key")
    kind = action[0]
    require(len(action) == {"place-piece": 7, "single-cell": 3, "reroll": 3}[kind],
            "Diagnostic action key length")
    result = {"kind": kind}
    if kind == "place-piece":
        result.update(piece=action[3], row=action[4], col=action[5])
    elif kind == "single-cell":
        result.update(row=action[1], col=action[2])
    return result


def candidate_band(n):
    return "1-32" if n <= 32 else "33-128" if n <= 128 else "129-512" if n <= 512 else "513-4096"


def fixture_family(fixture, allowed):
    # Current TS createSyntheticFixture identifies family as `${family}-${seed}`.
    matches = [family for family in allowed if fixture["id"] == f'{family}-{fixture["seed"]}']
    require(len(matches) == 1, "Known synthetic fixture family identity")
    return matches[0]


def aggregate(rows, target):
    count = len(rows)
    values = [r["targets"][target] for r in rows]
    fixtures = {}
    for row, value in zip(rows, values):
        fixtures.setdefault(row["fixtureId"], []).append(int(value["correct"]))
    mean = lambda xs: sum(xs) / len(xs) if xs else None
    return {"examples": count, "correct": sum(v["correct"] for v in values),
            "top1": mean([int(v["correct"]) for v in values]),
            "top5": mean([int(v["top5"]) for v in values]),
            "firstLegalTop1": mean([int(v["firstLegal"]) for v in values]),
            "meanTeacherRank": mean([v["rank"] for v in values]),
            "meanNll": mean([v["nll"] for v in values]),
            "uniformMeanNll": mean([r["uniformNll"] for r in rows]),
            "meanMaxProbability": mean([r["probability"] for r in rows]),
            "meanTopMargin": mean([r["margin"] for r in rows if r["margin"] is not None]),
            "fixtures": len(fixtures),
            "fixtureMacroTop1": mean([mean(xs) for xs in fixtures.values()])}


def selection_counts(descriptions):
    def histogram(key):
        return dict(sorted(Counter(str(d[key]) for d in descriptions if key in d).items()))
    return {"examples": len(descriptions), "kind": histogram("kind"), "piece": histogram("piece"),
            "row": histogram("row"), "col": histogram("col"),
            "row0Col0": sum(d.get("row") == 0 and d.get("col") == 0 for d in descriptions)}


def summarize(rows):
    targets = {}
    for target in NAMES:
        groups = {}
        selectors = {"family": lambda r: r["family"], "candidates": lambda r: candidate_band(r["candidates"]),
                     "labels": lambda r: "same" if r["sameLabel"] else "different",
                     "teacher": lambda r: "complete" if r["complete"][target] else "incomplete",
                     "teacherAction": lambda r: r["teacherActions"][target]["kind"],
                     "teacherPiece": lambda r: r["teacherActions"][target].get("piece", "ability")}
        for axis, select in selectors.items():
            grouped = {}
            for row in rows:
                grouped.setdefault(select(row), []).append(row)
            groups[axis] = {key: aggregate(xs, target) for key, xs in sorted(grouped.items())}
        targets[target] = {"overall": aggregate(rows, target), "groups": groups,
                           "teacherSelection": selection_counts([r["teacherActions"][target] for r in rows])}
    return {"targets": targets, "modelSelection": selection_counts([r["selected"] for r in rows]),
            "firstLegalSelection": selection_counts([r["firstLegalAction"] for r in rows]),
            "predictionIsFirstLegal": sum(r["index"] == 0 for r in rows),
            "sameLabel": sum(r["sameLabel"] for r in rows),
            "differentLabel": sum(not r["sameLabel"] for r in rows)}


def inspect_split(model, examples, families):
    rows = []
    for e in examples:
        # Audit metadata, labels, candidate keys and raw state never enter forward.
        inputs = tensors(e)
        with torch.inference_mode():
            logits = model(*inputs)
            repeat = model(*inputs)
        require(torch.equal(logits, repeat), "Nondeterministic diagnostic logits")
        row = score(logits, e["labels"])
        row.update(id=e["id"], fixtureId=e["fixtureId"], family=families[e["fixtureId"]],
                   sameLabel=e["labels"]["A512"] == e["labels"]["A1024"],
                   complete={name: e["teachers"][name]["result"]["search"]["searchComplete"] for name in NAMES},
                   selected=action_description(e["candidateKeys"][row["index"]]),
                   firstLegalAction=action_description(e["candidateKeys"][0]),
                   teacherActions={name: action_description(e["candidateKeys"][e["labels"][name]]) for name in NAMES})
        rows.append(row)
    return rows


def diagnose(dataset, freeze_path, output):
    pack, protocol, manifest = load_pack(dataset, "train-dev")
    models, frozen_protocol, freeze = load_frozen(freeze_path)
    require(protocol == frozen_protocol and manifest["dataSha256"] == freeze["trainDevDataSha256"],
            "Frozen train/dev dataset identity")
    families = {e["fixture"]["id"]: fixture_family(e["fixture"], protocol["dataset"]["families"])
                for e in pack["episodes"]}
    raw, summary, agreement = {}, {}, {}
    for split in ("train", "dev"):
        raw[split] = {name: inspect_split(models[name], pack["examples"][split], families) for name in NAMES}
        summary[split] = {name: summarize(raw[split][name]) for name in NAMES}
        a, b = raw[split]["A512"], raw[split]["A1024"]
        require([r["id"] for r in a] == [r["id"] for r in b], "Shared diagnostic inputs")
        agreement[split] = {"examples": len(a), "samePrediction": sum(x["index"] == y["index"] for x, y in zip(a, b))}
    parameters = [torch.cat([p.detach().reshape(-1) for p in models[name].parameters()]) for name in NAMES]
    differences = {"parameters": len(parameters[0]), "exactlyEqual": torch.equal(*parameters),
                   "l2Distance": float(torch.linalg.vector_norm(parameters[0] - parameters[1]))}
    identity = {"freezeSha256": sha256(Path(freeze_path).read_bytes()), "dataSha256": manifest["dataSha256"],
                "protocolSha256": manifest["protocolSha256"], "models": freeze["models"],
                "diagnosticSourceSha256": sha256(Path(__file__).read_bytes())}
    report = {"status": "PASS", "at": datetime.now(timezone.utc).isoformat(), "identity": identity,
              "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform()},
              "summary": summary, "modelAgreement": agreement, "weightDifference": differences,
              "boundaries": {"testLoaded": False, "testScored": False, "teacherCalls": 0, "trainingRuns": 0,
                             "selectionChanged": False, "repeatLogitsExact": True, "sourceIdentityRebinding": False}}
    write_json(output / "raw.json", raw)
    write_json(output / "report.json", report)
    print(json.dumps({"status": "PASS", "modelAgreement": agreement, "weightDifference": differences}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        diagnose(args.dataset, args.freeze, output)
    except Exception as error:
        write_json(output / "failure.json", {"status": "FAIL", "errorType": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
