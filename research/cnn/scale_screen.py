"""Fixed five-scale, stored-component screening; no model, load, forward or training."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
from data import require, sha256
from component_audit import prepare as component_prepare, validate_record, common_row, comparison_summary, training_guard
from spatial_audit import controlled_row, compute_guard, ERRORS
from within_kind import summarize as conditional_summary
from spatial import Budget
from train import write_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "research/experiments/residual-scale-screen-v1.json"
SCALES = (0, 0.25, 0.5, 0.75, 1)


def scaled_logits(common, residual, scale):
    require(type(scale) in (int, float) and scale in SCALES, "Fixed five scales only")
    require(common.shape == residual.shape and common.ndim == 1 and len(common) > 0 and
            all(t.dtype == torch.float32 and t.device.type == "cpu" and torch.isfinite(t).all().item()
                for t in (common, residual)), "Finite CPU float32 component vectors")
    # Separate FP32 operations: do not fuse/reorder or center the stored components.
    multiplied = residual * torch.tensor(scale, dtype=torch.float32)
    values = common + multiplied
    require(torch.isfinite(values).all().item(), "Nonfinite scaled score; STOP")
    return values


def screen_row(example, full, values, endpoint=None):
    row = common_row(example, full, values)
    if endpoint is not None:
        require(all(row[k] == endpoint[k] for k in row if k != "fullScore") and
                all(v == endpoint["fullScore"][k] for k, v in row["fullScore"].items()), "Endpoint score/row exact; STOP")
        return endpoint
    return row


def full_aggregate(rows):
    mean = lambda values: sum(values)/len(values) if values else None
    targets = [r["fullScore"]["targets"]["A512"] for r in rows]
    return {"examples": len(rows), "correct": sum(r["globalCorrect"] for r in rows),
            "top5": sum(t["top5"] for t in targets), "meanRank": mean([t["rank"] for t in targets]),
            "meanNll": mean([t["nll"] for t in targets]),
            "errors": {k: sum(r["error"] == k for r in rows) for k in ERRORS}}


def full_summary(rows):
    selectors = {"teacherKind": lambda r: r["teacherKind"], "error": lambda r: r["error"],
                 "family": lambda r: r["family"], "searchComplete": lambda r: str(r["searchComplete"]).lower(),
                 "teacherFinalPhase": lambda r: r["teacherFinalPhase"], "legalKinds": lambda r: r["legalKinds"]}
    return {"overall": full_aggregate(rows), "groups": {name: {k: full_aggregate([r for r in rows if select(r) == k])
            for k in sorted({select(r) for r in rows})} for name, select in selectors.items()}}


def compact_comparison(rows, left, right):
    value = comparison_summary(rows, left, right)
    return {k: value[k] for k in ("leftModel", "rightModel", "overall")} | {"teacherKind": value["groups"]["teacherKind"]}


def prepare(source, experiment, original, recorded):
    config = json.loads(CONFIG.read_bytes())
    require(config["approval"] == "USER_APPROVED" and config["status"] == "FIXED_STORED_SCALE_SCREEN" and
            config["scales"] == list(SCALES) and config["source"]["examples"] == {"train": 339, "dev": 89} and
            config["source"]["candidates"] == 149597 and config["limits"] == {"modelCreationCheckpointLoadForwardTraining": 0,
                "teacherDomainEpisodeTestRolloutGeometry": 0, "seconds": 600, "workingSetBytes": 2147483648}, "Fixed approved scale contract")
    _, examples, prior, predicted, logits, paths, hashes, sources, _, families = component_prepare(source, experiment, original)
    recorded = Path(recorded)
    for name, expected in config["source"]["primaryHashes"].items():
        paths["primary/"+name] = recorded/name
        hashes["primary/"+name] = expected
        require(sha256(paths["primary/"+name].read_bytes()) == expected, "Original component hash: "+name)
    report, summary, raw = [json.loads((recorded/p).read_bytes()) for p in ("report.json", "summary.json", "raw.json")]
    original_inputs = {p: h for p, h in hashes.items() if not p.startswith("primary/")}
    require(report["status"] == "PASS" and report["mode"] == "infer" and
            report["identity"] == {"sourceHashes": sources, "inputHashes": original_inputs} and
            report["accounting"] == {"topLevelForwards": 856, "storedComponentRecords": 428, "trainingTeacherDomainTestRolloutGeometry": 0} and
            report["rawSha256"] == hashes["primary/raw.json"] and report["componentsSha256"] == hashes["primary/components.jsonl"] and
            all(summary[k] == report[k] for k in summary) and report["originalGate"] == prior["gate"], "Original component execution identity")
    ledger = [json.loads(line) for line in (recorded/"forward-count.jsonl").read_text(encoding="utf-8").splitlines()]
    require(ledger == [{"call": n, "limit": 856} for n in range(1, 857)], "Original 856 ledger; not new calls")
    for p in ("research/cnn/scale_screen.py", "research/experiments/residual-scale-screen-v1.json"):
        sources[p] = sha256((ROOT/p).read_bytes())
    return config, examples, prior, predicted, logits, paths, hashes, sources, families, raw, summary


def run(source, experiment, original, recorded, output):
    start = time.perf_counter()
    torch.set_num_threads(2)
    with training_guard(), compute_guard() as counter:
        config, examples, prior, predicted, old_logits, paths, inputs, sources, families, primary_raw, primary = prepare(source, experiment, original, recorded)
        identity = {"sourceHashes": sources, "inputHashes": inputs}
        write_json(output/"started.json", {"atUtc": datetime.now(timezone.utc).isoformat(), "identity": identity})
        budget = Budget(600, 2147483648, output, start)
        budget.sample("loaded")
        records = iter(json.loads(line) for line in (Path(recorded)/"components.jsonl").read_text(encoding="utf-8").splitlines())
        raw, summary, total_candidates, evaluations = {}, {}, 0, 0
        for split, count in (("train", 339), ("dev", 89)):
            controls = {m: [] for m in ("baseline", "common", "full")}
            scaled = {str(s): [] for s in SCALES}
            require(all(len(primary_raw[split]["models"][m]) == count for m in controls), "Original component row counts")
            for i, e in enumerate(examples[split]):
                common, residual, full = validate_record(next(records), e, split, old_logits["spatial"][split][i])
                baseline = controlled_row(e, predicted["baseline"][split][i], old_logits["baseline"][split][i], families[e["fixtureId"]])
                original_full = controlled_row(e, predicted["spatial"][split][i], old_logits["spatial"][split][i], families[e["fixtureId"]])
                original_common = common_row(e, original_full, common)
                for m, row in (("baseline", baseline), ("common", original_common), ("full", original_full)):
                    require(row == primary_raw[split]["models"][m][i], "Original component scored row exact")
                    controls[m].append(row)
                for scale in SCALES:
                    values = scaled_logits(common, residual, scale)
                    endpoint = original_common if scale == 0 else original_full if scale == 1 else None
                    if endpoint is not None:
                        require(torch.equal(values, common if scale == 0 else full), "Endpoint logits exact")
                    scaled[str(scale)].append(screen_row(e, original_full, values, endpoint))
                    evaluations += 1
                total_candidates += len(common)
                budget.sample("observation")
            summary[split] = {"baseline": {"conditional": conditional_summary(controls["baseline"]), "fullList": full_summary(controls["baseline"])}, "scales": {}}
            for scale in SCALES:
                key = str(scale)
                rows = {**controls, "scale:"+key: scaled[key]}
                summary[split]["scales"][key] = {"conditional": conditional_summary(scaled[key]), "fullList": full_summary(scaled[key]),
                    "comparisons": {"vs"+m.title(): compact_comparison(rows, m, "scale:"+key) for m in controls}}
            require(summary[split]["baseline"]["conditional"] == primary["summary"][split]["models"]["baseline"], "Original baseline aggregate")
            for key, model, name in (("0", "common", "baselineToCommon"), ("1", "full", "baselineToFull")):
                actual = summary[split]["scales"][key]
                require(actual["conditional"] == primary["summary"][split]["models"][model] and
                        actual["comparisons"]["vsBaseline"]["overall"] == primary["summary"][split]["comparisons"][name]["overall"] and
                        actual["comparisons"]["vsBaseline"]["teacherKind"] == primary["summary"][split]["comparisons"][name]["groups"]["teacherKind"], "Endpoint paired/conditional aggregate exact")
            require(comparison_summary(controls,"common","full") == primary["summary"][split]["comparisons"]["commonToFull"], "Original common-to-full aggregate")
            raw[split] = {"controls": controls, "scales": scaled}
            print(json.dumps({"split": split, "correct": {k: v["fullList"]["overall"]["correct"] for k,v in summary[split]["scales"].items()}}), flush=True)
        require(next(records, None) is None and total_candidates == 149597 and evaluations == 2140, "Fixed record/candidate/evaluation counts")
        for p, h in inputs.items(): require(sha256(paths[p].read_bytes()) == h, "Input changed after screen")
        for p, h in sources.items(): require(sha256((ROOT/p).read_bytes()) == h, "Source changed after screen")
        result = {"schemaVersion": 1, "id": config["id"], "status": "PASS", "method": config["method"], "identity": identity,
                  "summary": summary, "originalGate": prior["gate"], "scales": list(SCALES),
                  "accounting": {"observations": 428, "candidates": total_candidates, "scaleEvaluations": evaluations,
                    "candidateScaleScores": total_candidates*5, "modelCreationCheckpointLoadForwardTraining": 0,
                    "teacherDomainEpisodeTestRolloutGeometry": 0, **counter},
                  "limits": ["Known train/dev post-hoc screening; no held-out or causal evidence", "No selected scale or new gate; original gate FAIL retained",
                             "Jointly trained common-only is not independent baseline; teacher-kind oracle is not actual policy", "Original model/label/feature/loss/app unchanged"]}
        write_json(output/"raw.json", raw)
        write_json(output/"summary.json", result)
        write_json(output/"report.json", {**result, "atUtc": datetime.now(timezone.utc).isoformat(),
            "rawSha256": sha256((output/"raw.json").read_bytes()), "summarySha256": sha256((output/"summary.json").read_bytes()),
            "budget": budget.report(), "environment": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda, "threads": torch.get_num_threads()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "experiment", "original", "recorded", "output"): parser.add_argument("--"+name, required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    require(output.is_relative_to(ROOT/".research-output") and output != ROOT/".research-output", "New ignored output required")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(), "Commit screen source/config before execution")
    output.mkdir(parents=True, exist_ok=False)
    try: run(args.source, args.experiment, args.original, args.recorded, output)
    except Exception as error:
        write_json(output/"failure.json", {"status": "FAIL", "error": str(error), "preservation": "NOT_VERIFIED; separate original-file audit required"})
        raise


if __name__ == "__main__": main()
