"""Fixed 200/1000 update contrast, one fresh model per sequential child; no retries."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import torch
from data import require, sha256
from controlled_point_fit import flat_loss, score_row, summarize, fit_gate, validate_dataset, counted_model
from spatial import make_model, spatial_tensors, Budget
from paired import weights_hash
from train import write_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "research/experiments/controlled-fit-duration-v1.json"
UPDATES = {"control": 200, "candidate": 1000}


def canonical_state(value):
    """Full logical state, including param groups, indices, dtype, shape and values; no object IDs."""
    if isinstance(value, torch.Tensor):
        require(value.device.type == "cpu" and torch.isfinite(value).all().item(), "Finite CPU optimizer tensor")
        return {"tensor": {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.detach().tolist()}}
    if isinstance(value, dict):
        require(all(type(k) in (str, int) for k in value), "Logical state keys")
        return {"mapping": [[type(k).__name__, k, canonical_state(value[k])]
                            for k in sorted(value, key=lambda k: (type(k).__name__, str(k)))]}
    if isinstance(value, (tuple, list)): return [canonical_state(v) for v in value]
    require(value is None or type(value) in (str, bool, int, float), "Logical state scalar")
    require(type(value) is not float or math.isfinite(value), "Finite optimizer scalar")
    return value


def optimizer_fingerprint(state):
    return sha256(json.dumps(canonical_state(state), separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf8"))


class DurationLedger:
    def __init__(self, path, arm):
        require(arm in UPDATES, "Fixed duration arm")
        self.path, self.arm = Path(path), arm
        self.limits = {"initial": 16, "train": UPDATES[arm] * 16, "final1": 16, "final2": 16}
        self.counts = {k: 0 for k in self.limits}
        self.offset = 0 if arm == "control" else 3248
        with self.path.open("x", encoding="utf8"): pass

    def reserve(self, phase, step, example):
        require(phase in self.limits and self.counts[phase] < self.limits[phase] and
                self.offset + sum(self.counts.values()) < 19296 and type(example) is int and 0 <= example < 16,
                "Duration forward budget exhausted; STOP")
        current = next(k for k in self.limits if self.counts[k] < self.limits[k])
        require(phase == current and example == self.counts[phase] % 16 and
                step == (self.counts[phase] // 16 + 1 if phase == "train" else None), "Duration call order; STOP")
        record = {"call": self.offset + sum(self.counts.values()) + 1, "arm": self.arm, "phase": phase,
                  "phaseCall": self.counts[phase] + 1, "step": step, "example": example}
        with self.path.open("a", encoding="utf8", newline="\n") as stream:
            stream.write(json.dumps(record) + "\n"); stream.flush()
        self.counts[phase] += 1


def prefix_exact(prefix, original, control=None):
    require(prefix["step"] == 200 and prefix["weightsSha256"] == original["finalLogicalWeightsSha256"] and
            prefix["historySha256"] == original["historySha256"] and
            isinstance(prefix["optimizerSha256"], str) and len(prefix["optimizerSha256"]) == 64 and
            all(c in "0123456789abcdef" for c in prefix["optimizerSha256"]), "Original 200-prefix drift; STOP")
    if control is not None: require(prefix == control, "Concurrent model/optimizer/history prefix drift; STOP")


def load(run, arm):
    config, frozen = json.loads(CONFIG.read_bytes()), json.loads((run / "execution.json").read_bytes())
    proposal = ROOT / config["proposalPath"]
    p = json.loads(proposal.read_bytes())
    require(config["status"] == frozen["status"] == "FROZEN_BEFORE_EXECUTION" and config["approval"] == "USER_APPROVED" and
            config["proposalSha256"] == sha256(proposal.read_bytes()) and frozen["configSha256"] == sha256(CONFIG.read_bytes()) and
            config["contract"] == {k: p[k] for k in config["contract"]}, "Approved duration contract identity")
    c = config["contract"]
    require(c["contrast"]["controlUpdates"] == 200 and c["contrast"]["candidateUpdates"] == 1000 and
            c["forwardBudget"]["totalTopLevelMax"] == 19296 and c["forwardBudget"]["checkpointLoads"] == 0, "Fixed duration limits")
    for path, digest in {**c["sourceHashes"], **frozen["sourceHashes"]}.items():
        require(sha256((ROOT / path).read_bytes()) == digest, "Duration source changed: " + path)
    require(sha256((run / "dataset.json").read_bytes()) == frozen["datasetSha256"] == c["originalRun"]["datasetSha256"], "Exact original dataset bytes")
    original = Path(frozen["originalRunPath"])
    for name, key in [("execution.json", "executionSha256"), ("initial.json", "initialPredictionsSha256"),
                      ("history.json", "historySha256"), ("final1.json", "finalPredictionsSha256")]:
        require(sha256((original / name).read_bytes()) == c["originalRun"][key], "Original archive identity")
    if arm == "candidate":
        phase = json.loads((run / "candidate-input.json").read_bytes())
        require(phase["executionSha256"] == sha256((run / "execution.json").read_bytes()) and
                phase["controlReportSha256"] == sha256((run / "control/report.json").read_bytes()), "Control phase freeze")
        control = json.loads((run / "control/report.json").read_bytes())
        require(control["status"] == "PASS" and all(control["originalReproduction"].values()) and
                control["executionSha256"] == phase["executionSha256"] and control["modelForwards"] == 3248,
                "Exact control must precede candidate model creation")
        for name, digest in control["hashes"].items(): require(sha256((run / "control" / name).read_bytes()) == digest, "Control artifact drift")
    return config, frozen, validate_dataset(json.loads((run / "dataset.json").read_bytes())), original


def run_arm(run, arm):
    started = time.perf_counter()
    output = run / arm
    monitor = Budget(600, 2147483648, output, started)
    monitor.sample("start")
    require(torch.version.cuda is None, "Existing CPU-only PyTorch")
    torch.set_num_interop_threads(1)
    config, frozen, examples, original = load(run, arm)
    c, steps = config["contract"], UPDATES[arm]
    inputs = [spatial_tensors(e) for e in examples]
    monitor.sample("loaded")
    training = c["shared"]["model"]["config"]  # Legacy epochs12/batchSize8 construct the same model, not this schedule.
    model = make_model(training, "spatial")
    require(sum(p.numel() for p in model.parameters()) == 13292, "Fixed parameter count")
    initial_weights = weights_hash(model)
    require(initial_weights == c["originalRun"]["initialLogicalWeightsSha256"], "Initial weights drift; STOP")
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learningRate"], weight_decay=training["weightDecay"])
    ledger = DurationLedger(output / "forward-count.jsonl", arm)
    state = {"phase": "initial", "step": None, "example": 0}
    def evaluate(phase):
        model.eval(); state.update(phase=phase, step=None)
        rows = []
        with torch.inference_mode():
            for i, (e, inp) in enumerate(zip(examples, inputs)):
                state["example"] = i
                rows.append(score_row(model(*inp), e["label"], e["id"]))
                monitor.sample(phase)
        write_json(output / (phase + ".json"), rows)
        return rows
    history, prefix = [], None
    with counted_model(model, ledger, state):
        initial = evaluate("initial")
        require(sha256((output / "initial.json").read_bytes()) == c["originalRun"]["initialPredictionsSha256"], "Initial logits drift; STOP")
        for step in range(1, steps + 1):
            monitor.sample("update-before")
            model.train(); state.update(phase="train", step=step)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for i, (e, inp) in enumerate(zip(examples, inputs)):
                state["example"] = i
                loss = flat_loss(model(*inp), e["label"])
                losses.append(float(loss.detach())); (loss / 16).backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()), "Nonfinite gradient; STOP")
            optimizer.step()
            require(all(torch.isfinite(p).all().item() for p in model.parameters()), "Nonfinite weights; STOP")
            history.append({"step": step, "meanTrainingNll": sum(losses) / 16})
            if step == 200:
                write_json(output / "prefix-history.json", history)
                logical = canonical_state(optimizer.state_dict())
                write_json(output / "optimizer-step200.json", logical)
                prefix = {"step": 200, "weightsSha256": weights_hash(model),
                          "historySha256": sha256((output / "prefix-history.json").read_bytes()),
                          "optimizerSha256": optimizer_fingerprint(optimizer.state_dict())}
                prefix_exact(prefix, c["originalRun"], json.loads((run / "control/prefix.json").read_bytes()) if arm == "candidate" else None)
                write_json(output / "prefix.json", prefix)
                print(f"{arm} prefix200 exact; no extra forward", flush=True)
            monitor.sample("update-after")
            if step % 100 == 0: print(f"{arm} update {step}/{steps} loss={history[-1]['meanTrainingNll']:.8f}", flush=True)
        final1, final2 = evaluate("final1"), evaluate("final2")
    require((output / "final1.json").read_bytes() == (output / "final2.json").read_bytes(), "Final repeated logits drift; STOP")
    require(ledger.counts == ledger.limits, "Exact duration call accounting")
    write_json(output / "history.json", history)
    final_weights = weights_hash(model)
    summary = {"initial": summarize(initial), "final": summarize(final1), "fitGate": fit_gate(summarize(final1)), "updates": steps}
    reproduction = None
    if arm == "control":
        old = json.loads((original / "summary.json").read_bytes())
        reproduction = {"initial": sha256((output / "initial.json").read_bytes()) == c["originalRun"]["initialPredictionsSha256"],
                        "history": sha256((output / "history.json").read_bytes()) == c["originalRun"]["historySha256"],
                        "finalRows": sha256((output / "final1.json").read_bytes()) == c["originalRun"]["finalPredictionsSha256"],
                        "logicalWeights": final_weights == c["originalRun"]["finalLogicalWeightsSha256"],
                        "metrics": all(summary[k] == old[k] for k in ("initial", "final", "fitGate"))}
        require(all(reproduction.values()), "Original control reproduction drift; STOP")
    torch.save(model.state_dict(), output / "final-weights.pt")
    write_json(output / "summary.json", summary)
    for path, digest in frozen["sourceHashes"].items(): require(sha256((ROOT / path).read_bytes()) == digest, "Source changed during duration run")
    monitor.sample("finished")
    names = ["initial.json", "final1.json", "final2.json", "history.json", "prefix-history.json", "prefix.json", "optimizer-step200.json", "forward-count.jsonl", "final-weights.pt", "summary.json"]
    write_json(output / "report.json", {"status": "PASS", "atUtc": datetime.now(timezone.utc).isoformat(), "arm": arm,
        "executionSha256": sha256((run / "execution.json").read_bytes()), "datasetSha256": frozen["datasetSha256"],
        "hashes": {p: sha256((output / p).read_bytes()) for p in names}, "summary": summary, "prefix": prefix,
        "initialWeightsSha256": initial_weights, "finalWeightsSha256": final_weights, "originalReproduction": reproduction,
        "forwardCounts": ledger.counts, "modelForwards": sum(ledger.counts.values()), "globalOffset": ledger.offset,
        "topLevelModels": 1, "checkpointLoads": 0, "trainingRuns": 1, "budget": monitor.report(),
        "environment": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                        "threads": torch.get_num_threads(), "interopThreads": torch.get_num_interop_threads()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True); parser.add_argument("--arm", choices=tuple(UPDATES), required=True)
    args = parser.parse_args(); run = Path(args.run).resolve(); output = run / args.arm
    require(run.is_relative_to(ROOT / ".research-output") and run != ROOT / ".research-output" and output.is_dir() and
            not (output / "forward-count.jsonl").exists(), "Owned new arm only; no resume or retry")
    try: run_arm(run, args.arm)
    except BaseException as error:
        if not (output / "failure.json").exists(): write_json(output / "failure.json", {"status": "FAIL", "error": str(error), "retry": False})
        raise


if __name__ == "__main__": main()
