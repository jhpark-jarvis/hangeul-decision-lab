"""Approved same-CNN occupancy contrast. No data generation, loads, tuning or retries."""
import time
PROCESS_STARTED = time.perf_counter()
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import struct
import sys
from unittest.mock import patch
import torch
from torch.nn import functional as F
from data import require, sha256
from spatial import make_model, spatial_tensors, Budget
from controlled_point_fit import counted_model
from paired import weights_hash
from train import write_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "research/experiments/board-context-v1.json"
LIMITS = {"initial": 32, "train": 32000, "final1": 32, "final2": 32}


def loss(scores, label):
    require(scores.ndim == 1 and len(scores) == 32 and scores.dtype == torch.float32 and
            scores.device.type == "cpu" and torch.isfinite(scores).all().item() and
            type(label) is int and 0 <= label < 32, "Finite full32 CPU FP32 CE")
    value = F.cross_entropy(scores.unsqueeze(0), torch.tensor([label]))
    require(torch.isfinite(value).item(), "Nonfinite loss; STOP")
    return value


def score_row(scores, label, index):
    values = scores.detach().tolist()
    chosen = int(scores.argmax().item())
    return {"example": index, "label": label, "index": chosen, "correct": chosen == label,
            "nll": float(loss(scores, label).detach()), "logits": values,
            "rank": 1 + sum(v > values[label] or (v == values[label] and j < label)
                            for j, v in enumerate(values) if j != label),
            "margin": values[label] - max(v for j, v in enumerate(values) if j != label)}


def summarize(rows):
    require(len(rows) == 32 and [r["example"] for r in rows] == list(range(32)), "32 ordered examples")
    return {"examples": 32, "correct": sum(r["correct"] for r in rows),
            "meanNll": sum(r["nll"] for r in rows) / 32,
            "minMargin": min(r["margin"] for r in rows), "maxRank": max(r["rank"] for r in rows),
            "cohorts": {name: {"examples": 16, "correct": sum(r["correct"] for r in rows[parity::2])}
                        for name, parity in [("orderLucky", 0), ("hard", 1)]}}


def fit_gate(summary):
    checks = {"correct32": summary["correct"] == 32, "meanNllAtMost005": summary["meanNll"] <= 0.05,
              "allMarginsPositive": summary["minMargin"] > 0}
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def validate_examples(data):
    require(data["schemaVersion"] == 1 and len(data["examples"]) == 32, "Original P110 build schema")
    def vector(v, n):
        return isinstance(v, list) and len(v) == n and all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in v)
    for i, e in enumerate(data["examples"]):
        require(type(e["label"]) is int and 0 <= e["label"] < 32 and e["reference"]["complete"] is True and
                e["reference"]["qualityIndices"] == [e["label"]] and e["descriptor"]["familyIndex"] == i // 2 and
                e["descriptor"]["arm"] == ("early-aligned" if i % 2 == 0 else "late-aligned"), "Ordered complete unique quality roots")
        obs = e["observation"]
        require(len(obs["planes"]) == 3 and all(len(p) == 16 and all(vector(r, 10) for r in p) for p in obs["planes"]) and
                vector(obs["metadata"], 140), "Original observation shape")
        require(len(e["candidateFeatures"]) == len(e["candidateKeys"]) == len(e["footprints"]) == 32 and
                len(set(e["candidateKeys"])) == 32 and all(vector(v, 74) for v in e["candidateFeatures"]) and
                all(isinstance(v, str) and len(v) == 160 and set(v) <= {"0", "1"} for v in e["footprints"]), "Full ordered candidate inputs")
    return data["examples"]


def tensors(example, arm):
    require(arm in ("control", "candidate"), "Fixed arm")
    # spatial_tensors whitelists only original observation/features/footprints.
    values = spatial_tensors(example)
    require(tuple(values[0].shape) == (1, 3, 16, 10) and tuple(values[1].shape) == (1, 140) and
            tuple(values[2].shape) == (32, 74) and tuple(values[3].shape) == (32, 1, 16, 10), "Fixed native schema")
    if arm == "control":
        planes = values[0].clone()
        planes[:, 0].zero_()
        values = (planes, *values[1:])
    return values


def input_keys(inp):
    # Raw little-endian FP32 bytes also independently reconstructed by the Node reader.
    context = inp[0].flatten().tolist() + inp[1].flatten().tolist()
    return [sha256(struct.pack("<854f", *(context + inp[2][j].tolist() + inp[3][j].flatten().tolist()))) for j in range(32)]


def native_identity(rows, keys):
    seen = {}
    for e, row in enumerate(rows):
        for j, value in enumerate(row["logits"]):
            key = keys[e][j]
            bits = struct.pack("<f", value)
            require(key not in seen or seen[key] == bits, "Identical native input produced different logits; STOP")
            seen[key] = bits
    return {"uniqueInputs": len(seen), "candidateRows": 1024, "allIdenticalInputLogitsExact": True}


class BoardLedger:
    def __init__(self, path, arm):
        require(arm in ("control", "candidate"), "Fixed arm")
        self.path, self.arm = Path(path), arm
        self.offset = 0 if arm == "control" else 32096
        self.counts = {k: 0 for k in LIMITS}
        with self.path.open("x", encoding="utf8"): pass

    def reserve(self, phase, step, example):
        require(phase in LIMITS and self.counts[phase] < LIMITS[phase] and sum(self.counts.values()) < 32096 and
                type(example) is int, "Forward budget; STOP")
        require(phase == next(k for k in LIMITS if self.counts[k] < LIMITS[k]) and
                example == self.counts[phase] % 32 and
                step == (self.counts[phase] // 32 + 1 if phase == "train" else None), "Pre-call phase/step/example order; STOP")
        row = {"call": self.offset + sum(self.counts.values()) + 1, "arm": self.arm, "phase": phase,
               "phaseCall": self.counts[phase] + 1, "step": step, "example": example}
        with self.path.open("a", encoding="utf8", newline="\n") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
        self.counts[phase] += 1


def run(base, arm):
    out = base / arm
    budget = Budget(600, 2147483648, out, PROCESS_STARTED)
    budget.sample("start")
    c = json.loads(CONFIG.read_bytes())
    f = json.loads((base / "execution.json").read_bytes())
    require(c["approval"] == "USER_APPROVED" and c["status"] == f["status"] == "FROZEN_BEFORE_EXECUTION" and
            f["configSha256"] == sha256(CONFIG.read_bytes()), "Approved frozen execution")
    require(c["contract"]["budget"]["totalTopLevelForwardMax"] == 64192 and c["contract"]["fit"]["updatesEach"] == 1000, "Fixed training budget")
    for p, h in f["sourceHashes"].items(): require(sha256((ROOT / p).read_bytes()) == h, "Source drift " + p)
    require(sha256((base / "dataset.json").read_bytes()) == c["archiveHashes"]["build1.json"] == f["datasetSha256"], "Original data bytes")
    examples = validate_examples(json.loads((base / "dataset.json").read_bytes()))
    if arm == "candidate":
        phase = json.loads((base / "candidate-input.json").read_bytes())
        require(phase["executionSha256"] == sha256((base / "execution.json").read_bytes()) and
                phase["controlReportSha256"] == sha256((base / "control/report.json").read_bytes()), "Control-before-candidate freeze")
    require(torch.version.cuda is None, "Existing CPU-only runtime")
    torch.set_num_interop_threads(1)
    inputs = [tensors(e, arm) for e in examples]
    keys = [input_keys(inp) for inp in inputs]
    write_json(out / "input-identities.json", keys)
    write_json(out / "model-intent.json", {"topLevelModels": 1, "checkpointLoads": 0, "arm": arm})
    training = c["contract"]["model"]["config"]
    model = make_model(training, "spatial")
    require(sum(p.numel() for p in model.parameters()) == 13292, "Same original parameter count")
    initial_weights = weights_hash(model)
    if arm == "candidate":
        require(initial_weights == json.loads((base / "control/report.json").read_bytes())["initialLogicalWeightsSha256"], "Same initial logical weights")
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learningRate"], weight_decay=training["weightDecay"])
    ledger = BoardLedger(out / "forward-count.jsonl", arm)
    state = {"phase": "initial", "step": None, "example": 0}
    def evaluate(phase):
        model.eval()
        state.update(phase=phase, step=None)
        rows = []
        with torch.inference_mode():
            for i, (e, inp) in enumerate(zip(examples, inputs)):
                budget.sample(phase + "-before")
                state["example"] = i
                rows.append(score_row(model(*inp), e["label"], i))
                budget.sample(phase + "-after")
        write_json(out / (phase + ".json"), rows)
        if arm == "control":
            native_identity(rows, keys)
            require(summarize(rows)["correct"] <= 16, "Pointwise control bound violated; STOP")
        return rows
    def blocked(*args, **kwargs): raise ValueError("No checkpoint load")
    history = []
    with patch("torch.load", blocked), counted_model(model, ledger, state):
        initial = evaluate("initial")
        for step in range(1, 1001):
            budget.sample("train-before")
            model.train()
            state.update(phase="train", step=step)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for i, (e, inp) in enumerate(zip(examples, inputs)):
                state["example"] = i
                value = loss(model(*inp), e["label"])
                losses.append(float(value.detach()))
                (value / 32).backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()), "Finite gradients")
            optimizer.step()
            require(all(torch.isfinite(p).all().item() for p in model.parameters()), "Finite weights")
            history.append({"step": step, "meanTrainingNll": sum(losses) / 32})
            budget.sample("train-after")
            if step % 100 == 0: print(f"{arm} update {step}/1000 loss={history[-1]['meanTrainingNll']:.8f}", flush=True)
        final1, final2 = evaluate("final1"), evaluate("final2")
    require((out / "final1.json").read_bytes() == (out / "final2.json").read_bytes(), "Final repeats exact")
    require(ledger.counts == LIMITS, "Exact forward accounting")
    write_json(out / "history.json", history)
    torch.save(model.state_dict(), out / "final-weights.pt")
    summary = {"arm": arm, "updates": 1000, "initial": summarize(initial), "final": summarize(final1), "fitGate": fit_gate(summarize(final1))}
    write_json(out / "summary.json", summary)
    for p, h in f["sourceHashes"].items(): require(sha256((ROOT / p).read_bytes()) == h, "Source changed during run")
    budget.sample("finished")
    names = ["initial.json", "final1.json", "final2.json", "history.json", "input-identities.json", "forward-count.jsonl", "model-intent.json", "summary.json", "final-weights.pt"]
    write_json(out / "report.json", {"status": "PASS", "arm": arm, "atUtc": datetime.now(timezone.utc).isoformat(),
        "executionSha256": sha256((base / "execution.json").read_bytes()), "hashes": {p: sha256((out / p).read_bytes()) for p in names},
        "summary": summary, "forwardCounts": ledger.counts, "parameters": 13292, "topLevelModels": 1, "checkpointLoads": 0,
        "initialLogicalWeightsSha256": initial_weights, "finalLogicalWeightsSha256": weights_hash(model),
        "nativeIdentity": native_identity(final1, keys) if arm == "control" else None, "budget": budget.report(),
        "environment": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                        "threads": torch.get_num_threads(), "interopThreads": torch.get_num_interop_threads()}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--arm", choices=["control", "candidate"], required=True)
    args = parser.parse_args()
    base = Path(args.run).resolve()
    out = base / args.arm
    require(base.is_relative_to(ROOT / ".research-output") and out.is_dir() and not (out / "model-intent.json").exists(), "New frozen arm; no resume/retry")
    try: run(base, args.arm)
    except BaseException as error:
        if not (out / "failure.json").exists(): write_json(out / "failure.json", {"status": "FAIL", "error": str(error), "retry": False})
        raise
