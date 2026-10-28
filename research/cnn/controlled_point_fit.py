"""One approved fit-only run. No previous data, checkpoints, dev selection or retries."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import torch
from torch.nn import functional as F
from data import require, sha256
from spatial import make_model, spatial_tensors, Budget
from paired import weights_hash
from train import write_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "research/experiments/controlled-point-fit-v1.json"
LIMITS = {"initial": 16, "train": 3200, "final1": 16, "final2": 16}


def flat_loss(scores, label):
    require(scores.ndim == 1 and len(scores) == 31 and scores.dtype == torch.float32 and
            scores.device.type == "cpu" and torch.isfinite(scores).all().item() and
            type(label) is int and 0 <= label < 31, "Finite CPU float32 full-list fit loss")
    value = F.cross_entropy(scores.unsqueeze(0), torch.tensor([label]))
    require(torch.isfinite(value).item(), "Nonfinite fit loss; STOP")
    return value


def score_row(scores, label, identifier):
    nll = float(flat_loss(scores, label).detach())
    values = scores.detach().tolist()
    index = int(scores.argmax().item())
    rivals = [v for i, v in enumerate(values) if i != label]
    rank = 1 + sum(v > values[label] or (v == values[label] and i < label)
                   for i, v in enumerate(values) if i != label)
    return {"id": identifier, "label": label, "index": index, "correct": index == label,
            "nll": nll, "rank": rank, "margin": values[label] - max(rivals), "logits": values}


def summarize(rows):
    require(len(rows) == 16 and len({r["id"] for r in rows}) == 16, "Fit metric denominator")
    return {"examples": 16, "correct": sum(r["correct"] for r in rows),
            "meanNll": sum(r["nll"] for r in rows) / 16,
            "minimumTargetMargin": min(r["margin"] for r in rows),
            "maximumTargetRank": max(r["rank"] for r in rows)}


def fit_gate(summary):
    checks = {"allCorrect": summary["correct"] == 16,
              "meanNllAtMost005": summary["meanNll"] <= 0.05,
              "everyTargetMarginPositive": summary["minimumTargetMargin"] > 0}
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


class Ledger:
    def __init__(self, path):
        self.path, self.counts = Path(path), {k: 0 for k in LIMITS}
        with self.path.open("x", encoding="utf8"): pass

    def reserve(self, phase, step, example):
        require(phase in LIMITS and self.counts[phase] < LIMITS[phase] and
                sum(self.counts.values()) < 3248 and type(example) is int and 0 <= example < 16,
                "Forward budget exhausted or bad phase; STOP")
        expected_phase = next(k for k in LIMITS if self.counts[k] < LIMITS[k])
        require(phase == expected_phase and example == self.counts[phase] % 16 and
                step == (self.counts[phase] // 16 + 1 if phase == "train" else None),
                "Forward phase/step/example order; STOP")
        row = {"call": sum(self.counts.values()) + 1, "phase": phase,
               "phaseCall": self.counts[phase] + 1, "step": step, "example": example}
        with self.path.open("a", encoding="utf8", newline="\n") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()  # Reservation is durable before __call__, including failed calls.
        self.counts[phase] += 1


@contextmanager
def counted_model(model, ledger, state):
    original = torch.nn.Module.__call__
    allowed = {id(m) for m in model.modules()}
    def call(module, *args, **kwargs):
        require(id(module) in allowed, "Unregistered model forward; STOP")
        if module is model: ledger.reserve(state["phase"], state["step"], state["example"])
        return original(module, *args, **kwargs)
    try:
        torch.nn.Module.__call__ = call
        yield
    finally:
        torch.nn.Module.__call__ = original


def validate_dataset(data):
    require(data["schemaVersion"] == 1 and data["id"] == "controlled-point-fit-data-v1" and
            data["split"] == "fit-only" and len(data["examples"]) == 16 and
            data["counts"]["examples"] == 16 and data["counts"]["candidates"] == 496 and
            0 < data["counts"]["referenceNodes"] <= 4096 and data["counts"]["outcomeTransitions"] == 496 and
            data["counts"]["dfsCalls"] == 0, "Isolated fixed fit data")
    def vector(v, n):
        return isinstance(v, list) and len(v) == n and all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in v)
    for row, e in enumerate(data["examples"]):
        require(e["id"] == f"controlled-fit-{row}" and type(e["label"]) is int and
                0 <= e["label"] < 31 and e["reference"]["complete"] is True and
                e["reference"]["unique"] is True and 0 < e["reference"]["visitedNodes"] <= 256,
                "Fit identity and complete unique reference")
        obs = e["observation"]
        require(set(obs) == {"planes", "metadata"} and len(obs["planes"]) == 3 and
                all(len(p) == 16 and all(vector(r, 10) and all(v in (0, 1) for v in r) for r in p) for p in obs["planes"]) and
                vector(obs["metadata"], 140), "Fit observation schema")
        board = obs["planes"][0]
        require(all(v == 0 for p in obs["planes"][1:] for r in p for v in r), "No fit items")
        require(all(board[r][c] == int(c != (3 * row) % 10) if r == row else
                    board[r][c] == int(c not in ((3 * r) % 10, (3 * r + 1) % 10))
                    for r in range(16) for c in range(10)), "Fixed fit board geometry")
        keys = [["single-cell", r, c] for r in range(16) for c in range(10) if not board[r][c]]
        require(len(keys) == 31 and [json.loads(k) for k in e["candidateKeys"]] == keys and
                keys[e["label"]] == ["single-cell", row, (3 * row) % 10] and
                len(e["candidateFeatures"]) == len(e["footprints"]) == 31, "Full ordered point-only mask/label")
        for i, (key, features, bits) in enumerate(zip(keys, e["candidateFeatures"], e["footprints"])):
            _, r, c = key
            expected = "".join("1" if k == 10 * r + c else "0" for k in range(160))
            require(vector(features, 74) and features[:3] == [0, 1, 0] and
                    features[25:27] == [r / 15, c / 9] and
                    features[52] == (1 / 16 if i == e["label"] else 0) and bits == expected,
                    "Fit features/footprint geometry and outcome shortcut")
    return data["examples"]


def load(output):
    config, frozen = json.loads(CONFIG.read_bytes()), json.loads((output / "execution.json").read_bytes())
    proposal = ROOT / config["proposalPath"]
    p = json.loads(proposal.read_bytes())
    require(config["status"] == frozen["status"] == "FROZEN_BEFORE_EXECUTION" and
            config["approval"] == "USER_APPROVED" and sha256(CONFIG.read_bytes()) == frozen["configSha256"] and
            sha256(proposal.read_bytes()) == config["proposalSha256"] and
            config["contract"] == {k: p[k] for k in config["contract"]}, "Approved frozen fit contract")
    require(config["contract"]["forwardBudget"]["totalTopLevelMax"] == 3248 and
            config["contract"]["fit"]["steps"] == 200 and config["contract"]["fit"]["runs"] == 1,
            "Fixed fit budget")
    for path, digest in {**p["sourceHashes"], **frozen["sourceHashes"]}.items():
        require(sha256((ROOT / path).read_bytes()) == digest, "Fit source changed: " + path)
    body = (output / "dataset.json").read_bytes()
    require(sha256(body) == frozen["datasetSha256"], "Frozen fit dataset hash")
    return config, frozen, validate_dataset(json.loads(body))


def run(output):
    started = time.perf_counter()
    monitor = Budget(600, 2147483648, output, started)
    monitor.sample("start")
    require(torch.version.cuda is None, "Existing CPU-only runtime required")
    torch.set_num_interop_threads(1)
    config, frozen, examples = load(output)
    training = config["contract"]["model"]["config"]
    inputs = [spatial_tensors(e) for e in examples]
    monitor.sample("loaded")
    model = make_model(training, "spatial")  # Exactly one top-level model; no checkpoint deserialization.
    require(sum(p.numel() for p in model.parameters()) == 13292, "Original spatial parameter count")
    initial_weights = weights_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learningRate"], weight_decay=training["weightDecay"])
    ledger = Ledger(output / "forward-count.jsonl")
    state = {"phase": "initial", "step": None, "example": 0}
    def evaluate(phase):
        model.eval()
        state.update(phase=phase, step=None)
        rows = []
        with torch.inference_mode():
            for i, (e, inp) in enumerate(zip(examples, inputs)):
                state["example"] = i
                rows.append(score_row(model(*inp), e["label"], e["id"]))
                monitor.sample(phase)
        write_json(output / (phase + ".json"), rows)
        return rows
    history = []
    with counted_model(model, ledger, state):
        initial = evaluate("initial")
        for step in range(1, 201):
            monitor.sample("update-before")
            model.train()
            state.update(phase="train", step=step)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for i, (e, inp) in enumerate(zip(examples, inputs)):
                state["example"] = i
                loss = flat_loss(model(*inp), e["label"])
                losses.append(float(loss.detach()))
                (loss / 16).backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()), "Nonfinite gradient; STOP")
            optimizer.step()
            require(all(torch.isfinite(p).all().item() for p in model.parameters()), "Nonfinite weight; STOP")
            history.append({"step": step, "meanTrainingNll": sum(losses) / 16})
            monitor.sample("update-after")
            if step % 25 == 0: print(f"update {step}/200 loss={history[-1]['meanTrainingNll']:.8f}", flush=True)
        final1, final2 = evaluate("final1"), evaluate("final2")
    require((output / "final1.json").read_bytes() == (output / "final2.json").read_bytes(), "Final frozen repeats drift; STOP")
    require(ledger.counts == LIMITS, "Exact forward accounting")
    write_json(output / "history.json", history)
    # State dict only, private output; never loaded for evaluation or public verification.
    torch.save(model.state_dict(), output / "final-weights.pt")
    initial_summary, final_summary = summarize(initial), summarize(final1)
    summary = {"schemaVersion": 1, "id": "controlled-point-fit-v1", "technicalStatus": "PASS",
               "fitGate": fit_gate(final_summary), "initial": initial_summary, "final": final_summary,
               "updates": 200, "trainingRuns": 1, "topLevelForwardCalls": 3248,
               "finalRepeatsExact": True, "heldOutOrKindSelectionTested": False,
               "spatialMechanismTested": False, "appUsesDfs": True}
    write_json(output / "summary.json", summary)
    for path, digest in frozen["sourceHashes"].items():
        require(sha256((ROOT / path).read_bytes()) == digest, "Source changed during fit")
    monitor.sample("finished")
    names = ["dataset.json", "execution.json", "initial.json", "final1.json", "final2.json", "history.json", "forward-count.jsonl", "summary.json", "final-weights.pt"]
    write_json(output / "child-report.json", {"status": "PASS", "atUtc": datetime.now(timezone.utc).isoformat(),
        "summary": summary, "sourceHead": frozen["sourceHead"], "hashes": {p: sha256((output / p).read_bytes()) for p in names},
        "forwardCounts": ledger.counts, "initialWeightsSha256": initial_weights, "finalWeightsSha256": weights_hash(model),
        "topLevelModels": 1, "checkpointLoads": 0, "trainingRuns": 1, "budget": monitor.report(),
        "environment": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                        "threads": torch.get_num_threads(), "interopThreads": torch.get_num_interop_threads()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    require(output.is_relative_to(ROOT / ".research-output") and output != ROOT / ".research-output" and
            (output / "execution.json").is_file() and not (output / "forward-count.jsonl").exists(), "New owned frozen execution; no resume/retry")
    try:
        run(output)
    except BaseException as error:
        path = output / "child-failure.json"
        if not path.exists(): write_json(path, {"status": "FAIL", "error": str(error), "retry": False})
        raise


if __name__ == "__main__": main()
