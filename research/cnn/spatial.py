"""Fixed spatial-residual contrast; train/dev only, no domain or teacher execution."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
from torch import nn
from data import require, sha256
from model import CandidateCNN, configure, tensors
from outcome import OutcomeCandidateCNN, OUTCOME_SCHEMA, load_inputs, gate, verify_baseline
from paired import training_audit, weights_hash
from diagnose import score, summarize, action_description, fixture_family
from train import evaluate, rss_bytes, write_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "research/experiments/spatial-residual-v1.json"
SPATIAL_SCHEMA = {**OUTCOME_SCHEMA, "version": "legal-candidate-spatial-residual-v3", "footprintShape": [1, 16, 10]}


class Budget:
    def __init__(self, seconds, bytes_limit, output, started=None):
        self.start = time.perf_counter() if started is None else started
        self.seconds, self.bytes_limit, self.output = seconds, bytes_limit, output
        self.samples, self.maximum = 0, 0

    def sample(self, phase):
        elapsed, memory = time.perf_counter() - self.start, rss_bytes()
        require(elapsed <= self.seconds, "Child wall-time budget; STOP")
        require(memory is not None and memory <= self.bytes_limit, "Sampled working-set budget; STOP")
        self.samples += 1
        self.maximum = max(self.maximum, memory)
        if self.output:
            with (self.output / "memory.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps({"phase": phase, "elapsedSeconds": elapsed, "workingSetBytes": memory}) + "\n")

    def report(self):
        return {"samples": self.samples, "maxSampledWorkingSetBytes": self.maximum,
                "instantaneousPeakMeasured": False, "wallTimeSeconds": time.perf_counter() - self.start,
                "wallTimeLimitSeconds": self.seconds, "workingSetLimitBytes": self.bytes_limit}


class SpatialCandidateCNN(nn.Module):
    def __init__(self, config, common):
        super().__init__()
        self.common = common
        self.spatial = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(),
                                     nn.AdaptiveAvgPool2d((4, 5)), nn.Flatten(), nn.Linear(80, 1))
        nn.init.zeros_(self.spatial[-1].weight)
        nn.init.zeros_(self.spatial[-1].bias)
        self.chunk = 32

    def forward(self, planes, metadata, candidates, footprints):
        require(footprints.shape == (len(candidates), 1, 16, 10) and footprints.dtype == torch.float32 and
                footprints.device.type == "cpu" and torch.isfinite(footprints).all().item() and
                ((footprints == 0) | (footprints == 1)).all().item(), "Binary CPU footprint schema")
        original = self.common(planes, metadata, candidates)
        residual = []
        for start in range(0, len(candidates), self.chunk):
            mask = footprints[start:start + self.chunk]
            current = planes.expand(len(mask), -1, -1, -1)
            residual.append(self.spatial(torch.cat((current, mask), dim=1)).squeeze(1))
        logits = original + torch.cat(residual)
        require(torch.isfinite(logits).all().item(), "Nonfinite spatial logits; STOP")
        return logits


def spatial_tensors(example):
    # Only observation, candidate features and pre-encoded geometry enter forward.
    bits = example["footprints"]
    require(len(bits) == len(example["candidateFeatures"]) and
            all(isinstance(b, str) and len(b) == 160 and set(b) <= {"0", "1"} for b in bits),
            "Stored binary footprints")
    mask = torch.tensor([[int(c) for c in b] for b in bits], dtype=torch.float32).reshape(-1, 1, 16, 10)
    return (*tensors(example), mask)


def make_model(config, mode):
    configure(config)
    common = OutcomeCandidateCNN(config, CandidateCNN(config))
    return common if mode == "baseline" else SpatialCandidateCNN(config, common)


def track_forwards(model):
    counter = {"calls": 0}
    def count(*args): counter["calls"] += 1
    model.register_forward_pre_hook(count)
    return counter


def load(source, original, execution):
    config, frozen = json.loads(CONFIG.read_bytes()), json.loads(Path(execution).read_bytes())
    require(config["approval"] == "USER_APPROVED" and config["status"] == frozen["status"] ==
            "FROZEN_BEFORE_EXECUTION" and frozen["configSha256"] == sha256(CONFIG.read_bytes()), "Frozen spatial config")
    require(config["limits"] == {"footprintCandidates": 149597, "trainingRuns": 2, "epochs": 12,
            "childSeconds": 600, "workingSetBytes": 2147483648, "chunk": 32, "teacherDomainTestRollout": 0},
            "Fixed spatial limits")
    for p, h in frozen["sourceHashes"].items():
        require(sha256((ROOT / p).read_bytes()) == h, "Spatial source changed: " + p)
    original = Path(original)
    for p, h in config["source"]["outcomeHashes"].items():
        require(sha256((original / p).read_bytes()) == h, "Original outcome identity: " + p)
    require(sha256((Path(source) / "train-dev/dataset.json").read_bytes()) == config["source"]["datasetSha256"],
            "Original paired identity")
    pack, protocol, manifest, examples, old_frozen = load_inputs(source, original / "dataset.json", original / "execution.json")
    for k in ("seed", "epochs", "batchSize", "threads", "convChannels", "embedding", "poolRows", "poolCols", "learningRate", "weightDecay"):
        require(config["training"][k] == protocol["training"][k], "Original hyperparameter: " + k)
    body = (Path(execution).parent / "footprints.json").read_bytes()
    require(sha256(body) == frozen["footprintsSha256"], "Footprint hash")
    masks = json.loads(body)
    require(masks["schema"] == SPATIAL_SCHEMA and masks["candidates"] == 149597 and
            masks["sourceDataSha256"] == manifest["dataSha256"] and
            masks["outcomeDataSha256"] == old_frozen["augmentedSha256"] and
            set(masks["examples"]) == {"train", "dev"}, "Spatial train/dev-only identity")
    total = 0
    for split in ("train", "dev"):
        require(len(examples[split]) == len(masks["examples"][split]), "Spatial observation count")
        for e, m in zip(examples[split], masks["examples"][split]):
            require(set(m) == {"id", "footprints"} and m["id"] == e["id"] and
                    len(m["footprints"]) == len(e["candidateKeys"]) and
                    all(isinstance(b, str) and len(b) == 160 and set(b) <= {"0", "1"} for b in m["footprints"]),
                    "Ordered spatial binary input")
            e["footprints"] = m["footprints"]
            total += len(m["footprints"])
    require(total == 149597, "Exact candidate accounting")
    return config, frozen, pack, protocol, examples


def initial_control(examples, config, monitor):
    baseline = make_model(config, "baseline").eval()
    spatial = make_model(config, "spatial").eval()
    counts = [track_forwards(baseline), track_forwards(spatial)]
    with torch.inference_mode():
        for split in ("train", "dev"):
            for e in examples[split]:
                inp = spatial_tensors(e)
                require(torch.equal(baseline(*inp[:3]), spatial(*inp)), "Initial spatial logits drift; STOP")
                monitor.sample("initial-control")
    parameters = {"baseline": sum(p.numel() for p in baseline.parameters()),
                  "spatial": sum(p.numel() for p in spatial.parameters())}
    require(parameters == {"baseline": 12915, "spatial": 13292}, "Fixed parameter counts")
    return {"initialLogitsExact": True, "observations": 428, "parameters": parameters,
            "modelForwards": sum(c["calls"] for c in counts)}


def fit_fixed(model, train, dev, protocol, data_hash, checkpoint, mode, monitor):
    config = protocol["training"]
    tensor = tensors if mode == "baseline" else spatial_tensors
    train_inputs, dev_inputs = [tensor(e) for e in train], [tensor(e) for e in dev]
    monitor.sample("tensors")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learningRate"], weight_decay=config["weightDecay"])
    generator = torch.Generator().manual_seed(config["seed"])
    history, best = [], -1
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator).tolist()
        total_loss = 0.0
        for offset in range(0, len(order), config["batchSize"]):
            monitor.sample("batch-before")
            batch = order[offset:offset + config["batchSize"]]
            optimizer.zero_grad(set_to_none=True)
            for index in batch:
                loss = torch.nn.functional.cross_entropy(model(*train_inputs[index]).unsqueeze(0), torch.tensor([train[index]["label"]]))
                require(torch.isfinite(loss).item(), "Nonfinite spatial contrast loss; STOP")
                total_loss += float(loss.detach())
                (loss / len(batch)).backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()), "Nonfinite gradient; STOP")
            optimizer.step()
            monitor.sample("batch-after")
        model.eval()
        dev_metrics, _ = evaluate(model, dev, dev_inputs)
        monitor.sample("epoch-dev")
        selected = dev_metrics["top1"] > best
        if selected:
            best, best_epoch = dev_metrics["top1"], epoch
            torch.save({"state_dict": model.state_dict(), "schema": OUTCOME_SCHEMA if mode == "baseline" else SPATIAL_SCHEMA,
                        "protocol": protocol, "dataSha256": data_hash, "epoch": epoch}, checkpoint)
        history.append({"epoch": epoch, "trainLoss": total_loss / len(train), "dev": dev_metrics, "selected": selected})
        print(f"{mode} epoch {epoch}: dev {dev_metrics['correct']}/89", flush=True)
    return history, best_epoch


def diagnose_frozen(model, examples, families, mode, output, monitor):
    raw, summary, logits_records = {}, {}, {}
    tensor = tensors if mode == "baseline" else spatial_tensors
    for split in ("train", "dev"):
        rows, records = [], []
        for e in examples[split]:
            with torch.inference_mode():
                inp = tensor(e)
                logits = model(*inp)
                require(torch.equal(logits, model(*inp)), "Repeated frozen spatial logits drift")
            row = score(logits, e["labels"])
            row.update(id=e["id"], fixtureId=e["fixtureId"], family=families[e["fixtureId"]],
                sameLabel=e["labels"]["A512"] == e["labels"]["A1024"],
                complete={n: e["teachers"][n]["result"]["search"]["searchComplete"] for n in ("A512", "A1024")},
                teacherActions={n: action_description(e["candidateKeys"][e["labels"][n]]) for n in ("A512", "A1024")},
                selected=action_description(e["candidateKeys"][row["index"]]), firstLegalAction=action_description(e["candidateKeys"][0]))
            rows.append(row)
            records.append({"id": e["id"], "logits": logits.detach().tolist()})
            monitor.sample("frozen-diagnostic")
        raw[split], logits_records[split] = rows, records
        summary[split] = summarize(rows)
        summary[split]["targets"] = {"A512": summary[split]["targets"]["A512"]}
    write_json(output / "predictions.json", raw)
    write_json(output / "logits.json", logits_records)
    return raw, summary


def run(mode, source, original, execution, output):
    started = time.perf_counter()
    config = json.loads(CONFIG.read_bytes())
    torch.set_num_interop_threads(1)
    require(torch.version.cuda is None, "Existing CPU-only build required")
    monitor = Budget(config["limits"]["childSeconds"], config["limits"]["workingSetBytes"], output, started)
    monitor.sample("start")
    config, frozen, pack, protocol, examples = load(source, original, execution)
    monitor.sample("loaded")
    audit = training_audit(examples["train"], protocol["training"])
    execution_hash = sha256(Path(execution).read_bytes())
    if mode == "preflight":
        control = initial_control(examples, protocol["training"], monitor)
        write_json(output / "report.json", {"status": "PASS", "control": control, "audit": audit,
            "executionSha256": execution_hash, "budget": monitor.report(), "trainingRuns": 0, "testAccess": False})
        return
    prior = Path(execution).parent
    preflight = json.loads((prior / "preflight/report.json").read_bytes())
    require(preflight["status"] == "PASS" and preflight["executionSha256"] == execution_hash and
            preflight["control"]["initialLogitsExact"] and preflight["audit"] == audit, "Preflight before training")
    if mode == "spatial":
        baseline = json.loads((prior / "baseline/report.json").read_bytes())
        require(baseline["status"] == "PASS" and baseline["executionSha256"] == execution_hash and
                all(baseline["baselineReproduction"].values()), "Baseline reproduced before spatial training")
    model = make_model(protocol["training"], mode)
    counter = track_forwards(model)
    checkpoint = output / "checkpoint.pt"
    history, epoch = fit_fixed(model, examples["train"], examples["dev"], protocol,
                               frozen["originalHashes"]["dataset.json"], checkpoint, mode, monitor)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(saved["epoch"] == epoch and saved["protocol"] == protocol and saved["dataSha256"] ==
            frozen["originalHashes"]["dataset.json"] and saved["schema"] ==
            (OUTCOME_SCHEMA if mode == "baseline" else SPATIAL_SCHEMA), "Checkpoint identity")
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    reproduction = None
    if mode == "baseline":
        old = json.loads((Path(original) / "outcome/report.json").read_bytes())
        original_weights = torch.load(Path(original) / "outcome/checkpoint.pt", map_location="cpu", weights_only=True)
        reproduction = verify_baseline(model, history, epoch, audit, old, original_weights)
    families = {e["fixture"]["id"]: fixture_family(e["fixture"], protocol["dataset"]["families"]) for e in pack["episodes"]}
    raw, summary = diagnose_frozen(model, examples, families, mode, output, monitor)
    if mode == "baseline":
        require(raw == json.loads((Path(original) / "outcome/predictions.json").read_bytes()), "Baseline prediction scores exact; STOP")
    require(counter["calls"] == 5992, "Fixed training/evaluation forward accounting")
    for p, h in config["source"]["outcomeHashes"].items():
        require(sha256((Path(original) / p).read_bytes()) == h, "Original unchanged after fit")
    write_json(output / "report.json", {"status": "PASS", "mode": mode, "history": history, "selectedEpoch": epoch,
        "executionSha256": execution_hash, "audit": audit, "baselineReproduction": reproduction, "summary": summary,
        "checkpointSha256": sha256(checkpoint.read_bytes()), "weightsSha256": weights_hash(model),
        "predictionsSha256": sha256((output / "predictions.json").read_bytes()), "logitsSha256": sha256((output / "logits.json").read_bytes()),
        "modelForwards": counter["calls"], "trainingRuns": 1, "epochs": 12, "teacherDomainTestRollout": 0,
        "budget": monitor.report(), "environment": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                                                      "threads": torch.get_num_threads(), "interopThreads": torch.get_num_interop_threads()}})


def reproduce(source, original, execution, output):
    started = time.perf_counter()
    torch.set_num_interop_threads(1)
    config, frozen, pack, protocol, examples = load(source, original, execution)
    monitor = Budget(600, 2147483648, output, started)
    families = {e["fixture"]["id"]: fixture_family(e["fixture"], protocol["dataset"]["families"]) for e in pack["episodes"]}
    models, calls = {}, 0
    for mode in ("baseline", "spatial"):
        original_dir = Path(execution).parent / mode
        report = json.loads((original_dir / "report.json").read_bytes())
        require(report["status"] == "PASS" and report["executionSha256"] == sha256(Path(execution).read_bytes()), "Frozen report")
        checkpoint = original_dir / "checkpoint.pt"
        require(sha256(checkpoint.read_bytes()) == report["checkpointSha256"], "Frozen checkpoint hash")
        model = make_model(protocol["training"], mode).eval()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"], strict=True)
        require(weights_hash(model) == report["weightsSha256"], "Frozen logical weights")
        counter = track_forwards(model)
        dest = output / mode
        dest.mkdir()
        _, summary = diagnose_frozen(model, examples, families, mode, dest, monitor)
        for p, digest in (("predictions.json", report["predictionsSha256"]), ("logits.json", report["logitsSha256"])):
            require(sha256((dest / p).read_bytes()) == digest, "Frozen reproduced bytes: " + p)
        require(summary == report["summary"] and counter["calls"] == 856, "Reproduced summary/count")
        models[mode], calls = summary, calls + counter["calls"]
    write_json(output / "report.json", {"status": "PASS", "trainingRuns": 0, "modelForwards": calls,
        "teacherDomainTestRollout": 0, "summary": models, "budget": monitor.report(),
        "gate": gate(models["baseline"]["dev"]["targets"]["A512"], models["spatial"]["dev"]["targets"]["A512"])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "baseline", "spatial", "reproduce"), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--execution", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    require(output.is_relative_to(ROOT / ".research-output") and output != ROOT / ".research-output", "New ignored output required")
    if args.mode != "reproduce":
        require(output == Path(args.execution).resolve().parent / args.mode, "One mode output per execution")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(), "Commit source/config before execution")
    output.mkdir(parents=True, exist_ok=False)
    try:
        (reproduce if args.mode == "reproduce" else lambda *a: run(args.mode, *a))(args.source, args.original, args.execution, output)
    except Exception as error:
        write_json(output / "failure.json", {"status": "FAIL", "error": str(error), "additionalTrainingAllowed": False,
                                           "preservation": "NOT_VERIFIED; perform separate original-file audit"})
        raise


if __name__ == "__main__": main()
