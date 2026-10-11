"""Fixed train/dev-only action-outcome contrast; no test or teacher execution."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
from torch import nn
from data import SCHEMA, compact, require, sha256, vector
from model import CandidateCNN, configure, tensors
from paired import load_pack, load_frozen, own_examples, training_audit, weights_hash
from diagnose import score, summarize, action_description, fixture_family
from train import fit, evaluate, rss_bytes, write_json

OUTCOME_SCHEMA = {**SCHEMA, "version": "legal-candidate-outcome-v2", "candidateSize": 74}
ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "research/experiments/outcome-features-v1.json"


def validate_training_contract(original, extended):
    # The original protocol also has descriptive fields, not model parameters.
    for key in ("seed", "epochs", "batchSize", "threads", "convChannels", "embedding",
                "poolRows", "poolCols", "learningRate", "weightDecay"):
        require(original[key] == extended[key], "Fixed training contract: " + key)
    require(original["loss"] == "single-hard-label-cross-entropy" and
            original["sampleOrder"] == "same-paired-example-indices" and
            original["checkpointSelection"] == "own-teacher-highest-dev-top1-first-tie" and
            original["additionalTuning"] is False and original["runs"] == extended["runs"] == 2,
            "Original objective/order/selection contract")


class OutcomeCandidateCNN(CandidateCNN):
    def __init__(self, config, common):
        super().__init__(config)
        self.load_state_dict(common.state_dict(), strict=True)
        # Preserve the original 52-wide GEMM. Padding it to 74 can change rounding.
        self.outcome_candidate = nn.Linear(22, config["embedding"], bias=False)
        self.outcome_bias = nn.Linear(22, 1, bias=False)
        nn.init.zeros_(self.outcome_candidate.weight)
        nn.init.zeros_(self.outcome_bias.weight)

    def forward(self, planes, metadata, candidates):
        require(planes.shape == (1, 3, 16, 10) and metadata.shape == (1, 140) and
                candidates.ndim == 2 and 1 <= len(candidates) <= 4096 and candidates.shape[1] == 74,
                "Outcome tensor schema")
        require(all(t.device.type == "cpu" and torch.isfinite(t).all().item() for t in
                    (planes, metadata, candidates)), "CPU finite inputs required")
        require(((candidates >= 0) & (candidates <= 1)).all().item(), "Outcome input range")
        old, extra = candidates[:, :52].contiguous(), candidates[:, 52:]
        context = self.context(torch.cat((self.board(planes), metadata), dim=1))
        embedding = torch.tanh(self.candidate[0](old) + self.outcome_candidate(extra))
        logits = (embedding * context).sum(dim=1) / self.scale
        logits = logits + (self.bias(old) + self.outcome_bias(extra)).squeeze(1)
        require(torch.isfinite(logits).all().item(), "Nonfinite outcome logits")
        return logits


def gate(baseline, outcome):
    groups = outcome["groups"]["teacherAction"]
    require(baseline["overall"]["examples"] == outcome["overall"]["examples"] == 89,
            "Gate dev denominator")
    require(groups["single-cell"]["examples"] == 13 and groups["reroll"]["examples"] == 8,
            "Gate action denominators")
    checks = {"aboveFirstLegal": outcome["overall"]["correct"] >= 48,
              "gainAtLeastFive": outcome["overall"]["correct"] >= baseline["overall"]["correct"] + 5,
              "singleCellAtLeastThree": groups["single-cell"]["correct"] >= 3,
              "rerollAtLeastSeven": groups["reroll"]["correct"] >= 7}
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "freshTestExecuted": False, "additionalTuning": False}


def load_inputs(source, augmented, execution):
    config = json.loads(CONFIG.read_bytes())
    frozen_execution = json.loads(Path(execution).read_bytes())
    require(frozen_execution["status"] == "FROZEN_BEFORE_EXECUTION" and
            frozen_execution["configSha256"] == sha256(CONFIG.read_bytes()), "Execution config freeze")
    for path, digest in frozen_execution["sourceHashes"].items():
        require(sha256((ROOT / path).read_bytes()) == digest, "Execution source changed: " + path)
    pack, protocol, manifest = load_pack(Path(source) / "train-dev", "train-dev")
    _, frozen_protocol, freeze = load_frozen(Path(source) / "freeze.json")
    require(protocol == frozen_protocol and manifest["dataSha256"] == freeze["trainDevDataSha256"] ==
            config["source"]["dataSha256"], "Original source identity")
    require(sha256((Path(source) / "models/A512/report.json").read_bytes()) ==
            freeze["models"]["A512"]["reportSha256"], "Original baseline history hash")
    validate_training_contract(protocol["training"], config["training"])
    body = Path(augmented).read_bytes()
    require(sha256(body) == frozen_execution["augmentedSha256"], "Augmented hash")
    augmented = json.loads(body)
    require(augmented["schema"] == OUTCOME_SCHEMA and augmented["sourceDataSha256"] == manifest["dataSha256"]
            and set(augmented["examples"]) == {"train", "dev"}, "Dev-only outcome schema")
    examples = {}
    candidates = 0
    for split, count in (("train", 339), ("dev", 89)):
        old = own_examples(pack, split, "A512")
        new = augmented["examples"][split]
        require(len(old) == len(new) == count, "Fixed split count")
        examples[split] = []
        for a, b in zip(old, new):
            require(set(b) == {"id", "candidateFeatures"} and a["id"] == b["id"] and
                    len(a["candidateFeatures"]) == len(b["candidateFeatures"]), "Unchanged observation order")
            for f, g in zip(a["candidateFeatures"], b["candidateFeatures"]):
                vector(g, 74)
                require(f == g[:52], "Original candidate prefix changed")
            candidates += len(b["candidateFeatures"])
            examples[split].append({**a, "candidateFeatures": b["candidateFeatures"]})
    require(candidates == augmented["transitions"] == config["limits"]["additionalDomainTransitions"] == 149597,
            "Exact transition accounting")
    return pack, protocol, manifest, examples, frozen_execution


def initial_control(pack, examples, config):
    configure(config)
    baseline = CandidateCNN(config).eval()
    outcome = OutcomeCandidateCNN(config, baseline).eval()
    count = 0
    with torch.inference_mode():
        for split in ("train", "dev"):
            for old, new in zip(pack["examples"][split], examples[split]):
                require(torch.equal(baseline(*tensors(old)), outcome(*tensors(new))),
                        "Initial logits must be bit-exact")
                count += 1
    return {"initialLogitsExact": True, "observations": count,
            "commonWeightsSha256": weights_hash(baseline),
            "parameters": {"baseline": sum(p.numel() for p in baseline.parameters()),
                           "outcome": sum(p.numel() for p in outcome.parameters())}}


def fit_outcome(train, dev, protocol, data_hash, checkpoint):
    config = protocol["training"]
    configure(config)
    torch.set_num_interop_threads(1)
    require(torch.version.cuda is None, "CPU-only build required")
    model = OutcomeCandidateCNN(config, CandidateCNN(config))
    train_inputs, dev_inputs = [tensors(e) for e in train], [tensors(e) for e in dev]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learningRate"], weight_decay=config["weightDecay"])
    generator = torch.Generator().manual_seed(config["seed"])
    history, best = [], -1
    start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator).tolist()
        total_loss = 0.0
        for offset in range(0, len(order), config["batchSize"]):
            batch = order[offset:offset + config["batchSize"]]
            optimizer.zero_grad(set_to_none=True)
            for index in batch:
                loss = torch.nn.functional.cross_entropy(model(*train_inputs[index]).unsqueeze(0),
                                                         torch.tensor([train[index]["label"]]))
                require(torch.isfinite(loss).item(), "Nonfinite outcome loss")
                total_loss += float(loss.detach())
                (loss / len(batch)).backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()),
                    "Nonfinite outcome gradient")
            optimizer.step()
        model.eval()
        dev_metrics, _ = evaluate(model, dev, dev_inputs)
        selected = dev_metrics["top1"] > best
        if selected:
            best, best_epoch = dev_metrics["top1"], epoch
            torch.save({"state_dict": model.state_dict(), "schema": OUTCOME_SCHEMA, "protocol": protocol,
                        "dataSha256": data_hash, "epoch": epoch}, checkpoint)
        history.append({"epoch": epoch, "trainLoss": total_loss / len(train), "dev": dev_metrics, "selected": selected})
        print(f"Outcome epoch {epoch}: dev {dev_metrics['correct']}/89", flush=True)
    return model, history, best_epoch, (time.perf_counter() - start) * 1000


def diagnose(model, examples, families):
    rows = []
    for e in examples:
        with torch.inference_mode():
            logits = model(*tensors(e))
            require(torch.equal(logits, model(*tensors(e))), "Repeated logits changed")
        row = score(logits, e["labels"])
        row.update(id=e["id"], fixtureId=e["fixtureId"], family=families[e["fixtureId"]],
                   sameLabel=e["labels"]["A512"] == e["labels"]["A1024"],
                   complete={name: e["teachers"][name]["result"]["search"]["searchComplete"]
                             for name in ("A512", "A1024")},
                   teacherActions={name: action_description(e["candidateKeys"][e["labels"][name]])
                                   for name in ("A512", "A1024")},
                   selected=action_description(e["candidateKeys"][row["index"]]),
                   firstLegalAction=action_description(e["candidateKeys"][0]))
        rows.append(row)
    # Report A512 only; A1024 is unchanged audit metadata, never a training target.
    summary = summarize(rows)
    summary["targets"] = {"A512": summary["targets"]["A512"]}
    return rows, summary


def verify_baseline(model, history, epoch, audit, original, saved):
    require(original["history"] == history and original["selectedEpoch"] == epoch and original["audit"] == audit,
            "Baseline history/epoch/order did not reproduce; STOP")
    require(set(model.state_dict()) == set(saved["state_dict"]) and
            all(torch.equal(v, saved["state_dict"][k]) for k, v in model.state_dict().items()),
            "Baseline weights did not reproduce; STOP")
    return {"historyExact": True, "weightsExact": True, "selectedEpochExact": True, "orderExact": True}


def run(mode, source, augmented, execution, output):
    start, before = time.perf_counter(), rss_bytes()
    pack, protocol, manifest, examples, frozen = load_inputs(source, augmented, execution)
    audit = training_audit(own_examples(pack, "train", "A512"), protocol["training"])
    execution_hash = sha256(Path(execution).read_bytes())
    if mode != "preflight":
        control = json.loads((Path(execution).parent / "preflight/report.json").read_bytes())
        require(control["status"] == "PASS" and control["executionSha256"] == execution_hash and
                control["audit"] == audit and control["control"]["initialLogitsExact"] and
                control["control"]["observations"] == 428, "Initial preflight required before training")
    if mode == "outcome":
        baseline = json.loads((Path(execution).parent / "baseline/report.json").read_bytes())
        require(baseline["status"] == "PASS" and baseline["executionSha256"] == execution_hash and
                baseline["audit"] == audit and all(baseline["baselineReproduction"].values()),
                "Reproduced baseline required before outcome training")
    if mode == "preflight":
        control = initial_control(pack, examples, protocol["training"])
        require(control["parameters"] == {"baseline": 12189, "outcome": 12915}, "Parameter counts")
        write_json(output / "report.json", {"status": "PASS", "control": control, "audit": audit,
                                           "executionSha256": execution_hash,
                                           "testLoaded": False, "trainingRuns": 0})
        return
    checkpoint = output / "checkpoint.pt"
    if mode == "baseline":
        train, dev = [own_examples(pack, split, "A512") for split in ("train", "dev")]
        model, history, epoch, train_ms, _ = fit(train, dev, protocol, manifest["dataSha256"], checkpoint)
    else:
        train, dev = examples["train"], examples["dev"]
        model, history, epoch, train_ms = fit_outcome(train, dev, protocol, frozen["augmentedSha256"], checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(saved["epoch"] == epoch and saved["schema"] == (SCHEMA if mode == "baseline" else OUTCOME_SCHEMA)
            and saved["protocol"] == protocol and saved["dataSha256"] ==
            (manifest["dataSha256"] if mode == "baseline" else frozen["augmentedSha256"]), "New checkpoint identity")
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    reproduced = None
    if mode == "baseline":
        original = json.loads((Path(source) / "models/A512/report.json").read_bytes())
        old = torch.load(Path(source) / "models/A512/checkpoint.pt", map_location="cpu", weights_only=True)
        reproduced = verify_baseline(model, history, epoch, audit, original, old)
    families = {e["fixture"]["id"]: fixture_family(e["fixture"], protocol["dataset"]["families"])
                for e in pack["episodes"]}
    raw, summary = {}, {}
    for split, es in (("train", train), ("dev", dev)):
        raw[split], summary[split] = diagnose(model, es, families)
    # Recheck the original frozen files and weights after training, without test access.
    load_frozen(Path(source) / "freeze.json")
    report = {"status": "PASS", "mode": mode, "selectedEpoch": epoch, "history": history,
              "executionSha256": execution_hash,
              "audit": audit, "baselineReproduction": reproduced, "summary": summary,
              "checkpointSha256": sha256(checkpoint.read_bytes()), "weightsSha256": weights_hash(model),
              "testLoaded": False, "testScored": False, "teacherCalls": 0, "trainingRuns": 1,
              "epochs": 12, "timings": {"trainingDevMs": train_ms, "totalMs": (time.perf_counter()-start)*1000},
              "memory": {"beforeBytes": before, "afterBytes": rss_bytes(), "peak": False},
              "environment": {"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                              "threads": torch.get_num_threads(), "interopThreads": torch.get_num_interop_threads()}}
    write_json(output / "predictions.json", raw)
    write_json(output / "report.json", report)
    print(json.dumps({"status": "PASS", "mode": mode, "selectedEpoch": epoch,
                      "dev": summary["dev"]["targets"]["A512"]["overall"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "baseline", "outcome"), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--augmented", required=True)
    parser.add_argument("--execution", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    require(output.resolve() == (Path(args.execution).resolve().parent / args.mode),
            "Exactly one output per frozen execution/mode")
    output.mkdir(parents=True, exist_ok=False)
    try:
        run(args.mode, args.source, args.augmented, args.execution, output)
    except Exception as error:
        write_json(output / "failure.json", {"status": "FAIL", "errorType": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
