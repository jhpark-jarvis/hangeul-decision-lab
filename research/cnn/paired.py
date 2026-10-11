"""Dev-only paired training and post-freeze evaluation; domain remains in TS."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
from data import SCHEMA, compact, require, sha256, vector, integer
from model import CandidateCNN, configure, predict, tensors
from train import fit, write_json, rss_bytes, metrics

NAMES = ("A512", "A1024")
ROOT = Path(__file__).resolve().parents[2]


def weights_hash(model):
    return sha256(compact({k: v.detach().tolist() for k, v in model.state_dict().items()}))


def validate_pack(pack, protocol, stage):
    require(pack["status"] == "PASS" and pack["schema"] == SCHEMA and pack["stage"] == stage,
            "Paired stage/schema")
    require(pack["protocolId"] == protocol["id"] and protocol["status"] == "FROZEN_PAIRED_EXPERIMENT",
            "Paired protocol identity")
    splits = ("train", "dev") if stage == "train-dev" else ("test",)
    for split in ("train", "dev", "test"):
        examples = pack["examples"][split]
        require(bool(examples) == (split in splits), "Empty or premature paired split")
        seen = set()
        seeds = protocol["dataset"]["reservedTestSeeds" if split == "test" else split + "Seeds"]
        for e in examples:
            require(e["split"] == split and e["seed"] in seeds and e["id"] not in seen, "Paired split/id")
            seen.add(e["id"])
            require(integer(e["actionIndex"], 0, 5), "Action horizon")
            obs = e["observation"]
            require(set(obs) == {"planes", "metadata"} and len(obs["planes"]) == 3, "Observation schema")
            for plane in obs["planes"]:
                require(len(plane) == 16, "Plane rows")
                for row in plane:
                    vector(row, 10, True)
            vector(obs["metadata"], 140)
            keys, features = e["candidateKeys"], e["candidateFeatures"]
            require(1 <= len(keys) <= 4096 and len(set(keys)) == len(keys) and
                    all(isinstance(k, str) for k in keys) and len(features) == len(keys), "Candidates")
            for f in features:
                vector(f, 52)
            for name in NAMES:
                require(integer(e["labels"][name], 0, len(keys) - 1), "Paired label index")
                t = e["teachers"][name]
                require(t["result"]["search"]["maxNodes"] == protocol["teachers"][name]["maxNodes"],
                        "Teacher budget")
    return pack


def load_pack(directory, stage):
    path = Path(directory)
    body = (path / "dataset.json").read_bytes()
    proto_body = (path / "protocol.json").read_bytes()
    protocol = json.loads(proto_body)
    manifest = json.loads((path / "manifest.json").read_bytes())
    require(sha256(body) == manifest["dataSha256"] and sha256(proto_body) == manifest["protocolSha256"],
            "Paired data/protocol hash")
    require(manifest["status"] == "PASS" and manifest["stage"] == stage, "Manifest stage")
    return validate_pack(json.loads(body), protocol, stage), protocol, manifest


def own_examples(pack, split, name):
    return [{**e, "label": e["labels"][name]} for e in pack["examples"][split]]


def training_audit(examples, config):
    configure(config)
    initial = weights_hash(CandidateCNN(config))
    generator = torch.Generator().manual_seed(config["seed"])
    order = [torch.randperm(len(examples), generator=generator).tolist() for _ in range(config["epochs"])]
    return {"initialWeightsSha256": initial, "sampleOrderSha256": sha256(compact(order)),
            "trainingIdsSha256": sha256(compact([e["id"] for e in examples]))}


def train_pair(dataset, name, output):
    start, before = time.perf_counter(), rss_bytes()
    pack, protocol, manifest = load_pack(dataset, "train-dev")
    train, dev = own_examples(pack, "train", name), own_examples(pack, "dev", name)
    audit = training_audit(train, protocol["training"])
    checkpoint = output / "checkpoint.pt"
    _, history, epoch, train_ms, _ = fit(train, dev, protocol, manifest["dataSha256"], checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(saved["epoch"] == epoch and saved["schema"] == SCHEMA and
            saved["protocol"] == protocol and saved["dataSha256"] == manifest["dataSha256"], "Saved identity")
    report = {"status": "PASS", "teacher": name, "protocolSha256": manifest["protocolSha256"],
              "dataSha256": manifest["dataSha256"], "checkpointSha256": sha256(checkpoint.read_bytes()),
              "selectedEpoch": epoch, "selection": "own-teacher-highest-dev-top1-first-tie",
              "history": history, "audit": audit, "testLoaded": False, "testScored": False,
              "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
              "timings": {"trainMs": train_ms, "totalMs": (time.perf_counter()-start)*1000},
              "memory": {"beforeBytes": before, "afterBytes": rss_bytes(), "peak": False}}
    write_json(output / "report.json", report)
    print(json.dumps({"status": "PASS", "teacher": name, "selectedEpoch": epoch,
                      "dev": history[epoch-1]["dev"]}), flush=True)


def load_frozen(freeze_path):
    freeze_path = Path(freeze_path).resolve()
    freeze = json.loads(freeze_path.read_bytes())
    require(freeze["status"] == "FROZEN_FOR_TEST", "Both checkpoints must be frozen")
    protocol = json.loads((freeze_path.parent / "protocol.json").read_bytes())
    require(sha256((freeze_path.parent / "protocol.json").read_bytes()) == freeze["protocolSha256"],
            "Frozen protocol changed")
    for relative, digest in freeze["sourceHashes"].items():
        require(sha256((ROOT / relative).read_bytes()) == digest, "Frozen source changed: " + relative)
    configure(protocol["training"])
    models = {}
    for name in NAMES:
        record = freeze["models"][name]
        path = freeze_path.parent / record["checkpoint"]
        require(sha256(path.read_bytes()) == record["sha256"], "Frozen checkpoint changed")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        require(saved["protocol"] == protocol and saved["schema"] == SCHEMA and
                saved["dataSha256"] == freeze["trainDevDataSha256"] and saved["epoch"] == record["selectedEpoch"],
                "Frozen model identity")
        model = CandidateCNN(protocol["training"])
        model.load_state_dict(saved["state_dict"], strict=True)
        model.eval()
        models[name] = model
    return models, protocol, freeze


def evaluate_pair(dataset, freeze_path, output):
    models, protocol, freeze = load_frozen(freeze_path)
    pack, test_protocol, manifest = load_pack(dataset, "test")
    require(protocol == test_protocol and manifest["freezeSha256"] == sha256(Path(freeze_path).read_bytes()),
            "Test freeze identity")
    pairs = pack["examples"]["test"]
    records, cross = {}, {}
    for name in NAMES:
        indices = [predict(models[name], tensors(e)) for e in pairs]
        require(all(0 <= i < len(e["candidateKeys"]) for e, i in zip(pairs, indices)), "Illegal prediction")
        records[name] = [{"id": e["id"], "index": i, "key": e["candidateKeys"][i]} for e, i in zip(pairs, indices)]
        cross[name] = {target: metrics(own_examples(pack, "test", target), indices) for target in NAMES}
    first = {target: metrics(own_examples(pack, "test", target), [0]*len(pairs)) for target in NAMES}
    write_json(output / "predictions.json", records)
    write_json(output / "report.json", {"status": "PASS", "freezeSha256": sha256(Path(freeze_path).read_bytes()),
                                        "dataSha256": manifest["dataSha256"], "crossTeacher": cross,
                                        "firstLegal": first, "finitePredictions": True,
                                        "postTestTuning": False, "models": freeze["models"]})
    print(json.dumps({"status": "PASS", "crossTeacher": cross}), flush=True)


def worker(freeze_path):
    models, _, _ = load_frozen(freeze_path)
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("stop"):
            break
        require(request["model"] in NAMES and 1 <= len(request["examples"]) <= 8, "Inference batch")
        start = time.perf_counter()
        indices = [predict(models[request["model"]], tensors(e)) for e in request["examples"]]
        print(json.dumps({"id": request["id"], "indices": indices,
                          "inferenceMs": (time.perf_counter()-start)*1000}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("train", "evaluate", "worker"))
    parser.add_argument("--dataset")
    parser.add_argument("--teacher", choices=NAMES)
    parser.add_argument("--freeze")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.mode == "worker":
        require(args.freeze and not args.output, "Worker arguments")
        worker(args.freeze)
        return
    require(args.dataset and args.output, "Dataset/output required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        if args.mode == "train":
            require(args.teacher and not args.freeze, "Dev-only arguments")
            train_pair(args.dataset, args.teacher, output)
        else:
            require(args.freeze and not args.teacher, "Frozen evaluation arguments")
            evaluate_pair(args.dataset, args.freeze, output)
    except Exception as error:
        write_json(output / "failure.json", {"status": "FAIL", "error": str(error), "errorType": type(error).__name__})
        raise


if __name__ == "__main__":
    main()
