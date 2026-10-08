"""Bounded CPU teacher imitation. Dev selects; test is evaluated afterwards."""
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import torch
from data import SCHEMA, load_dataset, require, sha256
from model import CandidateCNN, configure, predict, tensors


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def rss_bytes():
    # Windows working set at the observation instant, not a sampled peak.
    if os.name != "nt":
        return None
    from ctypes import wintypes

    class Memory(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                                               "PagefileUsage", "PeakPagefileUsage")]
    kernel, psapi = ctypes.WinDLL("kernel32", use_last_error=True), ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    info = Memory()
    info.cb = ctypes.sizeof(info)
    return info.WorkingSetSize if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(info), info.cb) else None


def metrics(examples, indices):
    correct = sum(i == e["label"] for e, i in zip(examples, indices))
    baseline = sum(e["label"] == 0 for e in examples)
    groups = {}
    for e, i in zip(examples, indices):
        groups.setdefault(e["fixtureId"], []).append(int(i == e["label"]))
    return {"examples": len(examples), "correct": correct,
            "top1": correct / len(examples) if examples else None,
            "firstLegalTop1": baseline / len(examples) if examples else None,
            "fixtures": len(groups),
            "fixtureMacroTop1": sum(sum(v) / len(v) for v in groups.values()) / len(groups) if groups else None}


def evaluate(model, examples, inputs):
    indices = [predict(model, inp) for inp in inputs]
    return metrics(examples, indices), indices


def latency_summary(samples):
    ordered = sorted(samples)
    return {"count": len(samples), "meanMs": sum(samples) / len(samples),
            "p95Ms": ordered[math.ceil(0.95 * len(ordered)) - 1], "maxMs": ordered[-1]}


def fit(train, dev, protocol, data_hash, checkpoint):
    """This function has no test split. It selects exclusively on dev."""
    config = protocol["training"]
    configure(config)
    torch.set_num_interop_threads(1)
    require(torch.version.cuda is None, "This pilot requires the CPU-only build")
    model = CandidateCNN(config)
    train_inputs, dev_inputs = [tensors(e) for e in train], [tensors(e) for e in dev]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learningRate"], weight_decay=config["weightDecay"])
    generator = torch.Generator().manual_seed(config["seed"])
    history, best = [], -1
    train_start = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator).tolist()
        total_loss = 0.0
        for offset in range(0, len(order), config["batchSize"]):
            batch = order[offset:offset + config["batchSize"]]
            optimizer.zero_grad(set_to_none=True)
            for index in batch:
                loss = torch.nn.functional.cross_entropy(model(*train_inputs[index]).unsqueeze(0), torch.tensor([train[index]["label"]]))
                require(torch.isfinite(loss).item(), "Nonfinite training loss")
                total_loss += float(loss.detach())
                (loss / len(batch)).backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()), "Nonfinite gradient")
            optimizer.step()
        model.eval()
        dev_metrics, _ = evaluate(model, dev, dev_inputs)
        selected = dev_metrics["top1"] > best
        if selected:
            best = dev_metrics["top1"]
            best_epoch = epoch
            # Own generated state dict only. Loading always uses weights_only=True.
            torch.save({"state_dict": model.state_dict(), "schema": SCHEMA, "protocol": protocol,
                        "dataSha256": data_hash, "epoch": epoch}, checkpoint)
        history.append({"epoch": epoch, "trainLoss": total_loss / len(train), "dev": dev_metrics, "selected": selected})
        print(f"Epoch {epoch}/{config['epochs']}: loss={history[-1]['trainLoss']:.4f}, dev top1={best:.4f} (best)", flush=True)
    train_ms = (time.perf_counter() - train_start) * 1000
    return model, history, best_epoch, train_ms, dev_inputs


def run(directory, output):
    start, before = time.perf_counter(), rss_bytes()
    pack, protocol, manifest = load_dataset(directory)
    load_ms = (time.perf_counter() - start) * 1000
    checkpoint = Path(output) / "checkpoint.pt"
    model, history, best_epoch, train_ms, dev_inputs = fit(
        pack["examples"]["train"], pack["examples"]["dev"], protocol, manifest["dataSha256"], checkpoint)
    restore_start = time.perf_counter()
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(saved["schema"] == SCHEMA and saved["protocol"] == protocol and saved["dataSha256"] == manifest["dataSha256"], "Checkpoint identity")
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    checkpoint_load_ms = (time.perf_counter() - restore_start) * 1000
    # First test evaluation occurs only after all training/dev selection is done.
    test = pack["examples"]["test"]
    encode_start = time.perf_counter()
    test_inputs = [tensors(e) for e in test]
    test_tensor_ms = (time.perf_counter() - encode_start) * 1000
    warmup_start = time.perf_counter()
    for _ in range(5):
        predict(model, dev_inputs[0])
    warmup_ms = (time.perf_counter() - warmup_start) * 1000
    samples, whole_samples, indices, records = [], [], [], []
    for example, inp in zip(test, test_inputs):
        pure_start = time.perf_counter()
        index = predict(model, inp)
        elapsed = (time.perf_counter() - pure_start) * 1000
        require(0 <= index < len(example["candidateKeys"]), "Illegal prediction index")
        indices.append(index)
        records.append({"id": example["id"], "index": index, "key": example["candidateKeys"][index], "inferenceMs": elapsed})
        samples.append(elapsed)
    # Fixed 3 repeats measure inference; they never select/tune the model.
    for _ in range(2):
        for inp, expected in zip(test_inputs, indices):
            pure_start = time.perf_counter()
            require(predict(model, inp) == expected, "Nondeterministic inference")
            samples.append((time.perf_counter() - pure_start) * 1000)
    for example, expected in zip(test, indices):
        whole_start = time.perf_counter()
        actual = predict(model, tensors(example))
        require(actual == expected, "Encoding inference mismatch")
        _ = example["candidateKeys"][actual]
        whole_samples.append((time.perf_counter() - whole_start) * 1000)
    test_metrics = metrics(test, indices)
    subgroups = {}
    for complete in (True, False):
        pairs = [(e, i) for e, i in zip(test, indices) if e["teacher"]["search"]["searchComplete"] is complete]
        subgroups["complete" if complete else "incomplete"] = metrics([e for e, _ in pairs], [i for _, i in pairs])
    sources = {str(p.relative_to(Path(__file__).parent)): sha256(p.read_bytes())
               for p in sorted(Path(__file__).parent.glob("*")) if p.is_file() and p.suffix in (".py", ".txt", ".json")}
    report = {"status": "PASS", "protocol": protocol, "dataSha256": manifest["dataSha256"],
              "checkpointSha256": sha256(checkpoint.read_bytes()), "sourceHashes": sources,
              "environment": {"python": sys.version, "torch": torch.__version__, "platform": platform.platform(),
                              "cudaBuild": torch.version.cuda, "threads": torch.get_num_threads(), "interopThreads": torch.get_num_interop_threads(),
                              "deterministicAlgorithms": torch.are_deterministic_algorithms_enabled()},
              "parameters": sum(p.numel() for p in model.parameters()), "history": history,
              "selectedEpoch": best_epoch, "selection": "highest-dev-top1-first-tie", "test": test_metrics,
              "teacherSubgroups": subgroups, "counts": manifest["counts"], "skips": len(pack["skips"]),
              "timings": {"dataLoadValidationMs": load_ms, "trainingDevMs": train_ms,
                          "checkpointLoadMs": checkpoint_load_ms, "testTensorConstructionMs": test_tensor_ms,
                          "warmupMs": warmup_ms, "pureInference": latency_summary(samples),
                          "tensorToKey": latency_summary(whole_samples), "totalMs": (time.perf_counter() - start) * 1000},
              "memory": {"definition": "Windows process working set before data load / after evaluation; not peak",
                         "beforeBytes": before, "afterBytes": rss_bytes()},
              "finitePredictions": True, "repeatedPredictionsEqual": True,
              "domainPredictionValidation": "NOT_RUN: execute scripts/research/verify-imitation.mjs separately",
              "limits": ["48 synthetic episodes; teacher-state imitation only", "bounded DFS is not an optimal oracle",
                         "correlated states within fixture; no statistical superiority or survival conclusion",
                         "no real-game, closed-loop learned-policy episodes, RL or application integration",
                         "pure inference excludes JSON, tensor encoding, domain candidates, IPC and UI"]}
    write_json(Path(output) / "predictions.json", {"dataSha256": manifest["dataSha256"], "records": records})
    write_json(Path(output) / "report.json", report)
    print(json.dumps({"status": report["status"], "selectedEpoch": best_epoch, "test": test_metrics, "timings": report["timings"]}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    # Never overwrite an experiment (including a failed one).
    output.mkdir(parents=True, exist_ok=False)
    try:
        run(args.dataset, output)
    except Exception as error:
        write_json(output / "failure.json", {"status": "FAIL", "errorType": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
