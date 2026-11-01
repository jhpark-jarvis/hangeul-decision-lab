"""One approved frozen CPU evaluation, at most 320 forwards; no optimizer or training."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch
import torch
from data import require, sha256
from controlled_point_fit import score_row, validate_dataset, counted_model
from spatial import make_model, spatial_tensors, Budget
from paired import weights_hash
from train import write_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "research/experiments/frozen-point-transfer-v1.json"
LIMITS = {"fit-control": 16, "transfer": 144, "fit-repeat": 16, "transfer-repeat": 144}


class TransferLedger:
    def __init__(self, path):
        self.path, self.counts = Path(path), {k: 0 for k in LIMITS}
        with self.path.open("x", encoding="utf8"): pass

    def reserve(self, phase, step, example):
        require(phase in LIMITS and type(example) is int and step is None and sum(self.counts.values()) < 320,
                "Frozen forward limit/phase; STOP")
        current = next(k for k in LIMITS if self.counts[k] < LIMITS[k])
        require(phase == current and example == self.counts[phase], "Frozen forward order; STOP")
        row = {"call": sum(self.counts.values()) + 1, "phase": phase, "phaseCall": example + 1, "step": None, "example": example}
        with self.path.open("a", encoding="utf8", newline="\n") as stream:
            stream.write(json.dumps(row) + "\n"); stream.flush()
        self.counts[phase] += 1


def summary(rows, size):
    require(len(rows) == size and len({r["id"] for r in rows}) == size, "Separate metric denominator")
    return {"examples": size, "correct": sum(r["correct"] for r in rows),
            "meanNll": sum(r["nll"] for r in rows) / size,
            "minimumTargetMargin": min(r["margin"] for r in rows),
            "maximumTargetRank": max(r["rank"] for r in rows)}


def validate_transfer(data, fit):
    require(data["schemaVersion"] == 1 and data["id"] == "frozen-point-transfer-data-v1" and data["split"] == "controlled-transfer-only",
            "Isolated transfer schema")
    counts = data["counts"]
    require(counts["examples"] == 144 and counts["candidates"] == counts["outcomeTransitions"] == 4464 and
            counts["dfsCalls"] == 0 and counts["referenceNodeCap"] == 36864 and 0 < counts["referenceNodes"] <= 36864,
            "Transfer construction limits")
    optimized = {10 * r + (3 * r) % 10 for r in range(16)}
    indices = [i for i in range(160) if i not in optimized]
    examples = data["examples"]
    require(len(examples) == 144, "Transfer example count")
    def vector(v, n):
        return isinstance(v, list) and len(v) == n and all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in v)
    for e, index in zip(examples, indices):
        row, col = divmod(index, 10)
        require(e["id"] == f"controlled-transfer-{index}" and e["fixtureIndex"] == index and
                type(e["label"]) is int and 0 <= e["label"] < 31 and e["reference"]["complete"] is True and
                e["reference"]["unique"] is True and 0 < e["reference"]["visitedNodes"] <= 256,
                "Transfer partition/reference identity")
        obs = e["observation"]
        require(set(obs) == {"planes", "metadata"} and len(obs["planes"]) == 3 and
                all(len(p) == 16 and all(vector(r, 10) and all(v in (0, 1) for v in r) for r in p) for p in obs["planes"]) and
                vector(obs["metadata"], 140) and obs["metadata"] == fit[0]["observation"]["metadata"], "Transfer observation schema")
        board = obs["planes"][0]
        require(all(v == 0 for p in obs["planes"][1:] for r in p for v in r) and
                all(board[r][c] == int(c != col) if r == row else board[r][c] == int(c not in ((3*r)%10, (3*r+1)%10))
                    for r in range(16) for c in range(10)), "Transfer controlled board geometry")
        keys = [["single-cell", r, c] for r in range(16) for c in range(10) if not board[r][c]]
        require(len(keys) == 31 and [json.loads(k) for k in e["candidateKeys"]] == keys and keys[e["label"]] == ["single-cell", row, col]
                and len(e["candidateFeatures"]) == len(e["footprints"]) == 31, "Transfer ordered legal mask/target")
        for i, (key, features, bits) in enumerate(zip(keys, e["candidateFeatures"], e["footprints"])):
            _, r, c = key
            require(vector(features, 74) and features[:3] == [0,1,0] and features[25:27] == [r/15, c/9] and
                    features[52] == (1/16 if i == e["label"] else 0) and
                    bits == "".join("1" if k == 10*r+c else "0" for k in range(160)), "Transfer features/footprint")
    require(sum(e["reference"]["visitedNodes"] for e in examples) == counts["referenceNodes"], "Reference sum")
    return examples


@contextmanager
def no_training():
    def blocked(*args, **kwargs): raise ValueError("Training/optimizer/extra checkpoint load forbidden; STOP")
    original_train = torch.nn.Module.train
    def eval_only(module, mode=True):
        require(mode is False, "Training mode forbidden; STOP")
        return original_train(module, False)
    with patch("torch.optim.Optimizer.__init__", blocked), patch("torch.Tensor.backward", blocked), \
            patch("torch.autograd.backward", blocked), patch("torch.load", blocked), patch("torch.nn.Module.train", eval_only):
        yield


def run(output):
    started = time.perf_counter()
    monitor = Budget(600, 2147483648, output, started); monitor.sample("start")
    config, frozen = json.loads(CONFIG.read_bytes()), json.loads((output / "execution.json").read_bytes())
    proposal = ROOT / config["proposalPath"]
    p = json.loads(proposal.read_bytes())
    require(config["approval"] == "USER_APPROVED" and config["status"] == frozen["status"] == "FROZEN_BEFORE_EXECUTION" and
            sha256(CONFIG.read_bytes()) == frozen["configSha256"] and sha256(proposal.read_bytes()) == config["proposalSha256"] and
            config["contract"] == {k:p[k] for k in config["contract"]}, "Approved frozen transfer contract")
    require(config["contract"]["forwardBudget"]["totalTopLevelMax"] == 320 and config["boundaries"]["trainingRuns"] == 0,
            "Fixed evaluation budget")
    for path, digest in frozen["sourceHashes"].items(): require(sha256((ROOT / path).read_bytes()) == digest, "Frozen source drift: " + path)
    for name, digest in frozen["inputHashes"].items(): require(sha256((output / name).read_bytes()) == digest, "Frozen input drift: " + name)
    fit = validate_dataset(json.loads((output / "fit-dataset.json").read_bytes()))
    transfer = validate_transfer(json.loads((output / "transfer-dataset.json").read_bytes()), fit)
    checkpoint = config["contract"]["checkpoint"]
    require(sha256((output / "fit-dataset.json").read_bytes()) == checkpoint["fitDatasetSha256"] and
            sha256((output / "fit-expected.json").read_bytes()) == checkpoint["fitPredictionsSha256"], "Original fit identity")
    weight_body = (Path(frozen["originalRunPath"]) / "candidate/final-weights.pt").read_bytes()
    require(sha256(weight_body) == checkpoint["serializationSha256"] and torch.version.cuda is None, "Frozen CPU checkpoint identity")
    torch.set_num_interop_threads(1)
    fit_inputs, transfer_inputs = [spatial_tensors(e) for e in fit], [spatial_tensors(e) for e in transfer]
    monitor.sample("validated")
    original_load = torch.load
    with no_training():
        write_json(output / "model-intent.json", {"topLevelModels":1, "trainingRuns":0})
        model = make_model(config["modelConfig"], "spatial")
        require(sum(p.numel() for p in model.parameters()) == 13292, "Same spatial model")
        write_json(output / "load-intent.json", {"checkpointLoads":1, "weightsOnly":True, "serializationSha256":checkpoint["serializationSha256"]})
        state_dict = original_load(io.BytesIO(weight_body), map_location="cpu", weights_only=True)
        require(isinstance(state_dict, dict) and all(isinstance(v, torch.Tensor) and v.device.type == "cpu" and
                v.dtype == torch.float32 and torch.isfinite(v).all().item() for v in state_dict.values()), "Finite CPU state dict")
        model.load_state_dict(state_dict, strict=True)
        require(weights_hash(model) == checkpoint["logicalWeightsSha256"], "Loaded logical weights mismatch")
        model.requires_grad_(False); model.eval()
        ledger = TransferLedger(output / "forward-count.jsonl")
        state = {"phase":"fit-control", "step":None, "example":0}
        def evaluate(phase, examples, inputs):
            state.update(phase=phase, step=None)
            rows = []
            with torch.inference_mode():
                for i, (e, inp) in enumerate(zip(examples, inputs)):
                    monitor.sample(phase + "-before"); state["example"] = i
                    rows.append(score_row(model(*inp), e["label"], e["id"]))
                    monitor.sample(phase + "-after")
            write_json(output / (phase + ".json"), rows)
            return rows
        with counted_model(model, ledger, state):
            fit_rows = evaluate("fit-control", fit, fit_inputs)
            require((output / "fit-control.json").read_bytes() == (output / "fit-expected.json").read_bytes(), "Fit control mismatch; STOP before transfer")
            transfer_rows = evaluate("transfer", transfer, transfer_inputs)
            evaluate("fit-repeat", fit, fit_inputs); evaluate("transfer-repeat", transfer, transfer_inputs)
        for a,b in [("fit-control","fit-repeat"),("transfer","transfer-repeat")]:
            require((output/(a+".json")).read_bytes() == (output/(b+".json")).read_bytes(), "Frozen repeat drift")
        require(ledger.counts == LIMITS and weights_hash(model) == checkpoint["logicalWeightsSha256"], "Frozen weights or accounting drift")
    fit_summary, transfer_summary = summary(fit_rows,16), summary(transfer_rows,144)
    checks = {"allCorrect":transfer_summary["correct"]==144,"meanNllAtMost005":transfer_summary["meanNll"]<=0.05,
              "everyTargetMarginPositive":transfer_summary["minimumTargetMargin"]>0}
    result = {"technicalStatus":"PASS","fit":fit_summary,"transfer":transfer_summary,
              "transferGate":{"status":"PASS" if all(checks.values()) else "FAIL","checks":checks},
              "topLevelModels":1,"checkpointLoads":1,"topLevelForwards":320,"trainingRuns":0,"optimizerUpdates":0}
    write_json(output / "summary.json",result)
    for path,digest in frozen["sourceHashes"].items(): require(sha256((ROOT/path).read_bytes())==digest,"Source changed during evaluation")
    monitor.sample("finished")
    names = ["fit-control.json","transfer.json","fit-repeat.json","transfer-repeat.json","forward-count.jsonl","summary.json","model-intent.json","load-intent.json"]
    write_json(output/"child-report.json",{"status":"PASS","atUtc":datetime.now(timezone.utc).isoformat(),"executionSha256":sha256((output/"execution.json").read_bytes()),"hashes":{n:sha256((output/n).read_bytes()) for n in names},"summary":result,"forwardCounts":ledger.counts,"loadedLogicalWeightsSha256":checkpoint["logicalWeightsSha256"],"finalLogicalWeightsSha256":checkpoint["logicalWeightsSha256"],"budget":monitor.report(),"environment":{"python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,"threads":torch.get_num_threads(),"interopThreads":torch.get_num_interop_threads()}})


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",required=True)
    output=Path(parser.parse_args().output).resolve()
    require(output.is_relative_to(ROOT/".research-output") and (output/"execution.json").is_file() and
            not (output/"model-intent.json").exists(),"New owned frozen output; no resume/retry")
    try: run(output)
    except BaseException as error:
        if not (output/"child-failure.json").exists(): write_json(output/"child-failure.json",{"status":"FAIL","error":str(error),"retry":False})
        raise
