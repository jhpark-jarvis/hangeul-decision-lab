"""Approved P118 adapter: exact opposing anchors; immutable V1 training loop."""
import time
PROCESS_STARTED = time.perf_counter()
import argparse
import json
import math
from pathlib import Path
import struct
from unittest.mock import patch
import board_context_fit as original
from data import require, sha256
from train import write_json

CONFIG = original.ROOT / "research/experiments/paired-target-identity-v1.json"
PHASES = ("initial", "final1", "final2")


def fp32_bits(value):
    require(type(value) in (int, float) and math.isfinite(value), "Finite score")
    bits = struct.pack("<f", value)
    require(struct.unpack("<f", bits)[0] == value, "Native FP32 score only")
    return bits


def ulp_distance(a, b):
    def order(v):
        n = struct.unpack("<I", fp32_bits(v))[0]
        return 0xffffffff - n if n >= 0x80000000 else n + 0x80000000
    return abs(order(a) - order(b))


def paired_identity(rows, keys):
    require(len(rows) > 0 and len(rows) % 2 == 0 and len(rows) == len(keys), "Complete pairs")
    groups = {}
    for i, row in enumerate(rows):
        require(len(row["logits"]) == len(keys[i]) and len(set(keys[i])) == len(keys[i]), "Unique ordered inputs")
        require(type(row["label"]) is int and 0 <= row["label"] < len(keys[i]), "Valid target")
        for j, value in enumerate(row["logits"]):
            fp32_bits(value)
            groups.setdefault(keys[i][j], []).append((i, j, value))
    for i in range(0, len(rows), 2):
        a, b, ka, kb = rows[i], rows[i + 1], keys[i], keys[i + 1]
        target_a, target_b = ka[a["label"]], kb[b["label"]]
        require(target_a != target_b and target_a in kb and target_b in ka, "Opposing shared target anchors")
        a_in_b, b_in_a = kb.index(target_a), ka.index(target_b)
        require((a["label"] < b_in_a) == (a_in_b < b["label"]), "Same first-index tie priority")
        require(fp32_bits(a["logits"][a["label"]]) == fp32_bits(b["logits"][a_in_b]), "Target A differs; STOP")
        require(fp32_bits(a["logits"][b_in_a]) == fp32_bits(b["logits"][b["label"]]), "Target B differs; STOP")
    mismatches, max_gap, max_ulp, same_index, different_index, targets = 0, 0, 0, 0, 0, 0
    for g in groups.values():
        if all(fp32_bits(v) == fp32_bits(g[0][2]) for _, _, v in g): continue
        mismatches += 1
        max_gap = max(max_gap, max(v for _, _, v in g) - min(v for _, _, v in g))
        max_ulp = max(max_ulp, max(ulp_distance(a[2], b[2]) for a in g for b in g))
        if len({j for _, j, _ in g}) == 1: same_index += 1
        else: different_index += 1
        targets += sum(rows[i]["label"] == j for i, j, _ in g)
    pairs = len(rows) // 2
    return {"anchors": {"pairs": pairs, "anchorComparisons": pairs * 2, "scoreOccurrences": pairs * 4,
                        "exactTargetAnchorScores": True, "sameTiePriority": True, "nativeCorrectUpperBound": pairs},
            "globalDrift": {"candidateRows": sum(len(r["logits"]) for r in rows), "uniqueInputs": len(groups),
                            "repeatedInputGroups": sum(len(g) > 1 for g in groups.values()), "mismatchGroups": mismatches,
                            "maximumAbsoluteGap": max_gap, "maximumUlpGap": max_ulp, "sameIndexMismatchGroups": same_index,
                            "differentIndexMismatchGroups": different_index}, "mismatchTargetOccurrences": targets}


class IdentityRecorder:
    def __init__(self, output):
        self.output, self.calls = Path(output), 0

    def __call__(self, rows, keys):
        require(self.calls < 4 and len(rows) == len(keys) == 32 and all(len(k) == 32 for k in keys), "Fixed identity callback schedule")
        result = paired_identity(rows, keys)
        if self.calls < 3:
            write_json(self.output / (PHASES[self.calls] + "-identity.json"), result)
        else:
            require(result == json.loads((self.output / "final1-identity.json").read_bytes()), "Report identity matches final1")
        self.calls += 1
        return result


def run(base, arm):
    out = base / arm
    recorder = IdentityRecorder(out)
    require(json.loads(CONFIG.read_bytes())["id"] == "paired-target-identity-v1", "Separate approved execution config")
    # The patch replaces the numerical acceptance callback only, not the model,
    # loss, tensors, seed, optimizer, update schedule, or pre-call ledger.
    with patch.object(original, "CONFIG", CONFIG), patch.object(original, "native_identity", recorder), patch.object(original, "PROCESS_STARTED", PROCESS_STARTED):
        original.run(base, arm)
    require(recorder.calls == (4 if arm == "control" else 0), "No extra identity callback")
    names = [p + "-identity.json" for p in PHASES] if arm == "control" else []
    write_json(out / "paired-report.json", {"status": "PASS", "arm": arm,
        "baseReportSha256": sha256((out / "report.json").read_bytes()),
        "identityCalls": recorder.calls, "identityRule": "PAIRED_TARGET_EXACT_TOLERANCE_ZERO",
        "phaseHashes": {p: sha256((out / p).read_bytes()) for p in names},
        "additionalModelLoadForwardTraining": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--arm", choices=["control", "candidate"], required=True)
    args = parser.parse_args()
    base = Path(args.run).resolve()
    out = base / args.arm
    require(base.is_relative_to(original.ROOT / ".research-output") and out.is_dir() and not (out / "model-intent.json").exists(), "Fresh frozen arm; no retry")
    try: run(base, args.arm)
    except BaseException as error:
        if not (out / "failure.json").exists(): write_json(out / "failure.json", {"status": "FAIL", "error": str(error), "retry": False})
        raise
