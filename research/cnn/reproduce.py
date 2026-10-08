"""Rerun train/dev only and compare exact weights/history; never score test."""
import argparse
from pathlib import Path
import torch
from data import SCHEMA, load_dataset, require, sha256
from train import fit, write_json
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output, reference = Path(args.output), Path(args.reference)
    output.mkdir(parents=True, exist_ok=False)
    try:
        pack, protocol, manifest = load_dataset(args.dataset)
        original = torch.load(reference / "checkpoint.pt", map_location="cpu", weights_only=True)
        report = json.loads((reference / "report.json").read_text(encoding="utf-8"))
        require(original["schema"] == SCHEMA and original["protocol"] == protocol and original["dataSha256"] == manifest["dataSha256"], "Reference identity")
        require(sha256((reference / "checkpoint.pt").read_bytes()) == report["checkpointSha256"], "Reference checkpoint hash")
        _, history, epoch, _, _ = fit(pack["examples"]["train"], pack["examples"]["dev"], protocol,
                                     manifest["dataSha256"], output / "checkpoint.pt")
        repeated = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
        require(epoch == original["epoch"] == report["selectedEpoch"], "Selected epoch changed")
        require(history == report["history"], "Training/dev history changed")
        require(original["state_dict"].keys() == repeated["state_dict"].keys(), "Weight structure changed")
        require(all(torch.equal(value, repeated["state_dict"][key]) for key, value in original["state_dict"].items()), "Training weights changed")
        result = {"status": "PASS", "dataSha256": manifest["dataSha256"], "selectedEpoch": epoch,
                  "exactWeightsEqual": True, "exactTrainDevHistoryEqual": True, "testScored": False,
                  "sourceHashes": {p.name: sha256(p.read_bytes()) for p in Path(__file__).parent.glob("*.py")}}
        write_json(output / "reproduction.json", result)
        print(json.dumps(result, indent=2))
    except Exception as error:
        write_json(output / "failure.json", {"status": "FAIL", "errorType": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
