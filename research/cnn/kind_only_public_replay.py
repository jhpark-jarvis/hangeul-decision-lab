"""Independent public saved-result reader with an explicit private frozen-config input.

Only the non-computational authorization text may differ in the public export.
Every original manifest hash, including the supplied config bytes, remains exact.
No original runner/source/manifest is rewritten; no model is constructed or loaded.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import torch
import kind_only as runner
import kind_only_replay as original_reader
from data import require, sha256, compact
from train import write_json
from point_objective_replay import readonly_data

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = "research/experiments/kind-only-objective-v1.json"


def require_export_projection(original, exported):
    require(isinstance(original, dict) and isinstance(exported, dict) and
            isinstance(original.get("authorization"), str) and isinstance(exported.get("authorization"), str), "Explicit authorization metadata")
    require({k:v for k,v in original.items() if k!="authorization"} ==
            {k:v for k,v in exported.items() if k!="authorization"}, "Only human authorization text may differ; computation contract exact")


def manifest_input(path, frozen_config):
    return Path(frozen_config) if path == CONFIG_PATH else ROOT/path


def load_with_frozen_config(source, original, experiment, execution, frozen_config):
    c, x = json.loads(Path(frozen_config).read_bytes()), json.loads(Path(execution).read_bytes())
    require(c["approval"] == "USER_APPROVED" and c["status"] == x["status"] == "FROZEN_BEFORE_EXECUTION" and
            x["configSha256"] == sha256(Path(frozen_config).read_bytes()), "Approved execution identity")
    exported = json.loads((ROOT / CONFIG_PATH).read_bytes())
    require_export_projection(c, exported)
    proposal = ROOT / "research/experiments/kind-only-objective-proposal-v1.json"
    p = json.loads(proposal.read_bytes())
    require(sha256(proposal.read_bytes()) == c["proposalSha256"] and all(c[k] == p[k] for k in
            ("source", "model", "objective", "training", "forwardBudget", "resources", "screening", "inference", "priorControlHashes")), "Exact approved proposal contract")
    require(c["forwardBudget"]["totalTopLevelMax"] == 12840 and c["training"]["runs"] == 2 and c["training"]["epochs"] == 12 and
            c["objective"]["kindWeight"] == 0.5 and c["objective"]["pointWeight"] == 0 and c["resources"]["perChildWallSeconds"] == 600 and
            c["resources"]["totalSupervisorWallSeconds"] == 1860 and c["resources"]["sampledWorkingSetBytes"] == 2147483648, "Fixed loss and budgets")
    require(x["sourceHashes"][CONFIG_PATH] == x["configSha256"], "Original config source identity")
    for path, digest in x["sourceHashes"].items():
        actual = manifest_input(path, frozen_config)
        require(sha256(actual.read_bytes()) == digest, "Frozen source changed: "+path)
    for path, digest in c["source"]["spatialHashes"].items(): require(sha256((Path(experiment)/path).read_bytes()) == digest, "Original spatial input changed: "+path)
    require(set(x["previousControlPaths"]) == set(c["priorControlHashes"]) == {"combined", "pointOnly"}, "Both completed flat controls required")
    for arm, previous_path in x["previousControlPaths"].items():
        previous = Path(previous_path)
        for path, digest in c["priorControlHashes"][arm].items():
            require(sha256((previous/path).read_bytes()) == digest, "Previous flat control input changed: "+arm+"/"+path)
    _, _, pack, protocol, examples = readonly_data(source, original, Path(experiment)/"execution.json")
    for k in ("seed", "epochs", "batchSize", "threads", "learningRate", "weightDecay"):
        require(c["training"][k] == protocol["training"][k], "Original hyperparameter")
    for split, count in (("train",339),("dev",89)):
        require(len(examples[split]) == count, "Original split counts")
        for e in examples[split]: runner.kind_indices(e)
    old = json.loads((Path(experiment)/"spatial/report.json").read_bytes())
    generator = torch.Generator().manual_seed(protocol["training"]["seed"])
    order = [torch.randperm(339,generator=generator).tolist() for _ in range(12)]
    audit = {**old["audit"], "sampleOrderSha256": sha256(compact(order)), "trainingIdsSha256": sha256(compact([e["id"] for e in examples["train"]]))}
    require(audit == old["audit"], "Original explicit shuffle/observation identity")
    return c, x, examples, protocol, pack, old, order, audit



@contextmanager
def config_input_adapter(path):
    original = runner.load
    try:
        runner.load = lambda source, old, experiment, execution: load_with_frozen_config(source, old, experiment, execution, path)
        yield
    finally:
        runner.load = original


def replay(source, original, experiment, execution, recorded, frozen_config, output):
    frozen_config = Path(frozen_config).resolve()
    with config_input_adapter(frozen_config):
        original_reader.replay(source, original, experiment, execution, recorded, output)
    report = json.loads((Path(output)/"report.json").read_bytes())
    report.update(publicReaderSha256=sha256(Path(__file__).read_bytes()),
        frozenConfigSha256=sha256(frozen_config.read_bytes()),
        exportedConfigSha256=sha256((ROOT/CONFIG_PATH).read_bytes()),
        configResolution="One explicit original private config input; all frozen manifest hash values exact, no model/score override",
        onlyExportMetadataField="authorization", originalManifestRewritten=False,
        originalSourceBytesRewritten=False, registeredReadonlyLoadAdapter=True)
    write_json(Path(output)/"public-config-report.json", report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source","original","experiment","execution","recorded","frozen-config","output"):
        parser.add_argument("--"+name,required=True)
    a=parser.parse_args(); output=Path(a.output).resolve()
    require(output.is_relative_to(ROOT/".research-output") and output!=ROOT/".research-output","New ignored output required")
    require(not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip(),"Commit public reader before execution")
    output.mkdir(parents=True,exist_ok=False)
    try:replay(a.source,a.original,a.experiment,a.execution,a.recorded,a.frozen_config,output)
    except Exception as error:
        write_json(output/"failure.json",{"status":"FAIL","error":str(error),"additionalTrainingAllowed":False,"preservation":"NOT_VERIFIED; separate original-file audit required"}); raise


if __name__=="__main__":main()
