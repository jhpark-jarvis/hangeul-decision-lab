"""Read-only point-only result reader; model creation/load/forward/training remain blocked."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import torch
import point_only as frozen_runner
from component_audit import prepare as component_prepare, training_guard
from spatial_audit import compute_guard
from data import require, sha256
from train import write_json

ROOT=Path(__file__).resolve().parents[2]


def readonly_data(source, original, execution):
    experiment=Path(execution).parent
    _,examples,_,_,_,paths,hashes,_,protocol,_=component_prepare(source,experiment,original)
    require(sha256(Path(execution).read_bytes())==hashes["execution.json"],"Original spatial execution hash")
    frozen=json.loads(Path(execution).read_bytes())
    config_path=ROOT/"research/experiments/spatial-residual-v1.json"
    require(sha256(config_path.read_bytes())==frozen["configSha256"],"Original spatial config identity")
    source=Path(source); body=(source/"train-dev/dataset.json").read_bytes()
    require(sha256(body)==hashes["dataset"],"Original paired pack identity")
    pack=json.loads(body); freeze=json.loads((source/"freeze.json").read_bytes())
    require(sha256((source/"protocol.json").read_bytes())==freeze["protocolSha256"],"Original protocol identity")
    for record in freeze["models"].values():
        require(sha256((source/record["checkpoint"]).read_bytes())==record["sha256"],"Original paired checkpoint bytes; no deserialization")
    own={split:[{**e,"label":e["labels"]["A512"]} for e in examples[split]] for split in ("train","dev")}
    return json.loads(config_path.read_bytes()),frozen,pack,protocol,own


@contextmanager
def data_only_loader():
    original=frozen_runner.spatial_load
    try:
        frozen_runner.spatial_load=readonly_data
        yield
    finally:frozen_runner.spatial_load=original


def replay(source,original,experiment,execution,recorded,output):
    recorded=Path(recorded); output=Path(output)
    reader_hash=sha256(Path(__file__).read_bytes())
    freeze=json.loads((recorded/"freeze.json").read_bytes()); report=json.loads((recorded/"report.json").read_bytes())
    require(freeze["status"]=="FROZEN_DEV_ONLY" and report["status"]=="PASS" and
            freeze["reportSha256"]==sha256((recorded/"report.json").read_bytes()) and
            freeze["executionSha256"]==sha256(Path(execution).read_bytes()) and
            report["ledgerSha256"]==sha256((recorded/"forward-count.jsonl").read_bytes()),"Successful frozen execution identity")
    torch.set_num_threads(2)
    with training_guard(),compute_guard() as counter,data_only_loader():
        frozen_runner.assemble(source,original,experiment,execution,recorded,output)
        for p in ("raw.json","summary.json"):
            require(sha256((output/p).read_bytes())==sha256((recorded/p).read_bytes())==report[p.replace('.json','')+"Sha256"],"Readonly original bytes exact")
        require(sha256(Path(__file__).read_bytes())==reader_hash,"Reader source changed")
        write_json(output/"report.json",{"status":"PASS","mode":"readonly-replay","additionalModelLoadForwardTraining":0,**counter,
            "readerSha256":reader_hash,"originalTrainingSourcesRebound":False,"dataLoader":"Versioned data-only adapter; original runner/source/input/manifest hashes unchanged",
            "rawSha256":sha256((output/"raw.json").read_bytes()),"summarySha256":sha256((output/"summary.json").read_bytes())})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source","original","experiment","execution","recorded","output"):parser.add_argument("--"+name,required=True)
    a=parser.parse_args(); output=Path(a.output).resolve()
    require(output.is_relative_to(ROOT/".research-output") and output!=ROOT/".research-output","New ignored output required")
    require(not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip(),"Commit reader before execution")
    output.mkdir(parents=True,exist_ok=False)
    try:replay(a.source,a.original,a.experiment,a.execution,a.recorded,output)
    except Exception as error:
        write_json(output/"failure.json",{"status":"FAIL","error":str(error),"additionalTrainingAllowed":False,"preservation":"NOT_VERIFIED; separate original-file audit required"});raise


if __name__=="__main__":main()
