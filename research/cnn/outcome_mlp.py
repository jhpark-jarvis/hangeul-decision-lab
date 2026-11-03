"""Approved cold spatial versus outcome-only MLP, one model per bounded CPU child."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import torch
from torch import nn
from data import require, sha256
from model import configure
from controlled_point_fit import flat_loss, score_row, summarize, validate_dataset, counted_model
from frozen_point_transfer import validate_transfer
from spatial import make_model as make_spatial, spatial_tensors, Budget
from paired import weights_hash
from train import write_json

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "research/experiments/outcome-mlp-v1.json"
LIMITS = {"initial":16, "train":16000, "fit1":16, "transfer1":144, "fit2":16, "transfer2":144}


class OutcomeMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(22,32,bias=True)
        self.activation = nn.ReLU()
        self.output = nn.Linear(32,1,bias=True)

    def forward(self, candidates):
        require(candidates.shape == (31,22) and candidates.dtype == torch.float32 and candidates.device.type == "cpu" and
                torch.isfinite(candidates).all().item() and ((candidates>=0)&(candidates<=1)).all().item(), "Only finite 31x22 CPU outcome input")
        return self.output(self.activation(self.hidden(candidates))).squeeze(1)


def outcome_tensors(e):
    # Labels, keys, raw board, metadata, footprint and absolute candidate coordinates are excluded.
    values = torch.tensor([f[52:74] for f in e["candidateFeatures"]],dtype=torch.float32)
    require(values.shape == (31,22) and torch.isfinite(values).all().item(), "Outcome-only input")
    return (values,)


class MlpLedger:
    def __init__(self,path,arm):
        require(arm in ("control","candidate"),"Fixed arm")
        self.path,self.arm=Path(path),arm
        self.offset=0 if arm=="control" else 16336
        self.counts={k:0 for k in LIMITS}
        with self.path.open("x",encoding="utf8"): pass

    def reserve(self,phase,step,example):
        require(phase in LIMITS and self.counts[phase]<LIMITS[phase] and sum(self.counts.values())<16336 and
                self.offset+sum(self.counts.values())<32672 and type(example) is int,"MLP budget/phase; STOP")
        current=next(k for k in LIMITS if self.counts[k]<LIMITS[k])
        expected=self.counts[phase]%16 if phase=="train" else self.counts[phase]
        require(phase==current and example==expected and step==(self.counts[phase]//16+1 if phase=="train" else None),"MLP call order; STOP")
        row={"call":self.offset+sum(self.counts.values())+1,"arm":self.arm,"phase":phase,"phaseCall":self.counts[phase]+1,"step":step,"example":example}
        with self.path.open("a",encoding="utf8",newline="\n") as stream:
            stream.write(json.dumps(row)+"\n");stream.flush()
        self.counts[phase]+=1


def gate(s):
    checks={"allCorrect":s["correct"]==s["examples"],"meanNllAtMost005":s["meanNll"]<=0.05,"everyTargetMarginPositive":s["minimumTargetMargin"]>0}
    return {"status":"PASS" if all(checks.values()) else "FAIL","checks":checks}


def summarize_transfer(rows):
    require(len(rows)==144 and len({r["id"] for r in rows})==144,"Separate transfer denominator")
    return {"examples":144,"correct":sum(r["correct"] for r in rows),"meanNll":sum(r["nll"] for r in rows)/144,
            "minimumTargetMargin":min(r["margin"] for r in rows),"maximumTargetRank":max(r["rank"] for r in rows)}


def run(output,arm):
    started=time.perf_counter();monitor=Budget(600,2147483648,output,started);monitor.sample("start")
    config=json.loads(CONFIG.read_bytes());frozen=json.loads((output.parent/"execution.json").read_bytes())
    proposal=ROOT/config["proposalPath"];p=json.loads(proposal.read_bytes())
    require(config["approval"]=="USER_APPROVED" and config["status"]==frozen["status"]=="FROZEN_BEFORE_EXECUTION" and
            sha256(CONFIG.read_bytes())==frozen["configSha256"] and config["proposalSha256"]==sha256(proposal.read_bytes()) and
            config["contract"]=={k:p[k] for k in config["contract"]},"Approved MLP execution identity")
    c=config["contract"]
    require(c["forwardBudget"]["totalTopLevelMax"]==32672 and c["shared"]["training"]["updates"]==1000 and
            c["contrast"]["candidate"]["parameters"]==769,"Fixed MLP limits")
    for path,h in frozen["sourceHashes"].items():require(sha256((ROOT/path).read_bytes())==h,"MLP source drift: "+path)
    for name,h in frozen["inputHashes"].items():require(sha256((output.parent/name).read_bytes())==h,"MLP input drift: "+name)
    fit=validate_dataset(json.loads((output.parent/"fit-dataset.json").read_bytes()))
    transfer=validate_transfer(json.loads((output.parent/"transfer-dataset.json").read_bytes()),fit)
    if arm=="candidate":
        phase=json.loads((output.parent/"candidate-input.json").read_bytes())
        require(phase["executionSha256"]==sha256((output.parent/"execution.json").read_bytes()) and phase["controlReportSha256"]==sha256((output.parent/"control/report.json").read_bytes()),"Control phase freeze")
        require(json.loads((output.parent/"control/report.json").read_bytes())["controlExact"] is True,"Exact control before candidate")
    require(torch.version.cuda is None,"Existing CPU-only runtime")
    torch.set_num_interop_threads(1)
    tensor=spatial_tensors if arm=="control" else outcome_tensors
    fit_inputs=[tensor(e) for e in fit];transfer_inputs=[tensor(e) for e in transfer]
    write_json(output/"model-intent.json",{"topLevelModels":1,"checkpointLoads":0,"arm":arm})
    configure(c["shared"]["modelConfig"])
    model=make_spatial(c["shared"]["modelConfig"],"spatial") if arm=="control" else OutcomeMlp()
    require(sum(v.numel() for v in model.parameters())==(13292 if arm=="control" else 769),"Fixed model parameter count")
    initial_weights=weights_hash(model)
    if arm=="control":require(initial_weights==c["original"]["initialLogicalWeightsSha256"],"Original control initial weights")
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
    ledger=MlpLedger(output/"forward-count.jsonl",arm);state={"phase":"initial","step":None,"example":0}
    def evaluate(phase,examples,inputs):
        model.eval();state.update(phase=phase,step=None);rows=[]
        with torch.inference_mode():
            for i,(e,inp) in enumerate(zip(examples,inputs)):
                monitor.sample(phase+"-before");state["example"]=i
                rows.append(score_row(model(*inp),e["label"],e["id"]))
                monitor.sample(phase+"-after")
        write_json(output/(phase+".json"),rows);return rows
    def blocked_load(*args,**kwargs):raise ValueError("No checkpoint loads; STOP")
    history=[]
    with patch("torch.load",blocked_load),counted_model(model,ledger,state):
        initial=evaluate("initial",fit,fit_inputs)
        if arm=="control":require(sha256((output/"initial.json").read_bytes())==c["original"]["initialFitLogitsSha256"],"Original initial control logits")
        for step in range(1,1001):
            monitor.sample("train-before");model.train();state.update(phase="train",step=step);optimizer.zero_grad(set_to_none=True);losses=[]
            for i,(e,inp) in enumerate(zip(fit,fit_inputs)):
                state["example"]=i;loss=flat_loss(model(*inp),e["label"]);losses.append(float(loss.detach()));(loss/16).backward()
            require(all(v.grad is None or torch.isfinite(v.grad).all().item() for v in model.parameters()),"Nonfinite gradient; STOP")
            optimizer.step();require(all(torch.isfinite(v).all().item() for v in model.parameters()),"Nonfinite weights; STOP")
            history.append({"step":step,"meanTrainingNll":sum(losses)/16});monitor.sample("train-after")
            if step%200==0:print(f"{arm} update {step}/1000 loss={history[-1]['meanTrainingNll']:.8f}",flush=True)
        write_json(output/"history.json",history)
        if arm=="control":
            require(sha256((output/"history.json").read_bytes())==c["original"]["history1000Sha256"] and
                    weights_hash(model)==c["original"]["finalLogicalWeightsSha256"],"Original control history/final weights")
        fit_rows=evaluate("fit1",fit,fit_inputs)
        if arm=="control":require(sha256((output/"fit1.json").read_bytes())==c["original"]["finalFitLogitsSha256"],"Control fit exact")
        transfer_rows=evaluate("transfer1",transfer,transfer_inputs)
        if arm=="control":require(sha256((output/"transfer1.json").read_bytes())==c["original"]["finalTransferLogitsSha256"],"Control transfer exact")
        evaluate("fit2",fit,fit_inputs);evaluate("transfer2",transfer,transfer_inputs)
    for a,b in [("fit1","fit2"),("transfer1","transfer2")]:require((output/(a+".json")).read_bytes()==(output/(b+".json")).read_bytes(),"Final repeats mismatch")
    require(ledger.counts==LIMITS,"Exact arm ledger")
    torch.save(model.state_dict(),output/"final-weights.pt")
    fit_summary,transfer_summary=summarize(fit_rows),summarize_transfer(transfer_rows)
    summary={"arm":arm,"initial":summarize(initial),"fit":fit_summary,"transfer":transfer_summary,"fitGate":gate(fit_summary),"transferGate":gate(transfer_summary),"updates":1000}
    write_json(output/"summary.json",summary)
    for path,h in frozen["sourceHashes"].items():require(sha256((ROOT/path).read_bytes())==h,"Source changed during MLP execution")
    monitor.sample("finished")
    names=["initial.json","history.json","fit1.json","transfer1.json","fit2.json","transfer2.json","forward-count.jsonl","summary.json","final-weights.pt","model-intent.json"]
    write_json(output/"report.json",{"status":"PASS","atUtc":datetime.now(timezone.utc).isoformat(),"arm":arm,"executionSha256":sha256((output.parent/"execution.json").read_bytes()),"hashes":{n:sha256((output/n).read_bytes()) for n in names},"summary":summary,"forwardCounts":ledger.counts,"globalOffset":ledger.offset,"topLevelModels":1,"checkpointLoads":0,"trainingRuns":1,"parameters":13292 if arm=="control" else 769,"initialLogicalWeightsSha256":initial_weights,"finalLogicalWeightsSha256":weights_hash(model),"controlExact":True if arm=="control" else None,"budget":monitor.report(),"environment":{"python":sys.version,"torch":torch.__version__,"cuda":torch.version.cuda,"threads":torch.get_num_threads(),"interopThreads":torch.get_num_interop_threads()}})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",required=True);parser.add_argument("--arm",choices=["control","candidate"],required=True);args=parser.parse_args()
    run_dir=Path(args.run).resolve();output=run_dir/args.arm
    require(run_dir.is_relative_to(ROOT/".research-output") and (run_dir/"execution.json").is_file() and output.is_dir() and not (output/"model-intent.json").exists(),"New frozen arm only; no resume")
    try:run(output,args.arm)
    except BaseException as error:
        if not (output/"failure.json").exists():write_json(output/"failure.json",{"status":"FAIL","error":str(error),"retry":False})
        raise
