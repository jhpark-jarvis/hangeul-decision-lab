"""Fixed same-MLP coverage contrast; no checkpoint loads, tuning or data generation."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import torch
from torch.nn import functional as F
from data import require,sha256
from model import configure
from outcome_mlp import OutcomeMlp,gate
from controlled_point_fit import counted_model,validate_dataset,summarize,score_row as old_score_row
from spatial import Budget
from paired import weights_hash
from train import write_json

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/"research/experiments/coverage-training-v1.json"
LIMITS={"anchor":16,"train":16000,"fit1":16,"dev1":16,"final1":16,"fit2":16,"dev2":16,"final2":16}


class CoverageMlp(OutcomeMlp):
    # Same inherited layers/constructor/state keys, only the candidate-count guard changes.
    def forward(self,candidates):
        require(candidates.ndim==2 and candidates.shape[0] in (31,46) and candidates.shape[1]==22 and
                candidates.dtype==torch.float32 and candidates.device.type=="cpu" and torch.isfinite(candidates).all().item() and
                ((candidates>=0)&(candidates<=1)).all().item(),"Fixed31-anchor/46 CPU outcome inputs")
        return self.output(self.activation(self.hidden(candidates))).squeeze(1)


def tensors(e,n):
    require(n in (31,46),"Fixed candidate counts")
    values=torch.tensor([f[52:74] for f in e["candidateFeatures"]],dtype=torch.float32)
    require(values.shape==(n,22) and torch.isfinite(values).all().item() and ((values>=0)&(values<=1)).all().item(),"Outcome-only finite shape")
    return (values,)


def loss(scores,label):
    require(scores.ndim==1 and len(scores) in (31,46) and scores.dtype==torch.float32 and scores.device.type=="cpu" and
            torch.isfinite(scores).all().item() and type(label) is int and 0<=label<len(scores),"Finite full-list FP32 CE")
    value=F.cross_entropy(scores.unsqueeze(0),torch.tensor([label]));require(torch.isfinite(value).item(),"Finite loss");return value


def score_row(scores,label,identifier):
    value=float(loss(scores,label).detach());v=scores.detach().tolist();index=int(scores.argmax().item())
    return {"id":identifier,"label":label,"index":index,"correct":index==label,"nll":value,
            "rank":1+sum(x>v[label] or (x==v[label] and i<label) for i,x in enumerate(v) if i!=label),
            "margin":v[label]-max(x for i,x in enumerate(v) if i!=label),"logits":v}


class CoverageLedger:
    def __init__(self,path,arm):
        require(arm in ("control","diverse"),"Fixed arm");self.path=Path(path);self.arm=arm
        self.offset=0 if arm=="control" else 16112;self.counts={k:0 for k in LIMITS}
        with self.path.open("x",encoding="utf8"):pass
    def reserve(self,phase,step,example):
        require(phase in LIMITS and self.counts[phase]<LIMITS[phase] and sum(self.counts.values())<16112 and type(example) is int,"Coverage forward cap; STOP")
        require(phase==next(k for k in LIMITS if self.counts[k]<LIMITS[k]) and example==self.counts[phase]%16 and
                step==(self.counts[phase]//16+1 if phase=="train" else None),"Coverage call order; STOP")
        row={"call":self.offset+sum(self.counts.values())+1,"arm":self.arm,"phase":phase,"phaseCall":self.counts[phase]+1,"step":step,"example":example}
        with self.path.open("a",encoding="utf8",newline="\n") as f:f.write(json.dumps(row)+"\n");f.flush()
        self.counts[phase]+=1


def run(out,arm):
    started=time.perf_counter();budget=Budget(600,2147483648,out,started);budget.sample("start")
    c=json.loads(CONFIG.read_bytes());f=json.loads((out.parent/"execution.json").read_bytes())
    require(c["approval"]=="USER_APPROVED" and c["status"]==f["status"]=="FROZEN_BEFORE_EXECUTION" and sha256(CONFIG.read_bytes())==f["configSha256"],"Approved frozen execution")
    require(c["training"]["totalTopLevelForwardMax"]==32224 and c["training"]["updatesEach"]==1000,"Fixed CPU training")
    for p,h in f["sourceHashes"].items():require(sha256((ROOT/p).read_bytes())==h,"Source drift "+p)
    for p,h in f["inputHashes"].items():require(sha256((out.parent/p).read_bytes())==h,"Input drift "+p)
    require(sha256((out.parent/"dataset.json").read_bytes())==c["dataSha256"] and sha256((out.parent/"anchor-dataset.json").read_bytes())==c["anchor"]["datasetSha256"],"Data identities")
    anchor=validate_dataset(json.loads((out.parent/"anchor-dataset.json").read_bytes()))
    d=json.loads((out.parent/"dataset.json").read_bytes());require(d["id"]=="coverage-data-v1","Data schema")
    fit=d["splits"][arm];dev=d["splits"]["dev"];final=d["splits"]["final"]
    for rows in (fit,dev,final):
        require(len(rows)==16 and len({e["id"] for e in rows})==16,"Fixed split denominator")
        for e in rows:require(len(e["candidateFeatures"])==46 and len(e["candidateKeys"])==46 and type(e["label"]) is int and 0<=e["label"]<46,"46 ordered candidates")
    if arm=="diverse":
        phase=json.loads((out.parent/"diverse-input.json").read_bytes());require(phase["executionSha256"]==sha256((out.parent/"execution.json").read_bytes()) and phase["controlReportSha256"]==sha256((out.parent/"control/report.json").read_bytes()),"Control phase freeze")
        require(json.loads((out.parent/"control/report.json").read_bytes())["anchorExact"] is True,"Control anchor before diverse")
    require(torch.version.cuda is None,"CPU-only runtime");torch.set_num_interop_threads(1)
    inputs={"anchor":[tensors(e,31) for e in anchor],"fit":[tensors(e,46) for e in fit],"dev":[tensors(e,46) for e in dev],"final":[tensors(e,46) for e in final]}
    write_json(out/"model-intent.json",{"topLevelModels":1,"checkpointLoads":0,"arm":arm})
    configure(c["modelConfig"]);model=CoverageMlp();require(sum(p.numel() for p in model.parameters())==769,"Same769 parameters")
    initial=weights_hash(model);require(initial==c["anchor"]["weightsSha256"],"Original MLP initial weights exact")
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001,weight_decay=0.0001)
    ledger=CoverageLedger(out/"forward-count.jsonl",arm);state={"phase":"anchor","step":None,"example":0}
    def evaluate(phase,rows,values):
        model.eval();state.update(phase=phase,step=None);result=[]
        with torch.inference_mode():
            for i,(e,inp) in enumerate(zip(rows,values)):
                budget.sample(phase+"-before");state["example"]=i;scores=model(*inp)
                result.append((old_score_row if phase=="anchor" else score_row)(scores,e["label"],e["id"]));budget.sample(phase+"-after")
        write_json(out/(phase+".json"),result);return result
    def blocked(*args,**kwargs):raise ValueError("No checkpoint loads")
    history=[];results={}
    with patch("torch.load",blocked),counted_model(model,ledger,state):
        evaluate("anchor",anchor,inputs["anchor"]);require(sha256((out/"anchor.json").read_bytes())==c["anchor"]["logitsSha256"],"Original old16 MLP initial logits exact")
        for step in range(1,1001):
            budget.sample("train-before");model.train();state.update(phase="train",step=step);optimizer.zero_grad(set_to_none=True);losses=[]
            for i,(e,inp) in enumerate(zip(fit,inputs["fit"])):
                state["example"]=i;v=loss(model(*inp),e["label"]);losses.append(float(v.detach()));(v/16).backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()),"Finite gradients")
            optimizer.step();require(all(torch.isfinite(p).all().item() for p in model.parameters()),"Finite weights")
            history.append({"step":step,"meanTrainingNll":sum(losses)/16});budget.sample("train-after")
            if step%200==0:print(f"{arm} update {step}/1000 loss={history[-1]['meanTrainingNll']:.8f}",flush=True)
        for repeat in (1,2):
            for split,rows in [("fit",fit),("dev",dev),("final",final)]:results[split]=evaluate(split+str(repeat),rows,inputs[split])
    for split in ("fit","dev","final"):require((out/(split+"1.json")).read_bytes()==(out/(split+"2.json")).read_bytes(),"Final repeats exact")
    require(ledger.counts==LIMITS,"Exact ledger")
    write_json(out/"history.json",history);torch.save(model.state_dict(),out/"final-weights.pt")
    summary={"arm":arm,"updates":1000,"splits":{s:{"metrics":summarize(rows),"gate":gate(summarize(rows))} for s,rows in results.items()}}
    write_json(out/"summary.json",summary)
    for p,h in f["sourceHashes"].items():require(sha256((ROOT/p).read_bytes())==h,"Source changed")
    budget.sample("finished");names=["anchor.json","history.json","fit1.json","dev1.json","final1.json","fit2.json","dev2.json","final2.json","forward-count.jsonl","final-weights.pt","model-intent.json","summary.json"]
    write_json(out/"report.json",{"status":"PASS","arm":arm,"atUtc":datetime.now(timezone.utc).isoformat(),"executionSha256":sha256((out.parent/"execution.json").read_bytes()),"hashes":{p:sha256((out/p).read_bytes()) for p in names},"summary":summary,"anchorExact":True,"initialLogicalWeightsSha256":initial,"finalLogicalWeightsSha256":weights_hash(model),"parameters":769,"topLevelModels":1,"checkpointLoads":0,"forwardCounts":ledger.counts,"budget":budget.report(),"environment":{"python":sys.version,"torch":torch.__version__,"threads":torch.get_num_threads(),"interopThreads":torch.get_num_interop_threads(),"cuda":torch.version.cuda}})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",required=True);parser.add_argument("--arm",choices=["control","diverse"],required=True);args=parser.parse_args()
    base=Path(args.run).resolve();out=base/args.arm
    require(base.is_relative_to(ROOT/".research-output") and out.is_dir() and not (out/"model-intent.json").exists(),"New frozen arm only")
    try:run(out,args.arm)
    except BaseException as error:
        if not (out/"failure.json").exists():write_json(out/"failure.json",{"status":"FAIL","error":str(error),"automaticRetry":False})
        raise
