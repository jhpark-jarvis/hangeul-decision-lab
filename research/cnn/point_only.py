"""Fixed point-only objective contrast; original models and completed experiments stay immutable."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import torch
from torch.nn import functional as F
from data import require, sha256, compact
from spatial import make_model, spatial_tensors, fit_fixed, diagnose_frozen, Budget, SPATIAL_SCHEMA
from paired import weights_hash
from train import write_json, evaluate
from diagnose import fixture_family
from spatial_audit import controlled_row, compute_guard, KINDS
from component_audit import training_guard, comparison_summary
from within_kind import summarize as conditional_summary
from scale_screen import full_summary
from point_objective_replay import readonly_data as spatial_load

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "research/experiments/point-only-objective-v1.json"


def kind_indices(example):
    features, keys = example["candidateFeatures"], example["candidateKeys"]
    require(len(features) == len(keys) and 1 <= len(keys) <= 4096 and len(set(keys)) == len(keys), "Full ordered legal candidates")
    kinds = []
    for f, key in zip(features, keys):
        hot = f[:3]
        require(len(f) == 74 and len(hot) == 3 and all(v in (0, 1) for v in hot) and sum(hot) == 1, "One-hot kind features")
        index = hot.index(1)
        require(json.loads(key)[0] == KINDS[index], "Stored key/kind feature agreement")
        kinds.append(index)
    require(type(example["label"]) is int and 0 <= example["label"] < len(keys) and
            example["label"] == example["labels"]["A512"], "Unchanged hard label")
    return kinds


def loss_parts(logits, kinds, label):
    require(logits.ndim == 1 and logits.dtype == torch.float32 and logits.device.type == "cpu" and
            torch.isfinite(logits).all().item() and len(logits) == len(kinds) and type(label) is int and
            0 <= label < len(kinds) and all(type(k) is int and 0 <= k < 3 for k in kinds), "Finite full-list objective input")
    target = kinds[label]
    present = sorted(set(kinds))
    groups = [[i for i, k in enumerate(kinds) if k == g] for g in present]
    flat = F.cross_entropy(logits.unsqueeze(0), torch.tensor([label]))
    kind = logits.new_zeros(())  # Kind auxiliary is exactly disabled, not evaluated.
    point = logits.new_zeros(())
    if target == 1:
        indices = groups[present.index(1)]
        point = F.cross_entropy(logits[indices].unsqueeze(0), torch.tensor([indices.index(label)]))
    total = flat + 0.5 * point
    require(all(torch.isfinite(t).item() for t in (flat, kind, point, total)), "Finite objective terms")
    return flat, kind, point, total


@contextmanager
def forward_ledger(models, path, mode, state, start):
    original = torch.nn.Module.__call__
    top = {id(m) for m in models}
    allowed = {id(m) for model in models for m in model.modules()}
    limits = {"preflight": {"initial": 856}, "control": {"train": 4068, "epoch-dev": 1068, "frozen": 856},
              "candidate": {"train": 4068, "epoch-dev": 1068, "frozen": 856}}
    counts = {p: 0 for p in limits[mode]}
    def call(module, *args, **kwargs):
        require(id(module) in allowed, "Unregistered model call; STOP")
        if id(module) in top:
            phase = "initial" if mode == "preflight" else "frozen" if state["stage"] == "frozen" else "train" if module.training else "epoch-dev"
            require(counts[phase] < limits[mode][phase] and start + sum(counts.values()) < 12840, "Forward budget exhausted; STOP")
            counts[phase] += 1
            with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps({"call": start+sum(counts.values()), "limit": 12840, "mode": mode, "phase": phase, "phaseCall": counts[phase]})+"\n")
        return original(module, *args, **kwargs)
    try:
        torch.nn.Module.__call__ = call
        yield counts
    finally: torch.nn.Module.__call__ = original


def load(source, original, experiment, execution):
    c, x = json.loads(CONFIG.read_bytes()), json.loads(Path(execution).read_bytes())
    require(c["approval"] == "USER_APPROVED" and c["status"] == x["status"] == "FROZEN_BEFORE_EXECUTION" and
            x["configSha256"] == sha256(CONFIG.read_bytes()), "Approved execution identity")
    proposal = ROOT / "research/experiments/point-only-objective-proposal-v1.json"
    p = json.loads(proposal.read_bytes())
    require(sha256(proposal.read_bytes()) == c["proposalSha256"] and all(c[k] == p[k] for k in
            ("source", "model", "objective", "training", "forwardBudget", "resources", "screening", "inference")), "Exact approved proposal contract")
    require(c["forwardBudget"]["totalTopLevelMax"] == 12840 and c["training"]["runs"] == 2 and c["training"]["epochs"] == 12 and
            c["objective"]["kindWeight"] == 0 and c["objective"]["pointWeight"] == 0.5 and c["resources"]["perChildWallSeconds"] == 600 and
            c["resources"]["totalSupervisorWallSeconds"] == 1860 and c["resources"]["sampledWorkingSetBytes"] == 2147483648, "Fixed loss and budgets")
    for path, digest in x["sourceHashes"].items(): require(sha256((ROOT/path).read_bytes()) == digest, "Frozen source changed: "+path)
    for path, digest in c["source"]["spatialHashes"].items(): require(sha256((Path(experiment)/path).read_bytes()) == digest, "Original spatial input changed: "+path)
    previous=Path(x["previousControlPath"])
    for path,digest in c["previousControlHashes"].items():
        require(sha256((previous/path).read_bytes())==digest,"Previous flat control input changed: "+path)
    _, _, pack, protocol, examples = spatial_load(source, original, Path(experiment)/"execution.json")
    for k in ("seed", "epochs", "batchSize", "threads", "learningRate", "weightDecay"):
        require(c["training"][k] == protocol["training"][k], "Original hyperparameter")
    for split, count in (("train",339),("dev",89)):
        require(len(examples[split]) == count, "Original split counts")
        for e in examples[split]: kind_indices(e)
    old = json.loads((Path(experiment)/"spatial/report.json").read_bytes())
    generator = torch.Generator().manual_seed(protocol["training"]["seed"])
    order = [torch.randperm(339,generator=generator).tolist() for _ in range(12)]
    audit = {**old["audit"], "sampleOrderSha256": sha256(compact(order)), "trainingIdsSha256": sha256(compact([e["id"] for e in examples["train"]]))}
    require(audit == old["audit"], "Original explicit shuffle/observation identity")
    return c, x, examples, protocol, pack, old, order, audit


def append_terms(trace, terms):
    trace.append([float(t.detach()) for t in terms])


def validate_calls(records):
    require(len(records)==12840 and [r["call"] for r in records]==list(range(1,12841)),"Original complete ledger; no new calls")
    offset=0
    for mode,limits in (("preflight",{"initial":856}),("control",{"train":4068,"epoch-dev":1068,"frozen":856}),
                        ("candidate",{"train":4068,"epoch-dev":1068,"frozen":856})):
        block=records[offset:offset+sum(limits.values())]; counts={k:0 for k in limits}
        for r in block:
            require(r["mode"]==mode and r["phase"] in limits and r["limit"]==12840,"Ledger mode/phase/limit")
            counts[r["phase"]]+=1
            require(r["phaseCall"]==counts[r["phase"]],"Ledger phase sequence")
        require(counts==limits,"Complete ledger phase quotas")
        offset+=len(block)


def trace_summary(trace, auxiliary):
    require(len(trace) == 4068, "Twelve full train traces")
    rows = []
    for epoch in range(12):
        block = trace[epoch*339:(epoch+1)*339]
        row = {"epoch": epoch+1, "examples": 339, "pointExamples": 52,
               **{key: sum(v[i] for v in block)/339 for i,key in enumerate(("flatMean","kindMean","pointMean","auxTotalMean"))}}
        row["optimizedMean"] = row["auxTotalMean"] if auxiliary else row["flatMean"]
        rows.append(row)
    return rows


def fit_aux(model, examples, protocol, data_hash, checkpoint, monitor, kinds, orders):
    train, dev = examples["train"], examples["dev"]
    inputs, dev_inputs = [spatial_tensors(e) for e in train], [spatial_tensors(e) for e in dev]
    monitor.sample("tensors")
    config = protocol["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learningRate"], weight_decay=config["weightDecay"])
    generator = torch.Generator().manual_seed(config["seed"])
    history, trace, best = [], [], -1
    for epoch in range(1,13):
        model.train(); order = torch.randperm(339,generator=generator).tolist(); total = 0.0
        require(order == orders[epoch-1], "Actual candidate shuffle exact")
        for offset in range(0,339,8):
            monitor.sample("batch-before"); batch = order[offset:offset+8]; optimizer.zero_grad(set_to_none=True)
            for index in batch:
                terms = loss_parts(model(*inputs[index]), kinds[index], train[index]["label"])
                append_terms(trace,terms); total += float(terms[3].detach()); (terms[3]/len(batch)).backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()), "Finite gradients; STOP")
            optimizer.step(); monitor.sample("batch-after")
        model.eval(); dev_metrics,_ = evaluate(model,dev,dev_inputs); monitor.sample("epoch-dev")
        selected = dev_metrics["top1"] > best
        if selected:
            best,best_epoch = dev_metrics["top1"],epoch
            torch.save({"state_dict":model.state_dict(),"schema":SPATIAL_SCHEMA,"protocol":protocol,"dataSha256":data_hash,"epoch":epoch},checkpoint)
        history.append({"epoch":epoch,"trainLoss":total/339,"dev":dev_metrics,"selected":selected})
        print(f"candidate epoch {epoch}: dev {dev_metrics['correct']}/89",flush=True)
    return history,best_epoch,trace


def run(mode,source,original,experiment,execution,output):
    started=time.perf_counter(); torch.set_num_interop_threads(1)
    require(torch.version.cuda is None,"Existing CPU-only build required")
    monitor=Budget(600,2147483648,output,started)
    c,x,examples,protocol,pack,old,order,audit=load(source,original,experiment,execution)
    torch.set_num_threads(2); monitor.sample("loaded")
    root=Path(execution).parent; execution_hash=sha256(Path(execution).read_bytes())
    state={"stage":"initial"}; ledger=root/"forward-count.jsonl"
    if mode=="preflight":
        require(not ledger.exists(),"Exclusive preflight ledger")
        a,b=[make_model(protocol["training"],"spatial").eval() for _ in range(2)]
        initial=weights_hash(a); require(initial==weights_hash(b) and sum(p.numel() for p in a.parameters())==13292,"Identical original spatial initialization")
        with forward_ledger([a,b],ledger,mode,state,0) as counts,torch.inference_mode():
            for split in ("train","dev"):
                for e in examples[split]:
                    inputs=spatial_tensors(e); require(torch.equal(a(*inputs),b(*inputs)),"Initial paired logits exact"); monitor.sample("initial")
        require(counts=={"initial":856},"Initial count")
        require(weights_hash(a)==initial==weights_hash(b),"Initial forward weights unchanged")
        write_json(output/"report.json",{"status":"PASS","executionSha256":execution_hash,"initialWeightsSha256":initial,"initialLogitsExact":True,"audit":audit,"phaseCounts":counts,"modelForwards":856,"trainingRuns":0,"budget":monitor.report()}); return
    pre=json.loads((root/"preflight/report.json").read_bytes())
    require(pre["status"]=="PASS" and pre["executionSha256"]==execution_hash and pre["audit"]==audit,"Preflight before training")
    if mode=="candidate":
        control=json.loads((root/"control/report.json").read_bytes())
        require(control["status"]=="PASS" and control["baselineReproduction"] and control["executionSha256"]==execution_hash,"Control exact before candidate")
    records=[json.loads(line) for line in ledger.read_text().splitlines()]
    start=856 if mode=="control" else 6848
    require(len(records)==start and [r["call"] for r in records]==list(range(1,start+1)),"Global call sequence before child")
    model=make_model(protocol["training"],"spatial")
    require(weights_hash(model)==pre["initialWeightsSha256"],"Cold initial weights exact")
    kinds=[kind_indices(e) for e in examples["train"]]; require(sum(k[e["label"]]==1 for k,e in zip(kinds,examples["train"]))==52,"Original point observations")
    checkpoint=output/"checkpoint.pt"; state["stage"]="fit"; trace=[]
    with forward_ledger([model],ledger,mode,state,start) as counts:
        handle=None
        if mode=="control":
            def observe(module,args,logits):
                if not module.training:return
                n=len(trace); index=order[n//339][n%339]
                require(torch.equal(args[2],torch.tensor(examples["train"][index]["candidateFeatures"],dtype=torch.float32)),"Actual control input order")
                with torch.no_grad():append_terms(trace,loss_parts(logits.detach(),kinds[index],examples["train"][index]["label"]))
            handle=model.register_forward_hook(observe)
            try:history,epoch=fit_fixed(model,examples["train"],examples["dev"],protocol,c["source"]["outcomeDatasetSha256"],checkpoint,"spatial",monitor)
            finally:handle.remove()
        else:history,epoch,trace=fit_aux(model,examples,protocol,c["source"]["outcomeDatasetSha256"],checkpoint,monitor,kinds,order)
        saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
        require(saved["epoch"]==epoch and saved["schema"]==SPATIAL_SCHEMA and saved["protocol"]==protocol and saved["dataSha256"]==c["source"]["outcomeDatasetSha256"],"Own selected checkpoint identity")
        model.load_state_dict(saved["state_dict"],strict=True); model.eval()
        if mode=="control":require(history==old["history"] and epoch==old["selectedEpoch"]==11 and weights_hash(model)==old["weightsSha256"],"Original spatial history/epoch/weights exact; STOP")
        state["stage"]="frozen"
        families={e["fixture"]["id"]:fixture_family(e["fixture"],protocol["dataset"]["families"]) for e in pack["episodes"]}
        raw,summary=diagnose_frozen(model,examples,families,"spatial",output,monitor)
    require(counts=={"train":4068,"epoch-dev":1068,"frozen":856},"Exact per-arm calls")
    loss_history=trace_summary(trace,mode=="candidate")
    require(all(h["trainLoss"]==t["optimizedMean"] for h,t in zip(history,loss_history)),"Loss trace denominator and optimizer objective exact")
    reproduction=mode=="control"
    if reproduction:
        require(raw==json.loads((Path(experiment)/"spatial/predictions.json").read_bytes()) and summary==old["summary"] and
                sha256((output/"logits.json").read_bytes())==old["logitsSha256"],"Original spatial predictions/logits exact; STOP")
    if mode=="control":
        previous=Path(x["previousControlPath"])
        prior=json.loads((previous/"control/report.json").read_bytes())
        require(history==prior["history"] and epoch==prior["selectedEpoch"]==11 and weights_hash(model)==prior["weightsSha256"] and audit==prior["audit"],"Previous flat control logical/history/shuffle exact; STOP")
        for path in ("predictions.json","logits.json","loss-history.json"):
            if path=="loss-history.json":continue # Old trace includes the then-active kind diagnostic; optimizer/flat trace is checked separately.
            require(sha256((output/path).read_bytes())==prior["artifactHashes"][path],"Previous flat control full output exact; STOP")
        prior_losses=json.loads((previous/"control/loss-history.json").read_bytes())
        require(all(a["flatMean"]==b["flatMean"] and a["optimizedMean"]==b["optimizedMean"] for a,b in zip(loss_history,prior_losses)),"Previous optimized flat trace exact")
    write_json(output/"loss-history.json",loss_history)
    write_json(output/"report.json",{"status":"PASS","mode":mode,"executionSha256":execution_hash,"audit":audit,"history":history,"selectedEpoch":epoch,
        "baselineReproduction":reproduction,"initialWeightsSha256":pre["initialWeightsSha256"],"weightsSha256":weights_hash(model),"summary":summary,"phaseCounts":counts,"modelForwards":5992,"trainingRuns":1,
        "artifactHashes":{p:sha256((output/p).read_bytes()) for p in ("checkpoint.pt","predictions.json","logits.json","loss-history.json")},"budget":monitor.report()})
    if mode=="candidate":assemble(source,original,experiment,execution,root,root)


def assemble(source,original,experiment,execution,recorded,output):
    c,x,examples,protocol,pack,old,order,audit=load(source,original,experiment,execution)
    execution_hash=sha256(Path(execution).read_bytes()); recorded=Path(recorded)
    ledger=[json.loads(line) for line in (recorded/"forward-count.jsonl").read_text().splitlines()]
    validate_calls(ledger)
    models,raw,artifacts={}, {}, {}
    families={e["fixture"]["id"]:fixture_family(e["fixture"],protocol["dataset"]["families"]) for e in pack["episodes"]}
    for mode in ("control","candidate"):
        report=json.loads((recorded/mode/"report.json").read_bytes())
        require(report["status"]=="PASS" and report["executionSha256"]==execution_hash and report["audit"]==audit and
                report["modelForwards"]==5992 and report["phaseCounts"]=={"train":4068,"epoch-dev":1068,"frozen":856},"Frozen arm identity")
        for p,h in report["artifactHashes"].items():require(sha256((recorded/mode/p).read_bytes())==h,"Frozen arm artifact")
        history=report["history"]; require(len(history)==12 and [h["epoch"] for h in history]==list(range(1,13)),"Fixed twelve epochs")
        require(report["selectedEpoch"]==max(range(12),key=lambda i:history[i]["dev"]["correct"])+1,"Original dev first-epoch selection")
        if mode=="control":require(report["baselineReproduction"] and history==old["history"] and report["weightsSha256"]==old["weightsSha256"],"Control exact in replay")
        predictions=json.loads((recorded/mode/"predictions.json").read_bytes()); logits=json.loads((recorded/mode/"logits.json").read_bytes())
        losses=json.loads((recorded/mode/"loss-history.json").read_bytes())
        models[mode]={"selectedEpoch":report["selectedEpoch"],"lossHistory":losses,"splits":{}}; raw[mode]={}
        for split in ("train","dev"):
            require(len(predictions[split])==len(logits[split])==len(examples[split]),"Full frozen row counts")
            rows=[controlled_row(e,p,l,families[e["fixtureId"]]) for e,p,l in zip(examples[split],predictions[split],logits[split])]
            points=[r for r in rows if r["teacherKind"]=="single-cell"]
            models[mode]["splits"][split]={"conditional":conditional_summary(rows),"fullList":full_summary(rows),"pointCorrectKind":sum(r["selectedKind"]=="single-cell" for r in points)}
            raw[mode][split]=rows
        artifacts[mode]={**report["artifactHashes"],"report.json":sha256((recorded/mode/"report.json").read_bytes())}
    b=models["control"]["splits"]["dev"]; a=models["candidate"]["splits"]["dev"]
    checks={"aboveFirstLegal":a["fullList"]["overall"]["correct"]>=48,"gainAtLeastFive":a["fullList"]["overall"]["correct"]>=b["fullList"]["overall"]["correct"]+5,
            "singleCellAtLeastThree":a["fullList"]["groups"]["teacherKind"]["single-cell"]["correct"]>=3,"rerollAtLeastSeven":a["fullList"]["groups"]["teacherKind"]["reroll"]["correct"]>=7}
    mechanism={"pointKindAtLeastSeven":a["pointCorrectKind"]>=7,"pointOracleAtLeastFive":a["conditional"]["groups"]["teacherKind"]["single-cell"]["teacherKindOracleCorrect"]>=5}
    result={"schemaVersion":1,"id":c["id"],"status":"PASS","identity":{"executionSha256":execution_hash,"sourceHashes":x["sourceHashes"],"artifactHashes":artifacts},
            "models":models,"paired":{split:comparison_summary({m:raw[m][split] for m in raw},"control","candidate") for split in ("train","dev")},
            "gate":{"status":"PASS" if all(checks.values()) else "FAIL","checks":checks},"mechanism":{"status":"PASS" if all(mechanism.values()) else "FAIL","checks":mechanism},
            "accounting":{"originalTopLevelForwards":12840,"trainingRuns":2,"epochsPerRun":12,"teacherDomainTestRolloutGeometry":0},
            "limits":["Known train/dev single-seed screen; no held-out or policy survival claim","One point auxiliary term versus concurrent flat only; kind-only and interaction remain unmeasured","Technical PASS is separate from original four-gate screening and mechanism checks","No app adoption, extra tuning, new label/data/model architecture/test"]}
    coordinate={"oracleAtLeastFive":a["conditional"]["groups"]["teacherKind"]["single-cell"]["teacherKindOracleCorrect"]>=5,
                "oracleGainAtLeastThree":a["conditional"]["groups"]["teacherKind"]["single-cell"]["teacherKindOracleCorrect"]>=b["conditional"]["groups"]["teacherKind"]["single-cell"]["teacherKindOracleCorrect"]+3}
    result["coordinateMechanism"]={"status":"PASS" if all(coordinate.values()) else "FAIL","checks":coordinate}
    result["pointJoint"]={m:{split:point_joint(raw[m][split]) for split in ("train","dev")} for m in raw}
    result["pointPaired"]={split:point_paired(raw["control"][split],raw["candidate"][split]) for split in ("train","dev")}
    write_json(output/"raw.json",raw); write_json(output/"summary.json",result)
    return result



def point_joint(rows):
    points=[r for r in rows if r["teacherKind"]=="single-cell"]
    result={"examples":len(points),"bothCorrect":0,"kindOnly":0,"coordinateOnly":0,"neither":0}
    for row in points:
        kind=row["selectedKind"]=="single-cell"; coordinate=row["conditional"]["oracleCorrect"]
        require(row["globalCorrect"]==(kind and coordinate),"Actual point equals kind/coordinate intersection")
        result["bothCorrect" if kind and coordinate else "kindOnly" if kind else "coordinateOnly" if coordinate else "neither"]+=1
    return result


def point_paired(before,after):
    a=[r for r in before if r["teacherKind"]=="single-cell"]; b=[r for r in after if r["teacherKind"]=="single-cell"]
    require([r["id"] for r in a]==[r["id"] for r in b],"Fixed point paired observations")
    result={}
    for metric in ("actual","kind","oracle"):
        value=lambda r:r["globalCorrect"] if metric=="actual" else r["selectedKind"]=="single-cell" if metric=="kind" else r["conditional"]["oracleCorrect"]
        result[metric]={"examples":len(a),"bothCorrect":0,"gain":0,"regression":0,"bothWrong":0}
        for x,y in zip(a,b):
            old,new=value(x),value(y)
            result[metric]["bothCorrect" if old and new else "gain" if new else "regression" if old else "bothWrong"]+=1
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("preflight","control","candidate"),required=True)
    for name in ("source","original","experiment","execution","output"):parser.add_argument("--"+name,required=True)
    parser.add_argument("--recorded"); args=parser.parse_args(); output=Path(args.output).resolve()
    require(output.is_relative_to(ROOT/".research-output") and output!=ROOT/".research-output","New ignored output required")
    require(not subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip(),"Commit source/config before execution")
    require(bool(args.recorded)==(args.mode=="replay"),"Recorded output only for replay")
    if args.mode!="replay":require(output==Path(args.execution).resolve().parent/args.mode,"Exclusive mode directory")
    output.mkdir(parents=True,exist_ok=False)
    try:
        if args.mode=="replay":
            with training_guard(),compute_guard() as counter:
                frozen=json.loads((Path(args.recorded)/"freeze.json").read_bytes())
                require(frozen["reportSha256"]==sha256((Path(args.recorded)/"report.json").read_bytes()) and frozen["executionSha256"]==sha256(Path(args.execution).read_bytes()),"Frozen successful run")
                torch.set_num_threads(2); result=assemble(args.source,args.original,args.experiment,args.execution,args.recorded,output)
                require(sha256((output/"raw.json").read_bytes())==sha256((Path(args.recorded)/"raw.json").read_bytes()) and
                        sha256((output/"summary.json").read_bytes())==sha256((Path(args.recorded)/"summary.json").read_bytes()),"Saved replay byte exact")
                write_json(output/"report.json",{"status":"PASS","mode":"replay","additionalModelLoadForwardTraining":0,**counter,"rawSha256":sha256((output/"raw.json").read_bytes()),"summarySha256":sha256((output/"summary.json").read_bytes())})
        else:run(args.mode,args.source,args.original,args.experiment,args.execution,output)
    except Exception as error:
        write_json(output/"failure.json",{"status":"FAIL","error":str(error),"additionalTrainingAllowed":False,"preservation":"NOT_VERIFIED; separate original-file audit required"}); raise


if __name__=="__main__":main()
