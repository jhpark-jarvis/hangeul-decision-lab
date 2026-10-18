"""Synthetic loss/gradient/budget controls; no real-data training or forwards."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from point_objective import loss_parts, kind_indices, forward_ledger, trace_summary, validate_calls, main
from spatial import fit_fixed
from spatial_audit import compute_guard


class PointTests(unittest.TestCase):
    def test_flat_ce_native_exact_and_aux_gradient_coordinate_signal(self):
        s=torch.tensor([0.,0.,0.],requires_grad=True)
        flat,kind,point,total=loss_parts(s,[0,1,1],2)
        self.assertTrue(torch.equal(flat,torch.nn.functional.cross_entropy(s.unsqueeze(0),torch.tensor([2]))))
        self.assertAlmostEqual(float(kind.detach()),0.69314718,places=6)
        self.assertAlmostEqual(float(point.detach()),0.69314718,places=6)
        total.backward()
        self.assertTrue(torch.isfinite(s.grad).all())
        self.assertLess(float(s.grad[2]),float(s.grad[1]))
        self.assertAlmostEqual(float(s.grad.sum()),0,places=6)

    def test_kind_count_correction_and_nonpoint_zero_singleton(self):
        a=loss_parts(torch.zeros(3),[0,0,1],2)
        b=loss_parts(torch.zeros(5),[0,0,0,0,1],4)
        self.assertTrue(torch.equal(a[1],b[1]))
        self.assertEqual(float(a[2]),0)
        self.assertEqual(float(loss_parts(torch.zeros(2),[0,0],0)[1]),0)
        self.assertEqual(float(loss_parts(torch.zeros(2),[0,1],0)[2]),0)

    def test_bad_inputs_and_onehot_label_key_mismatch_stop(self):
        for values,kinds,label in (([float('nan')],[1],0),([0.],[3],0),([0.],[1],1),([0.],[True],0)):
            with self.assertRaises(ValueError):loss_parts(torch.tensor(values),kinds,label)
        e={'candidateFeatures':[[0,1,0]+[0]*71],'candidateKeys':['["single-cell",0,0]'],'label':0,'labels':{'A512':0}}
        self.assertEqual(kind_indices(e),[1])
        for change in ({'label':1},{'candidateKeys':['["reroll","x",0]']},{'candidateFeatures':[[1,1,0]+[0]*71]}):
            with self.assertRaises(ValueError):kind_indices({**e,**change})

    def test_ledger_pre_call_unknown_module_budget_and_restoration(self):
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root/'.research-output') as d:
            model,other=nn.Identity(),nn.Identity(); original=nn.Module.__call__; path=Path(d)/'calls'
            handle=model.register_forward_pre_hook(lambda m,a:self.assertEqual(json.loads(path.read_text())['call'],12840))
            with forward_ledger([model],path,'preflight',{'stage':'initial'},12839) as counts:
                model(torch.zeros(1))
                with self.assertRaises(ValueError):model(torch.zeros(1))
                with self.assertRaises(ValueError):other(torch.zeros(1))
            handle.remove();self.assertEqual(counts,{'initial':1});self.assertIs(nn.Module.__call__,original)

    def test_phase_routing_train_eval_frozen_and_exception_restores(self):
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root/'.research-output') as d:
            model=nn.Identity();state={'stage':'fit'}
            with forward_ledger([model],Path(d)/'calls','control',state,856) as counts:
                model.train();model(torch.zeros(1));model.eval();model(torch.zeros(1));state['stage']='frozen';model(torch.zeros(1))
            self.assertEqual(counts,{'train':1,'epoch-dev':1,'frozen':1})

    def test_full_phase_ledger_validation_no_model(self):
        rows=[]
        for mode,phases in [('preflight',[('initial',856)]),('control',[('train',4068),('epoch-dev',1068),('frozen',856)]),('candidate',[('train',4068),('epoch-dev',1068),('frozen',856)])]:
            for phase,n in phases:
                for i in range(n):rows.append({'call':len(rows)+1,'limit':12840,'mode':mode,'phase':phase,'phaseCall':i+1})
        with compute_guard():validate_calls(rows)
        for bad in (rows[:-1],[{**rows[0],'limit':12841}]+rows[1:],[{**rows[0],'phaseCall':2}]+rows[1:]):
            with self.assertRaises(ValueError):validate_calls(bad)

    def test_detached_loss_observer_preserves_original_fit_history_and_weights(self):
        torch.manual_seed(7)
        class Toy(nn.Module):
            def __init__(self):super().__init__();self.layer=nn.Linear(2,1)
            def forward(self,x):return self.layer(x).squeeze(1)
        a=Toy();b=copy.deepcopy(a)
        train=[{'label':2},{'label':1}];dev=[{'label':2,'fixtureId':'a'},{'label':1,'fixtureId':'a'}]
        x=torch.tensor([[0.,1.],[1.,0.],[1.,1.]])
        monitor=type('Monitor',(),{'sample':lambda self,phase:None})()
        config={'training':{'learningRate':0.001,'weightDecay':0.0001,'seed':7,'epochs':2,'batchSize':8}}
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root/'.research-output') as d,patch('spatial.spatial_tensors',return_value=(x,)):
            first=fit_fixed(a,train,dev,config,'toy',Path(d)/'a.pt','spatial',monitor)
            def observer(m,args,out):
                if m.training:
                    with torch.no_grad():loss_parts(out.detach(),[0,1,1],2)
            handle=b.register_forward_hook(observer)
            with forward_ledger([b],Path(d)/'calls','control',{'stage':'fit'},0):
                second=fit_fixed(b,train,dev,config,'toy',Path(d)/'b.pt','spatial',monitor)
            handle.remove()
        self.assertEqual(first,second)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))

    def test_trace_observation_denominator_includes_zeros(self):
        trace=[[1.,2.,0.,2.] for _ in range(4068)];trace[0]=[1.,2.,2.,3.]
        result=trace_summary(trace,True)
        self.assertEqual(result[0]['pointMean'],2/339)
        self.assertEqual(result[0]['optimizedMean'],result[0]['auxTotalMean'])
        self.assertEqual(trace_summary(trace,False)[0]['optimizedMean'],1)
        with self.assertRaises(ValueError):trace_summary(trace[:-1],True)

    def test_replay_compute_block_and_cli_dirty_no_execution(self):
        for action in (lambda:nn.Identity(),lambda:torch.load('missing')):
            with self.assertRaises(ValueError):
                with compute_guard():action()
        args=['point_objective.py','--mode','preflight','--source','none','--original','none','--experiment','none','--execution','none','--output','.research-output/point-unused']
        with patch.object(sys,'argv',args),patch('point_objective.subprocess.check_output',return_value=' M source'),patch('point_objective.run') as run:
            with self.assertRaisesRegex(ValueError,'Commit'):main()
            run.assert_not_called()

    def test_exclusive_failure_output_preserved_no_actual_forward(self):
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root/'.research-output') as d:
            args=['point_objective.py','--mode','preflight','--source','none','--original','none','--experiment','none','--execution',str(Path(d)/'execution.json'),'--output',str(Path(d)/'preflight')]
            with patch.object(sys,'argv',args),patch('point_objective.subprocess.check_output',return_value=''),patch('point_objective.torch.set_num_interop_threads'):
                with self.assertRaises(FileNotFoundError):main()
                failure=(Path(d)/'preflight/failure.json').read_bytes()
                with self.assertRaises(FileExistsError):main()
                self.assertEqual(failure,(Path(d)/'preflight/failure.json').read_bytes())
                self.assertFalse((Path(d)/'forward-count.jsonl').exists())


if __name__=='__main__':unittest.main()
