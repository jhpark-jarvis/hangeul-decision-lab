"""Point-only FP32 objective and stored diagnostic boundaries on synthetic tensors."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
import point_only as runner
from point_only import loss_parts, point_joint, point_paired
from spatial_audit import compute_guard


class PointOnlyTests(unittest.TestCase):
    def test_native_ce_and_exact_gradient_reference(self):
        a=torch.tensor([0.2,0.7,-0.3],requires_grad=True)
        b=a.detach().clone().requires_grad_()
        flat,kind,point,total=loss_parts(a,[0,1,1],2)
        expected=torch.nn.functional.cross_entropy(b.unsqueeze(0),torch.tensor([2]))+0.5*torch.nn.functional.cross_entropy(b[1:].unsqueeze(0),torch.tensor([1]))
        self.assertTrue(torch.equal(total,expected));self.assertEqual(float(kind),0)
        total.backward();expected.backward();self.assertTrue(torch.equal(a.grad,b.grad))
        self.assertTrue(torch.equal(flat,torch.nn.functional.cross_entropy(a.unsqueeze(0),torch.tensor([2]))))
        self.assertGreater(float(point.detach()),0)

    def test_point_extra_gradient_zero_on_other_kind_and_nonpoint_zero(self):
        a=torch.zeros(3,requires_grad=True)
        _,kind,point,_=loss_parts(a,[0,1,1],2)
        point.backward();self.assertEqual(float(a.grad[0]),0);self.assertEqual(float(kind),0)
        for kinds,label in (([0,0],1),([0,1],0),([1],0)):
            flat,kind,point,total=loss_parts(torch.zeros(len(kinds)),kinds,label)
            self.assertEqual(float(point),0);self.assertEqual(float(kind),0);self.assertTrue(torch.equal(flat,total))

    def test_actual_batch_denominator_with_zeros(self):
        tensors=[torch.zeros(3,requires_grad=True) for _ in range(3)]
        terms=[loss_parts(s,[0,1,1],label)[3] for s,label in zip(tensors,[2,0,0])]
        (sum(terms)/3).backward()
        ref=torch.zeros(3,requires_grad=True)
        native=torch.nn.functional.cross_entropy(ref.unsqueeze(0),torch.tensor([2]))+0.5*torch.nn.functional.cross_entropy(ref[1:].unsqueeze(0),torch.tensor([1]))
        (native/3).backward()
        self.assertTrue(torch.equal(tensors[0].grad,ref.grad))
        self.assertFalse(torch.equal(tensors[0].grad,ref.grad*3/8))

    def test_invalid_float_kind_index_and_nonfinite_rejected(self):
        for values,kinds,label in (([float('inf')],[1],0),([0.],[True],0),([0.],[1],True),([0.],[3],0),([0.],[1],1)):
            with self.assertRaises(ValueError):loss_parts(torch.tensor(values),kinds,label)

    def test_joint_and_paired_observation_identity(self):
        def row(i,k,c):return {'id':str(i),'teacherKind':'single-cell','selectedKind':'single-cell' if k else 'reroll','globalCorrect':k and c,'conditional':{'oracleCorrect':c}}
        a=[row(0,True,False),row(1,False,True),row(2,False,False)]
        b=[row(0,True,True),row(1,False,False),row(2,True,False)]
        self.assertEqual(point_joint(a),{'examples':3,'bothCorrect':0,'kindOnly':1,'coordinateOnly':1,'neither':1})
        p=point_paired(a,b);self.assertEqual(p['actual']['gain'],1);self.assertEqual(p['oracle']['regression'],1);self.assertEqual(p['kind']['gain'],1)
        with self.assertRaises(ValueError):point_paired(a,list(reversed(b)))
        with self.assertRaises(ValueError):point_joint([{**a[0],'globalCorrect':True}])

    def test_data_loader_and_replay_compute_guard_are_pure(self):
        self.assertEqual(runner.spatial_load.__module__,'point_objective_replay')
        for action in (lambda:nn.Identity(),lambda:torch.load('missing')):
            with self.assertRaises(ValueError):
                with compute_guard():action()

    def test_failure_output_exclusive_no_model_or_forward(self):
        import sys
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root/'.research-output') as d:
            args=['point_only.py','--mode','preflight','--source','none','--original','none','--experiment','none','--execution',str(Path(d)/'execution.json'),'--output',str(Path(d)/'preflight')]
            with patch.object(sys,'argv',args),patch('point_only.subprocess.check_output',return_value=''),patch('point_only.torch.set_num_interop_threads'):
                with self.assertRaises(FileNotFoundError):runner.main()
                failure=(Path(d)/'preflight/failure.json').read_bytes()
                with self.assertRaises(FileExistsError):runner.main()
                self.assertEqual(failure,(Path(d)/'preflight/failure.json').read_bytes())
                self.assertFalse((Path(d)/'forward-count.jsonl').exists())


if __name__=='__main__':unittest.main()
