"""Kind-only native FP32 contract on synthetic tensors; no real-data model calls."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F
import kind_only as runner
from kind_only import loss_parts, kind_indices, point_joint, point_paired, error_transitions
from spatial_audit import compute_guard


def reference(s, kinds, label):
    present=sorted(set(kinds))
    values=torch.stack([torch.logsumexp(s[[i for i,k in enumerate(kinds) if k==g]],0)-math.log(kinds.count(g)) for g in present])
    return F.cross_entropy(s.unsqueeze(0),torch.tensor([label]))+0.5*F.cross_entropy(values.unsqueeze(0),torch.tensor([present.index(kinds[label])]))


class KindOnlyTests(unittest.TestCase):
    def test_native_fp32_loss_and_gradient_exact_reference(self):
        a=torch.tensor([0.2,0.7,-0.3,0.1],requires_grad=True)
        b=a.detach().clone().requires_grad_()
        flat,kind,point,total=loss_parts(a,[0,1,1,2],2)
        expected=reference(b,[0,1,1,2],2)
        self.assertTrue(torch.equal(total,expected));self.assertEqual(float(point),0)
        total.backward();expected.backward();self.assertTrue(torch.equal(a.grad,b.grad))
        self.assertTrue(torch.equal(flat,F.cross_entropy(a.unsqueeze(0),torch.tensor([2]))))
        self.assertGreater(float(kind.detach()),0)

    def test_point_auxiliary_never_evaluated_even_for_point_target(self):
        with patch('kind_only.F.cross_entropy',wraps=F.cross_entropy) as ce:
            _,_,point,_=loss_parts(torch.zeros(3),[0,1,1],2)
        self.assertEqual(ce.call_count,2);self.assertEqual(float(point),0)

    def test_single_present_kind_extra_loss_and_gradient_zero(self):
        s=torch.tensor([2.,1.,-1.],requires_grad=True)
        flat,kind,point,total=loss_parts(s,[1,1,1],1)
        self.assertEqual(float(kind.detach()),0);self.assertEqual(float(point),0)
        kind.backward(retain_graph=True);self.assertTrue(torch.equal(s.grad,torch.zeros(3)))
        self.assertTrue(torch.equal(flat,total));total.backward();self.assertTrue(torch.isfinite(s.grad).all())

    def test_absent_kinds_and_candidate_count_normalization(self):
        _,kind,_,_=loss_parts(torch.zeros(3),[2,2,0],2)
        self.assertTrue(torch.equal(kind,F.cross_entropy(torch.zeros(1,2),torch.tensor([0]))))
        first=loss_parts(torch.tensor([2.,-1.,0.]),[0,0,1],2)[1]
        cloned=loss_parts(torch.tensor([2.,-1.,2.,-1.,0.]),[0,0,0,0,1],4)[1]
        torch.testing.assert_close(first,cloned,rtol=1e-6,atol=1e-6)

    def test_same_target_kind_has_no_coordinate_specific_extra_target(self):
        s=torch.tensor([2.,1.,0.])
        a=loss_parts(s,[1,1,0],0);b=loss_parts(s,[1,1,0],1)
        self.assertTrue(torch.equal(a[1],b[1]));self.assertFalse(torch.equal(a[0],b[0]))

    def test_stable_large_common_shift(self):
        s=torch.tensor([2.,1.,0.])
        torch.testing.assert_close(loss_parts(s,[0,1,1],2)[1],loss_parts(s+1000,[0,1,1],2)[1],rtol=1e-4,atol=1e-4)

    def test_kind_analytical_gradient_and_finite_difference(self):
        s=torch.tensor([0.2,0.7,-0.3,0.1],requires_grad=True);kinds=[0,1,1,2]
        kind=loss_parts(s,kinds,2)[1];kind.backward()
        groups=[[0],[1,2],[3]]
        logits=torch.stack([torch.logsumexp(s[g],0)-math.log(len(g)) for g in groups]).detach()
        p=logits.softmax(0)
        for g,indices in enumerate(groups):
            expected=(p[g]-int(g==1))*s.detach()[indices].softmax(0)
            torch.testing.assert_close(s.grad[indices],expected,rtol=1e-6,atol=1e-7)
        for i in range(4):
            plus=s.detach().clone();minus=plus.clone();plus[i]+=0.01;minus[i]-=0.01
            diff=(loss_parts(plus,kinds,2)[1]-loss_parts(minus,kinds,2)[1])/0.02
            torch.testing.assert_close(diff,s.grad[i],rtol=0.002,atol=2e-5)

    def test_actual_tail_batch_mean_includes_zero_kind_terms(self):
        labels=[2,0,1];kinds=[[0,1,1],[0,0,0],[2,2,2]]
        tensors=[torch.zeros(3,requires_grad=True) for _ in range(3)]
        for s,k,y in zip(tensors,kinds,labels):(loss_parts(s,k,y)[3]/3).backward()
        ref=torch.zeros(3,requires_grad=True);(reference(ref,kinds[0],labels[0])/3).backward()
        self.assertTrue(torch.equal(tensors[0].grad,ref.grad))
        self.assertFalse(torch.equal(tensors[0].grad,ref.grad*3/8))

    def test_invalid_loss_inputs_rejected(self):
        for values,kinds,label in (([float('inf')],[1],0),([float('nan')],[1],0),([] ,[],0),([0.],[True],0),([0.],[1],True),([0.],[3],0),([0.],[1],1),([0.],[0,1],0)):
            with self.assertRaises(ValueError):loss_parts(torch.tensor(values),kinds,label)
        with self.assertRaises(ValueError):loss_parts(torch.zeros(1,dtype=torch.float64),[0],0)

    def test_kind_feature_key_label_and_order_identity(self):
        e={'candidateFeatures':[[1,0,0]+[0]*71,[0,1,0]+[0]*71],
           'candidateKeys':[json.dumps(['place-piece',0]),json.dumps(['single-cell',0])],
           'label':1,'labels':{'A512':1}}
        self.assertEqual(kind_indices(e),[0,1])
        for changed in ({**e,'label':0},{**e,'candidateKeys':[e['candidateKeys'][0]]*2},
                        {**e,'candidateKeys':list(reversed(e['candidateKeys']))},
                        {**e,'candidateFeatures':[[1,1,0]+[0]*71,e['candidateFeatures'][1]]}):
            with self.assertRaises(ValueError):kind_indices(changed)

    def test_joint_paired_and_error_transition_denominators(self):
        def row(i,k,c):return {'id':str(i),'teacherKind':'single-cell','selectedKind':'single-cell' if k else 'reroll','globalCorrect':k and c,'conditional':{'oracleCorrect':c},'error':'exact-label' if k and c else 'wrong-kind' if not k else 'different-position'}
        a=[row(0,True,False),row(1,False,True),row(2,False,False)]
        b=[row(0,True,True),row(1,False,False),row(2,True,False)]
        self.assertEqual(point_joint(a),{'examples':3,'bothCorrect':0,'kindOnly':1,'coordinateOnly':1,'neither':1})
        p=point_paired(a,b);self.assertEqual(p['actual']['gain'],1);self.assertEqual(p['oracle']['regression'],1)
        self.assertEqual(sum(error_transitions(a,b).values()),3)
        with self.assertRaises(ValueError):error_transitions(a,list(reversed(b)))
        with self.assertRaises(ValueError):point_paired(a,list(reversed(b)))
        with self.assertRaises(ValueError):point_joint([{**a[0],'globalCorrect':True}])

    def test_data_loader_and_replay_guard_no_real_models(self):
        self.assertEqual(runner.spatial_load.__module__,'point_objective_replay')
        for action in (lambda:nn.Identity(),lambda:torch.load('missing')):
            with self.assertRaises(ValueError):
                with compute_guard():action()

    def test_failure_output_exclusive_no_model_or_forward(self):
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root/'.research-output') as d:
            args=['kind_only.py','--mode','preflight','--source','none','--original','none','--experiment','none','--execution',str(Path(d)/'execution.json'),'--output',str(Path(d)/'preflight')]
            with patch.object(sys,'argv',args),patch('kind_only.subprocess.check_output',return_value=''),patch('kind_only.torch.set_num_interop_threads'):
                with self.assertRaises(FileNotFoundError):runner.main()
                before=(Path(d)/'preflight/failure.json').read_bytes()
                with self.assertRaises(FileExistsError):runner.main()
                self.assertEqual(before,(Path(d)/'preflight/failure.json').read_bytes())
                self.assertFalse((Path(d)/'forward-count.jsonl').exists())


if __name__=='__main__':unittest.main()
