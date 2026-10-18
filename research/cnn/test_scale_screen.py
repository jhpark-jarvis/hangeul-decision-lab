"""Toy arithmetic/contract checks; no real-data model calls or training."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from scale_screen import scaled_logits, screen_row, full_aggregate, full_summary, compact_comparison, main
from component_audit import training_guard
from spatial_audit import controlled_row, compute_guard
from test_spatial_audit import fixture


class ScaleTests(unittest.TestCase):
    def test_fixed_five_float32_endpoints_and_order(self):
        c, r = torch.tensor([2., 1., 0.]), torch.tensor([-1., 2., 0.])
        with training_guard(), compute_guard():
            scores = [scaled_logits(c,r,s) for s in (0,0.25,0.5,0.75,1)]
        self.assertTrue(torch.equal(scores[0], c))
        self.assertTrue(torch.equal(scores[-1], c+r))
        self.assertEqual(scores[2].tolist(), [1.5,2.,0.])
        self.assertEqual(c.tolist(), [2.,1.,0.])
        self.assertEqual(r.tolist(), [-1.,2.,0.])

    def test_unapproved_scales_types_shape_dtype_and_nonfinite_rejected(self):
        c=torch.zeros(3)
        for s in (-1,0.1,2,True,float('nan')):
            with self.assertRaises(ValueError): scaled_logits(c,c,s)
        for r in (torch.zeros(2),torch.zeros(3,dtype=torch.float64),torch.tensor([0.,float('inf'),0.])):
            with self.assertRaises(ValueError): scaled_logits(c,r,0.5)
        with self.assertRaises(ValueError): scaled_logits(torch.zeros(1,3),torch.zeros(1,3),1)

    def test_scaled_overflow_stops(self):
        t=torch.tensor([torch.finfo(torch.float32).max])
        with self.assertRaisesRegex(ValueError,'Nonfinite'): scaled_logits(t,t,1)

    def test_endpoint_full_score_metadata_and_tie_first_index_exact(self):
        e,p,l=fixture((3.,2.,1.)); full=controlled_row(e,p,l,'sparse')
        with compute_guard():
            row=screen_row(e,full,torch.tensor([3.,2.,1.]),full)
            tie=screen_row(e,full,torch.tensor([1.,1.,1.]))
        self.assertIs(row,full)
        self.assertEqual(tie['selectedIndex'],0)
        with self.assertRaisesRegex(ValueError,'Endpoint'): screen_row(e,full,torch.tensor([1.,2.,3.]),full)

    def test_full_nll_errors_and_group_denominators_include_empty_case(self):
        e,p,l=fixture((3.,2.,1.)); full=controlled_row(e,p,l,'sparse')
        with compute_guard(): summary=full_summary([full,full])
        overall=summary['overall']
        self.assertEqual(overall['examples'],2)
        self.assertEqual(sum(overall['errors'].values()),2)
        self.assertEqual(overall['meanNll'],full['fullScore']['targets']['A512']['nll'])
        for groups in summary['groups'].values(): self.assertEqual(sum(r['examples'] for r in groups.values()),2)
        self.assertIsNone(full_aggregate([])['meanNll'])

    def test_paired_rescue_and_explicit_scale_names_no_policy_selection(self):
        e,p,l=fixture((3.,2.,1.)); full=controlled_row(e,p,l,'sparse')
        worse=screen_row(e,full,torch.tensor([1.,2.,0.]))
        with compute_guard(): value=compact_comparison({'common':[worse],'scale:0.5':[full]},'common','scale:0.5')
        self.assertEqual(value['rightModel'],'scale:0.5')
        self.assertEqual(value['overall']['actualHardLabel']['improvement'],1)
        self.assertEqual(value['overall']['actualHardLabel']['regression'],0)

    def test_compute_training_guards_block_and_restore(self):
        original=torch.load
        with training_guard():
            with self.assertRaises(ValueError): torch.tensor(1.,requires_grad=True).backward()
        for call in (lambda:torch.nn.Identity(),lambda:torch.load('absent')):
            with self.assertRaises(ValueError):
                with compute_guard(): call()
        self.assertIs(torch.load,original)

    def test_cli_dirty_and_outside_output_stop_before_run(self):
        args=['scale_screen.py','--source','absent','--experiment','absent','--original','absent','--recorded','absent','--output','.research-output/unused-scale-test']
        with patch.object(sys,'argv',args),patch('scale_screen.subprocess.check_output',return_value=' M file'),patch('scale_screen.run') as run:
            with self.assertRaisesRegex(ValueError,'Commit'): main()
            run.assert_not_called()
        args[-1]='.'
        with patch.object(sys,'argv',args):
            with self.assertRaisesRegex(ValueError,'ignored'): main()

    def test_missing_input_failure_exclusive_output_no_forward_ledger(self):
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root/'.research-output') as directory:
            output=Path(directory)/'out'
            args=['scale_screen.py','--source','absent','--experiment','absent','--original','absent','--recorded','absent','--output',str(output)]
            with patch.object(sys,'argv',args),patch('scale_screen.subprocess.check_output',return_value=''):
                with self.assertRaises(FileNotFoundError): main()
                before=(output/'failure.json').read_bytes()
                self.assertEqual(json.loads(before)['preservation'],'NOT_VERIFIED; separate original-file audit required')
                with self.assertRaises(FileExistsError): main()
                self.assertEqual((output/'failure.json').read_bytes(),before)
                self.assertFalse((output/'forward-count.jsonl').exists())


if __name__ == '__main__': unittest.main()
