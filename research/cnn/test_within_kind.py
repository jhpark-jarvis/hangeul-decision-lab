"""Frozen scoring/budget checks; actual experiment inference is not run here."""
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import torch
from within_kind import conditional_score, aggregate, controlled_row, bounded_forward, ROOT, main
from diagnose import score

KEYS = ['["place-piece","a",0,"DOT",0,0,[[true]]]', '["single-cell",0,0]', '["single-cell",1,0]', '["reroll","a",0]']


class WithinKindTests(unittest.TestCase):
    def test_oracle_recovery_is_separate_from_global_prediction(self):
        logits = torch.tensor([5., 1., 2., 0.])
        r = conditional_score(logits, KEYS, 2)
        self.assertEqual(int(logits.argmax()), 0)
        self.assertEqual(r['oracleIndex'], 2)
        self.assertTrue(r['oracleCorrect'])
        self.assertAlmostEqual(r['oracleNll'], math.log(1+math.exp(-1)), places=6)
        self.assertEqual(r['kindMass']['single-cell']['candidates'], 2)
        self.assertEqual(r['kindMass']['single-cell']['uniformMass'], 0.5)
        self.assertAlmostEqual(sum(x['probabilityMass'] for x in r['kindMass'].values()), 1)

    def test_stable_ties_preserve_original_full_index_and_labels(self):
        r = conditional_score(torch.tensor([0., 3., 3., 0.]), KEYS, 2)
        self.assertEqual(r['oracleIndex'], 1)
        self.assertEqual(r['oracleRank'], 2)
        self.assertFalse(r['oracleCorrect'])
        self.assertAlmostEqual(r['oracleNll'], math.log(2), places=6)

    def test_kind_mass_is_stable_for_large_offsets_and_absent_kinds(self):
        r = conditional_score(torch.tensor([1e20, 1e20, 1e20, 1e20]), KEYS, 0)
        self.assertAlmostEqual(r['kindMass']['single-cell']['probabilityMass'], 0.5)
        one = conditional_score(torch.tensor([1000.]), [KEYS[1]], 0)
        self.assertEqual(one['oracleRank'], 1)
        self.assertEqual(one['oracleNll'], 0)
        self.assertEqual(one['kindMass']['reroll']['probabilityMass'], 0)
        self.assertEqual(one['kindMass']['reroll']['candidates'], 0)

    def test_nonfinite_invalid_label_mask_and_key_fail_closed(self):
        for logits, keys, label in ((torch.tensor([float('nan')]), [KEYS[1]], 0),
                                   (torch.zeros(4), KEYS, True), (torch.zeros(4), KEYS, 4),
                                   (torch.zeros(4), KEYS[:3], 0), (torch.zeros(2), [KEYS[1]]*2, 0),
                                   (torch.zeros(1), ['["other"]'], 0)):
            with self.assertRaises(ValueError): conditional_score(logits, keys, label)

    def test_original_score_control_rejects_drift_before_oracle_statistics(self):
        logits = torch.tensor([5., 1., 2., 0.])
        example = {'id':'x','fixtureId':'f','candidateKeys':KEYS,'labels':{'A512':2,'A1024':0}}
        old = {**score(logits, example['labels']), 'id':'x'}
        audit = {'id':'x','fixtureId':'f','correct':False,'teacherKind':'single-cell','family':'sparse',
                 'error':'wrong-kind','searchComplete':False,'teacherFinalPhase':'playing','legalKinds':'mixed'}
        self.assertFalse(controlled_row(logits, example, old, audit)['globalCorrect'])
        with self.assertRaisesRegex(ValueError, 'STOP'):
            controlled_row(logits, example, {**old, 'index':2}, audit)

    def test_forward_budget_prevents_the_extra_model_call(self):
        model = Mock(return_value=torch.tensor([0.]))
        budget = {'calls':0,'limit':2}
        bounded_forward(model, (), budget); bounded_forward(model, (), budget)
        with self.assertRaisesRegex(ValueError, 'STOP'): bounded_forward(model, (), budget)
        self.assertEqual(model.call_count, 2)
        self.assertEqual(budget['calls'], 2)

    def test_row_weighted_aggregate_and_empty_group_are_not_policy_claims(self):
        conditional = conditional_score(torch.tensor([5., 1., 2., 0.]), KEYS, 2)
        rows = [{'fixtureId':'one','globalCorrect':False,'error':'wrong-kind','conditional':conditional}]*2
        r = aggregate(rows)
        self.assertEqual(r['globalHardCorrect'], 0)
        self.assertEqual(r['teacherKindOracleCorrect'], 2)
        self.assertEqual(r['wrongKindOracleRecoveries'], 2)
        self.assertEqual(r['kindMass']['single-cell']['meanUniformMass'], 0.5)
        self.assertIsNone(aggregate([])['teacherKindOracleTop1'])

    def test_cli_missing_source_failure_and_output_reuse_have_zero_model_calls(self):
        parent = ROOT/'.research-output'; parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent, prefix='test-within-kind-') as directory:
            base = Path(directory); self.assertTrue(base.resolve().is_relative_to(parent))
            output = base/'failed'
            command = ['within_kind.py', '--mode','infer',
                       '--source',str(base/'missing'),'--experiment',str(base/'missing'),
                       '--audit-raw',str(base/'missing.json'),'--output',str(output)]
            with patch.object(sys,'argv',command), patch('within_kind.subprocess.check_output',return_value=''):
                with self.assertRaises(FileNotFoundError): main()
            failure = (output/'failure.json').read_bytes()
            self.assertFalse((output/'forward-count.jsonl').exists())
            with patch.object(sys,'argv',command), patch('within_kind.subprocess.check_output',return_value=''):
                with self.assertRaises(FileExistsError): main()
            self.assertEqual((output/'failure.json').read_bytes(),failure)

    def test_dirty_source_stops_before_output_or_inference(self):
        with patch.object(sys,'argv',['within_kind.py','--mode','infer','--source','missing',
                                     '--experiment','missing','--audit-raw','missing',
                                     '--output',str(ROOT/'.research-output/never-created-within-kind')]), \
                patch('within_kind.subprocess.check_output',return_value=' M source.py'), \
                patch('within_kind.run') as run:
            with self.assertRaisesRegex(ValueError,'Commit'): main()
            run.assert_not_called()


if __name__ == '__main__': unittest.main()
