"""Only tensors/scalars/ledger; no research networks, optimizers or loads."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import coverage_training as runner
from controlled_point_fit import flat_loss


class CoverageTrainingTests(unittest.TestCase):
    def setUp(self):
        for name in ["torch.nn.Module.__init__","torch.optim.Optimizer.__init__","torch.load"]:
            p=patch(name,side_effect=AssertionError("No extra NN/optimizer/load"));p.start();self.addCleanup(p.stop)
    def test_same_native_ce_for_anchor_and46_count_guard(self):
        s=torch.linspace(-1,1,31)
        self.assertEqual(runner.loss(s,15).item(),flat_loss(s,15).item())
        row=runner.score_row(torch.zeros(46),1,"synthetic")
        self.assertEqual((row["index"],row["rank"],row["margin"]),(0,2,0))
        for s in [torch.zeros(32),torch.full((46,),float("nan"))]:
            with self.assertRaises(ValueError):runner.loss(s,0)
    def test_whitelist_excludes_board_coordinates_keys_and_label(self):
        e={"candidateFeatures":[[object()]*52+[0.5]*22]*46,"label":object(),"observation":object(),"footprints":object()}
        values,=runner.tensors(e,46);self.assertEqual(tuple(values.shape),(46,22));self.assertTrue(torch.equal(values,torch.full((46,22),0.5)))
        with self.assertRaises(ValueError):runner.tensors(e,31)
    def test_both_ledgers_full32224_without_network(self):
        with tempfile.TemporaryDirectory(prefix="coverage-training-ledger-") as d:
            for arm in ["control","diverse"]:
                ledger=runner.CoverageLedger(Path(d)/arm,arm)
                for phase,n in runner.LIMITS.items():
                    for i in range(n):ledger.reserve(phase,i//16+1 if phase=="train" else None,i%16)
                rows=[json.loads(s) for s in ledger.path.read_text().splitlines()]
                self.assertEqual(len(rows),16112);self.assertEqual(rows[-1]["call"],16112 if arm=="control" else 32224)
                with self.assertRaises(ValueError):ledger.reserve("final2",None,0)
    def test_wrong_phase_step_index_and_reuse_reject_before_write(self):
        with tempfile.TemporaryDirectory(prefix="coverage-training-guard-") as d:
            path=Path(d)/"ledger";ledger=runner.CoverageLedger(path,"control")
            for args in [("train",1,0),("anchor",1,0),("anchor",None,1)]:
                with self.assertRaises(ValueError):ledger.reserve(*args)
            self.assertEqual(path.read_bytes(),b"")
            with self.assertRaises(FileExistsError):runner.CoverageLedger(path,"control")


if __name__=="__main__":unittest.main()
