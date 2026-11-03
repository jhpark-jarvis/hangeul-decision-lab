"""Tensor/scalar/ledger tests; no research model, optimizer or checkpoint load."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import outcome_mlp as runner


class OutcomeMlpTests(unittest.TestCase):
    def setUp(self):
        for name in ["torch.nn.Module.__init__", "torch.load", "torch.optim.Optimizer.__init__"]:
            guard=patch(name,side_effect=AssertionError("No extra NN/optimizer/load"))
            guard.start(); self.addCleanup(guard.stop)

    def test_input_whitelist_ignores_all_context_and_label(self):
        e={"candidateFeatures":[[object()]*52+[i/22 for i in range(22)] for _ in range(31)],
           "observation":object(),"label":object(),"candidateKeys":object(),"footprints":object()}
        values,=runner.outcome_tensors(e)
        self.assertEqual(tuple(values.shape),(31,22));self.assertEqual(values.dtype,torch.float32)
        self.assertTrue(torch.equal(values[0],torch.tensor([i/22 for i in range(22)])))
        with self.assertRaises(ValueError):runner.outcome_tensors({"candidateFeatures":[[0.]*74]})
        with self.assertRaises(ValueError):runner.outcome_tensors({"candidateFeatures":[[0.]*52+[float("nan")]*22]*31})

    def test_both_complete_ledgers_no_model(self):
        with tempfile.TemporaryDirectory(prefix="outcome-mlp-ledger-") as d:
            for arm in ["control","candidate"]:
                ledger=runner.MlpLedger(Path(d)/arm,arm)
                for phase,n in runner.LIMITS.items():
                    for i in range(n):ledger.reserve(phase,i//16+1 if phase=="train" else None,i%16 if phase=="train" else i)
                rows=[json.loads(s) for s in ledger.path.read_text().splitlines()]
                self.assertEqual(len(rows),16336);self.assertEqual(rows[-1]["call"],16336 if arm=="control" else 32672)
                with self.assertRaises(ValueError):ledger.reserve("transfer2",None,144)

    def test_ledger_rejects_phase_step_example_and_reuse_before_write(self):
        with tempfile.TemporaryDirectory(prefix="outcome-mlp-guard-") as d:
            path=Path(d)/"ledger";ledger=runner.MlpLedger(path,"control")
            for args in [("train",1,0),("initial",1,0),("initial",None,1)]:
                with self.assertRaises(ValueError):ledger.reserve(*args)
            self.assertEqual(path.read_bytes(),b"")
            with self.assertRaises(FileExistsError):runner.MlpLedger(path,"control")

    def test_native_scoring_ties_and_all_gates(self):
        row=runner.score_row(torch.zeros(31),1,"synthetic")
        self.assertEqual(row["index"],0);self.assertEqual(row["rank"],2);self.assertFalse(row["correct"])
        base={"examples":144,"correct":144,"meanNll":0.05,"minimumTargetMargin":0.01}
        self.assertEqual(runner.gate(base)["status"],"PASS")
        for k,v in [("correct",143),("meanNll",0.0501),("minimumTargetMargin",0)]:
            self.assertEqual(runner.gate({**base,k:v})["status"],"FAIL")
        with self.assertRaises(ValueError):runner.summarize_transfer([row]*144)


if __name__=="__main__":unittest.main()
