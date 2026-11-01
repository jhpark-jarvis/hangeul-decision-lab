"""Scalar/tensor/ledger guards only, no extra research models or checkpoint loads."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import frozen_point_transfer as runner


class FrozenTransferTests(unittest.TestCase):
    def setUp(self):
        guards = [patch("torch.nn.Module.__init__", side_effect=AssertionError("No extra NN")),
                  patch("torch.load", side_effect=AssertionError("No checkpoint loads")),
                  patch("torch.optim.Optimizer.__init__", side_effect=AssertionError("No optimizers"))]
        for guard in guards: guard.start()
        self.addCleanup(lambda: [g.stop() for g in reversed(guards)])

    def test_all_320_reservations_and_order_without_model(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"ledger"; ledger=runner.TransferLedger(path)
            for phase,n in runner.LIMITS.items():
                for i in range(n): ledger.reserve(phase,None,i)
            rows=[json.loads(s) for s in path.read_text().splitlines()]
            self.assertEqual(len(rows),320);self.assertEqual(rows[16]["phase"],"transfer")
            self.assertEqual(rows[-1]["call"],320)
            with self.assertRaises(ValueError): ledger.reserve("transfer-repeat",None,144)

    def test_mismatch_stops_before_transfer_and_reuse_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"ledger";ledger=runner.TransferLedger(path)
            for args in [("transfer",None,0),("fit-control",None,1),("fit-control",1,0)]:
                with self.assertRaises(ValueError): ledger.reserve(*args)
            self.assertEqual(path.read_bytes(),b"")
            with self.assertRaises(FileExistsError): runner.TransferLedger(path)

    def test_fit_and_transfer_denominators_do_not_mix(self):
        rows=[{"id":str(i),"correct":True,"nll":0.02,"margin":1.0,"rank":1} for i in range(144)]
        self.assertEqual(runner.summary(rows,144)["correct"],144)
        with self.assertRaises(ValueError): runner.summary(rows,16)
        with self.assertRaises(ValueError): runner.summary([rows[0]]*144,144)

    def test_no_training_guard_blocks_backward_load_optimizer_and_training_mode(self):
        with runner.no_training():
            for f in [lambda:torch.load("unused"),lambda:torch.tensor(1.,requires_grad=True).backward(),
                      lambda:torch.autograd.backward(torch.tensor(1.)),lambda:torch.optim.Optimizer.__init__(None,[],{}),
                      lambda:torch.nn.Module.train(None,True)]:
                with self.assertRaises(ValueError):f()

    def test_scoring_reuses_native_float32_and_tie_order_without_model(self):
        row=runner.score_row(torch.zeros(31),1,"synthetic")
        self.assertEqual(row["index"],0);self.assertEqual(row["rank"],2)
        self.assertEqual(row["margin"],0);self.assertFalse(row["correct"])
        with self.assertRaises(ValueError):runner.score_row(torch.full((31,),float("nan")),1,"bad")


if __name__=="__main__":unittest.main()
