"""Scalar/tensor/ledger tests only. No NN, optimizer, checkpoint load or training."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
import board_context_fit as runner


class BoardContextTests(unittest.TestCase):
    def setUp(self):
        for name in ["torch.nn.Module.__init__", "torch.optim.Optimizer.__init__", "torch.load", "board_context_fit.make_model"]:
            guard = patch(name, side_effect=AssertionError("No additional NN/optimizer/load"))
            guard.start()
            self.addCleanup(guard.stop)

    def test_native_ce_ties_and_guard(self):
        scores = torch.linspace(-1, 1, 32)
        self.assertEqual(runner.loss(scores, 20).item(), F.cross_entropy(scores.unsqueeze(0), torch.tensor([20])).item())
        row = runner.score_row(torch.zeros(32), 1, 0)
        self.assertEqual((row["index"], row["rank"], row["margin"]), (0, 2, 0))
        self.assertAlmostEqual(row["nll"], math.log(32), places=6)
        for values in [torch.zeros(31), torch.full((32,), float("nan")), torch.zeros(32, dtype=torch.float64)]:
            with self.assertRaises(ValueError): runner.loss(values, 0)

    def test_only_occupancy_is_masked_and_inputs_are_whitelisted(self):
        e = {"observation": {"planes": [[[1] * 10 for _ in range(16)], [[0] * 10 for _ in range(16)], [[1] * 10 for _ in range(16)]], "metadata": [0.5] * 140},
             "candidateFeatures": [[0.25] * 74 for _ in range(32)], "footprints": ["1" + "0" * 159] * 32,
             "state": object(), "label": object(), "descriptor": object(), "reference": object()}
        control, candidate = runner.tensors(e, "control"), runner.tensors(e, "candidate")
        self.assertTrue(torch.equal(control[0][:, 0], torch.zeros((1, 16, 10))))
        self.assertTrue(torch.equal(control[0][:, 1:], candidate[0][:, 1:]))
        for a, b in zip(control[1:], candidate[1:]): self.assertTrue(torch.equal(a, b))
        self.assertEqual(e["observation"]["planes"][0][0], [1] * 10)
        self.assertEqual(len(runner.input_keys(control)), 32)
        self.assertNotEqual(runner.input_keys(control), runner.input_keys(candidate))

    def test_identity_checks_exact_fp32_without_forward(self):
        rows = [{"logits": [0.5, 1.0]}, {"logits": [1.0, 0.5]}]
        keys = [["a", "b"], ["b", "a"]]
        self.assertEqual(runner.native_identity(rows, keys)["uniqueInputs"], 2)
        rows[1]["logits"][0] = 1.25
        with self.assertRaises(ValueError): runner.native_identity(rows, keys)

    def test_complete64192_scalar_reservations_no_model(self):
        with tempfile.TemporaryDirectory(prefix="board-ledger-") as d:
            for arm in ["control", "candidate"]:
                ledger = runner.BoardLedger(Path(d) / arm, arm)
                for phase, n in runner.LIMITS.items():
                    for i in range(n): ledger.reserve(phase, i // 32 + 1 if phase == "train" else None, i % 32)
                rows = [json.loads(s) for s in ledger.path.read_text().splitlines()]
                self.assertEqual(len(rows), 32096)
                self.assertEqual(rows[-1]["call"], 32096 if arm == "control" else 64192)
                with self.assertRaises(ValueError): ledger.reserve("final2", None, 0)

    def test_rejected_reservations_do_not_write_and_disk_error_does_not_count(self):
        with tempfile.TemporaryDirectory(prefix="board-ledger-guard-") as d:
            path = Path(d) / "ledger"
            ledger = runner.BoardLedger(path, "control")
            for args in [("train", 1, 0), ("initial", 1, 0), ("initial", None, 1)]:
                with self.assertRaises(ValueError): ledger.reserve(*args)
            self.assertEqual(path.read_bytes(), b"")
            with patch("pathlib.Path.open", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError): ledger.reserve("initial", None, 0)
            self.assertEqual(sum(ledger.counts.values()), 0)
            with self.assertRaises(FileExistsError): runner.BoardLedger(path, "control")

    def test_quality_gate_is_separate_from_technical_validity(self):
        self.assertEqual(runner.fit_gate({"correct": 32, "meanNll": 0.01, "minMargin": 0.1})["status"], "PASS")
        for values in [{"correct": 16, "meanNll": 0.8, "minMargin": -1}, {"correct": 32, "meanNll": 0.06, "minMargin": 0.1}, {"correct": 32, "meanNll": 0.01, "minMargin": 0}]:
            self.assertEqual(runner.fit_gate(values)["status"], "FAIL")


if __name__ == "__main__": unittest.main()
