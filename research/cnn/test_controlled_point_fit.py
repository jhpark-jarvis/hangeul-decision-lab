"""Synthetic scalar/native FP32 checks only; zero NN creation, forward or training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F
import controlled_point_fit as runner


class ControlledFitTests(unittest.TestCase):
    def setUp(self):
        self.guards = [patch("torch.nn.Module.__init__", side_effect=AssertionError("No model construction")),
                       patch("torch.load", side_effect=AssertionError("No checkpoint load")),
                       patch("torch.optim.Adam", side_effect=AssertionError("No training"))]
        for p in self.guards: p.start()
        self.addCleanup(lambda: [p.stop() for p in reversed(self.guards)])

    def test_native_fp32_loss_gradient_and_fullbatch_normalization(self):
        a = torch.linspace(-1, 1, 31, requires_grad=True)
        b = a.detach().clone().requires_grad_()
        loss = runner.flat_loss(a, 7)
        expected = F.cross_entropy(b.unsqueeze(0), torch.tensor([7]))
        self.assertTrue(torch.equal(loss, expected))
        (loss / 16).backward(); (expected / 16).backward()
        self.assertTrue(torch.equal(a.grad, b.grad))

    def test_bad_loss_schema_label_nonfinite(self):
        for x, y in [(torch.zeros(30), 1), (torch.zeros(31, dtype=torch.float64), 1),
                     (torch.full((31,), float("nan")), 0), (torch.zeros(31), -1), (torch.zeros(31), True)]:
            with self.assertRaises(ValueError): runner.flat_loss(x, y)

    def test_first_index_tie_rank_margin(self):
        row = runner.score_row(torch.zeros(31), 1, "synthetic")
        self.assertEqual(row["index"], 0); self.assertEqual(row["rank"], 2)
        self.assertEqual(row["margin"], 0); self.assertFalse(row["correct"])

    def test_gate_keeps_accuracy_nll_margin_separate(self):
        p = {"correct": 16, "meanNll": 0.05, "minimumTargetMargin": 1}
        self.assertEqual(runner.fit_gate(p)["status"], "PASS")
        for key, value in [("correct", 15), ("meanNll", 0.050001), ("minimumTargetMargin", 0)]:
            self.assertEqual(runner.fit_gate({**p, key: value})["status"], "FAIL")

    def test_metrics_reject_duplicate_or_wrong_denominator(self):
        for rows in [[], [{"id":"x"}] * 16]:
            with self.assertRaises(ValueError): runner.summarize(rows)

    def test_ledger_records_all_reserved_calls_before_execution_and_rejects_extra(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.jsonl"
            ledger = runner.Ledger(path)
            for phase, count in runner.LIMITS.items():
                for i in range(count): ledger.reserve(phase, i // 16 + 1 if phase == "train" else None, i % 16)
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual(len(rows), 3248); self.assertEqual(rows[-1]["call"], 3248)
            with self.assertRaises(ValueError): ledger.reserve("final2", None, 0)
            self.assertEqual(len(path.read_text().splitlines()), 3248)

    def test_ledger_rejects_phase_step_order_without_writing(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ledger.jsonl"
            ledger = runner.Ledger(path)
            for values in [("train", 1, 0), ("initial", None, 1), ("initial", 1, 0)]:
                with self.assertRaises(ValueError): ledger.reserve(*values)
            self.assertEqual(path.read_bytes(), b"")
            with self.assertRaises(FileExistsError): runner.Ledger(path)


if __name__ == "__main__": unittest.main()
