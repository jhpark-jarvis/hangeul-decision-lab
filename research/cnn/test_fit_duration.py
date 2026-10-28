"""Synthetic tensors/state and ledger only; model creation/load and Adam are forbidden."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
import fit_duration as runner


class DurationTests(unittest.TestCase):
    def setUp(self):
        guards = [patch("torch.nn.Module.__init__", side_effect=AssertionError("No extra models")),
                  patch("torch.load", side_effect=AssertionError("No loads")),
                  patch("torch.optim.Adam", side_effect=AssertionError("No training"))]
        for g in guards: g.start()
        self.addCleanup(lambda: [g.stop() for g in reversed(guards)])

    def test_optimizer_hash_is_order_independent_and_includes_state_and_groups(self):
        state = {"state": {0: {"step": torch.tensor(200.), "exp_avg": torch.tensor([0.1, 0.2])}}, "param_groups": [{"params": [0], "lr": 0.001}]}
        reordered = {"param_groups": state["param_groups"], "state": state["state"]}
        digest = runner.optimizer_fingerprint(state)
        self.assertEqual(digest, runner.optimizer_fingerprint(reordered))
        for changed in [{**state, "param_groups": [{"params": [0], "lr": 0.002}]},
                        {**state, "state": {0: {"step": torch.tensor(199.), "exp_avg": torch.tensor([0.1, 0.2])}}}]:
            self.assertNotEqual(digest, runner.optimizer_fingerprint(changed))

    def test_optimizer_tensor_dtype_shape_and_nonfinite(self):
        a = runner.optimizer_fingerprint(torch.tensor([1.]))
        self.assertNotEqual(a, runner.optimizer_fingerprint(torch.tensor([[1.]])))
        self.assertNotEqual(a, runner.optimizer_fingerprint(torch.tensor([1.], dtype=torch.float64)))
        for value in [torch.tensor([float("nan")]), {"bad": float("inf")}, object()]:
            with self.assertRaises(ValueError): runner.optimizer_fingerprint(value)

    def test_prefix_exact_to_original_and_concurrent_only(self):
        original = {"finalLogicalWeightsSha256": "a" * 64, "historySha256": "b" * 64}
        prefix = {"step": 200, "weightsSha256": "a" * 64, "historySha256": "b" * 64, "optimizerSha256": "c" * 64}
        runner.prefix_exact(prefix, original, prefix.copy())
        for key in ("weightsSha256", "historySha256", "optimizerSha256"):
            with self.assertRaises(ValueError): runner.prefix_exact({**prefix, key: "d" * 64}, original, prefix)

    def test_complete_phase_and_global_reservations_without_model_calls(self):
        with tempfile.TemporaryDirectory() as d:
            total = 0
            for arm in runner.UPDATES:
                path = Path(d) / (arm + ".jsonl"); ledger = runner.DurationLedger(path, arm)
                for phase, n in ledger.limits.items():
                    for i in range(n): ledger.reserve(phase, i // 16 + 1 if phase == "train" else None, i % 16)
                rows = [json.loads(x) for x in path.read_text().splitlines()]
                self.assertEqual(rows[0]["call"], total + 1); total += len(rows)
                self.assertEqual(rows[-1]["call"], total)
                with self.assertRaises(ValueError): ledger.reserve("final2", None, 0)
            self.assertEqual(total, 19296)

    def test_ledger_rejects_bad_arm_phase_step_order_and_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError): runner.DurationLedger(Path(d) / "bad", "extra")
            path = Path(d) / "ledger"; ledger = runner.DurationLedger(path, "control")
            for values in [("train", 1, 0), ("initial", None, 1), ("initial", 1, 0)]:
                with self.assertRaises(ValueError): ledger.reserve(*values)
            self.assertEqual(path.read_bytes(), b"")
            with self.assertRaises(FileExistsError): runner.DurationLedger(path, "control")


if __name__ == "__main__": unittest.main()
