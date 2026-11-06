"""No NN, optimizer or loads; scalar identity, recorder and adapter tests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import paired_target_fit as runner


class PairedTargetTests(unittest.TestCase):
    def setUp(self):
        for name in ["torch.nn.Module.__init__", "torch.optim.Optimizer.__init__", "torch.load", "board_context_fit.make_model"]:
            guard = patch(name, side_effect=AssertionError("No extra NN/optimizer/load"))
            guard.start()
            self.addCleanup(guard.stop)

    def pair(self):
        return [{"label": 0, "logits": [2, 1, 0]}, {"label": 1, "logits": [2, 1, 2**-149]}], [["a", "b", "x"], ["a", "b", "x"]]

    def test_irrelevant_one_ulp_is_measured_not_corrected(self):
        rows, keys = self.pair()
        before = json.dumps(rows)
        r = runner.paired_identity(rows, keys)
        self.assertEqual(r["anchors"]["nativeCorrectUpperBound"], 1)
        self.assertEqual(r["globalDrift"]["maximumUlpGap"], 1)
        self.assertEqual(r["mismatchTargetOccurrences"], 0)
        self.assertEqual(json.dumps(rows), before)

    def test_target_one_ulp_and_changed_tie_priority_stop(self):
        rows, keys = self.pair()
        rows[1]["logits"][0] = 2 + 2**-22
        with self.assertRaises(ValueError): runner.paired_identity(rows, keys)
        rows = [{"label": 0, "logits": [2, 1, 0]}, {"label": 0, "logits": [1, 2, 0]}]
        with self.assertRaises(ValueError): runner.paired_identity(rows, [["a", "b", "x"], ["b", "a", "x"]])

    def test_nonfinite_non_fp32_signed_zero_and_missing_anchor(self):
        for value in [float("nan"), float("inf"), 1 / 3]:
            with self.assertRaises(ValueError): runner.fp32_bits(value)
        self.assertEqual(runner.ulp_distance(-0.0, 0.0), 1)
        rows, keys = self.pair()
        keys[1][0] = "z"
        with self.assertRaises(ValueError): runner.paired_identity(rows, keys)

    def test_recorder_three_phases_and_report_only_no_forward(self):
        with tempfile.TemporaryDirectory(prefix="paired-recorder-") as d:
            rows = [{"label": i % 2, "logits": [2, 1] + [0] * 30} for i in range(32)]
            keys = [[str(j) for j in range(32)] for _ in range(32)]
            recorder = runner.IdentityRecorder(d)
            for _ in range(4): recorder(rows, keys)
            self.assertEqual(len(list(Path(d).glob("*-identity.json"))), 3)
            with self.assertRaises(ValueError): recorder(rows, keys)

    def test_scoped_adapter_restores_original_hook_and_configuration(self):
        old_hook, old_config = runner.original.native_identity, runner.original.CONFIG
        with patch.object(runner, "CONFIG", Path("synthetic-config")), patch("pathlib.Path.read_bytes", return_value=b'{"id":"paired-target-identity-v1"}'), patch.object(runner.original, "run", side_effect=RuntimeError("synthetic stop")):
            with self.assertRaises(RuntimeError): runner.run(Path("synthetic"), "control")
        self.assertIs(runner.original.native_identity, old_hook)
        self.assertEqual(runner.original.CONFIG, old_config)


if __name__ == "__main__": unittest.main()
