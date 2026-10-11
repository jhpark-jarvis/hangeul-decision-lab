"""Outcome contrast controls without training the real experiment."""
import unittest
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
from model import CandidateCNN, configure, tensors
from outcome import OutcomeCandidateCNN, gate, verify_baseline, validate_training_contract

CONFIG = {"seed": 20261008, "epochs": 12, "batchSize": 8, "threads": 2,
          "convChannels": 8, "embedding": 32, "poolRows": 4, "poolCols": 5,
          "learningRate": 0.001, "weightDecay": 0.0001}


class OutcomeTests(unittest.TestCase):
    def setUp(self):
        configure(CONFIG)
        self.baseline = CandidateCNN(CONFIG).eval()
        self.outcome = OutcomeCandidateCNN(CONFIG, self.baseline).eval()

    def test_exact_initial_logits_with_nonzero_extra_and_variable_masks(self):
        for count in (1, 33, 160, 512, 1024):
            p, m, c = torch.rand(1, 3, 16, 10), torch.rand(1, 140), torch.rand(count, 52)
            extra = torch.rand(count, 22)
            with torch.inference_mode():
                self.assertTrue(torch.equal(self.baseline(p, m, c), self.outcome(p, m, torch.cat((c, extra), 1))))
        self.assertEqual(sum(p.numel() for p in self.baseline.parameters()), 12189)
        self.assertEqual(sum(p.numel() for p in self.outcome.parameters()), 12915)

    def test_zero_path_can_receive_gradient_without_an_optimizer_or_real_training(self):
        c = torch.rand(8, 74)
        loss = self.outcome(torch.rand(1, 3, 16, 10), torch.rand(1, 140), c)[0]
        loss.backward()
        self.assertGreater(float(self.outcome.outcome_candidate.weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.outcome.outcome_bias.weight.grad.abs().sum()), 0)

    def test_input_whitelist_and_range_fail_closed(self):
        e = {"observation": {"planes": [[[0]*10 for _ in range(16)] for _ in range(3)],
                             "metadata": [0]*140}, "candidateFeatures": [[0]*74],
             "state": "FORBIDDEN", "labels": "FORBIDDEN", "future": "FORBIDDEN"}
        p, m, c = tensors(e)
        self.assertEqual(self.outcome(p, m, c).shape, (1,))
        for bad in (float("nan"), -0.1, 1.1):
            c[0, 73] = bad
            with self.assertRaises(ValueError): self.outcome(p, m, c)
        with self.assertRaises(ValueError): self.outcome(p, m, torch.zeros(1, 52))

    def test_all_gates_are_required_and_denominators_fixed(self):
        def summary(n, single=3, reroll=7):
            return {"overall": {"examples": 89, "correct": n}, "groups": {"teacherAction": {
                "single-cell": {"examples": 13, "correct": single}, "reroll": {"examples": 8, "correct": reroll}}}}
        self.assertEqual(gate(summary(29), summary(48))["status"], "PASS")
        for baseline, outcome in ((44, summary(48)), (29, summary(47)), (29, summary(48, 2)), (29, summary(48, 3, 6))):
            self.assertEqual(gate(summary(baseline), outcome)["status"], "FAIL")
        bad = summary(48); bad["groups"]["teacherAction"]["reroll"]["examples"] = 10
        with self.assertRaises(ValueError): gate(summary(29), bad)

    def test_corrupt_execution_stops_before_data_or_training_and_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            execution = root / "execution.json"
            execution.write_text(json.dumps({"status": "INVALID"}))
            output = root / "preflight"
            command = [sys.executable, str(Path(__file__).parent / "outcome.py"), "--mode", "preflight",
                       "--source", str(root / "absent"), "--augmented", str(root / "absent.json"),
                       "--execution", str(execution), "--output", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            failure = (output / "failure.json").read_bytes()
            self.assertIn("Execution config freeze", json.loads(failure)["error"])
            self.assertFalse((output / "checkpoint.pt").exists())
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual((output / "failure.json").read_bytes(), failure)

    def test_baseline_history_and_weight_drift_stop_the_contrast(self):
        history, audit = [{"epoch": 11, "dev": 29}], {"order": "same"}
        original = {"history": history, "selectedEpoch": 11, "audit": audit}
        saved = {"state_dict": {k: v.clone() for k, v in self.baseline.state_dict().items()}}
        self.assertTrue(all(verify_baseline(self.baseline, history, 11, audit, original, saved).values()))
        with self.assertRaisesRegex(ValueError, "STOP"):
            verify_baseline(self.baseline, history, 12, audit, original, saved)
        saved["state_dict"]["bias.bias"][0] += 1
        with self.assertRaisesRegex(ValueError, "STOP"):
            verify_baseline(self.baseline, history, 11, audit, original, saved)

    def test_original_protocol_descriptions_are_not_extended_hyperparameters(self):
        directory = Path(__file__).parent
        original = json.loads((directory / "paired-protocol.json").read_bytes())["training"]
        extended = json.loads((directory.parent / "experiments/outcome-features-v1.json").read_bytes())["training"]
        validate_training_contract(original, extended)
        for key, bad in (("epochs", 13), ("seed", 7)):
            with self.assertRaises(ValueError): validate_training_contract(original, {**extended, key: bad})
        with self.assertRaises(ValueError): validate_training_contract({**original, "loss": "different"}, extended)


if __name__ == "__main__": unittest.main()
