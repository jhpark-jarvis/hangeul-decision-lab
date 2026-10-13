"""Synthetic controls; no real training/data or additional experiment runs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from model import configure
from spatial import SpatialCandidateCNN, spatial_tensors, make_model, Budget, main
from outcome import gate

CONFIG = {"seed": 20261008, "epochs": 12, "batchSize": 8, "threads": 2, "convChannels": 8,
          "embedding": 32, "poolRows": 4, "poolCols": 5, "learningRate": 0.001, "weightDecay": 0.0001}


class SpatialTests(unittest.TestCase):
    def setUp(self):
        configure(CONFIG)
        self.baseline = make_model(CONFIG, "baseline").eval()
        self.spatial = make_model(CONFIG, "spatial").eval()

    def test_zero_residual_exact_and_fixed_parameters_for_variable_full_masks(self):
        self.assertEqual(sum(p.numel() for p in self.baseline.parameters()), 12915)
        self.assertEqual(sum(p.numel() for p in self.spatial.parameters()), 13292)
        for n in (1, 31, 32, 33, 65, 512):
            p, m, c = torch.rand(1, 3, 16, 10), torch.rand(1, 140), torch.rand(n, 74)
            f = torch.randint(0, 2, (n, 1, 16, 10)).float()
            with torch.inference_mode():
                self.assertTrue(torch.equal(self.baseline(p, m, c), self.spatial(p, m, c, f)))

    def test_chunk32_preserves_full_order_against_independent_one_candidate_branch(self):
        p, m, c = torch.rand(1, 3, 16, 10), torch.rand(1, 140), torch.rand(65, 74)
        f = torch.randint(0, 2, (65, 1, 16, 10)).float()
        with torch.no_grad(): self.spatial.spatial[-1].weight.fill_(0.05)
        batches = []
        hook = self.spatial.spatial.register_forward_pre_hook(lambda module, args: batches.append(len(args[0])))
        with torch.inference_mode(): actual = self.spatial(p, m, c, f)
        hook.remove()
        self.assertEqual(batches, [32, 32, 1])
        with torch.inference_mode():
            expected = self.baseline(p, m, c) + torch.cat([
                self.spatial.spatial(torch.cat((p, f[i:i+1]), dim=1)).squeeze(1) for i in range(65)])
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_zero_last_layer_receives_gradient_without_optimizer_or_real_fit(self):
        output = self.spatial(torch.rand(1, 3, 16, 10), torch.rand(1, 140), torch.rand(3, 74), torch.ones(3, 1, 16, 10))
        output[0].backward()
        self.assertGreater(float(self.spatial.spatial[-1].weight.grad.abs().sum()), 0)
        self.assertEqual(float(self.spatial.spatial[0].weight.grad.abs().sum()), 0)

    def test_only_four_tensor_inputs_and_private_binary_storage(self):
        e = {"observation": {"planes": [[[0]*10 for _ in range(16)] for _ in range(3)], "metadata": [0]*140},
             "candidateFeatures": [[0]*74, [0]*74], "footprints": ["0"*160, "0"*159+"1"],
             "candidateKeys": "FORBIDDEN", "labels": "FORBIDDEN", "state": "FORBIDDEN", "future": "FORBIDDEN"}
        inputs = spatial_tensors(e)
        self.assertEqual(len(inputs), 4)
        self.assertEqual(inputs[3].shape, (2, 1, 16, 10))
        self.assertEqual(inputs[3][1, 0, 15, 9].item(), 1)
        for bad in (["0"*159], ["x"*160, "0"*160], ["0"*160]):
            with self.assertRaises(ValueError): spatial_tensors({**e, "footprints": bad})

    def test_nonfinite_fractional_shape_and_candidate_range_fail_closed(self):
        p, m, c, f = torch.zeros(1, 3, 16, 10), torch.zeros(1, 140), torch.zeros(1, 74), torch.zeros(1, 1, 16, 10)
        for value in (float("nan"), -1, 0.5, 2):
            f[0, 0, 0, 0] = value
            with self.assertRaises(ValueError): self.spatial(p, m, c, f)
        f.zero_()
        with self.assertRaises(ValueError): self.spatial(p, m, c, f.double())
        with self.assertRaises(ValueError): self.spatial(p, m, c, torch.zeros(2, 1, 16, 10))
        c[0, 73] = float("inf")
        with self.assertRaises(ValueError): self.spatial(p, m, c, f)

    def test_memory_and_wall_time_stop_and_sampling_definition(self):
        with patch("spatial.time.perf_counter", return_value=10), patch("spatial.rss_bytes", return_value=100):
            monitor = Budget(600, 200, None)
            monitor.sample("batch")
            self.assertEqual(monitor.report()["maxSampledWorkingSetBytes"], 100)
            self.assertFalse(monitor.report()["instantaneousPeakMeasured"])
        with patch("spatial.time.perf_counter", return_value=611), patch("spatial.rss_bytes", return_value=100):
            with self.assertRaisesRegex(ValueError, "wall-time"): Budget(600, 200, None, 10).sample("batch")
        for memory in (201, None):
            with patch("spatial.rss_bytes", return_value=memory):
                with self.assertRaisesRegex(ValueError, "working-set"): Budget(600, 200, None).sample("batch")

    def test_wrong_execution_stops_before_fit_and_preserves_exclusive_failure(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root / ".research-output") as directory:
            base = Path(directory)
            execution = base / "execution.json"
            execution.write_text(json.dumps({"status": "BAD"}))
            args = ["spatial.py", "--mode", "preflight", "--source", str(base / "absent"),
                    "--original", str(base / "absent"), "--execution", str(execution), "--output", str(base / "preflight")]
            with patch.object(sys, "argv", args), patch("spatial.subprocess.check_output", return_value=""), patch("spatial.rss_bytes", return_value=1):
                with self.assertRaises(ValueError): main()
                failure = (base / "preflight/failure.json").read_bytes()
                with self.assertRaises(FileExistsError): main()
                self.assertEqual((base / "preflight/failure.json").read_bytes(), failure)
                self.assertFalse((base / "preflight/checkpoint.pt").exists())

    def test_real_cli_dirty_guard_and_no_tuning_argument(self):
        with patch.object(sys, "argv", ["spatial.py", "--mode", "preflight", "--source", "absent", "--original", "absent", "--execution", ".research-output/guard/execution.json", "--output", ".research-output/guard/preflight"]), patch("spatial.subprocess.check_output", return_value=" M file"):
            with self.assertRaisesRegex(ValueError, "Commit source"): main()
        result = subprocess.run([sys.executable, str(Path(__file__).parent / "spatial.py"), "--epochs", "13"], capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_original_baseline_gate_and_spatial_contract_stay_fixed(self):
        config = json.loads((Path(__file__).parent.parent / "experiments/spatial-residual-v1.json").read_bytes())
        self.assertEqual(config["limits"], {"footprintCandidates": 149597, "trainingRuns": 2, "epochs": 12,
            "childSeconds": 600, "workingSetBytes": 2147483648, "chunk": 32, "teacherDomainTestRollout": 0})
        def summary(n, point, reroll):
            return {"overall": {"examples": 89, "correct": n}, "groups": {"teacherAction": {
                "single-cell": {"examples": 13, "correct": point}, "reroll": {"examples": 8, "correct": reroll}}}}
        self.assertEqual(gate(summary(29, 0, 8), summary(48, 3, 7))["status"], "PASS")
        self.assertEqual(gate(summary(29, 0, 8), summary(48, 2, 8))["status"], "FAIL")


if __name__ == "__main__": unittest.main()
