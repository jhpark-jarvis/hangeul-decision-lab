"""Toy component controls only; no real frozen-data forwards or training."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from spatial import make_model
from component_audit import capture, bounded_model_calls, validate_record, training_guard, common_row, residual_row, residual_summary, comparison_summary, main
from spatial_audit import compute_guard
from test_spatial_audit import fixture
from spatial_audit import controlled_row, pair_rows, paired_table

CONFIG = {"seed": 20261008, "epochs": 12, "batchSize": 8, "threads": 2, "convChannels": 8,
          "embedding": 32, "poolRows": 4, "poolCols": 5, "learningRate": 0.001, "weightDecay": 0.0001}


class ComponentTests(unittest.TestCase):
    def test_original_forward_hooks_exact_sum_repeats_chunk32_and_removed(self):
        model = make_model(CONFIG, "spatial").eval()
        with torch.no_grad(): model.spatial[-1].weight.fill_(0.01)
        inputs = (torch.rand(1,3,16,10), torch.rand(1,140), torch.rand(65,74), torch.randint(0,2,(65,1,16,10)).float())
        budget = {"calls": 0, "limit": 2}
        with training_guard(), bounded_model_calls(model, budget):
            a, b = capture(model, inputs), capture(model, inputs)
        self.assertEqual(budget["calls"], 2)
        self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)))
        self.assertTrue(torch.equal(a[0]+a[1], a[2]))
        self.assertFalse(model.common._forward_hooks)
        self.assertFalse(model.spatial._forward_hooks)

    def test_call_budget_unknown_module_and_ledger_before_call_stop(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root / ".research-output") as directory:
            ledger = Path(directory)/"calls.jsonl"
            model, other = nn.Identity(), nn.Identity()
            budget = {"calls": 0, "limit": 1, "ledger": ledger}
            hook = model.register_forward_pre_hook(lambda module, args: self.assertEqual(json.loads(ledger.read_text())["call"], 1))
            with bounded_model_calls(model, budget):
                model(torch.zeros(1))
                with self.assertRaisesRegex(ValueError, "budget"): model(torch.zeros(1))
                with self.assertRaisesRegex(ValueError, "registered"): other(torch.zeros(1))
            hook.remove()
            self.assertEqual(budget["calls"], 1)

    def test_training_backward_optimizer_true_mode_blocked_and_restored(self):
        model = nn.Linear(1,1)
        original = nn.Module.train
        with training_guard():
            model.eval()
            with self.assertRaises(ValueError): model.train()
            with self.assertRaises(ValueError): torch.optim.Adam(model.parameters())
            with self.assertRaises(ValueError): torch.tensor(1., requires_grad=True).backward()
        self.assertIs(nn.Module.train, original)

    def test_replay_validates_saved_identity_sum_original_full_and_float32_without_model(self):
        e, _, original = fixture((3.,2.,1.))
        record = {"id": e["id"], "split": "dev", "common": [2.,2.,0.], "residual": [1.,0.,1.], "full": [3.,2.,1.]}
        with compute_guard(): values = validate_record(record, e, "dev", original)
        self.assertTrue(torch.equal(values[0]+values[1], values[2]))
        for bad in ({**record, "id": "wrong"}, {**record, "residual": [0.,0.,1.]},
                    {**record, "common": [0.1,2.,0.]}, {**record, "full": [float("nan"),2.,1.]}):
            with self.assertRaises(ValueError): validate_record(bad,e,"dev",original)

    def test_common_only_scoring_and_paired_rescue_do_not_call_model(self):
        e, p, l = fixture((3.,2.,1.))
        full = controlled_row(e,p,l,"sparse")
        with compute_guard(): common = common_row(e,full,torch.tensor([1.,2.,0.]))
        result = paired_table(pair_rows([common],[full]))
        self.assertEqual(result["improvement"], 1)
        self.assertEqual(common["error"], "different-position")
        self.assertEqual(common["teacherKind"], full["teacherKind"])
        named = comparison_summary({"common": [common], "full": [full]}, "common", "full")
        self.assertEqual((named["leftModel"], named["rightModel"]), ("common", "full"))
        self.assertEqual(named["overall"]["actualHardLabel"]["leftCorrect"], 0)
        self.assertEqual(named["overall"]["actualHardLabel"]["rightCorrect"], 1)

    def test_residual_observation_stats_weight_rows_and_reroll_span(self):
        e, _, _ = fixture()
        r = residual_row(e,torch.tensor([1.,2.,0.]),torch.tensor([1.,-1.,0.]))
        self.assertEqual(r["teacherMinusCommonWinnerResidual"], 2)
        self.assertEqual(r["residualSpan"], 2)
        self.assertEqual(residual_summary([r,r])["examples"], 2)
        self.assertIsNone(residual_summary([])["meanAbsResidual"])
        self.assertIsNone(residual_summary([r])["maxWithinRerollResidualSpan"])

    def test_capture_failure_removes_hooks_and_does_not_alter_weights(self):
        model = make_model(CONFIG,"spatial").eval()
        before = {k:v.clone() for k,v in model.state_dict().items()}
        with self.assertRaises(ValueError): capture(model,(torch.zeros(1,3,16,10),torch.zeros(1,140),torch.zeros(1,74),torch.full((1,1,16,10),float("nan"))))
        self.assertFalse(model.common._forward_hooks)
        self.assertFalse(model.spatial._forward_hooks)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()))

    def test_cli_missing_inputs_and_existing_failure_preserved_without_forward(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root / ".research-output") as directory:
            base = Path(directory)
            args = ["component_audit.py", "--mode", "infer", "--source", str(base/"absent"), "--experiment", str(base/"absent"), "--original", str(base/"absent"), "--output", str(base/"out")]
            with patch.object(sys,"argv",args), patch("component_audit.subprocess.check_output",return_value=""):
                with self.assertRaises(FileNotFoundError): main()
                failure = (base/"out/failure.json").read_bytes()
                with self.assertRaises(FileExistsError): main()
                self.assertEqual((base/"out/failure.json").read_bytes(),failure)
                self.assertFalse((base/"out/forward-count.jsonl").exists())

    def test_cli_dirty_and_replay_missing_recorded_stop_before_process(self):
        args = ["component_audit.py", "--mode", "infer", "--source", "absent", "--experiment", "absent", "--original", "absent", "--output", ".research-output/unused-component-guard"]
        with patch.object(sys,"argv",args), patch("component_audit.subprocess.check_output",return_value=" M file"), patch("component_audit.process") as run:
            with self.assertRaisesRegex(ValueError,"Commit"): main()
            run.assert_not_called()
        args[2] = "replay"
        with patch.object(sys,"argv",args):
            with self.assertRaisesRegex(ValueError,"Recorded"): main()


if __name__ == "__main__": unittest.main()
