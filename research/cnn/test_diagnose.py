"""Numerical, input-boundary and failure preservation checks for diagnostics."""
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from diagnose import score, aggregate, inspect_split, diagnose, action_description, fixture_family
from test_paired import pair_fixture


class DiagnosticTests(unittest.TestCase):
    def test_stable_ties_rank_nll_and_single_candidate(self):
        result = score(torch.tensor([2.0, 2.0, 0.0]), {"A512": 1, "A1024": 2})
        self.assertEqual(result["index"], 0)
        self.assertEqual(result["margin"], 0)
        self.assertEqual(result["targets"]["A512"]["rank"], 2)
        self.assertEqual(result["targets"]["A1024"]["rank"], 3)
        self.assertAlmostEqual(result["targets"]["A512"]["nll"], math.log(2 + math.exp(-2)), places=6)
        one = score(torch.tensor([3.0]), {"A512": 0, "A1024": 0})
        self.assertIsNone(one["margin"])
        self.assertEqual(one["targets"]["A512"]["nll"], 0)
        self.assertEqual(one["uniformNll"], 0)

    def test_reject_nonfinite_or_illegal_labels_and_keys(self):
        for logits, labels in (([float("nan")], {"A512": 0, "A1024": 0}),
                               ([0.0], {"A512": True, "A1024": 0}),
                               ([0.0], {"A512": 0, "A1024": 1})):
            with self.assertRaises(ValueError):
                score(torch.tensor(logits), labels)
        with self.assertRaises(ValueError):
            action_description('["place-piece"]')

    def test_fixture_macro_uses_fixture_not_example_weighting(self):
        rows = []
        for fixture, label in (("long", 0), ("long", 0), ("short", 1)):
            row = score(torch.tensor([1.0, 0.0]), {"A512": label, "A1024": label})
            row["fixtureId"] = fixture
            rows.append(row)
        result = aggregate(rows, "A512")
        self.assertAlmostEqual(result["top1"], 2 / 3)
        self.assertEqual(result["fixtureMacroTop1"], 0.5)
        self.assertIsNone(aggregate([], "A512")["top1"])

    def test_family_requires_generator_identity(self):
        self.assertEqual(fixture_family({"id": "pressure-9000", "seed": 9000}, ["sparse", "pressure"]), "pressure")
        with self.assertRaises(ValueError):
            fixture_family({"id": "pressure-9001", "seed": 9000}, ["sparse", "pressure"])

    def test_forward_never_receives_labels_keys_or_raw_state(self):
        pack, _ = pair_fixture()
        example = pack["examples"]["train"][0]
        example["candidateKeys"] = ['["single-cell",0,0]', '["single-cell",1,0]']
        for name in ("A512", "A1024"):
            example["teachers"][name]["result"]["search"]["searchComplete"] = False
        captured = []

        class Recorder:
            def __call__(self, *inputs):
                captured.append(tuple(x.clone() for x in inputs))
                return torch.tensor([1.0, 0.0])

        families = {example["fixtureId"]: "sparse"}
        inspect_split(Recorder(), [example], families)
        changed = copy.deepcopy(example)
        changed["labels"] = {"A512": 1, "A1024": 0}
        changed["candidateKeys"] = ['["single-cell",15,9]', '["single-cell",14,9]']
        changed["state"] = {"futureTape": "not-an-input"}
        inspect_split(Recorder(), [changed], families)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(captured[0], captured[2])))
        self.assertEqual(len(captured[0]), 3)

    def test_mismatched_dataset_stops_before_inference(self):
        pack, protocol = pair_fixture()
        with patch('diagnose.load_pack', return_value=(pack, protocol, {"dataSha256": "one"})), \
                patch('diagnose.load_frozen', return_value=({}, protocol, {"trainDevDataSha256": "two"})), \
                patch('diagnose.inspect_split') as inspect:
            with self.assertRaisesRegex(ValueError, "identity"):
                diagnose('unused', 'unused', Path('unused'))
            inspect.assert_not_called()

    def test_cli_failure_and_existing_output_preserve_marker(self):
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            output = base / 'failed'
            command = [sys.executable, str(Path(__file__).with_name('diagnose.py')), '--dataset',
                       str(base / 'missing'), '--freeze', str(base / 'missing-freeze'), '--output', str(output)]
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            marker = (output / 'failure.json').read_bytes()
            self.assertEqual(json.loads(marker)['status'], 'FAIL')
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual((output / 'failure.json').read_bytes(), marker)


if __name__ == '__main__':
    unittest.main()
