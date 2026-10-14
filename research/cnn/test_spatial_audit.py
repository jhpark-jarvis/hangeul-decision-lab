"""Toy stored-score controls; no model or real-data forwards/training."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from diagnose import score, action_description
from spatial_audit import error_kind, decode, controlled_row, pair_rows, paired_table, aggregate_pairs, summarize_pairs, compute_guard, main

KEYS = [json.dumps(a) for a in [["single-cell", 0, 0], ["single-cell", 1, 0], ["reroll", "held", 0]]]


def fixture(values=(2., 1., 0.), label=0, ident="toy"):
    e = {"id": ident, "fixtureId": "sparse-0", "candidateKeys": KEYS, "candidateFeatures": [[0]*52]*3,
         "labels": {"A512": label, "A1024": label}, "teachers": {n: {"result": {"search": {"searchComplete": False},
                "evaluation": {"phase": "playing"}}} for n in ("A512", "A1024")}}
    p = score(torch.tensor(values), e["labels"])
    p.update(id=ident, fixtureId=e["fixtureId"], family="sparse", sameLabel=True,
             complete={n: False for n in ("A512", "A1024")}, teacherActions={n: action_description(KEYS[label]) for n in ("A512", "A1024")},
             selected=action_description(KEYS[p["index"]]), firstLegalAction=action_description(KEYS[0]))
    return e, p, {"id": ident, "logits": list(values)}


class StoredSpatialTests(unittest.TestCase):
    def test_error_priority_matches_existing_typescript_classifier(self):
        a = ["place-piece", "held", 0, "L", 3, 4, [[True, False], [True, True]]]
        cases = [(a, "exact-label"), (["single-cell", 3, 4], "wrong-kind"),
                 ([a[0], "other", *a[2:]], "different-target"),
                 ([*a[:3], "OTHER", *a[4:]], "different-shape"),
                 ([*a[:4], 4, *a[5:]], "different-position")]
        for b, expected in cases: self.assertEqual(error_kind(json.dumps(a), json.dumps(b)), expected)
        # Run the original pure TS classifier on the same toy fixtures as an independent oracle.
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root / ".research-output") as directory:
            source = Path(directory) / "oracle.mjs"
            runtime_url = (root / "scripts/research/runtime.mjs").as_uri()
            source.write_text(f"import {{compileResearch}} from {json.dumps(runtime_url)};\nconst runtime=compileResearch({json.dumps(str(root))},'src/research/action-errors.ts');\nconst {{decodeStoredAction,errorKind}}=await import(runtime.url);\nconst target={json.dumps(json.dumps(a))};\nconst keys={json.dumps([json.dumps(b) for b, _ in cases])};\nconsole.log(JSON.stringify(keys.map(k=>errorKind(decodeStoredAction(target),decodeStoredAction(k)))));", encoding="utf-8")
            actual = subprocess.check_output(["node", str(source)], text=True, cwd=root)
            self.assertEqual(json.loads(actual), [expected for _, expected in cases])

    def test_bad_stored_keys_fail_before_scoring(self):
        for bad in ("null", "bad", json.dumps(["unknown"]), json.dumps(["single-cell", 16, 0]),
                    json.dumps(["reroll", "", 0]), json.dumps(["place-piece", "held", 0, "L", 0, 0, [[1]]])):
            with self.assertRaises(ValueError): decode(bad)

    def test_saved_full_score_ties_and_oracle_are_separate(self):
        e, p, l = fixture((1., 1., 2.), label=1)
        with compute_guard(): r = controlled_row(e, p, l, "sparse")
        self.assertFalse(r["globalCorrect"])
        self.assertEqual(r["error"], "wrong-kind")
        self.assertEqual(r["conditional"]["oracleIndex"], 0)
        self.assertEqual(r["conditional"]["oracleRank"], 2)

    def test_score_id_mask_float32_and_nonfinite_drift_stop(self):
        e, p, l = fixture()
        for bad in ({**p, "index": 1}, {**p, "margin": 9}, {**p, "family": "pressure"}):
            with self.assertRaises(ValueError): controlled_row(e, bad, l, "sparse")
        for bad in ({**l, "id": "other"}, {**l, "logits": [float("nan"), 1., 0.]}, {**l, "logits": [0.1, 1., 0.]}, {**l, "logits": [1.]}):
            with self.assertRaises(ValueError): controlled_row(e, p, bad, "sparse")

    def test_four_way_partition_and_net_gain_keep_observations_paired(self):
        a, b = [], []
        for i, (left, right) in enumerate([((2.,1.,0.),(2.,1.,0.)), ((1.,2.,0.),(2.,1.,0.)),
                                        ((2.,1.,0.),(1.,2.,0.)), ((1.,2.,0.),(1.,0.,2.))]):
            a.append(controlled_row(*fixture(left, ident=str(i)), "sparse"))
            b.append(controlled_row(*fixture(right, ident=str(i)), "sparse"))
        rows = pair_rows(a, b)
        table = paired_table(rows)
        self.assertEqual([table[k] for k in ("bothCorrect", "improvement", "regression", "bothWrong")], [1,1,1,1])
        self.assertEqual(table["netCorrectGain"], 0)
        self.assertEqual(aggregate_pairs(rows)["changedButBothWrong"], 1)
        with self.assertRaises(ValueError): pair_rows(a, list(reversed(b)))

    def test_confusion_transitions_and_empty_groups_have_exact_denominators(self):
        a = controlled_row(*fixture((1.,2.,0.)), "sparse")
        b = controlled_row(*fixture((1.,0.,2.)), "sparse")
        result = summarize_pairs(pair_rows([a], [b]))
        self.assertEqual(result["groups"]["teacherKind"]["single-cell"]["actualHardLabel"]["examples"], 1)
        agg = result["overall"]
        self.assertEqual(agg["errorTransition"]["different-position"]["wrong-kind"], 1)
        self.assertEqual(sum(sum(v.values()) for v in agg["kindConfusion"]["spatial"].values()), 1)
        self.assertEqual(paired_table([])["examples"], 0)

    def test_guard_blocks_model_init_forward_checkpoint_and_optimizer_and_restores(self):
        original = torch.nn.Module.__init__
        for call in (lambda: torch.nn.Linear(1,1), lambda: torch.load("absent.pt"),
                     lambda: torch.optim.Adam([torch.tensor(1., requires_grad=True)])):
            with self.assertRaisesRegex(ValueError, "forbidden"):
                with compute_guard(): call()
            self.assertIs(torch.nn.Module.__init__, original)
        module = torch.nn.Identity()
        with self.assertRaisesRegex(ValueError, "forbidden"):
            with compute_guard(): module(torch.zeros(1))

    def test_cli_missing_input_failure_and_existing_output_preserved(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(dir=root / ".research-output") as directory:
            base = Path(directory)
            args = ["spatial_audit.py", "--source", str(base / "absent"), "--experiment", str(base / "absent"), "--output", str(base / "out")]
            with patch.object(sys, "argv", args), patch("spatial_audit.subprocess.check_output", return_value=""):
                with self.assertRaises(FileNotFoundError): main()
                failure = (base / "out/failure.json").read_bytes()
                with self.assertRaises(FileExistsError): main()
                self.assertEqual((base / "out/failure.json").read_bytes(), failure)
                self.assertFalse((base / "out/summary.json").exists())

    def test_dirty_source_and_unapproved_extra_arguments_stop_before_run(self):
        with patch.object(sys, "argv", ["spatial_audit.py", "--source", "absent", "--experiment", "absent", "--output", ".research-output/audit-guard-unused"]), patch("spatial_audit.subprocess.check_output", return_value=" M file"), patch("spatial_audit.run") as run:
            with self.assertRaisesRegex(ValueError, "Commit"): main()
            run.assert_not_called()
        result = subprocess.run([sys.executable, str(Path(__file__).parent / "spatial_audit.py"), "--epochs", "13"], capture_output=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__": unittest.main()
