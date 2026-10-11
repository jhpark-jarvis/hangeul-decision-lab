"""Paired feature/freeze and identical-input controls; no reserved fixtures."""
import copy
import unittest
import torch
from data import SCHEMA
from paired import validate_pack, training_audit, own_examples
from test_cnn import fixture, CONFIG


def pair_fixture():
    old, old_protocol = fixture()
    protocol = {"id": "unit", "status": "FROZEN_PAIRED_EXPERIMENT",
                "dataset": {"trainSeeds": [1], "devSeeds": [2], "reservedTestSeeds": [3]},
                "teachers": {"A512": {"maxNodes": 512}, "A1024": {"maxNodes": 1024}},
                "training": CONFIG}
    examples = {"train": [], "dev": [], "test": []}
    for split in ("train", "dev"):
        e = copy.deepcopy(old["examples"][split][0])
        e["labels"] = {"A512": 0, "A1024": 1}
        e["teachers"] = {name: {"result": {"search": {"maxNodes": nodes}}}
                         for name, nodes in (("A512", 512), ("A1024", 1024))}
        examples[split].append(e)
    return {"status": "PASS", "schema": SCHEMA, "stage": "train-dev", "protocolId": "unit",
            "examples": examples}, protocol


class PairedTests(unittest.TestCase):
    def test_no_test_before_freeze_training(self):
        pack, p = pair_fixture()
        validate_pack(pack, p, "train-dev")
        pack["examples"]["test"] = copy.deepcopy(pack["examples"]["train"])
        with self.assertRaisesRegex(ValueError, "premature"):
            validate_pack(pack, p, "train-dev")

    def test_label_and_finite_feature_rejection(self):
        pack, p = pair_fixture()
        pack["examples"]["train"][0]["labels"]["A1024"] = 2
        with self.assertRaisesRegex(ValueError, "label index"):
            validate_pack(pack, p, "train-dev")
        pack, p = pair_fixture()
        pack["examples"]["train"][0]["candidateFeatures"][0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            validate_pack(pack, p, "train-dev")

    def test_same_initialization_order_and_inputs_labels_only_change(self):
        pack, p = pair_fixture()
        a, b = own_examples(pack, "train", "A512"), own_examples(pack, "train", "A1024")
        self.assertEqual(training_audit(a, CONFIG), training_audit(b, CONFIG))
        from model import tensors
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(tensors(a[0]), tensors(b[0]))))
        self.assertNotEqual(a[0]["label"], b[0]["label"])


if __name__ == "__main__":
    unittest.main()
