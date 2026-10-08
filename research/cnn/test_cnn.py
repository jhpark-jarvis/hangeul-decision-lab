"""Contract failures and meaningful numerical checks; no game-rule clone."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from data import SCHEMA, compact, load_dataset, sha256, validate_pack
from model import CandidateCNN, configure, predict, tensors
from train import metrics

CONFIG = {"seed": 31, "epochs": 2, "batchSize": 2, "threads": 2, "convChannels": 2,
          "embedding": 8, "poolRows": 2, "poolCols": 2, "learningRate": 0.01, "weightDecay": 0.0}


def fixture():
    spec = {"trainSeeds": [1], "devSeeds": [2], "testSeeds": [3], "families": ["sparse"],
            "maxActions": 1, "teacher": {"maxNodes": 8, "maxAlternatives": 1, "useMemoization": True}}
    protocol = {"schemaVersion": 1, "id": "numerical-test-only", "dataset": spec, "training": CONFIG}
    pack = {"status": "PASS", "schema": SCHEMA, "spec": spec, "examples": {}, "episodes": [], "skips": []}
    for split, seed in zip(("train", "dev", "test"), (1, 2, 3)):
        obs = {"planes": [[[0] * 10 for _ in range(16)] for _ in range(3)], "metadata": [0] * 140}
        features = [[0] * 52, [0] * 52]
        features[0][25], features[1][25] = 0, 1
        pack["examples"][split] = [{"id": split, "fixtureId": split, "split": split, "seed": seed,
                                    "actionIndex": 0, "stateKey": split, "observation": obs,
                                    "candidateKeys": ["left", "right"], "candidateFeatures": features,
                                    "label": 0, "teacher": {"latencyMs": 0, "search": {"searchComplete": False, "maxNodes": 8, "scope": "ordinary-pieces"}}}]
        pack["episodes"].append({"split": split, "fixture": {"id": split, "seed": seed}, "replayValid": True,
                                 "result": {"totals": {"actions": 1}, "ending": {"kind": "horizon"}}})
    return pack, protocol


class ContractTests(unittest.TestCase):
    def test_data_failures(self):
        original, protocol = fixture()
        self.assertIs(validate_pack(original, protocol), original)
        for index, mutate in enumerate((
            lambda p: p["examples"]["train"][0]["observation"]["metadata"].__setitem__(0, float("nan")),
            lambda p: p["examples"]["train"][0]["observation"]["planes"][0].pop(),
            lambda p: p["examples"]["train"][0].__setitem__("label", 2),
            lambda p: p["examples"]["train"][0].__setitem__("label", True),
            lambda p: p["examples"]["train"][0].__setitem__("candidateKeys", ["left", "left"]),
            lambda p: p["spec"]["devSeeds"].__setitem__(0, 1),
            lambda p: p["examples"]["dev"][0].__setitem__("stateKey", "train"),
            lambda p: p["schema"].__setitem__("version", "unknown"),
            lambda p: p["skips"].append({"split": "train", "fixtureId": "train", "reason": "teacher-abstention", "actionIndex": 0}),
        )):
            pack = copy.deepcopy(original)
            mutate(pack)
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    validate_pack(pack, protocol)

    def test_hash_and_output_failure_recovery(self):
        pack, protocol = fixture()
        with tempfile.TemporaryDirectory() as name:
            path = Path(name)
            body = compact(pack)
            (path / "dataset.json").write_bytes(body)
            (path / "protocol.json").write_bytes(compact(protocol))
            manifest = {"status": "PASS", "schema": SCHEMA, "spec": pack["spec"], "dataSha256": sha256(body), "protocolSha256": sha256(compact(protocol))}
            (path / "manifest.json").write_bytes(compact(manifest))
            load_dataset(path)
            (path / "dataset.json").write_bytes(body + b" ")
            with self.assertRaisesRegex(ValueError, "hash"):
                load_dataset(path)
            output = path / "failed"
            command = [sys.executable, str(Path(__file__).with_name("train.py")), "--dataset", str(path), "--output", str(output)]
            result = subprocess.run(command, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            failure = (output / "failure.json").read_bytes()
            self.assertEqual(json.loads(failure)["status"], "FAIL")
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual((output / "failure.json").read_bytes(), failure)

    def test_inputs_exclude_audit_and_labels_and_candidates_are_equivariant(self):
        pack, _ = fixture()
        example = pack["examples"]["train"][0]
        changed = copy.deepcopy(example)
        changed.update({"seed": 999, "label": 1, "candidateKeys": ["future-one", "future-two"],
                        "state": {"futureTape": "secret"}, "teacher": {"anything": "secret"}})
        configure(CONFIG)
        model = CandidateCNN(CONFIG).eval()
        inputs = tensors(example)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(inputs, tensors(changed))))
        self.assertTrue(torch.equal(model(*inputs), model(*tensors(changed))))
        original = model(*inputs)
        reordered = model(inputs[0], inputs[1], inputs[2].flip(0))
        torch.testing.assert_close(reordered, original.flip(0), rtol=0, atol=0)

    def test_learning_state_conditioning_and_weights_only_roundtrip(self):
        pack, _ = fixture()
        left = pack["examples"]["train"][0]
        right = copy.deepcopy(left)
        left["observation"]["metadata"][0] = 0
        right["observation"]["metadata"][0] = 1
        right["label"] = 1
        configure(CONFIG)
        model = CandidateCNN(CONFIG)
        opt = torch.optim.Adam(model.parameters(), lr=0.02)
        examples = [left, right]

        def loss():
            return sum(torch.nn.functional.cross_entropy(model(*tensors(e)).unsqueeze(0), torch.tensor([e["label"]])) for e in examples) / 2

        first = float(loss().detach())
        for _ in range(100):
            opt.zero_grad()
            value = loss()
            value.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all().item() for p in model.parameters()))
            opt.step()
        self.assertLess(float(loss().detach()), first / 3)
        self.assertEqual([predict(model, tensors(e)) for e in examples], [0, 1])
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "weights.pt"
            torch.save(model.state_dict(), path)
            loaded = CandidateCNN(CONFIG)
            loaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            torch.testing.assert_close(loaded(*tensors(left)), model(*tensors(left)), rtol=0, atol=0)
        configure(CONFIG)
        one, two = CandidateCNN(CONFIG), None
        configure(CONFIG)
        two = CandidateCNN(CONFIG)
        self.assertTrue(torch.equal(one(*tensors(left)), two(*tensors(left))))

    def test_nonfinite_and_schema_fail_closed(self):
        pack, _ = fixture()
        inp = tensors(pack["examples"]["train"][0])
        model = CandidateCNN(CONFIG)
        with self.assertRaises(ValueError):
            model(inp[0], inp[1], torch.zeros((0, 52)))
        inp[0][0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            model(*inp)
        inp[0][0, 0, 0, 0] = 0
        with torch.no_grad():
            next(model.parameters()).fill_(float("nan"))
        with self.assertRaises(ValueError):
            model(*inp)

    def test_empty_teacher_subgroup_not_zero_accuracy(self):
        self.assertIsNone(metrics([], [])["top1"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
