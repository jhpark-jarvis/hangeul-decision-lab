"""Validate the local TypeScript export; game rules stay in TypeScript."""
import hashlib
import json
import math
from pathlib import Path

SPLITS = ("train", "dev", "test")
PIECES = ["DOT", "LINE_3", "BRANCH_4", "DOUBLE_BRANCH_7", "DIAGONAL_3", "L_3",
          "DIAMOND_4", "CROWN_6", "HOOK_4", "LINE_5", "DOUBLE_ARM_RIGHT_6", "C_5",
          "STAR_9", "PI_10", "CROWN_7", "GRID_10", "MIEUM", "ZIGZAG_6", "DOUBLE_ARM_LEFT_6"]
SCHEMA = {"version": "legal-candidate-cnn-v1", "catalogVersion": "2026-10-02-user-labels-v2",
          "pieceIds": PIECES, "planeChannels": ["occupied", "single-cell-item", "reroll-item"],
          "rows": 16, "cols": 10, "shapeRows": 5, "shapeCols": 5,
          "metadataSize": 140, "candidateSize": 52, "maxCandidates": 4096}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def compact(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def vector(value, length, binary=False):
    require(isinstance(value, list) and len(value) == length, "Feature dimensions")
    require(all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 and
                (not binary or v in (0, 1)) for v in value), "Nonfinite/out-of-range feature")


def teacher_diagnostics(teacher, config):
    search = teacher["search"]
    require(type(search["searchComplete"]) is bool and search["maxNodes"] == config["maxNodes"] and
            search["scope"] in ("ordinary-pieces", "pieces-and-abilities"), "Teacher diagnostics")
    latency = teacher["latencyMs"]
    require(type(latency) in (int, float) and math.isfinite(latency) and latency >= 0, "Teacher clock")


def validate_pack(pack, protocol):
    require(pack["status"] == "PASS" and pack["schema"] == SCHEMA, "Dataset status/schema")
    spec = protocol["dataset"]
    require(protocol["schemaVersion"] == 1 and pack["spec"] == spec, "Protocol mismatch")
    all_seeds = []
    for split in SPLITS:
        seeds = spec[f"{split}Seeds"]
        require(isinstance(seeds, list) and seeds and all(integer(n, 0, 2**32 - 1) for n in seeds), "Invalid seeds")
        all_seeds.extend(seeds)
    require(len(set(all_seeds)) == len(all_seeds) <= 32, "Overlapping seeds")
    require(0 < len(spec["families"]) == len(set(spec["families"])) <= 2 and
            set(spec["families"]) <= {"sparse", "pressure"}, "Invalid families")
    require(integer(spec["maxActions"], 1, 12) and len(all_seeds) * len(spec["families"]) * spec["maxActions"] <= 512, "Dataset limit")
    teacher = spec["teacher"]
    require(integer(teacher["maxNodes"], 1, 512) and integer(teacher["maxAlternatives"], 1, 3) and type(teacher["useMemoization"]) is bool, "Teacher config")
    episode_map, seen_ids, split_states = {}, set(), set()
    for episode in pack["episodes"]:
        split, fixture = episode["split"], episode["fixture"]
        require(split in SPLITS and fixture["seed"] in spec[f"{split}Seeds"] and episode["replayValid"] is True, "Episode split/replay")
        key = (split, fixture["id"])
        require(key not in episode_map, "Duplicate episode")
        require(episode["result"]["ending"]["kind"] != "error", "Episode error")
        episode_map[key] = episode
    for split in SPLITS:
        require(sum(s == split for s, _ in episode_map) == len(spec[f"{split}Seeds"]) * len(spec["families"]), "Episode count")
        examples = pack["examples"][split]
        require(isinstance(examples, list) and examples, "Empty split")
        this_states = set()
        require(sum(e["result"]["totals"]["actions"] for (s, _), e in episode_map.items() if s == split) == len(examples), "Example coverage")
        for example in examples:
            require(example["id"] not in seen_ids, "Duplicate example")
            seen_ids.add(example["id"])
            require(example["split"] == split and example["seed"] in spec[f"{split}Seeds"], "Example split")
            episode = episode_map.get((split, example["fixtureId"]))
            require(episode is not None and example["seed"] == episode["fixture"]["seed"], "Example fixture")
            require(integer(example["actionIndex"], 0, spec["maxActions"] - 1), "Action index")
            require(example["stateKey"] not in split_states, "State leaked across splits")
            this_states.add(example["stateKey"])
            obs = example["observation"]
            require(set(obs) == {"planes", "metadata"}, "Unexpected model observation")
            require(len(obs["planes"]) == 3, "Plane channels")
            for plane in obs["planes"]:
                require(len(plane) == 16, "Plane rows")
                for row in plane:
                    vector(row, 10, True)
            vector(obs["metadata"], 140)
            keys, features = example["candidateKeys"], example["candidateFeatures"]
            require(isinstance(keys, list) and 1 <= len(keys) <= 4096 and all(isinstance(k, str) for k in keys) and len(set(keys)) == len(keys), "Candidate keys")
            require(isinstance(features, list) and len(features) == len(keys), "Candidate count")
            for feature in features:
                vector(feature, 52)
            require(integer(example["label"], 0, len(keys) - 1), "Illegal teacher label")
            teacher_diagnostics(example["teacher"], teacher)
        split_states.update(this_states)
    skip_episodes = set()
    for skip in pack["skips"]:
        key = (skip["split"], skip["fixtureId"])
        episode = episode_map.get(key)
        require(episode is not None and key not in skip_episodes and
                episode["result"]["ending"]["kind"] == "policy-abstention", "Skip episode")
        require(skip["reason"] == "teacher-abstention" and integer(skip["actionIndex"], 0, spec["maxActions"] - 1) and
                skip["actionIndex"] == episode["result"]["totals"]["actions"], "Skip action index")
        teacher_diagnostics(skip["teacher"], teacher)
        skip_episodes.add(key)
    require(sum(e["result"]["ending"]["kind"] == "policy-abstention" for e in pack["episodes"]) == len(skip_episodes), "Missing skip diagnostics")
    return pack


def load_dataset(directory):
    path = Path(directory)
    body = (path / "dataset.json").read_bytes()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((path / "protocol.json").read_text(encoding="utf-8"))
    require(sha256(body) == manifest["dataSha256"], "Dataset hash mismatch")
    require(sha256(compact(protocol)) == manifest["protocolSha256"], "Protocol hash mismatch")
    pack = validate_pack(json.loads(body), protocol)
    require(manifest["status"] == "PASS" and manifest["schema"] == SCHEMA and manifest["spec"] == pack["spec"], "Manifest mismatch")
    return pack, protocol, manifest
