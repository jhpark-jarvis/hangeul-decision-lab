"""Board-conditioned scoring over the complete variable-length legal list."""
import math
import torch
from torch import nn
from data import require


def validate_training(config):
    for key, low, high in [("seed", 0, 2**32 - 1), ("epochs", 1, 50), ("batchSize", 1, 32),
                           ("threads", 1, 2), ("convChannels", 1, 16), ("embedding", 1, 64),
                           ("poolRows", 1, 16), ("poolCols", 1, 10)]:
        require(type(config[key]) is int and low <= config[key] <= high, f"Training limit: {key}")
    for key in ("learningRate", "weightDecay"):
        require(type(config[key]) in (int, float) and math.isfinite(config[key]) and
                0 <= config[key] <= 0.1, f"Training rate: {key}")
    require(config["learningRate"] > 0, "Learning rate must be positive")
    return config


def configure(config):
    validate_training(config)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config["seed"])


def tensors(example):
    # Deliberately whitelist only these three inputs. No label/key/seed/raw state.
    obs = example["observation"]
    return (torch.tensor(obs["planes"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(obs["metadata"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(example["candidateFeatures"], dtype=torch.float32))


class CandidateCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        validate_training(config)
        c, embedding = config["convChannels"], config["embedding"]
        self.board = nn.Sequential(nn.Conv2d(3, c, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(c, c, 3, padding=1), nn.ReLU(),
                                   nn.AdaptiveAvgPool2d((config["poolRows"], config["poolCols"])), nn.Flatten())
        self.context = nn.Sequential(nn.Linear(c * config["poolRows"] * config["poolCols"] + 140, embedding), nn.Tanh())
        self.candidate = nn.Sequential(nn.Linear(52, embedding), nn.Tanh())
        self.bias = nn.Linear(52, 1)
        self.scale = math.sqrt(embedding)

    def forward(self, planes, metadata, candidates):
        require(planes.shape == (1, 3, 16, 10) and metadata.shape == (1, 140) and
                candidates.ndim == 2 and 1 <= candidates.shape[0] <= 4096 and candidates.shape[1] == 52, "Tensor schema")
        require(all(t.device.type == "cpu" and torch.isfinite(t).all().item() for t in (planes, metadata, candidates)), "CPU finite inputs required")
        context = self.context(torch.cat((self.board(planes), metadata), dim=1))
        logits = (self.candidate(candidates) * context).sum(dim=1) / self.scale + self.bias(candidates).squeeze(1)
        require(torch.isfinite(logits).all().item(), "Nonfinite model logits")
        return logits


def predict(model, inputs):
    with torch.inference_mode():
        return int(model(*inputs).argmax().item())
