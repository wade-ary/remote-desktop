"""Gesture model architectures and checkpoint loading for version1."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

NUM_FRAMES = 37
INPUT_SIZE = 63
NUM_CLASSES = 6
HIDDEN_SIZE = 128
NUM_LAYERS = 2
TRANSFORMER_NUM_LAYERS = 4
DROPOUT = 0.3
D_MODEL = 256
NHEAD = 4
DIM_FEEDFORWARD = 512

DEFAULT_LABELS = [
    "No gesture",
    "Swiping Left",
    "Swiping Right",
    "Turning Hand Clockwise",
    "Turning Hand Counterclockwise",
    "Zooming In With Two Fingers",
]

MODEL_REGISTRY: dict[str, type[nn.Module]] = {}


def _register(cls: type[nn.Module]) -> type[nn.Module]:
    MODEL_REGISTRY[cls.__name__] = cls
    return cls


@_register
class GestureBiLSTM(nn.Module):
    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        last_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        return self.classifier(self.dropout(last_hidden))


@_register
class GestureLSTM(nn.Module):
    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.classifier(self.dropout(hidden[-1]))


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention(x), dim=1)
        return (x * weights).sum(dim=1)


@_register
class GestureTransformer(nn.Module):
    def __init__(
        self,
        input_size=INPUT_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=TRANSFORMER_NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        max_len=NUM_FRAMES,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = AttentionPooling(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_embedding[:, : x.size(1), :]
        x = self.transformer(x)
        pooled = self.attn_pool(x)
        return self.classifier(self.dropout(pooled))


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_gesture_model(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[nn.Module, list[str], dict]:
    """Load a gesture checkpoint saved from version_1_model.ipynb."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = device or resolve_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_class = checkpoint.get("model_class")
    if model_class not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model_class '{model_class}'. Known: {known}")

    raw_hyperparameters = checkpoint.get("hyperparameters", {})
    allowed_keys = {
        "GestureBiLSTM": {"input_size", "hidden_size", "num_layers", "num_classes", "dropout"},
        "GestureLSTM": {"input_size", "hidden_size", "num_layers", "num_classes", "dropout"},
        "GestureTransformer": {
            "input_size", "d_model", "nhead", "num_layers", "dim_feedforward",
            "num_classes", "dropout", "max_len",
        },
    }
    model_kwargs = {
        k: v for k, v in raw_hyperparameters.items()
        if k in allowed_keys.get(model_class, set())
    }
    if model_class == "GestureTransformer" and "max_len" not in model_kwargs:
        model_kwargs["max_len"] = raw_hyperparameters.get("num_frames", NUM_FRAMES)

    model = MODEL_REGISTRY[model_class](**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    labels = checkpoint.get("labels", DEFAULT_LABELS)
    return model, labels, raw_hyperparameters
