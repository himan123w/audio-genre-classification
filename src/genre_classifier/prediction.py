"""Classify new audio tracks using a trained model checkpoint."""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from genre_classifier.features import extract_segments
from genre_classifier.model import GenreCNN
from genre_classifier.training import select_device


def predict(audio_path: Path, checkpoint_path: Path, top_k: int = 3) -> list[dict[str, Any]]:
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint_path}")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    device = select_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    classes: list[str] = checkpoint["classes"]
    config = checkpoint["feature_config"]
    segments = extract_segments(audio_path, **config)
    batch = torch.from_numpy(np.stack(segments)).unsqueeze(1).to(device)

    model = GenreCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        mean_probabilities = model(batch).softmax(dim=1).mean(dim=0)

    count = min(top_k, len(classes))
    probabilities, indices = torch.topk(mean_probabilities, k=count)
    return [
        {"genre": classes[index], "confidence": round(probability, 4)}
        for probability, index in zip(probabilities.cpu().tolist(), indices.cpu().tolist())
    ]

