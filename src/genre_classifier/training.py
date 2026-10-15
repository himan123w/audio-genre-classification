"""Training, checkpointing, and held-out evaluation."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from genre_classifier.data import MelSegmentDataset, load_manifest
from genre_classifier.model import GenreCNN


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_epoch(
    model: GenreCNN,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += batch_size
    return total_loss / count, correct / count


def _predict(
    model: GenreCNN, loader: DataLoader, device: torch.device
) -> np.ndarray:
    model.eval()
    probabilities = []
    with torch.no_grad():
        for features, _ in loader:
            logits = model(features.to(device))
            probabilities.extend(logits.softmax(dim=1).cpu().numpy())
    return np.asarray(probabilities)


def _aggregate_track_predictions(
    probabilities: np.ndarray,
    examples: list[dict[str, Any]],
    class_to_index: dict[str, int],
) -> tuple[list[int], list[int]]:
    if len(probabilities) != len(examples):
        raise ValueError(
            "The number of test predictions does not match the test examples."
        )
    track_probabilities: dict[str, list[np.ndarray]] = {}
    track_labels: dict[str, int] = {}
    for probability, example in zip(probabilities, examples):
        track_id = example["track_id"]
        label = class_to_index[example["genre"]]
        if track_id in track_labels and track_labels[track_id] != label:
            raise ValueError(f"Track {track_id!r} has inconsistent genre labels.")
        track_labels[track_id] = label
        track_probabilities.setdefault(track_id, []).append(probability)
    actual = []
    predicted = []
    for track_id, segments in track_probabilities.items():
        actual.append(track_labels[track_id])
        predicted.append(int(np.mean(segments, axis=0).argmax()))
    return actual, predicted


def _classification_metrics(
    actual: list[int], predicted: list[int], classes: list[str]
) -> dict[str, Any]:
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for true_label, predicted_label in zip(actual, predicted):
        matrix[true_label, predicted_label] += 1
    per_class = {}
    f1_values = []
    for index, name in enumerate(classes):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        f1_values.append(f1)
        per_class[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(matrix[index, :].sum()),
        }
    count = len(actual)
    return {
        "accuracy": sum(a == p for a, p in zip(actual, predicted)) / count,
        "macro_f1": float(np.mean(f1_values)),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def train(
    manifest_path: Path,
    output_dir: Path,
    *,
    epochs: int = 25,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    seed: int = 42,
    num_workers: int = 0,
) -> Path:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs and batch_size must be positive, and learning_rate > 0")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    manifest = load_manifest(manifest_path)
    classes: list[str] = manifest["classes"]
    if len(classes) < 2:
        raise ValueError("At least two genre classes are required to train a classifier.")
    class_to_index = {name: index for index, name in enumerate(classes)}
    datasets = {
        split: MelSegmentDataset(manifest_path, split, class_to_index)
        for split in ("train", "validation", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for split, dataset in datasets.items()
    }
    device = select_device()
    model = GenreCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    best_validation_accuracy = -1.0
    history = []

    print(f"Training on {device} with {len(classes)} genres.")
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = _run_epoch(
            model, loaders["train"], criterion, device, optimizer
        )
        validation_loss, validation_accuracy = _run_epoch(
            model, loaders["validation"], criterion, device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss {train_loss:.4f}, acc {train_accuracy:.3f} | "
            f"validation loss {validation_loss:.4f}, acc {validation_accuracy:.3f}"
        )
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "feature_config": manifest["feature_config"],
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_probabilities = _predict(model, loaders["test"], device)
    actual, predicted = _aggregate_track_predictions(
        test_probabilities, datasets["test"].examples, class_to_index
    )
    test_metrics = _classification_metrics(actual, predicted, classes)
    results = {
        "device": str(device),
        "best_validation_accuracy": best_validation_accuracy,
        "test": test_metrics,
        "history": history,
    }
    results_path = output_dir / "training_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(
        f"Test accuracy: {test_metrics['accuracy']:.3f} | "
        f"macro F1: {test_metrics['macro_f1']:.3f}"
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Detailed results: {results_path}")
    return checkpoint_path
