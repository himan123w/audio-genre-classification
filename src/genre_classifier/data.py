"""Dataset discovery, track-level splitting, and cached feature loading."""

import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from genre_classifier.features import AUDIO_EXTENSIONS, extract_segments


def import_fma(
    audio_dir: Path,
    metadata_path: Path,
    output_dir: Path,
) -> int:
    """Create genre folders of links for Creative Commons-licensed FMA tracks."""
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"FMA audio directory does not exist: {audio_dir}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"FMA metadata file does not exist: {metadata_path}")

    with metadata_path.open(encoding="utf-8", newline="") as metadata_file:
        reader = csv.reader(metadata_file)
        try:
            header_rows = [next(reader), next(reader)]
        except StopIteration as exc:
            raise ValueError(f"FMA metadata has no two-row header: {metadata_path}") from exc
        columns = list(zip(*header_rows))

        def column_index(name: str) -> int:
            matches = [
                index
                for index, labels in enumerate(columns)
                if tuple(part.strip().lower() for part in labels) == ("track", name)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one ('track', {name!r}) column in {metadata_path}."
                )
            return matches[0]

        genre_index = column_index("genre_top")
        license_index = column_index("license")
        created = 0
        for row in reader:
            if not row or not row[0].strip():
                continue
            if row[0].strip().lower() == "track_id":
                continue
            track_id = int(row[0])
            if max(genre_index, license_index) >= len(row):
                raise ValueError(f"Malformed FMA metadata row for track {track_id}.")
            genre = row[genre_index].strip()
            license_url = row[license_index].strip().lower()
            is_cc_license = (
                license_url.startswith(("attribution", "creative commons", "cc0"))
            )
            if not genre or not is_cc_license:
                continue

            source = audio_dir / f"{track_id // 1000:03d}" / f"{track_id:06d}.mp3"
            if not source.is_file():
                continue
            genre_dir = output_dir / genre
            genre_dir.mkdir(parents=True, exist_ok=True)
            destination = genre_dir / f"fma_{track_id:06d}.mp3"
            if destination.is_symlink() and destination.resolve() == source.resolve():
                continue
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(
                    f"Refusing to replace existing FMA import path: {destination}"
                )
            os.symlink(source.resolve(), destination)
            created += 1

    if created == 0 and not any(output_dir.glob("*/*")):
        raise ValueError(
            "No Creative Commons-licensed FMA audio tracks were imported. "
            "Check that the audio and metadata archives match."
        )
    print(
        f"Imported {created} Creative Commons-licensed FMA tracks found in "
        f"{audio_dir} into {output_dir}."
    )
    return created


def _split_tracks(
    tracks_by_genre: dict[str, list[Path]],
    *,
    test_size: float,
    validation_size: float,
    seed: int,
) -> dict[str, list[tuple[str, Path]]]:
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("test_size and validation_size must both be between 0 and 1")
    if test_size + validation_size >= 1:
        raise ValueError("test_size + validation_size must be less than 1")

    splits: dict[str, list[tuple[str, Path]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    rng = np.random.default_rng(seed)
    for genre, paths in sorted(tracks_by_genre.items()):
        if len(paths) < 3:
            raise ValueError(
                f"Genre {genre!r} has {len(paths)} track(s); at least 3 are "
                "required for track-level train/validation/test splits."
            )
        shuffled = list(paths)
        rng.shuffle(shuffled)
        test_count = max(1, round(len(shuffled) * test_size))
        validation_count = max(1, round(len(shuffled) * validation_size))
        while test_count + validation_count >= len(shuffled):
            if test_count >= validation_count and test_count > 1:
                test_count -= 1
            elif validation_count > 1:
                validation_count -= 1
            else:
                raise ValueError(f"Not enough tracks to split genre {genre!r}.")

        test_tracks = shuffled[:test_count]
        validation_tracks = shuffled[test_count : test_count + validation_count]
        train_tracks = shuffled[test_count + validation_count :]
        for split_name, split_tracks in (
            ("train", train_tracks),
            ("validation", validation_tracks),
            ("test", test_tracks),
        ):
            splits[split_name].extend((genre, path) for path in split_tracks)
    return splits


def prepare_dataset(
    data_dir: Path,
    output_dir: Path,
    *,
    sample_rate: int = 22050,
    segment_seconds: float = 3.0,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    seed: int = 42,
) -> Path:
    """Create cached Mel-spectrogram segments and a track-safe data manifest."""
    if sample_rate <= 0 or segment_seconds <= 0:
        raise ValueError("sample_rate and segment_seconds must be greater than zero")
    if n_mels <= 0 or n_fft <= 0 or hop_length <= 0:
        raise ValueError("n_mels, n_fft, and hop_length must be greater than zero")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Audio dataset directory does not exist: {data_dir}")
    tracks_by_genre: dict[str, list[Path]] = {}
    for genre_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        tracks = sorted(
            path
            for path in genre_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )
        if tracks:
            tracks_by_genre[genre_dir.name] = tracks
    if not tracks_by_genre:
        raise ValueError(
            f"No audio files found. Expected genre subfolders containing "
            f"{', '.join(sorted(AUDIO_EXTENSIONS))} files in {data_dir}."
        )

    splits = _split_tracks(
        tracks_by_genre,
        test_size=test_size,
        validation_size=validation_size,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    examples: list[dict[str, Any]] = []
    skipped_tracks: list[dict[str, str]] = []
    processed_tracks: dict[str, set[str]] = {
        genre: set() for genre in tracks_by_genre
    }
    track_indices = {
        (genre, path): index
        for genre, paths in sorted(tracks_by_genre.items())
        for index, path in enumerate(paths)
    }

    for split_name, tracks in splits.items():
        for genre, audio_path in tracks:
            track_index = track_indices[(genre, audio_path)]
            track_id = f"{genre}/{audio_path.relative_to(data_dir / genre).as_posix()}"
            try:
                segments = extract_segments(
                    audio_path,
                    sample_rate=sample_rate,
                    segment_seconds=segment_seconds,
                    n_mels=n_mels,
                    n_fft=n_fft,
                    hop_length=hop_length,
                )
            except (RuntimeError, ValueError) as exc:
                print(f"Skipping unreadable track {track_id}: {exc}")
                skipped_tracks.append({"track_id": track_id, "error": str(exc)})
                continue
            processed_tracks[genre].add(split_name)
            for segment_index, feature in enumerate(segments):
                feature_path = (
                    Path("features")
                    / genre
                    / f"track_{track_index:05d}_segment_{segment_index:04d}.npy"
                )
                full_feature_path = output_dir / feature_path
                full_feature_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(full_feature_path, feature)
                examples.append(
                    {
                        "feature": feature_path.as_posix(),
                        "genre": genre,
                        "split": split_name,
                        "track_id": track_id,
                        "segment": segment_index,
                    }
                )
            print(f"[{split_name}] {track_id}: {len(segments)} segment(s)")

    missing_splits = [
        f"{genre}/{split_name}"
        for genre, available_splits in processed_tracks.items()
        for split_name in ("train", "validation", "test")
        if split_name not in available_splits
    ]
    if missing_splits:
        raise ValueError(
            "Unreadable tracks left genres without examples in required splits: "
            + ", ".join(missing_splits)
        )

    manifest = {
        "schema_version": 1,
        "classes": sorted(tracks_by_genre),
        "feature_config": {
            "sample_rate": sample_rate,
            "segment_seconds": segment_seconds,
            "n_mels": n_mels,
            "n_fft": n_fft,
            "hop_length": hop_length,
        },
        "examples": examples,
        "skipped_tracks": skipped_tracks,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Prepared {len(examples)} segments from "
        f"{sum(map(len, tracks_by_genre.values())) - len(skipped_tracks)} readable "
        f"tracks; skipped {len(skipped_tracks)} unreadable track(s). "
        f"Manifest: {manifest_path}"
    )
    return manifest_path


class MelSegmentDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        class_to_index: dict[str, int],
    ) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.root = manifest_path.parent
        self.examples = [
            example for example in manifest["examples"] if example["split"] == split
        ]
        self.class_to_index = class_to_index
        if not self.examples:
            raise ValueError(f"No {split!r} examples found in {manifest_path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        example = self.examples[index]
        feature_path = self.root / example["feature"]
        try:
            feature = np.load(feature_path)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"Could not read feature file {feature_path}: {exc}") from exc
        if feature.ndim != 2:
            raise ValueError(f"Expected a 2D Mel-spectrogram in {feature_path}")
        return (
            torch.from_numpy(feature.copy()).unsqueeze(0),
            self.class_to_index[example["genre"]],
        )


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read dataset manifest {manifest_path}: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported dataset manifest version in {manifest_path}")
    return manifest
