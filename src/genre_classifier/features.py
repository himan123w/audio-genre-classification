"""Audio loading and fixed-size Mel-spectrogram extraction."""

from pathlib import Path

import librosa
import numpy as np


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aiff", ".aif"}
DB_FLOOR = -80.0


def extract_segments(
    audio_path: Path,
    *,
    sample_rate: int,
    segment_seconds: float,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> list[np.ndarray]:
    """Load a track and return one normalized Mel-spectrogram per segment."""
    try:
        audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Could not load audio file {audio_path}: {exc}") from exc

    segment_samples = int(sample_rate * segment_seconds)
    if segment_samples <= 0:
        raise ValueError("segment_seconds must be greater than zero")
    if audio.size == 0:
        raise ValueError(f"Audio file contains no samples: {audio_path}")

    segments = []
    for start in range(0, audio.size, segment_samples):
        segment = audio[start : start + segment_samples]
        if segment.size < segment_samples:
            segment = np.pad(segment, (0, segment_samples - segment.size))

        mel = librosa.feature.melspectrogram(
            y=segment,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        decibels = librosa.power_to_db(mel, ref=np.max)
        normalized = np.clip((decibels - DB_FLOOR) / -DB_FLOOR, 0.0, 1.0)
        segments.append(normalized.astype(np.float32))
    return segments
