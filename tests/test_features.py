import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from genre_classifier.features import extract_segments


class FeatureExtractionTests(unittest.TestCase):
    def test_extracts_fixed_size_normalized_segments(self) -> None:
        sample_rate = 8000
        sample_count = int(sample_rate * 1.25)
        samples = b"".join(
            struct.pack(
                "<h",
                round(12000 * math.sin(2 * math.pi * 440 * index / sample_rate)),
            )
            for index in range(sample_count)
        )

        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "tone.wav"
            with wave.open(str(audio_path), "wb") as audio_file:
                audio_file.setnchannels(1)
                audio_file.setsampwidth(2)
                audio_file.setframerate(sample_rate)
                audio_file.writeframes(samples)

            segments = extract_segments(
                audio_path,
                sample_rate=sample_rate,
                segment_seconds=1,
                n_mels=16,
                n_fft=256,
                hop_length=64,
            )

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].shape, segments[1].shape)
        self.assertEqual(segments[0].shape[0], 16)
        for segment in segments:
            self.assertEqual(segment.dtype, np.float32)
            self.assertTrue(np.isfinite(segment).all())
            self.assertGreaterEqual(float(segment.min()), 0.0)
            self.assertLessEqual(float(segment.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
