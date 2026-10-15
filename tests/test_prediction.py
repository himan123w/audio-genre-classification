import math
import struct
import tempfile
import unittest
import wave
from pathlib import Path

import torch

from genre_classifier.model import GenreCNN
from genre_classifier.prediction import predict


class PredictionTests(unittest.TestCase):
    def test_predicts_from_checkpoint_and_limits_top_k_to_class_count(self) -> None:
        sample_rate = 8000
        sample_count = sample_rate // 2
        samples = b"".join(
            struct.pack(
                "<h",
                round(12000 * math.sin(2 * math.pi * 440 * index / sample_rate)),
            )
            for index in range(sample_count)
        )
        classes = ["blues", "classical"]
        feature_config = {
            "sample_rate": sample_rate,
            "segment_seconds": 0.5,
            "n_mels": 16,
            "n_fft": 256,
            "hop_length": 64,
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_path = root / "tone.wav"
            checkpoint_path = root / "model.pt"
            with wave.open(str(audio_path), "wb") as audio_file:
                audio_file.setnchannels(1)
                audio_file.setsampwidth(2)
                audio_file.setframerate(sample_rate)
                audio_file.writeframes(samples)

            model = GenreCNN(num_classes=len(classes))
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "feature_config": feature_config,
                },
                checkpoint_path,
            )

            results = predict(audio_path, checkpoint_path, top_k=10)

        self.assertEqual(len(results), len(classes))
        self.assertEqual({result["genre"] for result in results}, set(classes))
        self.assertAlmostEqual(
            sum(result["confidence"] for result in results), 1.0, places=3
        )
        self.assertTrue(
            all(0.0 <= result["confidence"] <= 1.0 for result in results)
        )


if __name__ == "__main__":
    unittest.main()
