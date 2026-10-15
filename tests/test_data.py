import csv
import tempfile
import unittest
from pathlib import Path

from genre_classifier.data import _split_tracks, import_fma


class DatasetSplitTests(unittest.TestCase):
    def test_track_splits_are_disjoint_and_reproducible(self) -> None:
        tracks = {
            genre: [Path(f"{genre}-{index}.mp3") for index in range(10)]
            for genre in ("blues", "rock")
        }
        first = _split_tracks(
            tracks, test_size=0.2, validation_size=0.2, seed=17
        )
        second = _split_tracks(
            tracks, test_size=0.2, validation_size=0.2, seed=17
        )
        self.assertEqual(first, second)
        sets = {
            name: {track for _, track in values}
            for name, values in first.items()
        }
        self.assertFalse(sets["train"] & sets["validation"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["validation"] & sets["test"])


class FmaImportTests(unittest.TestCase):
    def test_imports_present_creative_commons_tracks_as_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_dir = root / "audio"
            metadata_path = root / "tracks.csv"
            output_dir = root / "labeled"

            with metadata_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["", "track", "track", "track"])
                writer.writerow(["", "id", "genre_top", "license"])
                writer.writerow(["track_id", "", "", ""])
                writer.writerow(
                    ["2", "2", "Hip-Hop", "Attribution-NonCommercial 3.0"]
                )
                writer.writerow(["3", "3", "Rock", "Public Domain"])
                writer.writerow(["4", "4", "Rock", "CC0 1.0 Universal"])

            present_track = audio_dir / "000" / "000002.mp3"
            present_track.parent.mkdir(parents=True)
            present_track.write_bytes(b"test audio")
            missing_track_id = 4

            self.assertEqual(import_fma(audio_dir, metadata_path, output_dir), 1)
            imported = output_dir / "Hip-Hop" / "fma_000002.mp3"
            self.assertTrue(imported.is_symlink())
            self.assertEqual(imported.resolve(), present_track.resolve())
            self.assertFalse((output_dir / "Rock" / "fma_000003.mp3").exists())
            self.assertFalse(
                (output_dir / "Rock" / f"fma_{missing_track_id:06d}.mp3").exists()
            )
            self.assertEqual(import_fma(audio_dir, metadata_path, output_dir), 0)


if __name__ == "__main__":
    unittest.main()
