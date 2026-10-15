"""Command-line interface for preparing, training, and using the classifier."""

import argparse
import json
from pathlib import Path

from genre_classifier.data import import_fma, prepare_dataset
from genre_classifier.prediction import predict
from genre_classifier.training import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genre-classifier",
        description="Train and run a 2D CNN music genre classifier.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fma_parser = commands.add_parser(
        "import-fma",
        help="Link Creative Commons tracks from an extracted FMA Small dataset.",
    )
    fma_parser.add_argument("--audio-dir", type=Path, default=Path("data/fma_small"))
    fma_parser.add_argument(
        "--metadata", type=Path, default=Path("data/fma_metadata/tracks.csv")
    )
    fma_parser.add_argument("--output-dir", type=Path, default=Path("data/fma_labeled"))

    prepare_parser = commands.add_parser(
        "prepare", help="Convert genre-organized audio into cached Mel-spectrograms."
    )
    prepare_parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    prepare_parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    prepare_parser.add_argument("--sample-rate", type=int, default=22050)
    prepare_parser.add_argument("--segment-seconds", type=float, default=3.0)
    prepare_parser.add_argument("--n-mels", type=int, default=128)
    prepare_parser.add_argument("--n-fft", type=int, default=2048)
    prepare_parser.add_argument("--hop-length", type=int, default=512)
    prepare_parser.add_argument("--test-size", type=float, default=0.2)
    prepare_parser.add_argument("--validation-size", type=float, default=0.1)
    prepare_parser.add_argument("--seed", type=int, default=42)

    train_parser = commands.add_parser("train", help="Train and evaluate the CNN.")
    train_parser.add_argument(
        "--manifest", type=Path, default=Path("data/processed/manifest.json")
    )
    train_parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    train_parser.add_argument("--epochs", type=int, default=25)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--learning-rate", type=float, default=0.001)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--num-workers", type=int, default=0)

    predict_parser = commands.add_parser(
        "predict", help="Predict the genre of one audio file."
    )
    predict_parser.add_argument("audio", type=Path)
    predict_parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/best_model.pt")
    )
    predict_parser.add_argument("--top-k", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "import-fma":
        import_fma(args.audio_dir, args.metadata, args.output_dir)
    elif args.command == "prepare":
        prepare_dataset(
            args.data_dir,
            args.output_dir,
            sample_rate=args.sample_rate,
            segment_seconds=args.segment_seconds,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            test_size=args.test_size,
            validation_size=args.validation_size,
            seed=args.seed,
        )
    elif args.command == "train":
        train(
            args.manifest,
            args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            num_workers=args.num_workers,
        )
    else:
        predictions = predict(args.audio, args.checkpoint, top_k=args.top_k)
        print(json.dumps(predictions, indent=2))


if __name__ == "__main__":
    main()
