# Audio Genre Classification

Train a 2D convolutional neural network (CNN) to classify music by genre. The
pipeline converts audio into fixed-size, normalized Mel-spectrograms, splits
whole tracks into train/validation/test sets, trains the model, reports
held-out metrics, and predicts genres for new tracks.

## 1. Set up

Use Python 3.10 or newer. From the project root:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

PyTorch automatically uses CUDA when available, Apple Silicon MPS when
available, and otherwise CPU.

## 2. Organize the audio

Put audio files in one directory per genre. WAV, MP3, FLAC, OGG, M4A, and AIFF
files are supported.

```text
data/raw/
├── blues/
│   ├── blues.00000.wav
│   └── blues.00001.wav
├── classical/
│   ├── classical.00000.wav
│   └── classical.00001.wav
└── rock/
    ├── rock.00000.wav
    └── rock.00001.wav
```

Any genre names are accepted. For a larger, balanced dataset, use FMA Small:
8,000 30-second tracks across eight genres, hosted by the dataset authors at
EPFL/ETHZ. Individual track licenses are included in its metadata; the import
command below only includes tracks with Creative Commons licenses. The audio
is not bundled with this project.

### Download and import FMA Small

Download the audio and metadata archives from the [official FMA repository
instructions](https://github.com/mdeff/fma#data). The dataset is hosted by the
Swiss Data Science Center:

```bash
mkdir -p data/downloads
curl -L --fail https://os.unil.cloud.switch.ch/fma/fma_small.zip \
  -o data/downloads/fma_small.zip
curl -L --fail https://os.unil.cloud.switch.ch/fma/fma_metadata.zip \
  -o data/downloads/fma_metadata.zip
echo "ade154f733639d52e35e32f5593efe5be76c6d70  data/downloads/fma_small.zip" | shasum -a 1 -c -
echo "f0df49ffe5f2a6008d7dc83c6915b31835dfe733  data/downloads/fma_metadata.zip" | shasum -a 1 -c -
python -m zipfile -e data/downloads/fma_small.zip data
python -m zipfile -e data/downloads/fma_metadata.zip data
python -m genre_classifier.cli import-fma
```

FMA Small requires about 7.2 GiB to download, plus room for extraction and
cached features. The import step creates links to the original audio rather
than copying it. Then set `--data-dir data/fma_labeled` in the preparation
command below. The FMA dataset is described in
[Defferrard et al. (ISMIR 2017)](https://arxiv.org/abs/1612.01840); follow the
dataset's attribution and individual track license terms.

## 3. Prepare Mel-spectrograms

```bash
python -m genre_classifier.cli prepare --data-dir data/raw
```

Each track is decoded as mono 22,050 Hz audio and divided into three-second
segments. The final partial segment is zero-padded. Each segment is converted
into a 128-bin power Mel-spectrogram, transformed to decibels, normalized, and
saved as a NumPy array in `data/processed/features/`.

The command writes `data/processed/manifest.json` with labels and feature
settings. Splits happen at the **track level before segmentation** to prevent
segments from the same track leaking between training and evaluation sets.
Each genre needs at least three tracks. Unreadable audio files are reported
and recorded in the manifest's `skipped_tracks` list; preparation fails if a
genre no longer has examples in every split.

Common preprocessing options:

```bash
python -m genre_classifier.cli prepare \
  --data-dir data/raw \
  --output-dir data/processed \
  --segment-seconds 3 \
  --n-mels 128 \
  --test-size 0.2 \
  --validation-size 0.1 \
  --seed 42
```

For FMA Small, 10-second segments and 64 Mel bins reduce preprocessing and
training time while still covering all 30 seconds of each track:

```bash
python -m genre_classifier.cli prepare \
  --data-dir data/fma_labeled \
  --output-dir data/processed/fma \
  --segment-seconds 10 \
  --n-mels 64 \
  --n-fft 1024
```

## 4. Train and evaluate

```bash
python -m genre_classifier.cli train \
  --manifest data/processed/manifest.json \
  --output-dir artifacts \
  --epochs 25 \
  --batch-size 32
```

To train on the prepared FMA data, point training at its manifest:

```bash
python -m genre_classifier.cli train \
  --manifest data/processed/fma/manifest.json \
  --output-dir artifacts/fma \
  --epochs 10 \
  --batch-size 64
```

The CNN uses four convolutional blocks and adaptive pooling. The checkpoint
with the best validation accuracy is saved to `artifacts/best_model.pt`; the
test set is evaluated after training. Training history, test accuracy, macro
F1, per-genre metrics, and a confusion matrix are saved to
`artifacts/training_results.json`. Test metrics are computed per track by
averaging the segment probabilities for each held-out track.

### FMA Small run in this workspace

The full FMA Small archive contains 8,000 tracks. The license-aware importer
linked 7,919 tracks across the eight genres; six unreadable audio files were
reported and excluded during preprocessing. With the commands above and a
10-epoch run, the track-level test accuracy was **57.51%** and macro F1 was
**0.549** on 1,584 held-out tracks. This is a baseline run, not a tuned model.

## 5. Predict a genre

```bash
python -m genre_classifier.cli predict path/to/song.mp3
```

The command averages model probabilities over all audio segments and prints
the top three genres and confidence scores as JSON. Choose a different number
of results with `--top-k 5`, or a different model with
`--checkpoint path/to/best_model.pt`.

## Project structure

```text
src/genre_classifier/
├── cli.py          # prepare, train, and predict commands
├── data.py         # track-level splits and cached feature dataset
├── features.py     # audio decoding and Mel-spectrogram extraction
├── model.py        # 2D CNN
├── prediction.py   # inference for a complete audio track
└── training.py     # training loop, checkpointing, and test metrics
```

Run the project tests with:

```bash
python -m unittest discover -s tests -v
```
