# signlang-det-training

[English](README.md) | [简体中文](README.zh-CN.md)

Training workflow for a reusable dynamic sign language feature encoder, packaged as a self-contained Kaggle notebook.

## Overview 🚀

This project trains `signlang_det`, a lightweight hand-motion encoder for prototype-based sign retrieval. The encoder learns general temporal representations from the Google Isolated Sign Language Recognition corpus and then adapts them to a target landmark collection.

The temporary classification heads are used only for optimization. They are not included in the exported encoder and do not restrict the vocabulary used at inference time.

The complete workflow is available in [signlang_det_kaggle_training.ipynb](signlang_det_kaggle_training.ipynb).

## Highlights ✨

- Shared `hand168-temporal` preprocessing across training, evaluation, and inference
- Wrist-relative normalization for two hands with explicit missing-hand handling
- Shared hand TCN branches followed by masked Transformer fusion
- Cross-entropy, supervised contrastive, and batch-hard triplet objectives
- Class-disjoint validation and test splits for open-vocabulary evaluation
- Pooled cosine retrieval with constrained frame-level DTW reranking
- Resumable memory-mapped preprocessing and atomic checkpoints
- Standalone encoder export with a SHA-256 fingerprint

## Training Workflow 🧭

The notebook runs the complete pipeline from top to bottom:

1. Validate the Kaggle CUDA runtime and selected GPU architecture.
2. Convert Google ASL parquet sequences into a resumable `hand168-temporal` cache.
3. Train general hand-motion representations with P-by-K sampling.
4. Select and evaluate the encoder on classes excluded from training.
5. Audit and preprocess the target object-NPY landmark collection.
6. Load only the selected encoder, create a new auxiliary classifier, and adapt the representation.
7. Evaluate pooled retrieval and Top-K DTW reranking.
8. Export the final encoder, configuration, metrics, splits, logs, and fingerprint.

### Feature contract

| Item | Contract |
|---|---|
| Input | `float32`, `B × 64 × 168` |
| Per-frame features | 2 hands × 21 landmarks × `(x, y, z, velocity)` |
| Padding value | `-100.0` |
| Encoder output | `float32`, `B × 64 × 128` |
| Valid output frames | L2-normalized |
| Padding output frames | All zeros |

## Data Requirements 📚

Attach both datasets to the Kaggle notebook before running it.

### Source corpus

The Google - Isolated Sign Language Recognition dataset must contain:

- `train.csv`
- `sign_to_prediction_index_map.json`
- the parquet files referenced by the `path` column in `train.csv`

Only `left_hand` and `right_hand` landmarks are used. Face and pose landmarks are ignored.

### Target landmarks

The target dataset must expose a `landmarks` directory containing object-NPY files. Each valid file must provide:

- a string label;
- a `float32` landmark array with shape `T × 100 × 3`.

The default target layout expects right-hand landmarks at indices `0:21` and left-hand landmarks at `21:42`. Adjust `target_right_slice` and `target_left_slice` if the attached dataset uses a different documented layout. Object files are loaded through a restricted pickle allow-list and invalid files are recorded in a rejection manifest.

## Run on Kaggle ▶️

1. Import or upload [signlang_det_kaggle_training.ipynb](signlang_det_kaggle_training.ipynb) to Kaggle.
2. Attach the source corpus and target landmark dataset.
3. Select a **single T4 GPU** accelerator.
4. Review the configuration cell and set explicit dataset paths if automatic discovery is ambiguous.
5. Set `SMOKE_TEST = True` for a short integration run, or keep it `False` for the documented training budget.
6. Choose **Run all** and retain the complete `/kaggle/working/signlang-det` output directory.

The notebook defaults to `num_workers = 0` and disables DataParallel for stable notebook restarts and efficient use of the lightweight encoder. Compatible checkpoints and an incomplete source cache are resumed automatically.

## Configuration ⚙️

The main settings are defined in the `Config` and `TrainSettings` dataclasses near the top of the notebook.

| Setting | Default | Purpose |
|---|---:|---|
| `source_root` | auto-discover | Root containing Google ASL metadata and parquet files |
| `target_landmarks` | auto-discover | Target `landmarks` directory |
| `work_dir` | `/kaggle/working/signlang-det` | Cache, checkpoint, metric, and export directory |
| `max_frames` | `64` | Maximum encoded sequence length |
| `min_frames` | `12` | Minimum accepted action length, matching the runtime segmenter |
| `max_input_frames` | `120` | Maximum accepted action length before rejection |
| `embedding_dim` | `128` | Frame and pooled feature dimension |
| `top_k` | `20` | Number of pooled candidates reranked with DTW |
| `dtw_window` | `12` | Constrained DTW window |
| `resume` | `True` | Restore compatible training state |
| `use_data_parallel` | `False` | Optional dual-GPU execution |

Production defaults use 30 epochs × 300 batches for representation training and 40 epochs × 200 batches for adaptation. These epochs contain a fixed number of P-by-K batches and do not represent full dataset passes.

## Outputs 📦

The notebook writes versioned artifacts under `/kaggle/working/signlang-det`, including:

- immutable epoch checkpoints plus `latest.pt` and `best.pt`;
- `signlang_det_encoder.pt`, containing only the final encoder contract and weights;
- deterministic split manifests and environment metadata;
- CSV and JSONL metrics, status files, and persistent training logs;
- rejected-sample manifests for both data sources;
- final retrieval evaluation results;
- the preprocessing contract identifier and encoder SHA-256 fingerprint.

Keep the encoder fingerprint and `hand168-temporal` preprocessing identifier with every dynamic prototype. A prototype produced by a different encoder or preprocessing contract must not be mixed with the exported model.

## Limitations and Calibration ⚠️

- Full training requires Kaggle-mounted datasets and a compatible CUDA runtime; repository-side checks cannot reproduce the production run without them.
- The current PyTorch CUDA build described by the workflow targets T4. CUDA availability alone does not guarantee architecture compatibility.
- The notebook does not provide action segmentation, a serving API, or a prototype management interface.
- Unknown-sign distance and margin thresholds cannot be inferred from training loss. They require separate calibration data containing known queries, difficult negatives, and unknown actions.
- Dataset licenses and usage restrictions remain the responsibility of their respective owners. This repository does not redistribute either training dataset.

## Project Files 🗂️

| File | Description |
|---|---|
| [signlang_det_kaggle_training.ipynb](signlang_det_kaggle_training.ipynb) | Self-contained Kaggle training and export workflow |
| [README.zh-CN.md](README.zh-CN.md) | Simplified Chinese documentation |
| [LICENSE](LICENSE) | Project license |

## License 📄

This project is released under the [Apache License 2.0](LICENSE).
