# Kaggle Continuous Delivery

The repository can publish a tagged notebook to one stable Kaggle kernel, wait for its training run across separate GitHub Actions executions, and attach the completed output to the matching GitHub Release.

## Setup

Create these GitHub Actions secrets:

| Secret | Purpose |
|---|---|
| `KAGGLE_API_TOKEN` | API token created in Kaggle Settings; the authenticated account owns the destination kernel |

The Kaggle account must have accepted the rules for [Google - Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs/data) and be able to access [ASL-preprocessing 7](https://www.kaggle.com/code/abdelrhmankaram/asl-preprocessing-7/output).

Optional GitHub Actions variables:

| Variable | Default | Purpose |
|---|---:|---|
| `KAGGLE_KERNEL_SLUG` | `signlang-det-training` | Stable destination kernel slug under the authenticated account |
| `KAGGLE_KERNEL_PRIVATE` | `true` | Whether the destination kernel is private |
| `RKNN_TARGET_PLATFORM` | `rk3588` | RKNN target platform passed to RKNN Toolkit 2; override it for another supported Rockchip target |

Kaggle's standard `kernel-metadata.json` sets `machine_shape` to `NvidiaTeslaT4`, enables GPU execution, and attaches the competition and notebook-output sources. The workflow pins an official Kaggle CLI release that passes this field to the kernel save request. The run will fail at Kaggle submission time if that accelerator is unavailable or not permitted for the account; it does not silently request a different GPU class.

Both workflows use Python 3.10. The worker conversion environment pins `setuptools<81`, `onnx==1.16.1`, `onnxruntime==1.23.2`, and `rknn-toolkit2==2.3.2`. ONNX 1.16.1 is required because RKNN Toolkit 2.3.2 still uses its legacy `onnx.mapping` API.

## Delivery flow

Pushing any Git tag creates a readable draft Release backed by hidden machine state as a durable FIFO queue item. One serialized worker advances a single transition on each invocation:

1. Resolve the account identity from `KAGGLE_API_TOKEN`, then upload the notebook from the exact tagged commit to that account's `${KAGGLE_KERNEL_SLUG}`.
2. Record the returned numeric Kaggle version in the draft Release. The Git tag is also injected into the uploaded copy as provenance, because Kaggle version names are numeric and cannot be replaced by arbitrary tag names.
3. Let the scheduled workflow check the active run every ten minutes.
4. Download successful output and require its exact file allowlist.
5. Verify and convert the final PT encoder to ONNX, then create non-quantized and INT8 RKNN models for `${RKNN_TARGET_PLATFORM}`. ONNX Runtime must numerically match PyTorch before either RKNN build starts.
6. Create a model manifest, upload the seven required assets, and publish the Release.

Only one queue item may be `starting` or `running`. Later tags remain queued. An already-running external version of the same stable kernel is allowed to finish before the queue continues.

If a draft Release's hidden tag becomes stale after a GitHub-side edit or migration, the worker repairs it from the Release's current `tag_name`. Before any upload, the resolved Git tag commit must still exactly match the queued `git_sha`.

Each published Release contains exactly:

- `signlang_det_encoder.pt`;
- `signlang_det_encoder.onnx`;
- `signlang_det_encoder.rknn`;
- `signlang_det_encoder.int8.rknn`;
- `model-manifest.json`, containing model identity/version, architecture and configuration, artifact and tokenizer locations, format-specific runtime dependencies, input/output contracts, provenance, target platform, sizes, and SHA-256 integrity data;
- `signlang_det_kaggle_training.ipynb` from the exact tagged commit;
- `notebook-output.tar.gz`, containing the INT8 calibration archive, two charts, two metric CSV files, and two training logs.

The Notebook selects up to 100 target-training samples with a deterministic random seed for INT8 calibration. Any missing or unexpected Notebook output prevents Release publication.

## Operations

Use **Kaggle CD - scheduled worker → Run workflow** to request an immediate poll. Each invocation still performs only one status check.

If a job reaches `failed`, run the same workflow with `retry_tag` set to its exact Git tag. The failed draft Release is returned to the queue. Transient API failures while a Kaggle version is active leave it recoverable for the next scheduled run.

The worker never waits for training inside one Actions run. Its concurrency group does not cancel in-progress ticks, and draft Releases preserve queue state between executions.
