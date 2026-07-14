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

The worker uses isolated jobs because steps in one GitHub Actions job share the same Python environment. A lightweight standard-library probe checks the GitHub Release queue first, so idle ticks do not install or authenticate the Kaggle client. All control-plane operations—queue updates, Kaggle submission and polling, Release publication, and failure finalization—use Python 3.11; only the Kaggle-facing steps install the pinned Kaggle CLI commit. Model conversion runs separately on Python 3.10 and pins `setuptools==79.0.1`, `numpy==1.26.4`, `torch==2.4.0`, `onnx==1.16.1`, `onnxruntime==1.23.2`, and `rknn-toolkit2==2.3.2`. ONNX 1.16.1 is required because RKNN Toolkit 2.3.2 still uses its legacy `onnx.mapping` API.

Every GitHub Action is pinned to a full commit. Write permissions are limited to the jobs that update queue or Release state, and API token environment variables are supplied only to the steps that use them. The conversion job has read-only repository permission, does not persist checkout credentials, and never receives Kaggle or GitHub API tokens.

## Delivery flow

Pushing any Git tag creates a readable draft Release backed by hidden machine state as a durable FIFO queue item. One serialized worker advances a single transition on each invocation:

1. Resolve the account identity from `KAGGLE_API_TOKEN`, then upload the notebook from the exact tagged commit to that account's `${KAGGLE_KERNEL_SLUG}`.
2. Lock the authenticated Kaggle kernel, returned numeric version, and RKNN target platform in the draft Release. The kernel and version together identify the external run; a version number is never accepted on its own. The uploaded Notebook is the unmodified file from the tagged commit; CD does not inject custom Notebook metadata.
3. Let the scheduled workflow check the active run every ten minutes.
4. Check that both the kernel and latest numeric version still match before and after downloading successful output. The downloader removes the Kaggle CLI's generated kernel log, requires the exact Notebook output allowlist, and passes the result to an isolated conversion job as a run-scoped GitHub Actions artifact.
5. In the Python 3.10 conversion job, safely validate the PT checkpoint and bounded calibration archive, then run the converter from the tagged commit. ONNX Runtime must numerically match PyTorch, and both generated RKNN files must pass simulator inference checks against ONNX.
6. Pass the seven prepared assets to an isolated publishing job, revalidate the draft Release state and every manifest SHA-256, upload the assets, and publish the Release. Conversion, artifact, or publication failures are written back to the draft as retryable failures.

Only one queue item may be `starting` or `running`. Later tags remain queued. An already-running external version of the same stable kernel is allowed to finish before the queue continues. The worker never adopts a numeric version unless its own `kernels_push` call returned that version or a retry references the same previously acknowledged completed version on the locked kernel. An ambiguous interrupted push may therefore create a duplicate training version on the next attempt, but it cannot be mistaken for an external run.

If a draft Release's hidden tag becomes stale after a GitHub-side edit or migration, the worker repairs it from the Release's current `tag_name`. Before any upload, the resolved Git tag commit must still exactly match the queued `git_sha`. Before downloading completed output, the authenticated account and configured slug must resolve to the locked kernel, and that kernel's latest numeric version must match the version recorded after submission. Legacy v1 queue items bind missing kernel or target fields once only when doing so is unambiguous; a recorded version without a kernel fails closed.

Each published Release contains exactly:

- `signlang_det_encoder.pt`;
- `signlang_det_encoder.onnx`;
- `signlang_det_encoder.rknn`;
- `signlang_det_encoder.int8.rknn`;
- `model-manifest.json`, containing model identity/version, architecture and configuration, artifact and tokenizer locations, format-specific runtime dependencies, input/output contracts, provenance, target platform, sizes, and SHA-256 integrity data;
- `signlang_det_kaggle_training.ipynb` from the exact tagged commit;
- `notebook-output.tar.gz`, containing the INT8 calibration archive, two charts, two metric CSV files, and two training logs.

The Notebook selects up to 100 target-training samples with a deterministic random seed for INT8 calibration. Conversion accepts only the expected regular calibration files and enforces compressed, expanded, member-count, and per-file size limits. Any missing or unexpected Notebook output prevents Release publication.

The manifest records I/O per model format. PT accepts a variable batch, while ONNX and both RKNN files use batch size 1. Accuracy-guarded hybrid quantization changes the INT8 model's feature input to `int8`, whose embedded scale and zero-point must be queried through RKNN Runtime, while retaining the embedding output as `float16`. The sequence-length input remains `int32`.

## Operations

Use **Kaggle CD - scheduled worker → Run workflow** on the repository's default branch to request an immediate poll. The workflow rejects manual runs from another branch or tag. Each invocation still performs only one status check.

When model conversion fails, failure finalization marks the Draft Release as `failed` and disables **Kaggle CD - scheduled worker**, stopping future scheduled polls. Pushing a new Git tag automatically re-enables the worker before requesting its immediate queue tick. A Release-publication failure does not disable the worker because model conversion already succeeded.

To retry the same failed tag after a conversion failure, first enable **Kaggle CD - scheduled worker** from the Actions page, then run it with `retry_tag` set to the exact Git tag. The failed draft Release is returned to the queue. A previously acknowledged completed Kaggle version is reused; otherwise a new version is submitted. Transient API failures while a running version is active leave it recoverable for the next scheduled run.

The Kaggle kernel and RKNN target are immutable after a queue item starts. Rotating the API token is safe when it resolves to the same Kaggle account, but changing the token owner, `KAGGLE_KERNEL_SLUG`, or `RKNN_TARGET_PLATFORM` makes an active or retried item fail with a configuration-drift message. Restore the locked configuration before retrying. Use a new Git tag when intentionally changing the destination kernel or RKNN target.

The worker never waits for training inside one Actions run. Its concurrency group does not cancel in-progress ticks, draft Releases preserve queue state between executions, and inter-job artifacts are retained for seven days. Re-running all jobs safely overwrites artifacts from the earlier attempt of the same workflow run. **Kaggle CD CI** runs the coordinator tests on Python 3.10 and 3.11 plus `actionlint` on branch pushes and pull requests.
