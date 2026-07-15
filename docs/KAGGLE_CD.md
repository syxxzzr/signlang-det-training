# Kaggle Continuous Delivery

The repository submits tagged notebook revisions to one stable Kaggle kernel. Training completion is handed back manually through a locked GitHub Issue; GitHub Actions then validates the uploaded ZIP, converts the model to ONNX and RKNN, and publishes the matching Release. There is no scheduled Kaggle polling and the Kaggle notebook contains no GitHub credential.

## Setup

Create this GitHub Actions secret:

| Secret | Purpose |
|---|---|
| `KAGGLE_API_TOKEN` | API token created in Kaggle Settings; the authenticated account owns the destination kernel |

The Kaggle account must have accepted the rules for [Google - Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs/data) and be able to access [ASL-preprocessing 7](https://www.kaggle.com/code/abdelrhmankaram/asl-preprocessing-7/output).

Optional GitHub Actions variables:

| Variable | Default | Purpose |
|---|---:|---|
| `KAGGLE_KERNEL_SLUG` | `signlang-det-training` | Stable destination kernel slug under the authenticated account |
| `KAGGLE_KERNEL_PRIVATE` | `true` | Whether the destination kernel is private |
| `RKNN_TARGET_PLATFORM` | `rk3588` | RKNN target passed to RKNN Toolkit 2 |

Kaggle metadata requests an `NvidiaTeslaT4`, enables GPU execution, and attaches the required competition and notebook-output source. The workflow pins the Kaggle CLI commit that supports this machine shape. Python packages in GitHub Actions use pip's default package index. Conversion pins ONNX 1.16.1 and Protobuf 4.25.4 because RKNN Toolkit 2.3.2 depends on their legacy APIs.

## Delivery flow

Pushing a Git tag performs the following steps:

1. **Kaggle CD - register tag** registers the tagged delivery as a Draft Release and creates its matching upload Issue.
2. The upload Issue is locked immediately. GitHub still lets repository writers comment on a locked conversation; no per-user allowlist is applied.
3. **Kaggle CD - submit next tag** submits the exact tagged notebook to Kaggle and records the returned kernel, numeric version, target platform, and upload Issue in the Draft Release.
4. GitHub Actions stops. It does not poll Kaggle or keep a runner waiting.

After training completes, download the Kaggle output ZIP and submit it through the locked Issue in either form:

- attach the ZIP directly to a new Issue comment; or
- upload it as an asset of the linked Draft Release, then create a comment containing `/convert <exact asset name>`.

The filename and extension are unrestricted. GitHub's own Issue attachment size limit still applies to direct attachments; use a Draft Release asset for a larger file. The downloaded content must be a valid ZIP containing exactly the expected notebook output tree.

Every attachment and every `/convert` command is a durable queue candidate. **Kaggle CD - convert Issue uploads** scans the complete comment history and processes unhandled candidates in comment order. A failed candidate receives a terminal failure comment and the workflow continues to the next queued candidate. If no candidate succeeds, the Issue remains open and another upload or command starts a new drain. Processing markers are non-terminal, so an interrupted workflow retries rather than losing a candidate.

The upload validator accepts only GitHub-hosted attachment or Release-asset downloads, caps compressed and expanded sizes and member count, rejects traversal, links, special files, encryption, duplicates, and unexpected output files, and determines ZIP validity from content instead of filename. Conversion then:

1. validates the PT checkpoint and bounded INT8 calibration archive;
2. exports ONNX and verifies it numerically against PyTorch;
3. builds non-quantized and INT8 RKNN models and runs simulator checks against ONNX;
4. generates the model manifest and reproducible notebook-output archive;
5. uploads the seven validated Release assets and removes a raw Draft Release input asset;
6. publishes the Release, posts success, closes the upload Issue, and dispatches **submit next tag** once for the next registered delivery.

Only one tag may be `starting` or `running`. Later tags retain their own Draft Releases and locked Issues but are not submitted until the active tag publishes successfully.

Each published Release contains exactly:

- `signlang_det_encoder.pt`;
- `signlang_det_encoder.onnx`;
- `signlang_det_encoder.rknn`;
- `signlang_det_encoder.int8.rknn`;
- `model-manifest.json`;
- `signlang_det_kaggle_training.ipynb` from the tagged commit;
- `notebook-output.tar.gz` containing the remaining validated notebook outputs.

## Operations

Watch the linked Kaggle page manually. A Kaggle error, timeout, or cancellation cannot notify GitHub in this design; the Draft Release and Issue remain open for inspection.

If Kaggle submission itself fails, run **Kaggle CD - submit next tag** from the default branch with `retry_tag` set to the failed tag. It requeues only a Draft Release already marked `failed`. A running delivery is never polled by this workflow.

GitHub may expose a Draft Release with a generated `untagged-<20 hex>` placeholder even though the queue body still records its registered Git tag. The coordinator ignores that placeholder and verifies the registered tag against the locked commit before uploading; other Release tag edits continue to repair stale queue metadata.

If an uploaded ZIP fails validation or conversion, inspect the bot's failure comment and add a corrected attachment or `/convert` command. Do not edit an old candidate: new comments provide stable FIFO identity and trigger the workflow. The Issue closes only after Release publication succeeds.

The conversion workflow may run for up to six hours while draining the candidates visible to it. Workflow concurrency prevents simultaneous drainers for the same Issue. Comments created during an active run trigger one later rescan, while the durable Issue history ensures candidates are not lost if GitHub replaces a pending concurrency run.

**Kaggle CD CI** compiles the coordinator and runs repository tests on Python 3.10 and 3.11, then checks all workflows with `actionlint` on branch pushes and pull requests.
