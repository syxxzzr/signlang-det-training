# Kaggle Continuous Delivery

The repository can publish a tagged notebook to one stable Kaggle kernel, wait for its training run across separate GitHub Actions executions, and attach the completed output to the matching GitHub Release.

## Setup

Create these GitHub Actions secrets:

| Secret | Purpose |
|---|---|
| `KAGGLE_USERNAME` | Owner of the destination Kaggle kernel |
| `KAGGLE_KEY` | Kaggle API token |

The Kaggle account must have accepted the rules for [Google - Isolated Sign Language Recognition](https://www.kaggle.com/competitions/asl-signs/data) and be able to access [ASL-preprocessing 7](https://www.kaggle.com/code/abdelrhmankaram/asl-preprocessing-7/output).

Optional GitHub Actions variables:

| Variable | Default | Purpose |
|---|---:|---|
| `KAGGLE_KERNEL_SLUG` | `signlang-det-training` | Stable destination kernel slug |
| `KAGGLE_POLL_INTERVAL_MINUTES` | `15` | Minimum time between Kaggle API status checks; minimum `5` |
| `KAGGLE_KERNEL_PRIVATE` | `true` | Whether the destination kernel is private |
| `KAGGLE_OUTPUT_PART_SIZE_MB` | `1900` | Maximum size of each Release archive part |

Kaggle's standard `kernel-metadata.json` sets `machine_shape` to `NvidiaTeslaT4`, enables GPU execution, and attaches the competition and notebook-output sources. The workflow pins an official Kaggle CLI release that passes this field to the kernel save request. The run will fail at Kaggle submission time if that accelerator is unavailable or not permitted for the account; it does not silently request a different GPU class.

## Delivery flow

Pushing any Git tag creates a draft Release whose JSON body is a durable FIFO queue item. One serialized worker advances a single transition on each invocation:

1. Upload the notebook from the exact tagged commit to `${KAGGLE_USERNAME}/${KAGGLE_KERNEL_SLUG}`.
2. Record the returned numeric Kaggle version in the draft Release. The Git tag is also injected into the uploaded copy as provenance, because Kaggle version names are numeric and cannot be replaced by arbitrary tag names.
3. Let the scheduled workflow wake every five minutes. It contacts Kaggle only when the configured polling interval has elapsed.
4. Download successful output, create a reproducible archive and checksums, upload them as Release assets, and publish the Release.

Only one queue item may be `starting` or `running`. Later tags remain queued. An already-running external version of the same stable kernel is allowed to finish before the queue continues.

Release assets contain `kaggle-output.tar.gz` or numbered parts, `kaggle-cd-manifest.json`, and `SHA256SUMS`. Reconstruct a split archive with:

```bash
cat kaggle-output.tar.gz.part-* > kaggle-output.tar.gz
sha256sum -c SHA256SUMS
tar -xzf kaggle-output.tar.gz
```

## Operations

Use **Kaggle CD - scheduled worker → Run workflow** to request an immediate poll. This bypasses the configured interval for that invocation but still performs only one status check.

If a job reaches `failed`, run the same workflow with `retry_tag` set to its exact Git tag. The failed draft Release is returned to the queue. Transient API failures while a Kaggle version is active leave it recoverable for the next scheduled run.

The worker never waits for training inside one Actions run. Its concurrency group does not cancel in-progress ticks, and draft Releases preserve queue state between executions.
