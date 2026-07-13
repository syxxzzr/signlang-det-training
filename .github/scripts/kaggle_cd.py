#!/usr/bin/env python3
"""Durable, scheduled Kaggle CD coordinator backed by draft GitHub releases."""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
import gzip
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


STATE_SCHEMA = "signlang-kaggle-cd-v1"
HANDOFF_SCHEMA = "signlang-kaggle-cd-handoff-v1"
ACTIVE_STATES = {"starting", "running"}
NOTEBOOK_PATH = "signlang_det_kaggle_training.ipynb"
NOTEBOOK_OUTPUT_ROOT = "signlang-det"
DEFAULT_KERNEL_SLUG = "signlang-det-training"
GOOGLE_ASL_COMPETITION = "asl-signs"
TARGET_KERNEL_SOURCE = "abdelrhmankaram/asl-preprocessing-7"
KAGGLE_MACHINE_SHAPE = "NvidiaTeslaT4"
STATE_MARKER = "signlang-kaggle-cd-state:"
NOTEBOOK_OUTPUT_FILES = (
    "signlang_det_encoder.pt",
    "int8_calibration.tar.gz",
    "figures/training_curves.png",
    "figures/retrieval_summary.png",
    "representation_training/metrics.csv",
    "domain_adaptation/metrics.csv",
    "representation_training/train.log",
    "domain_adaptation/train.log",
)
RELEASE_MODEL_FILES = (
    "signlang_det_encoder.pt",
    "signlang_det_encoder.onnx",
    "signlang_det_encoder.rknn",
    "signlang_det_encoder.int8.rknn",
)
RELEASE_ASSET_FILES = (
    *RELEASE_MODEL_FILES,
    NOTEBOOK_PATH,
    "notebook-output.tar.gz",
)


class TerminalDeliveryError(RuntimeError):
    """A deterministic delivery failure that requires an explicit retry."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: Optional[datetime] = None) -> str:
    return (value or utcnow()).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, received {value!r}")


def parse_release_state(release: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    if not release.get("draft"):
        return None
    body = release.get("body") or ""
    try:
        state = json.loads(body)
    except (TypeError, json.JSONDecodeError):
        start = body.find(f"<!-- {STATE_MARKER}")
        end = body.find(" -->", start)
        if start < 0 or end < 0:
            return None
        encoded = body[start + len(f"<!-- {STATE_MARKER}"):end].strip()
        try:
            decoded = base64.urlsafe_b64decode(encoded.encode()).decode()
            state = json.loads(decoded)
        except (UnicodeDecodeError, binascii.Error, json.JSONDecodeError, ValueError):
            return None
    if not isinstance(state, dict) or state.get("schema") != STATE_SCHEMA:
        return None
    release_tag = release.get("tag_name")
    if not isinstance(release_tag, str) or not release_tag:
        raise RuntimeError(f"CD release {release.get('id')} has no tag_name")
    if state.get("tag") != release_tag:
        previous_tag = state.get("tag")
        print(
            f"::warning::Repairing CD release {release.get('id')} tag mismatch: "
            f"state={previous_tag!r}, release={release_tag!r}",
            file=sys.stderr,
        )
        state["tag_recovered_from"] = previous_tag
        state["tag"] = release_tag
    return state


def render_release_body(state: Mapping[str, Any]) -> str:
    status = str(state.get("state", "unknown"))
    messages = {
        "queued": "⏳ Waiting in the delivery queue.",
        "starting": "🚀 Preparing the tagged notebook for Kaggle.",
        "running": "🏃 Training is running on Kaggle.",
        "failed": "❌ Kaggle delivery stopped before completion.",
    }
    rows = [
        f"| Status | **{html.escape(status.title())}** |",
        f"| Tag | `{html.escape(str(state.get('tag', '—')))}` |",
        f"| Commit | `{html.escape(str(state.get('git_sha', '—'))[:12])}` |",
        f"| Attempt | {int(state.get('attempt', 0))} |",
    ]
    if state.get("kaggle_version") is not None:
        rows.append(f"| Kaggle version | `{int(state['kaggle_version'])}` |")
    if state.get("kaggle_url"):
        url = html.escape(str(state["kaggle_url"]), quote=True)
        rows.append(f"| Notebook | [Open on Kaggle]({url}) |")

    sections = [
        "## Kaggle training delivery",
        "",
        messages.get(status, "Kaggle delivery status is being updated."),
        "",
        "| | |",
        "|---|---|",
        *rows,
    ]
    if status == "failed":
        failure = html.escape(str(state.get("failure") or "No failure detail was reported."))
        sections.extend([
            "", "### Failure", "",
            "> " + failure.replace("\n", "\n> "),
            "", "Run **Kaggle CD - scheduled worker** with `retry_tag` set to this tag after resolving the problem.",
        ])

    encoded = base64.urlsafe_b64encode(
        json.dumps(dict(state), separators=(",", ":"), sort_keys=True).encode()
    ).decode()
    sections.extend(["", "_This draft is maintained automatically by Kaggle CD._", "", f"<!-- {STATE_MARKER}{encoded} -->"])
    return "\n".join(sections) + "\n"


def release_name(state: Mapping[str, Any]) -> str:
    return f"{state['tag']} · Kaggle training · {str(state['state']).title()}"


def cd_releases(releases: Iterable[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], dict[str, Any]]]:
    parsed = []
    for release in releases:
        state = parse_release_state(release)
        if state is not None:
            parsed.append((release, state))
    return parsed


def select_work(releases: Sequence[Mapping[str, Any]]) -> tuple[Optional[Mapping[str, Any]], str]:
    parsed = cd_releases(releases)
    active = [(release, state) for release, state in parsed if state.get("state") in ACTIVE_STATES]
    if len(active) > 1:
        tags = [state.get("tag") for _, state in active]
        raise RuntimeError(f"CD queue invariant violated: multiple active releases: {tags}")
    if active:
        return active[0][0], "poll"
    queued = [(release, state) for release, state in parsed if state.get("state") == "queued"]
    if not queued:
        return None, "idle"
    queued.sort(key=lambda item: (item[1].get("queued_at", item[0].get("created_at", "")), item[0]["id"]))
    return queued[0][0], "start"


def build_kernel_metadata(username: str, slug: str, private: bool) -> dict[str, Any]:
    return {
        "id": f"{username}/{slug}",
        "title": " ".join(part.capitalize() for part in slug.split("-")),
        "code_file": NOTEBOOK_PATH,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": private,
        "enable_gpu": True,
        "enable_tpu": False,
        "machine_shape": KAGGLE_MACHINE_SHAPE,
        "enable_internet": False,
        "dataset_sources": [],
        "competition_sources": [GOOGLE_ASL_COMPETITION],
        "kernel_sources": [TARGET_KERNEL_SOURCE],
        "model_sources": [],
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: Sequence[str], *, cwd: Optional[Path] = None, input_text: Optional[str] = None) -> str:
    print("+", " ".join(command), flush=True)
    result = subprocess.run(
        list(command), cwd=cwd, input=input_text, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.returncode:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")
    return result.stdout


@dataclass
class Config:
    repository: str
    github_token: str
    kaggle_username: str
    kernel_slug: str
    kernel_private: bool
    rknn_target_platform: str

    @property
    def kernel_ref(self) -> str:
        return f"{self.kaggle_username}/{self.kernel_slug}"

    @classmethod
    def from_env(cls, *, require_kaggle: bool = True) -> "Config":
        required = ["GITHUB_REPOSITORY", "GH_TOKEN"]
        if require_kaggle:
            required.append("KAGGLE_API_TOKEN")
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise RuntimeError(f"Missing required environment values: {missing}")
        target_platform = os.environ.get("RKNN_TARGET_PLATFORM", "rk3588").strip()
        if not target_platform:
            raise ValueError("RKNN_TARGET_PLATFORM must not be empty")
        return cls(
            repository=os.environ["GITHUB_REPOSITORY"],
            github_token=os.environ["GH_TOKEN"],
            kaggle_username="",
            kernel_slug=os.environ.get("KAGGLE_KERNEL_SLUG", DEFAULT_KERNEL_SLUG),
            kernel_private=parse_bool(os.environ.get("KAGGLE_KERNEL_PRIVATE", "true")),
            rknn_target_platform=target_platform,
        )


class GitHubClient:
    def __init__(self, config: Config):
        self.config = config
        self.api_root = f"https://api.github.com/repos/{config.repository}"

    def request(self, method: str, path: str, payload: Optional[Mapping[str, Any]] = None) -> Any:
        url = path if path.startswith("https://") else self.api_root + path
        data = None if payload is None else json.dumps(dict(payload)).encode()
        request = urllib.request.Request(url, data=data, method=method)
        request.add_header("Accept", "application/vnd.github+json")
        request.add_header("Authorization", f"Bearer {self.config.github_token}")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read()
                return json.loads(body) if body else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(f"GitHub API {method} {url} failed: HTTP {exc.code}: {detail}") from exc

    def list_releases(self) -> list[dict[str, Any]]:
        releases, page = [], 1
        while True:
            batch = self.request("GET", f"/releases?per_page=100&page={page}")
            releases.extend(batch)
            if len(batch) < 100:
                return releases
            page += 1

    def get_release(self, release_id: int) -> dict[str, Any]:
        return self.request("GET", f"/releases/{release_id}")

    def create_queue_release(self, tag: str, git_sha: str) -> dict[str, Any]:
        if any(release.get("tag_name") == tag for release in self.list_releases()):
            raise RuntimeError(f"A GitHub release already exists for tag {tag}")
        state = {
            "schema": STATE_SCHEMA,
            "state": "queued",
            "tag": tag,
            "git_sha": git_sha,
            "queued_at": isoformat(),
            "updated_at": isoformat(),
            "attempt": 0,
        }
        return self.request("POST", "/releases", {
            "tag_name": tag,
            "target_commitish": git_sha,
            "name": release_name(state),
            "body": render_release_body(state),
            "draft": True,
            "prerelease": False,
        })

    def update_state(self, release_id: int, state: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(state)
        payload["updated_at"] = isoformat()
        return self.request("PATCH", f"/releases/{release_id}", {
            "body": render_release_body(payload),
            "name": release_name(payload),
        })

    def publish(self, release_id: int, tag: str, body: str) -> dict[str, Any]:
        return self.request("PATCH", f"/releases/{release_id}", {
            "name": tag,
            "body": body,
            "draft": False,
            "prerelease": False,
        })

    def upload_assets(self, tag: str, paths: Sequence[Path]) -> None:
        run(["gh", "release", "upload", tag, *[str(path) for path in paths], "--clobber"])


class KaggleClient:
    def __init__(self, config: Config):
        from kaggle.api.kaggle_api_extended import KaggleApi
        self.config = config
        self.api = KaggleApi()
        self.api.authenticate()
        username = self.api.config_values.get(self.api.CONFIG_NAME_USER)
        if not username:
            raise RuntimeError("Kaggle API token authentication did not resolve an account username")
        self.config.kaggle_username = str(username)

    @staticmethod
    def _kernel_missing(exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        status = (
            getattr(exc, "status", None)
            or getattr(exc, "status_code", None)
            or getattr(response, "status_code", None)
        )
        # GetKernel hides a nonexistent kernel behind 403 for some access-token
        # requests, and newer clients wrap that response in a ValueError without
        # retaining the status. The owner is always the authenticated token user,
        # so this cannot mask an attempt to read another account's private kernel.
        permission_denied = (
            isinstance(exc, ValueError)
            and "Permission 'kernels.get' was denied" in str(exc)
        )
        return status in {403, 404} or permission_denied

    def latest(self) -> Optional[dict[str, Any]]:
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
        owner, slug = self.config.kernel_ref.split("/", 1)
        request = ApiGetKernelRequest()
        request.user_name = owner
        request.kernel_slug = slug
        try:
            with self.api.build_kaggle_client() as client:
                response = client.kernels.kernels_api_client.get_kernel(request)
        except Exception as exc:
            if self._kernel_missing(exc):
                return None
            raise
        return {
            "version": int(response.metadata.current_version_number),
            "url": f"https://www.kaggle.com/code/{self.config.kernel_ref}",
        }

    def status(self) -> dict[str, str]:
        try:
            response = self.api.kernels_status(self.config.kernel_ref)
        except Exception as exc:
            if self._kernel_missing(exc):
                return {"status": "missing", "failure_message": ""}
            raise
        return {
            "status": response.status.name.lower(),
            "failure_message": response.failure_message or "",
        }

    def push(self, folder: Path) -> dict[str, Any]:
        result = self.api.kernels_push(str(folder))
        if result is None or result.error:
            raise RuntimeError(f"Kaggle kernel push failed: {getattr(result, 'error', 'no response')}")
        invalid = {
            "datasets": list(result.invalidDatasetSources or []),
            "competitions": list(result.invalidCompetitionSources or []),
            "kernels": list(result.invalidKernelSources or []),
        }
        if any(invalid.values()):
            raise RuntimeError(f"Kaggle rejected configured data sources: {invalid}")
        return {"version": int(result.versionNumber), "url": result.url}

    def download_output(self, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        output_root = re.escape(NOTEBOOK_OUTPUT_ROOT)
        expected_files = "|".join(map(re.escape, NOTEBOOK_OUTPUT_FILES))
        file_pattern = rf"^(?:{output_root}/)?(?:{expected_files})$"
        self.api.kernels_output(
            self.config.kernel_ref,
            str(destination),
            file_pattern=file_pattern,
            force=True,
            quiet=False,
        )
        kernel_log = destination / f"{self.config.kernel_slug}.log"
        if kernel_log.is_file() or kernel_log.is_symlink():
            kernel_log.unlink()


def state_for_release(release: Mapping[str, Any]) -> dict[str, Any]:
    state = parse_release_state(release)
    if state is None:
        raise RuntimeError(f"Release {release.get('id')} is not a Kaggle CD queue item")
    return state


def stage_upload(state: Mapping[str, Any], config: Config, folder: Path) -> None:
    tag, git_sha = str(state["tag"]), str(state["git_sha"])
    resolved = run(["git", "rev-parse", f"{tag}^{{commit}}"])
    if resolved.strip() != git_sha:
        raise RuntimeError(f"Tag {tag} resolves to {resolved.strip()}, expected {git_sha}")
    notebook_text = subprocess.check_output(["git", "show", f"{git_sha}:{NOTEBOOK_PATH}"], text=True)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / NOTEBOOK_PATH).write_text(notebook_text, encoding="utf-8")
    metadata = build_kernel_metadata(config.kaggle_username, config.kernel_slug, config.kernel_private)
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def latest_version(latest: Optional[Mapping[str, Any]]) -> Optional[int]:
    if latest is None or latest.get("version") is None:
        return None
    return int(latest["version"])


def version_matches(latest: Optional[Mapping[str, Any]], state: Mapping[str, Any]) -> bool:
    expected = state.get("kaggle_version")
    return expected is not None and latest_version(latest) == int(expected)


def version_mismatch_message(
    latest: Optional[Mapping[str, Any]], state: Mapping[str, Any]
) -> str:
    return (
        f"Kaggle latest version {latest_version(latest)!r} does not match "
        f"expected version {state.get('kaggle_version')!r} for tag {state.get('tag')!r}"
    )


def mark_failed(github: GitHubClient, release: Mapping[str, Any], state: dict[str, Any], message: str) -> None:
    state.update({"state": "failed", "failure": message, "failed_at": isoformat()})
    github.update_state(int(release["id"]), state)
    print(f"Kaggle CD failed for {state['tag']}: {message}", file=sys.stderr)


def fail_job(
    github: GitHubClient,
    release: Mapping[str, Any],
    state: dict[str, Any],
    message: str,
) -> None:
    mark_failed(github, release, state, message)
    raise RuntimeError(message)


def start_job(github: GitHubClient, kaggle: KaggleClient, release: Mapping[str, Any], state: dict[str, Any], config: Config) -> None:
    previous_version = state.get("kaggle_version")
    if state.get("state") == "queued":
        state.update({"state": "starting", "attempt": int(state.get("attempt", 0)) + 1, "failure": None})
        github.update_state(int(release["id"]), state)
    state.pop("kaggle_version_before_push", None)

    latest = kaggle.latest()
    current_status = kaggle.status()
    current_version = latest_version(latest)
    previous_completed = (
        previous_version is not None
        and current_version == int(previous_version)
        and current_status["status"] == "complete"
    )
    if previous_completed:
        state.update({
            "state": "running", "kaggle_kernel": config.kernel_ref,
            "kaggle_version": current_version, "kaggle_url": latest["url"],
            "started_at": state.get("started_at", isoformat()), "last_polled_at": None,
        })
        github.update_state(int(release["id"]), state)
        print(f"Recovered Kaggle version {current_version} for {state['tag']}")
        return
    if current_status["status"] in {"queued", "running", "new_script", "cancel_requested"}:
        state.update({
            "waiting_for_external_run": True,
            "last_polled_at": isoformat(),
            "external_status": current_status["status"],
        })
        github.update_state(int(release["id"]), state)
        print(f"Waiting for an existing Kaggle run with status {current_status['status']}")
        return

    with tempfile.TemporaryDirectory() as directory:
        upload_dir = Path(directory) / "upload"
        stage_upload(state, config, upload_dir)
        pushed = kaggle.push(upload_dir)
    state.update({
        "state": "running", "waiting_for_external_run": False,
        "kaggle_kernel": config.kernel_ref, "kaggle_version": pushed["version"],
        "kaggle_url": pushed["url"], "started_at": isoformat(), "last_polled_at": None,
    })
    github.update_state(int(release["id"]), state)
    print(f"Pushed {state['tag']} as Kaggle version {pushed['version']}")


def validate_notebook_outputs(output_root: Path) -> None:
    expected = {Path(relative) for relative in NOTEBOOK_OUTPUT_FILES}
    actual = {path.relative_to(output_root) for path in output_root.rglob("*") if path.is_file()}
    if actual != expected:
        missing = sorted(str(path) for path in expected - actual)
        unexpected = sorted(str(path) for path in actual - expected)
        raise TerminalDeliveryError(
            f"Kaggle notebook output allowlist mismatch: missing={missing}, unexpected={unexpected}"
        )


def find_notebook_output_root(download_dir: Path) -> Path:
    matches = [path.parent for path in download_dir.rglob("signlang_det_encoder.pt")]
    if len(matches) != 1:
        raise TerminalDeliveryError(f"Expected one notebook output root, found {len(matches)}")
    validate_notebook_outputs(matches[0])
    return matches[0]


def add_reproducible_file(bundle: tarfile.TarFile, path: Path, arcname: str) -> None:
    info = bundle.gettarinfo(str(path), arcname=arcname)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    with path.open("rb") as handle:
        bundle.addfile(info, handle)


def create_notebook_output_archive(output_root: Path, archive: Path) -> None:
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as bundle:
                for relative in NOTEBOOK_OUTPUT_FILES:
                    if relative != "signlang_det_encoder.pt":
                        add_reproducible_file(bundle, output_root / relative, relative)


def installed_version(distribution: str) -> str:
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return "unknown"


def create_model_manifest(
    pt_path: Path,
    asset_dir: Path,
    state: Mapping[str, Any],
    config: Config,
    converter_sha256: str,
) -> Path:
    import torch

    payload = torch.load(pt_path, map_location="cpu", weights_only=True)
    required = {
        "preprocessing",
        "encoder_fingerprint",
        "model_config",
        "input_contract",
        "output_contract",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise RuntimeError(f"PT export is missing model manifest fields: {missing}")
    quantization = {
        "signlang_det_encoder.pt": "none",
        "signlang_det_encoder.onnx": "none",
        "signlang_det_encoder.rknn": "none",
        "signlang_det_encoder.int8.rknn": "int8",
    }
    feature_contract = copy.deepcopy(payload["input_contract"])
    output_contract = copy.deepcopy(payload["output_contract"])
    fixed_features = copy.deepcopy(feature_contract)
    fixed_output = copy.deepcopy(output_contract)
    fixed_features["shape"][0] = 1
    fixed_output["shape"][0] = 1
    pt_io = {
        "inputs": {
            "features": feature_contract,
            "lengths": {"dtype": "int32", "shape": ["B"], "minimum": 1, "maximum": 64},
        },
        "outputs": {"frame_embeddings": output_contract},
    }
    fixed_io = {
        "inputs": {
            "features": fixed_features,
            "lengths": {"dtype": "int32", "shape": [1], "minimum": 1, "maximum": 64},
        },
        "outputs": {"frame_embeddings": fixed_output},
    }
    int8_io = copy.deepcopy(fixed_io)
    int8_io["inputs"]["features"]["dtype"] = "int8"
    int8_io["inputs"]["features"]["quantization_parameters"] = (
        "embedded; query scale and zero_point with RKNN Runtime"
    )
    int8_io["outputs"]["frame_embeddings"]["dtype"] = "int8"
    int8_io["outputs"]["frame_embeddings"]["quantization_parameters"] = (
        "embedded; query scale and zero_point with RKNN Runtime"
    )
    io_contracts = {
        "signlang_det_encoder.pt": pt_io,
        "signlang_det_encoder.onnx": fixed_io,
        "signlang_det_encoder.rknn": fixed_io,
        "signlang_det_encoder.int8.rknn": int8_io,
    }
    models = {}
    for name in RELEASE_MODEL_FILES:
        path = asset_dir / name
        if not path.is_file():
            raise RuntimeError(f"Cannot create model manifest; missing {name}")
        models[name] = {
            "format": path.suffix.removeprefix("."),
            "quantization": quantization[name],
            "io_contract": io_contracts[name],
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema": "signlang-model-release-v1",
        "model_identity": {
            "name": "signlang_det_encoder",
            "version": state["tag"],
            "task": "frame_level_sign_language_embedding",
        },
        "tag": state["tag"],
        "git_sha": state["git_sha"],
        "kaggle_version": state["kaggle_version"],
        "kaggle_url": state["kaggle_url"],
        "rknn_target_platform": config.rknn_target_platform,
        "preprocessing": payload["preprocessing"],
        "encoder_fingerprint": payload["encoder_fingerprint"],
        "architecture": {
            "name": "HandEncoder",
            "implementation": ".github/scripts/convert_models.py",
            "implementation_git_sha": state["git_sha"],
            "implementation_sha256": converter_sha256,
            "config": payload["model_config"],
        },
        "artifacts": {
            "pytorch_weights": "signlang_det_encoder.pt",
            "onnx_graph": "signlang_det_encoder.onnx",
            "rknn_graph": "signlang_det_encoder.rknn",
            "rknn_int8_graph": "signlang_det_encoder.int8.rknn",
            "tokenizer": None,
        },
        "tokenizer": {
            "required": False,
            "location": None,
            "reason": "The encoder consumes preprocessed hand landmark tensors, not tokens.",
        },
        "runtime_dependencies": {
            "pytorch": ["python==3.10.*", "numpy==1.26.4", "torch==2.4.0"],
            "onnx": ["python==3.10.*", "numpy==1.26.4", "onnxruntime==1.23.2"],
            "rknn": [
                "rknn-toolkit-lite2==2.3.2 or RKNN Runtime==2.3.2",
                f"target_platform=={config.rknn_target_platform}",
            ],
        },
        "input_contract": payload["input_contract"],
        "output_contract": payload["output_contract"],
        "conversion_environment": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "onnx": installed_version("onnx"),
            "onnxruntime": installed_version("onnxruntime"),
            "rknn_toolkit2": installed_version("rknn-toolkit2"),
            "setuptools": installed_version("setuptools"),
        },
        "models": models,
        "integrity": {
            "algorithm": "sha256",
            "files": {
                name: sha256_file(asset_dir / name) for name in RELEASE_ASSET_FILES
            },
        },
    }
    path = asset_dir / "model-manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def create_release_assets(
    output_dir: Path,
    asset_dir: Path,
    state: Mapping[str, Any],
    config: Config,
) -> list[Path]:
    output_root = find_notebook_output_root(output_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    pt_asset = asset_dir / "signlang_det_encoder.pt"
    shutil.copy2(output_root / pt_asset.name, pt_asset)
    create_notebook_output_archive(output_root, asset_dir / "notebook-output.tar.gz")

    notebook_text = subprocess.check_output(
        ["git", "show", f"{state['git_sha']}:{NOTEBOOK_PATH}"], text=True
    )
    notebook_asset = asset_dir / NOTEBOOK_PATH
    notebook_asset.write_text(notebook_text, encoding="utf-8")

    converter_source = subprocess.check_output(
        ["git", "show", f"{state['git_sha']}:.github/scripts/convert_models.py"]
    )
    current_converter = Path(__file__).with_name("convert_models.py")
    current_converter_sha256 = sha256_file(current_converter)
    run([
        sys.executable,
        str(current_converter),
        "--pt", str(pt_asset),
        "--calibration", str(output_root / "int8_calibration.tar.gz"),
        "--output-dir", str(asset_dir),
        "--target-platform", config.rknn_target_platform,
        "--validate-only",
    ])
    with tempfile.TemporaryDirectory() as directory:
        converter = Path(directory) / "convert_models.py"
        converter.write_bytes(converter_source)
        converter_sha256 = sha256_file(converter)
        run([
            sys.executable,
            str(converter),
            "--pt", str(pt_asset),
            "--calibration", str(output_root / "int8_calibration.tar.gz"),
            "--output-dir", str(asset_dir),
            "--target-platform", config.rknn_target_platform,
        ])
    if converter_sha256 != current_converter_sha256:
        run([
            sys.executable,
            str(current_converter),
            "--pt", str(pt_asset),
            "--calibration", str(output_root / "int8_calibration.tar.gz"),
            "--output-dir", str(asset_dir),
            "--target-platform", config.rknn_target_platform,
            "--verify-rknn-only",
        ])

    manifest_asset = create_model_manifest(
        pt_asset, asset_dir, state, config, converter_sha256
    )

    assets = [
        *[asset_dir / name for name in RELEASE_MODEL_FILES],
        manifest_asset,
        notebook_asset,
        asset_dir / "notebook-output.tar.gz",
    ]
    missing = [path.name for path in assets if not path.is_file()]
    if missing:
        raise RuntimeError(f"Model conversion did not create required Release assets: {missing}")
    return assets


def prepare_handoff(
    kaggle: KaggleClient,
    release: Mapping[str, Any],
    state: Mapping[str, Any],
    config: Config,
    handoff_dir: Path,
) -> None:
    latest = kaggle.latest()
    if not version_matches(latest, state):
        raise TerminalDeliveryError(version_mismatch_message(latest, state))
    if handoff_dir.exists():
        shutil.rmtree(handoff_dir)
    output_dir = handoff_dir / "output"
    kaggle.download_output(output_dir)
    downloaded_version = kaggle.latest()
    if not version_matches(downloaded_version, state):
        shutil.rmtree(handoff_dir)
        raise TerminalDeliveryError(
            "Kaggle version changed while downloading output: "
            + version_mismatch_message(downloaded_version, state)
        )
    find_notebook_output_root(output_dir)
    metadata = {
        "schema": HANDOFF_SCHEMA,
        "release_id": int(release["id"]),
        "repository": config.repository,
        "kaggle_kernel": config.kernel_ref,
        "rknn_target_platform": config.rknn_target_platform,
        "state": dict(state),
    }
    (handoff_dir / "release-state.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Prepared Kaggle output handoff for {state['tag']}")


def load_handoff(handoff_dir: Path) -> dict[str, Any]:
    path = handoff_dir / "release-state.json"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read workflow handoff metadata from {path}: {exc}") from exc
    if not isinstance(metadata, dict) or metadata.get("schema") != HANDOFF_SCHEMA:
        raise RuntimeError(f"Unsupported workflow handoff metadata in {path}")
    state = metadata.get("state")
    if not isinstance(state, dict) or state.get("state") != "running":
        raise RuntimeError("Workflow handoff does not describe a running CD release")
    return metadata


def convert_handoff(args: argparse.Namespace) -> None:
    metadata = load_handoff(args.handoff_dir)
    state = metadata["state"]
    kernel = str(metadata["kaggle_kernel"])
    if "/" not in kernel:
        raise RuntimeError(f"Invalid Kaggle kernel reference in workflow handoff: {kernel!r}")
    username, slug = kernel.split("/", 1)
    config = Config(
        repository=str(metadata["repository"]),
        github_token="",
        kaggle_username=username,
        kernel_slug=slug,
        kernel_private=True,
        rknn_target_platform=str(metadata["rknn_target_platform"]),
    )
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    assets = create_release_assets(
        args.handoff_dir / "output", args.output_dir / "assets", state, config
    )
    shutil.copy2(args.handoff_dir / "release-state.json", args.output_dir / "release-state.json")
    print(f"Prepared {len(assets)} Release assets for {state['tag']}")


def validate_release_assets(asset_dir: Path, metadata: Mapping[str, Any]) -> list[Path]:
    expected_names = {*RELEASE_ASSET_FILES, "model-manifest.json"}
    actual_names = {
        str(path.relative_to(asset_dir)) for path in asset_dir.rglob("*") if path.is_file()
    }
    if actual_names != expected_names:
        raise RuntimeError(
            "Converted Release asset set is invalid: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    manifest_path = asset_dir / "model-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read model manifest: {exc}") from exc
    state = metadata["state"]
    expected_manifest = {
        "schema": "signlang-model-release-v1",
        "tag": state["tag"],
        "git_sha": state["git_sha"],
        "kaggle_version": state["kaggle_version"],
        "kaggle_url": state["kaggle_url"],
        "rknn_target_platform": metadata["rknn_target_platform"],
    }
    mismatched = [
        key for key, value in expected_manifest.items() if manifest.get(key) != value
    ]
    if mismatched:
        raise RuntimeError(f"Model manifest handoff mismatch: {mismatched}")
    integrity = manifest.get("integrity") or {}
    if integrity.get("algorithm") != "sha256":
        raise RuntimeError("Model manifest does not use SHA-256 integrity metadata")
    recorded_hashes = integrity.get("files") or {}
    for name in RELEASE_ASSET_FILES:
        actual_hash = sha256_file(asset_dir / name)
        if recorded_hashes.get(name) != actual_hash:
            raise RuntimeError(f"Release asset SHA-256 mismatch: {name}")
    models = manifest.get("models") or {}
    for name in RELEASE_MODEL_FILES:
        model = models.get(name) or {}
        path = asset_dir / name
        if model.get("bytes") != path.stat().st_size:
            raise RuntimeError(f"Release model size mismatch: {name}")
        if model.get("sha256") != sha256_file(path):
            raise RuntimeError(f"Release model manifest hash mismatch: {name}")
    return [
        *[asset_dir / name for name in RELEASE_MODEL_FILES],
        manifest_path,
        asset_dir / NOTEBOOK_PATH,
        asset_dir / "notebook-output.tar.gz",
    ]


def publish_handoff(args: argparse.Namespace) -> None:
    metadata = load_handoff(args.handoff_dir)
    state = metadata["state"]
    config = Config.from_env(require_kaggle=False)
    if metadata.get("repository") != config.repository:
        raise RuntimeError("Workflow handoff repository does not match GITHUB_REPOSITORY")
    github = GitHubClient(config)
    release = github.get_release(int(metadata["release_id"]))
    current = state_for_release(release)
    immutable_keys = ("tag", "git_sha", "kaggle_version", "kaggle_url")
    mismatched = [key for key in immutable_keys if current.get(key) != state.get(key)]
    if mismatched or current.get("state") != "running":
        raise RuntimeError(
            f"Draft Release state changed after Kaggle handoff; mismatched fields: {mismatched}"
        )
    asset_dir = args.handoff_dir / "assets"
    assets = validate_release_assets(asset_dir, metadata)
    github.upload_assets(str(state["tag"]), assets)
    body = (
        f"Kaggle training completed for `{state['tag']}`.\n\n"
        f"- Git commit: `{state['git_sha']}`\n"
        f"- Kaggle kernel: `{metadata['kaggle_kernel']}`\n"
        f"- Kaggle version: `{state['kaggle_version']}`\n"
        f"- Kaggle URL: {state['kaggle_url']}\n"
        f"- RKNN target platform: `{metadata['rknn_target_platform']}`\n\n"
        "PT, ONNX, non-quantized RKNN, INT8 RKNN, the model manifest, the tagged notebook, "
        "and remaining notebook outputs are attached as seven Release assets.\n"
    )
    github.publish(int(release["id"]), str(state["tag"]), body)
    print(f"Published GitHub release {state['tag']}")


def fail_handoff(args: argparse.Namespace) -> None:
    metadata = load_handoff(args.handoff_dir)
    state = metadata["state"]
    config = Config.from_env(require_kaggle=False)
    if metadata.get("repository") != config.repository:
        raise RuntimeError("Workflow handoff repository does not match GITHUB_REPOSITORY")
    github = GitHubClient(config)
    release = github.get_release(int(metadata["release_id"]))
    if not release.get("draft"):
        print(f"Release {state['tag']} is already published; no failure state is needed")
        return
    current = state_for_release(release)
    if current.get("state") == "failed":
        print(f"Release {state['tag']} is already failed")
        return
    immutable_keys = ("tag", "git_sha", "kaggle_version", "kaggle_url")
    mismatched = [key for key in immutable_keys if current.get(key) != state.get(key)]
    if mismatched or current.get("state") != "running":
        raise RuntimeError(
            f"Draft Release state changed before failure finalization: {mismatched}"
        )
    detail = f"{args.phase} failed"
    if args.run_url:
        detail += f"; inspect {args.run_url}"
    mark_failed(github, release, current, detail)


def poll_job(
    github: GitHubClient,
    kaggle: KaggleClient,
    release: Mapping[str, Any],
    state: dict[str, Any],
    config: Config,
    handoff_dir: Path,
) -> None:
    if state.get("state") == "starting":
        start_job(github, kaggle, release, state, config)
        return
    latest = kaggle.latest()
    status = kaggle.status()
    state["last_polled_at"] = isoformat()
    if not version_matches(latest, state):
        message = version_mismatch_message(latest, state)
        if status["status"] in {"queued", "running", "new_script", "cancel_requested"}:
            state["external_status"] = status["status"]
            github.update_state(int(release["id"]), state)
            print(f"{message}; waiting for the active version to finish")
            return
        fail_job(github, release, state, message)
    if status["status"] in {"queued", "running", "new_script", "cancel_requested"}:
        state["kaggle_status"] = status["status"]
        github.update_state(int(release["id"]), state)
        print(f"Kaggle version {state['kaggle_version']} is {status['status']}")
    elif status["status"] == "complete":
        try:
            prepare_handoff(kaggle, release, state, config, handoff_dir)
        except TerminalDeliveryError as exc:
            fail_job(github, release, state, str(exc))
    elif status["status"] in {"error", "cancel_acknowledged"}:
        message = status["failure_message"] or f"Kaggle status: {status['status']}"
        fail_job(github, release, state, message)
    else:
        raise RuntimeError(f"Unknown Kaggle status: {status['status']}")


def enqueue(args: argparse.Namespace) -> None:
    config = Config.from_env(require_kaggle=False)
    release = GitHubClient(config).create_queue_release(args.tag, args.sha)
    print(f"Queued {args.tag} in draft release {release['id']}")


def retry(args: argparse.Namespace) -> None:
    config = Config.from_env(require_kaggle=False)
    github = GitHubClient(config)
    matches = [release for release in github.list_releases() if release.get("tag_name") == args.tag]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one release for {args.tag}, found {len(matches)}")
    release = matches[0]
    state = state_for_release(release)
    if state.get("state") != "failed":
        raise RuntimeError(f"Only failed jobs can be retried; {args.tag} is {state.get('state')}")
    state.update({"state": "queued", "failure": None, "queued_at": isoformat(), "last_polled_at": None})
    state.pop("kaggle_version_before_push", None)
    github.update_state(int(release["id"]), state)
    print(f"Requeued {args.tag}")


def tick(args: argparse.Namespace) -> None:
    config = Config.from_env(require_kaggle=True)
    github, kaggle = GitHubClient(config), KaggleClient(config)
    release, action = select_work(github.list_releases())
    if action == "idle":
        print("Kaggle CD queue is idle")
        return
    state = state_for_release(release)
    try:
        if action == "start":
            start_job(github, kaggle, release, state, config)
        else:
            poll_job(github, kaggle, release, state, config, args.handoff_dir)
    except Exception as exc:
        # Running API failures remain recoverable; startup failures release the queue.
        if state.get("state") == "starting":
            mark_failed(github, release, state, f"{type(exc).__name__}: {exc}")
        raise


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    enqueue_parser = commands.add_parser("enqueue")
    enqueue_parser.add_argument("--tag", required=True)
    enqueue_parser.add_argument("--sha", required=True)
    enqueue_parser.set_defaults(handler=enqueue)
    tick_parser = commands.add_parser("tick")
    tick_parser.add_argument("--handoff-dir", type=Path, required=True)
    tick_parser.set_defaults(handler=tick)
    retry_parser = commands.add_parser("retry")
    retry_parser.add_argument("--tag", required=True)
    retry_parser.set_defaults(handler=retry)
    convert_parser = commands.add_parser("convert-handoff")
    convert_parser.add_argument("--handoff-dir", type=Path, required=True)
    convert_parser.add_argument("--output-dir", type=Path, required=True)
    convert_parser.set_defaults(handler=convert_handoff)
    publish_parser = commands.add_parser("publish-handoff")
    publish_parser.add_argument("--handoff-dir", type=Path, required=True)
    publish_parser.set_defaults(handler=publish_handoff)
    fail_parser = commands.add_parser("fail-handoff")
    fail_parser.add_argument("--handoff-dir", type=Path, required=True)
    fail_parser.add_argument("--phase", required=True)
    fail_parser.add_argument("--run-url")
    fail_parser.set_defaults(handler=fail_handoff)
    return root


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
