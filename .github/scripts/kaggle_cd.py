#!/usr/bin/env python3
"""Durable, scheduled Kaggle CD coordinator backed by draft GitHub releases."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


STATE_SCHEMA = "signlang-kaggle-cd-v1"
ACTIVE_STATES = {"starting", "running"}
NOTEBOOK_PATH = "signlang_det_kaggle_training.ipynb"
DEFAULT_KERNEL_SLUG = "signlang-det-training"
GOOGLE_ASL_COMPETITION = "asl-signs"
TARGET_KERNEL_SOURCE = "abdelrhmankaram/asl-preprocessing-7"
KAGGLE_MACHINE_SHAPE = "NvidiaTeslaT4"


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
    try:
        state = json.loads(release.get("body") or "")
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(state, dict) or state.get("schema") != STATE_SCHEMA:
        return None
    if state.get("tag") != release.get("tag_name"):
        raise RuntimeError(f"CD release {release.get('id')} has a mismatched tag in its state")
    return state


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


def inject_provenance(notebook: Mapping[str, Any], tag: str, git_sha: str, repository: str) -> dict[str, Any]:
    output = copy.deepcopy(dict(notebook))
    metadata = output.setdefault("metadata", {})
    metadata.pop("kaggle", None)
    metadata.pop("accelerator", None)
    metadata["signlang_cd"] = {
        "release_tag": tag,
        "git_sha": git_sha,
        "repository": repository,
    }
    return output


def split_asset(path: Path, max_bytes: int) -> list[Path]:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    if path.stat().st_size <= max_bytes:
        return [path]
    parts = []
    with path.open("rb") as source:
        index = 1
        while True:
            chunk = source.read(max_bytes)
            if not chunk:
                break
            part = path.with_name(f"{path.name}.part-{index:04d}")
            part.write_bytes(chunk)
            parts.append(part)
            index += 1
    return parts


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


@dataclass(frozen=True)
class Config:
    repository: str
    github_token: str
    kaggle_username: str
    kaggle_key: str
    kernel_slug: str
    kernel_private: bool
    output_part_size_mb: int

    @property
    def kernel_ref(self) -> str:
        return f"{self.kaggle_username}/{self.kernel_slug}"

    @classmethod
    def from_env(cls, *, require_kaggle: bool = True) -> "Config":
        required = ["GITHUB_REPOSITORY", "GH_TOKEN"]
        if require_kaggle:
            required.extend(["KAGGLE_USERNAME", "KAGGLE_KEY"])
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            raise RuntimeError(f"Missing required environment values: {missing}")
        part_size = int(os.environ.get("KAGGLE_OUTPUT_PART_SIZE_MB", "1900"))
        if not 1 <= part_size <= 1900:
            raise ValueError("KAGGLE_OUTPUT_PART_SIZE_MB must be between 1 and 1900")
        return cls(
            repository=os.environ["GITHUB_REPOSITORY"],
            github_token=os.environ["GH_TOKEN"],
            kaggle_username=os.environ.get("KAGGLE_USERNAME", ""),
            kaggle_key=os.environ.get("KAGGLE_KEY", ""),
            kernel_slug=os.environ.get("KAGGLE_KERNEL_SLUG", DEFAULT_KERNEL_SLUG),
            kernel_private=parse_bool(os.environ.get("KAGGLE_KERNEL_PRIVATE", "true")),
            output_part_size_mb=part_size,
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
            "name": f"{tag} (Kaggle CD queued)",
            "body": json.dumps(state, indent=2, sort_keys=True),
            "draft": True,
            "prerelease": False,
        })

    def update_state(self, release_id: int, state: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(state)
        payload["updated_at"] = isoformat()
        return self.request("PATCH", f"/releases/{release_id}", {
            "body": json.dumps(payload, indent=2, sort_keys=True),
            "name": f"{payload['tag']} (Kaggle CD {payload['state']})",
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

    @staticmethod
    def _not_found(exc: Exception) -> bool:
        return getattr(exc, "status", None) == 404 or getattr(exc, "status_code", None) == 404

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
            if self._not_found(exc):
                return None
            raise
        try:
            notebook = json.loads(response.blob.source)
        except (TypeError, json.JSONDecodeError):
            notebook = {}
        return {
            "version": int(response.metadata.current_version_number),
            "provenance": notebook.get("metadata", {}).get("signlang_cd", {}),
            "url": f"https://www.kaggle.com/code/{self.config.kernel_ref}",
        }

    def status(self) -> dict[str, str]:
        try:
            response = self.api.kernels_status(self.config.kernel_ref)
        except Exception as exc:
            if self._not_found(exc):
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
        self.api.kernels_output(self.config.kernel_ref, str(destination), force=True, quiet=False)


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
    notebook = json.loads(notebook_text)
    injected = inject_provenance(notebook, tag, git_sha, config.repository)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / NOTEBOOK_PATH).write_text(json.dumps(injected, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    metadata = build_kernel_metadata(config.kaggle_username, config.kernel_slug, config.kernel_private)
    (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def provenance_matches(latest: Optional[Mapping[str, Any]], state: Mapping[str, Any]) -> bool:
    if latest is None:
        return False
    provenance = latest.get("provenance") or {}
    return provenance.get("release_tag") == state.get("tag") and provenance.get("git_sha") == state.get("git_sha")


def mark_failed(github: GitHubClient, release: Mapping[str, Any], state: dict[str, Any], message: str) -> None:
    state.update({"state": "failed", "failure": message, "failed_at": isoformat()})
    github.update_state(int(release["id"]), state)
    print(f"Kaggle CD failed for {state['tag']}: {message}", file=sys.stderr)


def start_job(github: GitHubClient, kaggle: KaggleClient, release: Mapping[str, Any], state: dict[str, Any], config: Config) -> None:
    previous_version = state.get("kaggle_version")
    if state.get("state") == "queued":
        state.update({"state": "starting", "attempt": int(state.get("attempt", 0)) + 1, "failure": None})
        github.update_state(int(release["id"]), state)

    latest = kaggle.latest()
    current_status = kaggle.status()
    recoverable = provenance_matches(latest, state) and (
        previous_version is None or int(latest["version"]) != int(previous_version)
    )
    if recoverable:
        state.update({
            "state": "running", "kaggle_kernel": config.kernel_ref,
            "kaggle_version": latest["version"], "kaggle_url": latest["url"],
            "started_at": state.get("started_at", isoformat()), "last_polled_at": None,
        })
        github.update_state(int(release["id"]), state)
        print(f"Recovered Kaggle version {latest['version']} for {state['tag']}")
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


def create_release_assets(output_dir: Path, asset_dir: Path, state: Mapping[str, Any], config: Config) -> list[Path]:
    asset_dir.mkdir(parents=True, exist_ok=True)
    archive = asset_dir / "kaggle-output.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                bundle.add(path, arcname=str(path.relative_to(output_dir)))
    output_assets = split_asset(archive, config.output_part_size_mb * 1024 * 1024)
    if output_assets != [archive]:
        archive.unlink()
    manifest = {
        "schema": STATE_SCHEMA,
        "tag": state["tag"], "git_sha": state["git_sha"],
        "kaggle_kernel": config.kernel_ref,
        "kaggle_version": state["kaggle_version"],
        "kaggle_url": state["kaggle_url"],
        "created_at": isoformat(),
        "output_assets": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in output_assets
        ],
    }
    manifest_path = asset_dir / "kaggle-cd-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum_paths = [*output_assets, manifest_path]
    sums = asset_dir / "SHA256SUMS"
    sums.write_text("".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths), encoding="utf-8")
    return [*output_assets, manifest_path, sums]


def finalize_job(github: GitHubClient, kaggle: KaggleClient, release: Mapping[str, Any], state: dict[str, Any], config: Config) -> None:
    latest = kaggle.latest()
    if not provenance_matches(latest, state) or int(latest["version"]) != int(state["kaggle_version"]):
        raise RuntimeError("Latest Kaggle notebook provenance/version does not match the queued tag")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        output_dir, asset_dir = root / "output", root / "assets"
        kaggle.download_output(output_dir)
        assets = create_release_assets(output_dir, asset_dir, state, config)
        github.upload_assets(str(state["tag"]), assets)
    body = (
        f"Kaggle training completed for `{state['tag']}`.\n\n"
        f"- Git commit: `{state['git_sha']}`\n"
        f"- Kaggle kernel: `{config.kernel_ref}`\n"
        f"- Kaggle version: `{state['kaggle_version']}`\n"
        f"- Kaggle URL: {state['kaggle_url']}\n\n"
        "Output archives, a manifest, and SHA-256 checksums are attached to this release.\n"
    )
    github.publish(int(release["id"]), str(state["tag"]), body)
    print(f"Published GitHub release {state['tag']}")


def poll_job(github: GitHubClient, kaggle: KaggleClient, release: Mapping[str, Any], state: dict[str, Any], config: Config) -> None:
    if state.get("state") == "starting":
        start_job(github, kaggle, release, state, config)
        return
    latest = kaggle.latest()
    status = kaggle.status()
    state["last_polled_at"] = isoformat()
    if not provenance_matches(latest, state) or int(latest["version"]) != int(state["kaggle_version"]):
        if status["status"] in {"queued", "running", "new_script", "cancel_requested"}:
            state["external_status"] = status["status"]
            github.update_state(int(release["id"]), state)
            print("A different Kaggle version is active; waiting to preserve serialization")
            return
        mark_failed(github, release, state, "Latest Kaggle version no longer matches this tag")
        return
    if status["status"] in {"queued", "running", "new_script", "cancel_requested"}:
        state["kaggle_status"] = status["status"]
        github.update_state(int(release["id"]), state)
        print(f"Kaggle version {state['kaggle_version']} is {status['status']}")
    elif status["status"] == "complete":
        finalize_job(github, kaggle, release, state, config)
    elif status["status"] in {"error", "cancel_acknowledged"}:
        message = status["failure_message"] or f"Kaggle status: {status['status']}"
        mark_failed(github, release, state, message)
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
            poll_job(github, kaggle, release, state, config)
    except Exception as exc:
        # External/transient failures keep active jobs recoverable. Starting failures
        # before a Kaggle version exists are terminal and release the queue.
        if state.get("state") == "starting" and not state.get("kaggle_version"):
            mark_failed(github, release, state, f"{type(exc).__name__}: {exc}")
            return
        raise


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    enqueue_parser = commands.add_parser("enqueue")
    enqueue_parser.add_argument("--tag", required=True)
    enqueue_parser.add_argument("--sha", required=True)
    enqueue_parser.set_defaults(handler=enqueue)
    tick_parser = commands.add_parser("tick")
    tick_parser.set_defaults(handler=tick)
    retry_parser = commands.add_parser("retry")
    retry_parser.add_argument("--tag", required=True)
    retry_parser.set_defaults(handler=retry)
    return root


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
