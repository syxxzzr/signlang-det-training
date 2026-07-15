#!/usr/bin/env python3
"""Durable Kaggle CD coordinator backed by draft Releases and locked Issues."""

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
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, NoReturn, Optional, Sequence


STATE_SCHEMA = "signlang-kaggle-cd-v1"
HANDOFF_SCHEMA = "signlang-kaggle-cd-handoff-v1"
ACTIVE_STATES = {"starting", "running"}
NOTEBOOK_PATH = "signlang_det_kaggle_training.ipynb"
DEFAULT_KERNEL_SLUG = "signlang-det-training"
GOOGLE_ASL_COMPETITION = "asl-signs"
TARGET_KERNEL_SOURCE = "abdelrhmankaram/asl-preprocessing-7"
KAGGLE_MACHINE_SHAPE = "NvidiaTeslaT4"
STATE_MARKER = "signlang-kaggle-cd-state:"
ISSUE_MARKER = "signlang-kaggle-cd-issue:"
ATTEMPT_MARKER = "signlang-kaggle-cd-attempt:"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 64
MAX_ARCHIVE_EXPANDED_BYTES = 4 * 1024 * 1024 * 1024
ISSUE_ATTACHMENT_PATTERN = re.compile(
    r"https://github\.com/user-attachments/(?:files/\d+/|assets/)[^\s)>\]]+"
)
CONVERT_COMMAND_PATTERN = re.compile(r"(?m)^[ \t]*/convert[ \t]+(.+?)[ \t]*$")
ATTEMPT_STATUS_PATTERN = re.compile(
    rf"<!-- {re.escape(ATTEMPT_MARKER)}(?P<id>\d+:\d+):(?P<status>failed|succeeded) -->"
)
GITHUB_UNTAGGED_RELEASE_PATTERN = re.compile(r"untagged-[0-9a-f]{20}", re.IGNORECASE)
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
DELIVERY_CONFIG_KEYS = ("kaggle_kernel", "rknn_target_platform")
RELEASE_IMMUTABLE_KEYS = (
    "tag",
    "git_sha",
    "kaggle_kernel",
    "kaggle_version",
    "kaggle_url",
    "rknn_target_platform",
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
        if (
            isinstance(previous_tag, str)
            and previous_tag
            and GITHUB_UNTAGGED_RELEASE_PATTERN.fullmatch(release_tag)
        ):
            print(
                f"::warning::Ignoring synthetic tag {release_tag!r} for CD release "
                f"{release.get('id')}; preserving queued tag {previous_tag!r}",
                file=sys.stderr,
            )
            return state
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
    if state.get("kaggle_kernel"):
        rows.append(f"| Kaggle kernel | `{html.escape(str(state['kaggle_kernel']))}` |")
    if state.get("kaggle_version") is not None:
        rows.append(f"| Kaggle version | `{int(state['kaggle_version'])}` |")
    if state.get("rknn_target_platform"):
        rows.append(f"| RKNN target | `{html.escape(str(state['rknn_target_platform']))}` |")
    if state.get("kaggle_url"):
        url = html.escape(str(state["kaggle_url"]), quote=True)
        rows.append(f"| Notebook | [Open on Kaggle]({url}) |")
    if state.get("issue_number") is not None:
        issue_number = int(state["issue_number"])
        rows.append(f"| Output upload | [Issue #{issue_number}](../../issues/{issue_number}) |")

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
            "", "Run **Kaggle CD - submit next tag** with `retry_tag` set to this tag after resolving the problem.",
        ])

    encoded = base64.urlsafe_b64encode(
        json.dumps(dict(state), separators=(",", ":"), sort_keys=True).encode()
    ).decode()
    sections.extend(["", "_This draft is maintained automatically by Kaggle CD._", "", f"<!-- {STATE_MARKER}{encoded} -->"])
    return "\n".join(sections) + "\n"


def release_name(state: Mapping[str, Any]) -> str:
    return f"{state['tag']} · Kaggle training · {str(state['state']).title()}"


def render_delivery_issue(release: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    binding = {
        "schema": STATE_SCHEMA,
        "release_id": int(release["id"]),
        "tag": str(state["tag"]),
        "git_sha": str(state["git_sha"]),
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(binding, separators=(",", ":"), sort_keys=True).encode()
    ).decode()
    tag = html.escape(str(state["tag"]))
    release_url = html.escape(str(release["html_url"]), quote=True)
    kaggle_url = state.get("kaggle_url")
    if kaggle_url:
        notebook_link_en = (
            f"[Open the submitted Kaggle Notebook]({html.escape(str(kaggle_url), quote=True)})"
        )
        notebook_link_zh = (
            f"[打开已提交的 Kaggle Notebook]({html.escape(str(kaggle_url), quote=True)})"
        )
    else:
        notebook_link_en = "Pending submission — refresh this Issue after the submit workflow finishes."
        notebook_link_zh = "等待提交——提交 workflow 完成后刷新本 Issue。"
    return "\n".join([
        f"# Kaggle output handoff — <code>{tag}</code>",
        "",
        "## English",
        "",
        "This Issue is created and locked automatically. GitHub still allows repository users "
        "with **Write** access to comment; other users cannot submit output.",
        "",
        "### 1. Download the completed Kaggle output",
        "",
        f"- Kaggle Notebook: {notebook_link_en}",
        f"- Draft Release: [open the delivery draft]({release_url})",
        "- Wait until Kaggle reports that the Notebook run is complete, then use its "
        "**Output → Download all** action. Submit the downloaded output ZIP, not the Notebook source.",
        "",
        "### 2. Submit a conversion candidate",
        "",
        "Choose either method:",
        "",
        "1. **Issue attachment:** create a new comment, attach one or more downloaded output ZIPs, "
        "and submit the comment. Each attachment is queued automatically; no command is required.",
        "2. **Draft Release asset:** upload the file to the linked Draft Release, save the draft, "
        "then create a new Issue comment containing `/convert <exact asset name>`. The complete "
        "text after `/convert` is treated as the asset name, including spaces.",
        "",
        "The filename and extension are unrestricted, but the downloaded content must be a valid "
        "ZIP with the expected Kaggle output tree. Direct Issue attachments are subject to "
        "GitHub's attachment-size limit; use a Draft Release asset for larger files.",
        "",
        "### Queue and status",
        "",
        "- Attachments and `/convert` commands are processed in comment order.",
        "- `⏳` means the candidate is being processed.",
        "- `❌` means that candidate failed. Add a **new comment** to retry; do not edit an old comment.",
        "- `✅` means conversion and validation succeeded. The workflow publishes the Release and "
        "closes this Issue automatically.",
        "- If every queued candidate fails, this Issue remains open and waits for another new comment.",
        "",
        "---",
        "",
        "## 中文说明",
        "",
        "本 Issue 由系统自动创建并锁定。GitHub 仍允许拥有仓库 **Write** 权限的用户评论，"
        "其他用户不能提交输出。",
        "",
        "### 1. 下载训练输出",
        "",
        f"- Kaggle Notebook：{notebook_link_zh}",
        f"- Draft Release：[打开交付草稿]({release_url})",
        "- 等待 Kaggle 显示 Notebook 运行完成，然后使用 **Output → Download all**。"
        "请提交下载得到的输出 ZIP，不要提交 Notebook 源文件。",
        "",
        "### 2. 提交转换候选包",
        "",
        "任选一种方式：",
        "",
        "1. **Issue 附件：**新建一条评论，附加一个或多个输出 ZIP 后发送。每个附件会自动入队，"
        "不需要命令。",
        "2. **Draft Release 资产：**先把文件上传到上方 Draft Release 并保存草稿，再新建评论："
        "`/convert <完整资产名>`。`/convert` 后的全部文字（包括空格）都作为资产名。",
        "",
        "文件名和扩展名不限，但内容必须是包含预期 Kaggle 输出目录的有效 ZIP。Issue 直接附件受 "
        "GitHub 大小限制；大文件请使用 Draft Release 资产。",
        "",
        "### 队列与状态",
        "",
        "- 附件和 `/convert` 命令按评论顺序依次处理。",
        "- `⏳`：正在处理。",
        "- `❌`：该候选包失败。请发一条**新评论**重试，不要编辑旧评论。",
        "- `✅`：转换和验证成功，workflow 会发布 Release 并自动关闭本 Issue。",
        "- 如果当前候选包全部失败，Issue 会保持开启，等待下一条新评论。",
        "",
        f"<!-- {ISSUE_MARKER}{encoded} -->",
        "",
    ])


def parse_issue_binding(issue: Mapping[str, Any]) -> dict[str, Any]:
    body = str(issue.get("body") or "")
    start = body.find(f"<!-- {ISSUE_MARKER}")
    end = body.find(" -->", start)
    if start < 0 or end < 0:
        raise RuntimeError(f"Issue {issue.get('number')} is not a Kaggle CD upload Issue")
    encoded = body[start + len(f"<!-- {ISSUE_MARKER}"):end].strip()
    try:
        binding = json.loads(base64.urlsafe_b64decode(encoded.encode()).decode())
    except (UnicodeDecodeError, binascii.Error, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Issue {issue.get('number')} has invalid Kaggle CD state") from exc
    required = {"schema", "release_id", "tag", "git_sha"}
    if not isinstance(binding, dict) or required - binding.keys():
        raise RuntimeError(f"Issue {issue.get('number')} has incomplete Kaggle CD state")
    if binding.get("schema") != STATE_SCHEMA:
        raise RuntimeError(f"Issue {issue.get('number')} uses an unsupported Kaggle CD schema")
    return binding


@dataclass(frozen=True)
class ConversionCandidate:
    candidate_id: str
    comment_id: int
    source_kind: str
    source: str


def issue_candidates(comments: Iterable[Mapping[str, Any]]) -> list[ConversionCandidate]:
    candidates: list[tuple[str, int, int, ConversionCandidate]] = []
    for comment in comments:
        user = comment.get("user") or {}
        if str(user.get("type", "")).lower() == "bot":
            continue
        body = str(comment.get("body") or "")
        comment_id = int(comment["id"])
        matches: list[tuple[int, str, str]] = []
        matches.extend(
            (match.start(), "release_asset", match.group(1).strip())
            for match in CONVERT_COMMAND_PATTERN.finditer(body)
        )
        matches.extend(
            (match.start(), "issue_attachment", match.group(0))
            for match in ISSUE_ATTACHMENT_PATTERN.finditer(body)
        )
        for index, (position, source_kind, source) in enumerate(sorted(matches)):
            if not source:
                continue
            candidate_id = f"{comment_id}:{index}"
            candidate = ConversionCandidate(
                candidate_id=candidate_id,
                comment_id=comment_id,
                source_kind=source_kind,
                source=source,
            )
            candidates.append((str(comment.get("created_at") or ""), comment_id, position, candidate))
    return [item[-1] for item in sorted(candidates, key=lambda item: item[:-1])]


def terminal_attempts(comments: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for comment in comments:
        user = comment.get("user") or {}
        if str(user.get("type", "")).lower() != "bot":
            continue
        for match in ATTEMPT_STATUS_PATTERN.finditer(str(comment.get("body") or "")):
            statuses[match.group("id")] = match.group("status")
    return statuses


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
        return active[0][0], "active"
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

    def list_release_assets(self, release_id: int) -> list[dict[str, Any]]:
        assets, page = [], 1
        while True:
            batch = self.request(
                "GET", f"/releases/{release_id}/assets?per_page=100&page={page}"
            )
            assets.extend(batch)
            if len(batch) < 100:
                return assets
            page += 1

    def delete_release_asset(self, asset_id: int, *, missing_ok: bool = False) -> None:
        try:
            self.request("DELETE", f"/releases/assets/{asset_id}")
        except RuntimeError as exc:
            if not (missing_ok and "HTTP 404:" in str(exc)):
                raise

    def get_issue(self, issue_number: int) -> dict[str, Any]:
        return self.request("GET", f"/issues/{issue_number}")

    def list_issue_comments(self, issue_number: int) -> list[dict[str, Any]]:
        comments, page = [], 1
        while True:
            batch = self.request(
                "GET", f"/issues/{issue_number}/comments?per_page=100&page={page}"
            )
            comments.extend(batch)
            if len(batch) < 100:
                return comments
            page += 1

    def create_issue_comment(self, issue_number: int, body: str) -> dict[str, Any]:
        return self.request("POST", f"/issues/{issue_number}/comments", {"body": body})

    def close_issue(self, issue_number: int) -> dict[str, Any]:
        return self.request(
            "PATCH", f"/issues/{issue_number}", {"state": "closed", "state_reason": "completed"}
        )

    def create_delivery_issue(
        self, release: Mapping[str, Any], state: Mapping[str, Any]
    ) -> dict[str, Any]:
        issue = self.request("POST", "/issues", {
            "title": f"Kaggle output upload · {state['tag']}",
            "body": render_delivery_issue(release, state),
        })
        self.request("PUT", f"/issues/{int(issue['number'])}/lock")
        return issue

    def update_delivery_issue(
        self, release: Mapping[str, Any], state: Mapping[str, Any]
    ) -> Optional[dict[str, Any]]:
        issue_number = state.get("issue_number")
        if issue_number is None:
            return None
        return self.request("PATCH", f"/issues/{int(issue_number)}", {
            "body": render_delivery_issue(release, state),
        })

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

    @staticmethod
    def _allowed_download_url(url: str) -> bool:
        parsed = urllib.parse.urlsplit(url)
        hostname = (parsed.hostname or "").lower()
        return (
            parsed.scheme == "https"
            and (
                hostname == "github.com"
                or hostname == "api.github.com"
                or hostname.endswith(".github.com")
                or hostname.endswith(".githubusercontent.com")
            )
        )

    def download(self, url: str, destination: Path) -> None:
        if not self._allowed_download_url(url):
            raise TerminalDeliveryError(f"Refusing non-GitHub download URL: {url}")
        request = urllib.request.Request(url, method="GET")
        request.add_header("Accept", "application/octet-stream")
        request.add_header("Authorization", f"Bearer {self.config.github_token}")
        request.add_header("X-GitHub-Api-Version", "2022-11-28")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                final_url = response.geturl()
                if not self._allowed_download_url(final_url):
                    raise TerminalDeliveryError(
                        f"GitHub download redirected to an unsupported host: {final_url}"
                    )
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_UPLOAD_BYTES:
                    raise TerminalDeliveryError(
                        f"Uploaded archive exceeds {MAX_UPLOAD_BYTES} bytes"
                    )
                destination.parent.mkdir(parents=True, exist_ok=True)
                total = 0
                with destination.open("wb") as output:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        total += len(block)
                        if total > MAX_UPLOAD_BYTES:
                            raise TerminalDeliveryError(
                                f"Uploaded archive exceeds {MAX_UPLOAD_BYTES} bytes"
                            )
                        output.write(block)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise RuntimeError(
                f"GitHub download failed: HTTP {exc.code}: {detail}"
            ) from exc


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
            "kaggle_kernel": self.config.kernel_ref,
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


def expected_delivery_config(config: Config) -> dict[str, str]:
    return {
        "kaggle_kernel": config.kernel_ref,
        "rknn_target_platform": config.rknn_target_platform,
    }


def reconcile_delivery_config(state: dict[str, Any], config: Config) -> bool:
    expected = expected_delivery_config(config)
    changed = False
    recorded_kernel = state.get("kaggle_kernel")
    if not recorded_kernel:
        if state.get("kaggle_version") is not None:
            raise TerminalDeliveryError(
                f"Tag {state.get('tag')!r} records Kaggle version "
                f"{state.get('kaggle_version')!r} without a kernel identity"
            )
        state["kaggle_kernel"] = expected["kaggle_kernel"]
        changed = True
    recorded_target = state.get("rknn_target_platform")
    if not recorded_target:
        state["rknn_target_platform"] = expected["rknn_target_platform"]
        changed = True

    mismatched = [
        key for key in DELIVERY_CONFIG_KEYS if state.get(key) != expected[key]
    ]
    if mismatched:
        detail = ", ".join(
            f"{key}: locked={state.get(key)!r}, current={expected[key]!r}"
            for key in mismatched
        )
        raise TerminalDeliveryError(
            f"Delivery configuration changed for tag {state.get('tag')!r}: {detail}"
        )
    return changed


def require_delivery_config(state: Mapping[str, Any], config: Config) -> None:
    expected = expected_delivery_config(config)
    missing = [key for key in DELIVERY_CONFIG_KEYS if not state.get(key)]
    mismatched = [
        key for key in DELIVERY_CONFIG_KEYS
        if state.get(key) and state.get(key) != expected[key]
    ]
    if missing or mismatched:
        raise TerminalDeliveryError(
            f"Delivery identity is not locked to the current configuration for tag "
            f"{state.get('tag')!r}: missing={missing}, mismatched={mismatched}"
        )


def version_matches(latest: Optional[Mapping[str, Any]], state: Mapping[str, Any]) -> bool:
    expected = state.get("kaggle_version")
    expected_kernel = state.get("kaggle_kernel")
    return (
        latest is not None
        and expected is not None
        and expected_kernel is not None
        and latest.get("kaggle_kernel") == expected_kernel
        and latest_version(latest) == int(expected)
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
) -> NoReturn:
    mark_failed(github, release, state, message)
    raise RuntimeError(message)


def persist_running_state(
    github: GitHubClient,
    release: Mapping[str, Any],
    state: dict[str, Any],
) -> None:
    try:
        updated_release = github.update_state(int(release["id"]), state)
    except Exception:
        # Preserve the startup failure path when Kaggle returned a version but
        # GitHub did not acknowledge the running state. The outer tick can then
        # record a failed state containing that exact version for a safe retry.
        state["state"] = "starting"
        raise
    try:
        github.update_delivery_issue(updated_release, state)
    except Exception as exc:
        print(
            f"::warning::Could not add the Kaggle link to Issue "
            f"#{state.get('issue_number')}: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


def start_job(github: GitHubClient, kaggle: KaggleClient, release: Mapping[str, Any], state: dict[str, Any], config: Config) -> None:
    previous_version = state.get("kaggle_version")
    state.pop("last_polled_at", None)
    transitioned = state.get("state") == "queued"
    if transitioned:
        state.update({"state": "starting", "attempt": int(state.get("attempt", 0)) + 1, "failure": None})
    removed_legacy_field = state.pop("kaggle_version_before_push", None) is not None
    try:
        config_migrated = reconcile_delivery_config(state, config)
    except TerminalDeliveryError as exc:
        fail_job(github, release, state, str(exc))
    if transitioned or removed_legacy_field or config_migrated:
        github.update_state(int(release["id"]), state)

    latest = kaggle.latest()
    current_status = kaggle.status()
    current_version = latest_version(latest)
    previous_completed = (
        previous_version is not None
        and version_matches(latest, state)
        and current_status["status"] == "complete"
    )
    if previous_completed:
        state.pop("last_submission_check_at", None)
        state.update({
            "state": "running", "kaggle_kernel": config.kernel_ref,
            "kaggle_version": current_version, "kaggle_url": latest["url"],
            "started_at": state.get("started_at", isoformat()),
        })
        persist_running_state(github, release, state)
        print(f"Recovered Kaggle version {current_version} for {state['tag']}")
        return
    if current_status["status"] in {"queued", "running", "new_script", "cancel_requested"}:
        state.update({
            "waiting_for_external_run": True,
            "last_submission_check_at": isoformat(),
            "external_status": current_status["status"],
        })
        github.update_state(int(release["id"]), state)
        print(f"Waiting for an existing Kaggle run with status {current_status['status']}")
        return

    with tempfile.TemporaryDirectory() as directory:
        upload_dir = Path(directory) / "upload"
        stage_upload(state, config, upload_dir)
        pushed = kaggle.push(upload_dir)
    state.pop("last_submission_check_at", None)
    state.update({
        "state": "running", "waiting_for_external_run": False,
        "kaggle_kernel": config.kernel_ref, "kaggle_version": pushed["version"],
        "kaggle_url": pushed["url"], "started_at": isoformat(),
    })
    persist_running_state(github, release, state)
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
    output_root = matches[0]
    validate_notebook_outputs(output_root)
    outside = [
        str(path.relative_to(download_dir))
        for path in download_dir.rglob("*")
        if path.is_file() and not path.is_relative_to(output_root)
    ]
    if outside:
        raise TerminalDeliveryError(
            f"Uploaded archive contains files outside the notebook output root: {sorted(outside)}"
        )
    return output_root


def extract_uploaded_zip(archive_path: Path, destination: Path) -> None:
    if archive_path.stat().st_size > MAX_UPLOAD_BYTES:
        raise TerminalDeliveryError(f"Uploaded archive exceeds {MAX_UPLOAD_BYTES} bytes")
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if not members:
                raise TerminalDeliveryError("Uploaded ZIP is empty")
            if len(members) > MAX_ARCHIVE_MEMBERS:
                raise TerminalDeliveryError(
                    f"Uploaded ZIP has {len(members)} members; maximum is {MAX_ARCHIVE_MEMBERS}"
                )
            expanded = sum(member.file_size for member in members)
            if expanded > MAX_ARCHIVE_EXPANDED_BYTES:
                raise TerminalDeliveryError(
                    f"Uploaded ZIP expands to {expanded} bytes; maximum is "
                    f"{MAX_ARCHIVE_EXPANDED_BYTES}"
                )
            seen: set[PurePosixPath] = set()
            for member in members:
                name = member.filename
                if "\\" in name or "\x00" in name or name.startswith("/"):
                    raise TerminalDeliveryError(f"Unsafe ZIP member path: {name!r}")
                stripped = name[:-1] if name.endswith("/") else name
                parts = stripped.split("/") if stripped else []
                if not parts or any(part in {"", ".", ".."} for part in parts):
                    raise TerminalDeliveryError(f"Unsafe ZIP member path: {name!r}")
                relative = PurePosixPath(*parts)
                if relative in seen:
                    raise TerminalDeliveryError(f"Duplicate ZIP member path: {name!r}")
                seen.add(relative)
                mode = member.external_attr >> 16
                file_type = mode & 0o170000
                if file_type not in {0, 0o040000, 0o100000}:
                    raise TerminalDeliveryError(f"Unsupported ZIP member type: {name!r}")
                if member.flag_bits & 0x1:
                    raise TerminalDeliveryError(f"Encrypted ZIP member is unsupported: {name!r}")
                target = destination.joinpath(*parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                copied = 0
                with archive.open(member) as source, target.open("wb") as output:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        copied += len(block)
                        if copied > member.file_size:
                            raise TerminalDeliveryError(
                                f"ZIP member exceeded its declared size: {name!r}"
                            )
                        output.write(block)
                if copied != member.file_size:
                    raise TerminalDeliveryError(
                        f"ZIP member size mismatch for {name!r}: {copied} != {member.file_size}"
                    )
    except zipfile.BadZipFile as exc:
        raise TerminalDeliveryError("Uploaded file is not a valid ZIP archive") from exc


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
    fixed_landmarks = copy.deepcopy(feature_contract)
    fixed_output = copy.deepcopy(output_contract)
    fixed_landmarks["shape"][0] = 1
    fixed_output["shape"][0] = 1
    pt_io = {
        "inputs": {
            "features": feature_contract,
            "lengths": {"dtype": "int32", "shape": ["B"], "minimum": 1, "maximum": 64},
        },
        "outputs": {"frame_embeddings": output_contract},
    }
    fixed_io = {
        "inputs": {"landmarks": fixed_landmarks},
        "outputs": {"frame_embeddings": fixed_output},
    }
    int8_io = copy.deepcopy(fixed_io)
    int8_io["inputs"]["landmarks"]["dtype"] = "int8"
    int8_io["inputs"]["landmarks"]["quantization_parameters"] = (
        "embedded; query scale and zero_point with RKNN Runtime"
    )
    int8_io["outputs"]["frame_embeddings"]["dtype"] = "float16"
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
        "kaggle_kernel": state["kaggle_kernel"],
        "kaggle_version": state["kaggle_version"],
        "kaggle_url": state["kaggle_url"],
        "rknn_target_platform": state["rknn_target_platform"],
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


def prepare_uploaded_handoff(
    release: Mapping[str, Any],
    state: Mapping[str, Any],
    config: Config,
    archive_path: Path,
    handoff_dir: Path,
) -> None:
    if state.get("state") != "running":
        raise TerminalDeliveryError(
            f"Draft Release is {state.get('state')!r}; output is accepted only after submission"
        )
    require_delivery_config(state, config)
    missing_identity = [
        key for key in RELEASE_IMMUTABLE_KEYS
        if state.get(key) is None or state.get(key) == ""
    ]
    if missing_identity:
        raise TerminalDeliveryError(
            f"Draft Release is missing delivery identity: {missing_identity}"
        )
    if handoff_dir.exists():
        shutil.rmtree(handoff_dir)
    output_dir = handoff_dir / "output"
    extract_uploaded_zip(archive_path, output_dir)
    find_notebook_output_root(output_dir)
    metadata = {
        "schema": HANDOFF_SCHEMA,
        "release_id": int(release["id"]),
        "repository": config.repository,
        "kaggle_kernel": state["kaggle_kernel"],
        "rknn_target_platform": state["rknn_target_platform"],
        "state": dict(state),
    }
    (handoff_dir / "release-state.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Prepared uploaded output handoff for {state['tag']}")


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
    for key in DELIVERY_CONFIG_KEYS:
        outer = metadata.get(key)
        if not isinstance(outer, str) or not outer:
            raise RuntimeError(f"Workflow handoff is missing {key}")
        if not state.get(key):
            state[key] = outer
        elif state.get(key) != outer:
            raise RuntimeError(f"Workflow handoff {key} does not match release state")
    missing_identity = [
        key for key in RELEASE_IMMUTABLE_KEYS
        if state.get(key) is None or state.get(key) == ""
    ]
    if missing_identity:
        raise RuntimeError(f"Workflow handoff is missing release identity: {missing_identity}")
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
    (args.output_dir / "release-state.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
        "kaggle_kernel": state["kaggle_kernel"],
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


def migrate_legacy_release_target(
    github: GitHubClient,
    release: Mapping[str, Any],
    current: dict[str, Any],
    expected: Mapping[str, Any],
) -> bool:
    target_key = "rknn_target_platform"
    if current.get(target_key) or not expected.get(target_key):
        return False
    stable_keys = tuple(
        key for key in RELEASE_IMMUTABLE_KEYS if key != target_key
    )
    if any(current.get(key) != expected.get(key) for key in stable_keys):
        return False
    current[target_key] = expected[target_key]
    github.update_state(int(release["id"]), current)
    return True


def publish_handoff_directory(handoff_dir: Path) -> str:
    metadata = load_handoff(handoff_dir)
    state = metadata["state"]
    config = Config.from_env(require_kaggle=False)
    if metadata.get("repository") != config.repository:
        raise RuntimeError("Workflow handoff repository does not match GITHUB_REPOSITORY")
    github = GitHubClient(config)
    release = github.get_release(int(metadata["release_id"]))
    current = state_for_release(release)
    migrate_legacy_release_target(github, release, current, state)
    mismatched = [
        key for key in RELEASE_IMMUTABLE_KEYS
        if current.get(key) != state.get(key)
    ]
    if mismatched or current.get("state") != "running":
        raise RuntimeError(
            f"Draft Release state changed after Kaggle handoff; mismatched fields: {mismatched}"
        )
    asset_dir = handoff_dir / "assets"
    assets = validate_release_assets(asset_dir, metadata)
    staged_assets = github.list_release_assets(int(release["id"]))
    github.upload_assets(str(state["tag"]), assets)
    for staged_asset in staged_assets:
        github.delete_release_asset(int(staged_asset["id"]), missing_ok=True)
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
    return str(state["tag"])


def publish_handoff(args: argparse.Namespace) -> None:
    publish_handoff_directory(args.handoff_dir)


def candidate_comment(
    candidate: ConversionCandidate, status: str, detail: Optional[str] = None
) -> str:
    labels = {
        "processing": "⏳ Converting queued Kaggle output",
        "failed": "❌ Kaggle output conversion failed",
        "succeeded": "✅ Kaggle output conversion succeeded",
    }
    source = html.escape(candidate.source)
    lines = [
        f"### {labels[status]}",
        "",
        f"Candidate `{candidate.candidate_id}` from comment #{candidate.comment_id}: "
        f"<code>{source}</code>",
    ]
    if detail:
        escaped = html.escape(detail[:6000]).replace("\n", "<br>")
        lines.extend(["", f"> {escaped}"])
    lines.extend(["", f"<!-- {ATTEMPT_MARKER}{candidate.candidate_id}:{status} -->"])
    return "\n".join(lines)


def download_candidate(
    github: GitHubClient,
    release: Mapping[str, Any],
    candidate: ConversionCandidate,
    destination: Path,
) -> Optional[int]:
    if candidate.source_kind == "issue_attachment":
        github.download(candidate.source, destination)
        return None
    if candidate.source_kind != "release_asset":
        raise RuntimeError(f"Unsupported conversion candidate kind: {candidate.source_kind}")
    assets = [
        asset for asset in github.list_release_assets(int(release["id"]))
        if asset.get("name") == candidate.source
    ]
    if len(assets) != 1:
        raise TerminalDeliveryError(
            f"Expected one Draft Release asset named {candidate.source!r}, found {len(assets)}"
        )
    asset = assets[0]
    github.download(str(asset["url"]), destination)
    return int(asset["id"])


def issue_release_context(
    github: GitHubClient, issue_number: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    issue = github.get_issue(issue_number)
    if not issue.get("locked"):
        raise RuntimeError(f"Kaggle CD upload Issue #{issue_number} must remain locked")
    binding = parse_issue_binding(issue)
    release = github.get_release(int(binding["release_id"]))
    if str(binding["tag"]) != str(release.get("tag_name")):
        raise RuntimeError(
            f"Issue #{issue_number} no longer matches Release tag {release.get('tag_name')!r}"
        )
    if release.get("draft"):
        state = state_for_release(release)
        state_mismatched = [
            key for key in ("tag", "git_sha") if str(state.get(key)) != str(binding[key])
        ]
        if state_mismatched or str(state.get("issue_number")) != str(issue_number):
            raise RuntimeError(
                f"Issue #{issue_number} no longer matches Draft Release state: "
                f"{state_mismatched or ['issue_number']}"
            )
    return issue, binding, release


def write_process_outputs(path: Optional[Path], *, published: bool, processed: int) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as output:
        output.write(f"published={'true' if published else 'false'}\n")
        output.write(f"processed={processed}\n")


def close_published_issue(
    github: GitHubClient, issue: Mapping[str, Any], release: Mapping[str, Any]
) -> bool:
    if release.get("draft"):
        return False
    if issue.get("state") != "closed":
        github.close_issue(int(issue["number"]))
    return True


def process_issue(args: argparse.Namespace) -> None:
    config = Config.from_env(require_kaggle=False)
    github = GitHubClient(config)
    issue_number = int(args.issue_number)
    processed = 0

    issue, _, release = issue_release_context(github, issue_number)
    if close_published_issue(github, issue, release):
        write_process_outputs(args.github_output, published=True, processed=processed)
        return

    while True:
        issue, _, release = issue_release_context(github, issue_number)
        if close_published_issue(github, issue, release):
            write_process_outputs(args.github_output, published=True, processed=processed)
            return
        state = state_for_release(release)
        comments = github.list_issue_comments(issue_number)
        completed = terminal_attempts(comments)
        pending = [
            candidate for candidate in issue_candidates(comments)
            if candidate.candidate_id not in completed
        ]
        if not pending:
            write_process_outputs(args.github_output, published=False, processed=processed)
            return
        candidate = pending[0]
        github.create_issue_comment(
            issue_number, candidate_comment(candidate, "processing")
        )
        processed += 1
        try:
            kernel = str(state.get("kaggle_kernel") or "")
            if "/" not in kernel:
                raise TerminalDeliveryError(
                    f"Draft Release has invalid Kaggle kernel identity: {kernel!r}"
                )
            username, slug = kernel.split("/", 1)
            conversion_config = Config(
                repository=config.repository,
                github_token="",
                kaggle_username=username,
                kernel_slug=slug,
                kernel_private=True,
                rknn_target_platform=str(state.get("rknn_target_platform") or ""),
            )
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive = root / "uploaded-output"
                download_candidate(github, release, candidate, archive)
                handoff_dir = root / "handoff"
                converted_dir = root / "converted"
                prepare_uploaded_handoff(
                    release, state, conversion_config, archive, handoff_dir
                )
                convert_handoff(argparse.Namespace(
                    handoff_dir=handoff_dir, output_dir=converted_dir
                ))
                publish_handoff_directory(converted_dir)
            github.create_issue_comment(
                issue_number,
                candidate_comment(
                    candidate,
                    "succeeded",
                    f"Published Release {state['tag']} with validated ONNX and RKNN assets.",
                ),
            )
            github.close_issue(issue_number)
            write_process_outputs(args.github_output, published=True, processed=processed)
            return
        except Exception as exc:
            latest_issue = github.get_issue(issue_number)
            latest_release = github.get_release(int(release["id"]))
            if close_published_issue(github, latest_issue, latest_release):
                write_process_outputs(args.github_output, published=True, processed=processed)
                return
            github.create_issue_comment(
                issue_number,
                candidate_comment(candidate, "failed", f"{type(exc).__name__}: {exc}"),
            )
            print(
                f"Conversion candidate {candidate.candidate_id} failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )


def register_tag(args: argparse.Namespace) -> None:
    config = Config.from_env(require_kaggle=False)
    github = GitHubClient(config)
    release = github.create_queue_release(args.tag, args.sha)
    state = state_for_release(release)
    issue = github.create_delivery_issue(release, state)
    state["issue_number"] = int(issue["number"])
    github.update_state(int(release["id"]), state)
    print(
        f"Queued {args.tag} in draft release {release['id']} with locked "
        f"Issue #{issue['number']}"
    )


def probe(_: argparse.Namespace) -> None:
    config = Config.from_env(require_kaggle=False)
    release, action = select_work(GitHubClient(config).list_releases())
    state = None if release is None else state_for_release(release)
    print(f"has_work={'false' if action == 'idle' else 'true'}")
    print(f"delivery_action={action}")
    submit_work = action == "start" or (
        state is not None and state.get("state") == "starting"
    )
    print(f"submit_work={'true' if submit_work else 'false'}")


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
    state.update({"state": "queued", "failure": None, "queued_at": isoformat()})
    state.pop("last_polled_at", None)
    state.pop("last_submission_check_at", None)
    state.pop("kaggle_version_before_push", None)
    github.update_state(int(release["id"]), state)
    print(f"Requeued {args.tag}")


def submit(_: argparse.Namespace) -> None:
    config = Config.from_env(require_kaggle=True)
    github, kaggle = GitHubClient(config), KaggleClient(config)
    release, action = select_work(github.list_releases())
    if action == "idle":
        print("Kaggle CD queue is idle")
        return
    state = state_for_release(release)
    if action == "active" and state.get("state") == "running":
        print(
            f"Kaggle version {state.get('kaggle_version')} for {state['tag']} is "
            f"awaiting manual output upload in Issue #{state.get('issue_number')}"
        )
        return
    try:
        start_job(github, kaggle, release, state, config)
    except Exception as exc:
        if state.get("state") == "starting":
            mark_failed(github, release, state, f"{type(exc).__name__}: {exc}")
        raise


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    register_parser = commands.add_parser("register-tag")
    register_parser.add_argument("--tag", required=True)
    register_parser.add_argument("--sha", required=True)
    register_parser.set_defaults(handler=register_tag)
    probe_parser = commands.add_parser("probe")
    probe_parser.set_defaults(handler=probe)
    submit_parser = commands.add_parser("submit")
    submit_parser.set_defaults(handler=submit)
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
    process_issue_parser = commands.add_parser("process-issue")
    process_issue_parser.add_argument("--issue-number", type=int, required=True)
    process_issue_parser.add_argument("--github-output", type=Path)
    process_issue_parser.set_defaults(handler=process_issue)
    return root


def main() -> None:
    args = parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
