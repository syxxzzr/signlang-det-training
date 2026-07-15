import importlib.util
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).parents[1]


def load_coordinator():
    script = ROOT / ".github" / "scripts" / "kaggle_cd.py"
    spec = importlib.util.spec_from_file_location("kaggle_cd_issue_test", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class IssueQueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_coordinator()

    def test_comment_sources_are_queued_in_text_order_without_filename_rules(self):
        attachment = (
            "https://github.com/user-attachments/files/12345/model.output-without-zip-suffix"
        )
        comments = [
            {
                "id": 20,
                "created_at": "2026-07-14T01:00:00Z",
                "user": {"type": "User", "login": "writer"},
                "author_association": "COLLABORATOR",
                "body": f"Attached: [model output]({attachment})\n/convert output package (final)",
            },
            {
                "id": 21,
                "created_at": "2026-07-14T01:01:00Z",
                "user": {"type": "Bot"},
                "body": "/convert must-not-be-queued",
            },
        ]

        candidates = self.module.issue_candidates(comments)

        self.assertEqual(
            [item.source_kind for item in candidates],
            ["issue_attachment", "release_asset"],
        )
        self.assertEqual(candidates[-1].source, "output package (final)")

    def test_untrusted_comment_during_unlock_window_is_ignored(self):
        comments = [{
            "id": 22,
            "created_at": "2026-07-14T01:02:00Z",
            "user": {"type": "User", "login": "outsider"},
            "author_association": "NONE",
            "body": "/convert untrusted asset",
        }]

        github = SimpleNamespace(collaborator_permission=lambda login: "")

        self.assertEqual(self.module.authorized_issue_candidates(github, comments), [])

    def test_write_permission_overrides_advisory_author_association(self):
        comments = [{
            "id": 24,
            "created_at": "2026-07-14T01:04:00Z",
            "user": {"type": "User", "login": "writer"},
            "author_association": "NONE",
            "body": "/convert valid asset",
        }]
        github = SimpleNamespace(collaborator_permission=lambda login: "write")

        candidates = self.module.authorized_issue_candidates(github, comments)

        self.assertEqual([candidate.source for candidate in candidates], ["valid asset"])

    def test_candidate_requires_write_permission(self):
        comments = [{
            "id": 23,
            "created_at": "2026-07-14T01:03:00Z",
            "user": {"type": "User", "login": "org-reader"},
            "author_association": "MEMBER",
            "body": "/convert member asset",
        }]
        github = SimpleNamespace(collaborator_permission=lambda login: "read")

        self.assertEqual(self.module.authorized_issue_candidates(github, comments), [])

    def test_missing_collaborator_has_no_candidate_permission(self):
        config = self.module.Config(
            repository="owner/repository",
            github_token="token",
            kaggle_username="",
            kernel_slug="kernel",
            kernel_private=True,
            rknn_target_platform="rk3588",
        )
        client = self.module.GitHubClient(config)

        def missing_permission(*args, **kwargs):
            raise RuntimeError("GitHub API failed: HTTP 404: not found")

        client.request = missing_permission

        self.assertEqual(client.collaborator_permission("former-writer"), "")

    def test_only_bot_terminal_markers_dequeue_candidates(self):
        marker = "<!-- signlang-kaggle-cd-attempt:42:0:failed -->"
        comments = [
            {"body": marker, "user": {"type": "User"}},
            {"body": marker, "user": {"type": "Bot", "login": "other-app[bot]"}},
            {
                "body": marker,
                "user": {"type": "Bot", "login": "github-actions[bot]"},
            },
        ]
        self.assertEqual(self.module.terminal_attempts(comments), {"42:0": "failed"})

    def test_delivery_issue_is_locked_after_creation(self):
        config = self.module.Config(
            repository="owner/repository",
            github_token="token",
            kaggle_username="",
            kernel_slug="kernel",
            kernel_private=True,
            rknn_target_platform="rk3588",
        )
        client = self.module.GitHubClient(config)
        calls = []

        def fake_request(method, path, payload=None):
            calls.append((method, path, payload))
            return {"number": 17}

        client.request = fake_request
        release = {"id": 9, "html_url": "https://example.invalid/release"}
        state = {"tag": "v1", "git_sha": "abc"}

        issue = client.create_delivery_issue(release, state)

        self.assertEqual(issue["number"], 17)
        self.assertEqual(calls[-1], ("PUT", "/issues/17/lock", None))

    def test_bot_comment_temporarily_unlocks_and_relocks_issue(self):
        config = self.module.Config(
            repository="owner/repository",
            github_token="token",
            kaggle_username="",
            kernel_slug="kernel",
            kernel_private=True,
            rknn_target_platform="rk3588",
        )
        client = self.module.GitHubClient(config)
        calls = []

        def fake_request(method, path, payload=None):
            calls.append((method, path, payload))
            if method == "POST":
                return {"id": 42, "body": payload["body"]}
            return None

        client.request = fake_request

        comment = client.create_locked_issue_comment(19, "processing")

        self.assertEqual(comment["id"], 42)
        self.assertEqual(calls, [
            ("DELETE", "/issues/19/lock", None),
            ("POST", "/issues/19/comments", {"body": "processing"}),
            ("PUT", "/issues/19/lock", None),
        ])

    def test_bot_comment_relocks_issue_when_comment_creation_fails(self):
        config = self.module.Config(
            repository="owner/repository",
            github_token="token",
            kaggle_username="",
            kernel_slug="kernel",
            kernel_private=True,
            rknn_target_platform="rk3588",
        )
        client = self.module.GitHubClient(config)
        calls = []

        def fake_request(method, path, payload=None):
            calls.append((method, path, payload))
            if method == "POST":
                raise RuntimeError("comment failed")
            return None

        client.request = fake_request

        with self.assertRaisesRegex(RuntimeError, "comment failed"):
            client.create_locked_issue_comment(19, "failure")

        self.assertEqual(calls[-1], ("PUT", "/issues/19/lock", None))

    def test_delivery_issue_is_bilingual_and_explains_both_handoffs(self):
        release = {"id": 9, "html_url": "https://example.invalid/release"}
        state = {
            "tag": "v1",
            "git_sha": "abc",
            "kaggle_url": "https://www.kaggle.com/code/owner/kernel",
        }

        body = self.module.render_delivery_issue(release, state)

        self.assertIn("## English", body)
        self.assertIn("## 中文说明", body)
        self.assertIn("Output → Download all", body)
        self.assertIn("Issue attachment", body)
        self.assertIn("/convert <exact asset name>", body)
        self.assertIn("Add a new comment to retry", body)
        self.assertIn("打开已提交的 Kaggle Notebook", body)
        self.assertIn("### 1. Download the completed Kaggle output 📥", body)
        self.assertIn("### 2. Submit a conversion candidate 📦", body)
        self.assertIn("### Queue and status 📋", body)

    def test_generated_github_bodies_avoid_double_asterisk_markup(self):
        release = {"id": 9, "html_url": "https://example.invalid/release"}
        state = {
            "state": "failed",
            "tag": "v1",
            "git_sha": "abc",
            "attempt": 1,
            "failure": "example",
        }

        bodies = (
            self.module.render_delivery_issue(release, state),
            self.module.render_release_body(state),
        )

        for body in bodies:
            self.assertNotIn("**", body)

    def test_queue_continues_after_failure_and_closes_on_first_success(self):
        state = {
            "schema": self.module.STATE_SCHEMA,
            "state": "running",
            "tag": "v1",
            "git_sha": "abc",
            "issue_number": 17,
            "kaggle_kernel": "owner/kernel",
            "kaggle_version": 3,
            "kaggle_url": "https://example.invalid/kernel",
            "rknn_target_platform": "rk3588",
        }
        release = {
            "id": 9,
            "html_url": "https://example.invalid/release",
            "tag_name": "v1",
            "target_commitish": "abc",
            "draft": True,
            "body": self.module.render_release_body(state),
        }
        issue = {
            "number": 17,
            "locked": True,
            "state": "open",
            "body": self.module.render_delivery_issue(release, state),
        }
        comments = [
            {
                "id": 100,
                "created_at": "2026-07-14T01:00:00Z",
                "user": {"type": "User", "login": "writer"},
                "author_association": "NONE",
                "body": "/convert first candidate",
            },
            {
                "id": 101,
                "created_at": "2026-07-14T01:01:00Z",
                "user": {"type": "User", "login": "writer"},
                "author_association": "COLLABORATOR",
                "body": "/convert second candidate",
            },
        ]
        attempted = []

        class FakeGitHub:
            def get_issue(self, number):
                return issue

            def get_release(self, release_id):
                return release

            def list_issue_comments(self, number):
                return list(comments)

            def collaborator_permission(self, username):
                return "write"

            def create_locked_issue_comment(self, number, body):
                comments.append({
                    "id": 1000 + len(comments),
                    "created_at": "2026-07-14T02:00:00Z",
                    "user": {"type": "Bot", "login": "github-actions[bot]"},
                    "body": body,
                })

            def close_issue(self, number):
                issue["state"] = "closed"

        fake_github = FakeGitHub()
        config = self.module.Config(
            repository="owner/repository",
            github_token="token",
            kaggle_username="",
            kernel_slug="kernel",
            kernel_private=True,
            rknn_target_platform="rk3588",
        )
        originals = {
            "from_env": self.module.Config.__dict__["from_env"],
            "github_client": self.module.GitHubClient,
            "download": self.module.download_candidate,
            "prepare": self.module.prepare_uploaded_handoff,
            "convert": self.module.convert_handoff,
            "publish": self.module.publish_handoff_directory,
        }

        converted = []

        def fake_download(github, current_release, candidate, destination):
            attempted.append(candidate.source)
            destination.write_bytes(b"valid archive placeholder")

        def fake_prepare(*args, **kwargs):
            if attempted[-1] == "first candidate":
                raise self.module.TerminalDeliveryError(
                    "Kaggle notebook output allowlist mismatch"
                )

        def fake_convert(*args, **kwargs):
            converted.append(attempted[-1])

        def fake_publish(directory):
            release["draft"] = False
            return "v1"

        self.module.Config.from_env = classmethod(lambda cls, **kwargs: config)
        self.module.GitHubClient = lambda current_config: fake_github
        self.module.download_candidate = fake_download
        self.module.prepare_uploaded_handoff = fake_prepare
        self.module.convert_handoff = fake_convert
        self.module.publish_handoff_directory = fake_publish
        try:
            with tempfile.TemporaryDirectory() as directory:
                github_output = Path(directory) / "outputs"
                self.module.process_issue(SimpleNamespace(
                    issue_number=17, github_output=github_output
                ))
                outputs = github_output.read_text(encoding="utf-8")
        finally:
            self.module.Config.from_env = originals["from_env"]
            self.module.GitHubClient = originals["github_client"]
            self.module.download_candidate = originals["download"]
            self.module.prepare_uploaded_handoff = originals["prepare"]
            self.module.convert_handoff = originals["convert"]
            self.module.publish_handoff_directory = originals["publish"]

        self.assertEqual(attempted, ["first candidate", "second candidate"])
        self.assertEqual(converted, ["second candidate"])
        self.assertEqual(issue["state"], "closed")
        self.assertIn("published=true", outputs)
        self.assertTrue(any(":failed -->" in comment["body"] for comment in comments))
        self.assertTrue(any(":succeeded -->" in comment["body"] for comment in comments))


class ReleaseStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_coordinator()

    def release(self, state, tag_name):
        return {
            "id": 354162538,
            "draft": True,
            "tag_name": tag_name,
            "body": json.dumps(state),
        }

    def test_synthetic_draft_tag_does_not_replace_queued_tag(self):
        state = {
            "schema": self.module.STATE_SCHEMA,
            "state": "queued",
            "tag": "v0.0.1-Alpha",
            "git_sha": "a" * 40,
        }

        parsed = self.module.parse_release_state(
            self.release(state, "untagged-b6af7d978bf098ad8043")
        )

        self.assertEqual(parsed["tag"], "v0.0.1-Alpha")
        self.assertNotIn("tag_recovered_from", parsed)
        self.assertTrue(
            self.module.release_matches_tag(
                self.release(state, "untagged-b6af7d978bf098ad8043"),
                "v0.0.1-Alpha",
            )
        )
        self.assertFalse(
            self.module.release_matches_tag(
                self.release(state, "untagged-b6af7d978bf098ad8043"),
                "v0.0.2",
            )
        )

        config = SimpleNamespace(
            kaggle_username="owner",
            kernel_slug="kernel",
            kernel_private=True,
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(self.module, "run", return_value="a" * 40 + "\n") as run,
            mock.patch.object(self.module.subprocess, "check_output", return_value="{}"),
        ):
            self.module.stage_upload(parsed, config, Path(directory))

        run.assert_called_once_with(
            ["git", "rev-parse", "v0.0.1-Alpha^{commit}"]
        )

    def test_real_release_tag_still_repairs_stale_state(self):
        state = {
            "schema": self.module.STATE_SCHEMA,
            "state": "queued",
            "tag": "v0.0.1-Alpha",
            "git_sha": "a" * 40,
        }

        parsed = self.module.parse_release_state(self.release(state, "v0.0.2"))

        self.assertEqual(parsed["tag"], "v0.0.2")
        self.assertEqual(parsed["tag_recovered_from"], "v0.0.1-Alpha")

    def test_persisted_synthetic_state_recovers_original_tag(self):
        placeholder = "untagged-b6af7d978bf098ad8043"
        state = {
            "schema": self.module.STATE_SCHEMA,
            "state": "failed",
            "tag": placeholder,
            "tag_recovered_from": "v0.0.1-Alpha",
            "git_sha": "a" * 40,
        }

        parsed = self.module.parse_release_state(self.release(state, placeholder))

        self.assertEqual(parsed["tag"], "v0.0.1-Alpha")
        self.assertEqual(parsed["synthetic_tag_recovered_from"], placeholder)

    def test_persisted_synthetic_state_without_original_tag_fails_closed(self):
        placeholder = "untagged-b6af7d978bf098ad8043"
        state = {
            "schema": self.module.STATE_SCHEMA,
            "state": "failed",
            "tag": placeholder,
            "git_sha": "a" * 40,
        }

        with self.assertRaisesRegex(RuntimeError, "without a valid recovery tag"):
            self.module.parse_release_state(self.release(state, placeholder))

    def test_reserved_placeholder_format_cannot_be_registered_or_retried(self):
        placeholder = "untagged-b6af7d978bf098ad8043"
        config = self.module.Config(
            repository="owner/repository",
            github_token="token",
            kaggle_username="",
            kernel_slug="kernel",
            kernel_private=True,
            rknn_target_platform="rk3588",
        )
        client = self.module.GitHubClient(config)

        with self.assertRaisesRegex(RuntimeError, "reserved Draft Release placeholder"):
            client.create_queue_release(placeholder, "a" * 40)
        with self.assertRaisesRegex(RuntimeError, "reserved Draft Release placeholder"):
            self.module.retry(SimpleNamespace(tag=placeholder))

    def test_issue_context_restores_synthetic_draft_release_tag(self):
        state = {
            "schema": self.module.STATE_SCHEMA,
            "state": "running",
            "tag": "v0.0.1-Alpha",
            "git_sha": "a" * 40,
            "issue_number": 19,
        }
        release = {
            "id": 9,
            "draft": True,
            "tag_name": "untagged-b9b033178acf4e9f7a1c",
            "html_url": "https://example.invalid/release/9",
            "body": self.module.render_release_body(state),
        }
        issue = {
            "number": 19,
            "locked": True,
            "body": self.module.render_delivery_issue(release, state),
        }
        config = self.module.Config(
            repository="owner/repository",
            github_token="token",
            kaggle_username="",
            kernel_slug="kernel",
            kernel_private=True,
            rknn_target_platform="rk3588",
        )
        client = self.module.GitHubClient(config)
        patches = []

        def fake_request(method, path, payload=None):
            if (method, path) == ("GET", "/issues/19"):
                return issue
            if (method, path) == ("GET", "/releases/9"):
                return release
            if (method, path) == ("PATCH", "/releases/9"):
                patches.append(payload)
                return {**release, **payload}
            raise AssertionError(f"Unexpected request: {method} {path}")

        client.request = fake_request

        _, binding, repaired = self.module.issue_release_context(client, 19)

        self.assertEqual(binding["tag"], "v0.0.1-Alpha")
        self.assertEqual(repaired["tag_name"], "v0.0.1-Alpha")
        self.assertEqual(patches[0]["tag_name"], "v0.0.1-Alpha")

    def test_issue_context_rejects_non_placeholder_tag_mismatch(self):
        state = {
            "schema": self.module.STATE_SCHEMA,
            "state": "running",
            "tag": "v1",
            "git_sha": "a" * 40,
            "issue_number": 19,
        }
        bound_release = {
            "id": 9,
            "html_url": "https://example.invalid/release/9",
        }
        issue = {
            "number": 19,
            "locked": True,
            "body": self.module.render_delivery_issue(bound_release, state),
        }
        changed_release = {
            **bound_release,
            "draft": True,
            "tag_name": "v2",
            "body": self.module.render_release_body(state),
        }
        github = SimpleNamespace(
            get_issue=lambda number: issue,
            get_release=lambda release_id: changed_release,
        )

        with self.assertRaisesRegex(RuntimeError, "no longer matches Release tag 'v2'"):
            self.module.issue_release_context(github, 19)

    def test_standalone_publish_restores_tag_before_asset_upload(self):
        state = {
            "schema": self.module.STATE_SCHEMA,
            "state": "running",
            "tag": "v0.0.1-Alpha",
            "git_sha": "a" * 40,
            "kaggle_kernel": "owner/kernel",
            "kaggle_version": 3,
            "kaggle_url": "https://example.invalid/kernel",
            "rknn_target_platform": "rk3588",
        }
        metadata = {
            "repository": "owner/repository",
            "release_id": 9,
            "kaggle_kernel": "owner/kernel",
            "rknn_target_platform": "rk3588",
            "state": state,
        }
        release = {
            "id": 9,
            "draft": True,
            "tag_name": "untagged-b9b033178acf4e9f7a1c",
            "body": self.module.render_release_body(state),
        }
        events = []

        class FakeGitHub:
            def get_release(self, release_id):
                return release

            def list_release_assets(self, release_id):
                events.append("list-assets")
                return []

            def update_state(self, release_id, current):
                events.append("restore-tag")
                return {**release, "tag_name": current["tag"]}

            def upload_assets(self, tag, assets):
                events.append("upload-assets")
                self.assertion_tag = tag

            def delete_release_asset(self, asset_id, missing_ok=False):
                raise AssertionError("No staged assets should be deleted")

            def publish(self, release_id, tag, body):
                events.append("publish")
                return {"draft": False, "tag_name": tag}

        github = FakeGitHub()
        config = self.module.Config(
            repository="owner/repository",
            github_token="token",
            kaggle_username="",
            kernel_slug="kernel",
            kernel_private=True,
            rknn_target_platform="rk3588",
        )
        with (
            mock.patch.object(self.module, "load_handoff", return_value=metadata),
            mock.patch.object(self.module.Config, "from_env", return_value=config),
            mock.patch.object(self.module, "GitHubClient", return_value=github),
            mock.patch.object(
                self.module,
                "validate_release_assets",
                return_value=[Path("validated-asset")],
            ),
            mock.patch.object(
                self.module,
                "verify_tag_commit",
                side_effect=lambda state: events.append("verify-tag"),
            ),
        ):
            published_tag = self.module.publish_handoff_directory(Path("unused"))

        self.assertEqual(published_tag, "v0.0.1-Alpha")
        self.assertEqual(github.assertion_tag, "v0.0.1-Alpha")
        self.assertEqual(
            events,
            [
                "list-assets",
                "verify-tag",
                "restore-tag",
                "upload-assets",
                "publish",
            ],
        )

    def test_duplicate_registration_finds_placeholder_backed_release(self):
        state = {
            "schema": self.module.STATE_SCHEMA,
            "state": "queued",
            "tag": "v0.0.1-Alpha",
            "git_sha": "a" * 40,
        }
        existing = self.release(state, "untagged-b9b033178acf4e9f7a1c")
        config = self.module.Config(
            repository="owner/repository",
            github_token="token",
            kaggle_username="",
            kernel_slug="kernel",
            kernel_private=True,
            rknn_target_platform="rk3588",
        )
        client = self.module.GitHubClient(config)
        client.list_releases = lambda: [existing]

        with self.assertRaisesRegex(RuntimeError, "already exists for tag v0.0.1-Alpha"):
            client.create_queue_release("v0.0.1-Alpha", "a" * 40)

    def test_retry_finds_placeholder_backed_release_and_restores_tag(self):
        state = {
            "schema": self.module.STATE_SCHEMA,
            "state": "failed",
            "tag": "v0.0.1-Alpha",
            "git_sha": "a" * 40,
            "failure": "transient failure",
        }
        release = self.release(state, "untagged-b9b033178acf4e9f7a1c")
        updates = []

        class FakeGitHub:
            def list_releases(self):
                return [release]

            def update_state(self, release_id, current):
                updates.append(dict(current))
                return {**release, "tag_name": current["tag"]}

        config = self.module.Config(
            repository="owner/repository",
            github_token="token",
            kaggle_username="",
            kernel_slug="kernel",
            kernel_private=True,
            rknn_target_platform="rk3588",
        )
        with (
            mock.patch.object(self.module.Config, "from_env", return_value=config),
            mock.patch.object(self.module, "GitHubClient", return_value=FakeGitHub()),
        ):
            self.module.retry(SimpleNamespace(tag="v0.0.1-Alpha"))

        self.assertEqual(updates[0]["tag"], "v0.0.1-Alpha")
        self.assertEqual(updates[0]["state"], "queued")
        self.assertIsNone(updates[0]["failure"])


class UploadedZipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_coordinator()

    def write_expected_archive(self, archive: Path, prefix: str = "arbitrary-root"):
        with zipfile.ZipFile(archive, "w") as bundle:
            for relative in self.module.NOTEBOOK_OUTPUT_FILES:
                path = f"{prefix}/{relative}" if prefix else relative
                bundle.writestr(path, relative.encode())

    def test_valid_zip_needs_no_filename_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "any filename"
            output = root / "output"
            self.write_expected_archive(archive)

            self.module.extract_uploaded_zip(archive, output)
            detected = self.module.find_notebook_output_root(output)

            self.assertEqual(detected, output / "arbitrary-root")

    def test_nested_output_accepts_kaggle_renderer_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "upload"
            output = root / "output"
            self.write_expected_archive(archive, "signlang-det")
            with zipfile.ZipFile(archive, "a") as bundle:
                bundle.writestr("__results___files/__results___22_0.png", b"rendered")
                bundle.writestr("__results___files/__results___22_1.png", b"rendered")

            self.module.extract_uploaded_zip(archive, output)
            detected = self.module.find_notebook_output_root(output)

            self.assertEqual(detected, output / "signlang-det")

    def test_flat_output_at_archive_root_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "upload"
            output = root / "output"
            self.write_expected_archive(archive, "")
            with zipfile.ZipFile(archive, "a") as bundle:
                bundle.writestr("__results___files/__results___22_0.png", b"rendered")

            self.module.extract_uploaded_zip(archive, output)
            detected = self.module.find_notebook_output_root(output)

            self.assertEqual(detected, output)

    def test_zip_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "upload"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../outside", b"bad")

            with self.assertRaisesRegex(self.module.TerminalDeliveryError, "Unsafe ZIP"):
                self.module.extract_uploaded_zip(archive, root / "output")

    def test_files_outside_output_root_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "upload"
            self.write_expected_archive(archive)
            with zipfile.ZipFile(archive, "a") as bundle:
                bundle.writestr("unrelated.txt", b"unexpected")
            output = root / "output"

            self.module.extract_uploaded_zip(archive, output)
            with self.assertRaisesRegex(
                self.module.TerminalDeliveryError, "outside the notebook output root"
            ):
                self.module.find_notebook_output_root(output)


if __name__ == "__main__":
    unittest.main()
