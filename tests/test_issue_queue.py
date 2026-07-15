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
                "user": {"type": "User"},
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

    def test_only_bot_terminal_markers_dequeue_candidates(self):
        marker = "<!-- signlang-kaggle-cd-attempt:42:0:failed -->"
        comments = [
            {"body": marker, "user": {"type": "User"}},
            {"body": marker, "user": {"type": "Bot"}},
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
        self.assertIn("Add a **new comment** to retry", body)
        self.assertIn("打开已提交的 Kaggle Notebook", body)

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
                "user": {"type": "User"},
                "body": "/convert first candidate",
            },
            {
                "id": 101,
                "created_at": "2026-07-14T01:01:00Z",
                "user": {"type": "User"},
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

            def create_issue_comment(self, number, body):
                comments.append({
                    "id": 1000 + len(comments),
                    "created_at": "2026-07-14T02:00:00Z",
                    "user": {"type": "Bot"},
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

        def fake_download(github, current_release, candidate, destination):
            attempted.append(candidate.source)
            if candidate.source == "first candidate":
                raise self.module.TerminalDeliveryError("bad candidate")
            destination.write_bytes(b"valid archive placeholder")

        def fake_publish(directory):
            release["draft"] = False
            return "v1"

        self.module.Config.from_env = classmethod(lambda cls, **kwargs: config)
        self.module.GitHubClient = lambda current_config: fake_github
        self.module.download_candidate = fake_download
        self.module.prepare_uploaded_handoff = lambda *args, **kwargs: None
        self.module.convert_handoff = lambda *args, **kwargs: None
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


class UploadedZipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_coordinator()

    def write_expected_archive(self, archive: Path):
        with zipfile.ZipFile(archive, "w") as bundle:
            for relative in self.module.NOTEBOOK_OUTPUT_FILES:
                bundle.writestr(f"arbitrary-root/{relative}", relative.encode())

    def test_valid_zip_needs_no_filename_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "any filename"
            output = root / "output"
            self.write_expected_archive(archive)

            self.module.extract_uploaded_zip(archive, output)
            detected = self.module.find_notebook_output_root(output)

            self.assertEqual(detected, output / "arbitrary-root")

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
