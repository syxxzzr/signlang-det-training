import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("kaggle_cd.py")
SPEC = importlib.util.spec_from_file_location("kaggle_cd", SCRIPT)
kaggle_cd = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = kaggle_cd
SPEC.loader.exec_module(kaggle_cd)


def release(release_id, tag, state, created_at):
    return {
        "id": release_id,
        "tag_name": tag,
        "draft": True,
        "created_at": created_at,
        "body": json.dumps({
            "schema": kaggle_cd.STATE_SCHEMA,
            "state": state,
            "tag": tag,
            "git_sha": "a" * 40,
            "queued_at": created_at,
        }),
    }


class QueueTests(unittest.TestCase):
    def test_selects_only_active_release(self):
        releases = [
            release(1, "v1.0.0", "queued", "2026-01-01T00:00:00Z"),
            release(2, "v1.1.0", "running", "2026-01-02T00:00:00Z"),
        ]
        selected, action = kaggle_cd.select_work(releases)
        self.assertEqual(selected["id"], 2)
        self.assertEqual(action, "poll")

    def test_selects_oldest_queued_release(self):
        releases = [
            release(2, "v1.1.0", "queued", "2026-01-02T00:00:00Z"),
            release(1, "v1.0.0", "queued", "2026-01-01T00:00:00Z"),
        ]
        selected, action = kaggle_cd.select_work(releases)
        self.assertEqual(selected["id"], 1)
        self.assertEqual(action, "start")

    def test_refuses_multiple_active_releases(self):
        releases = [
            release(1, "v1.0.0", "starting", "2026-01-01T00:00:00Z"),
            release(2, "v1.1.0", "running", "2026-01-02T00:00:00Z"),
        ]
        with self.assertRaisesRegex(RuntimeError, "multiple active"):
            kaggle_cd.select_work(releases)

    def test_ignores_non_cd_release_bodies(self):
        selected, action = kaggle_cd.select_work([{
            "id": 3, "tag_name": "old", "draft": True,
            "created_at": "2025-01-01T00:00:00Z", "body": "normal release notes",
        }])
        self.assertIsNone(selected)
        self.assertEqual(action, "idle")


class KaggleMetadataTests(unittest.TestCase):
    def test_uses_official_kernel_metadata_fields(self):
        metadata = kaggle_cd.build_kernel_metadata(
            username="alice", slug="signlang-det-training", private=True,
        )
        self.assertEqual(metadata["id"], "alice/signlang-det-training")
        self.assertTrue(metadata["enable_gpu"])
        self.assertFalse(metadata["enable_tpu"])
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertEqual(metadata["competition_sources"], ["asl-signs"])
        self.assertEqual(metadata["kernel_sources"], ["abdelrhmankaram/asl-preprocessing-7"])
        self.assertNotIn("dataSources", metadata)
        self.assertNotIn("hardware", metadata)

    def test_injects_tag_provenance_into_upload_copy(self):
        notebook = {"cells": [], "metadata": {"kernelspec": {"name": "python3"}}}
        result = kaggle_cd.inject_provenance(
            notebook, tag="v1.2.3", git_sha="b" * 40, repository="o/r",
        )
        self.assertEqual(result["metadata"]["signlang_cd"]["release_tag"], "v1.2.3")
        self.assertEqual(result["metadata"]["signlang_cd"]["git_sha"], "b" * 40)
        self.assertNotIn("kaggle", result["metadata"])

    def test_default_title_matches_stable_slug(self):
        metadata = kaggle_cd.build_kernel_metadata("alice", "signlang-det-training", True)
        self.assertEqual(metadata["title"], "Signlang Det Training")

    def test_custom_slug_gets_a_matching_title(self):
        metadata = kaggle_cd.build_kernel_metadata("alice", "custom-training-kernel", True)
        self.assertEqual(metadata["title"], "Custom Training Kernel")


class CoordinatorTests(unittest.TestCase):
    def test_waiting_for_external_run_does_not_increment_attempt(self):
        item = release(1, "v1.0.0", "starting", "2026-01-01T00:00:00Z")
        state = kaggle_cd.state_for_release(item)
        state["attempt"] = 1

        class GitHub:
            def __init__(self): self.states = []
            def update_state(self, _release_id, value): self.states.append(dict(value))

        class Kaggle:
            def latest(self): return None
            def status(self): return {"status": "running", "failure_message": ""}

        config = type("Config", (), {"kernel_ref": "alice/signlang-det-training"})()
        github = GitHub()
        kaggle_cd.start_job(github, Kaggle(), item, state, config)
        self.assertEqual(state["state"], "starting")
        self.assertEqual(state["attempt"], 1)
        self.assertTrue(state["waiting_for_external_run"])


class WorkflowTests(unittest.TestCase):
    def test_workflows_define_tag_queue_and_scheduled_serial_worker(self):
        root = SCRIPT.parents[2]
        enqueue = (root / ".github/workflows/kaggle-cd-enqueue.yml").read_text()
        worker = (root / ".github/workflows/kaggle-cd-worker.yml").read_text()
        self.assertIn("tags:\n      - \"*\"", enqueue)
        self.assertIn("actions: write", enqueue)
        self.assertIn('cron: "*/10 * * * *"', worker)
        self.assertIn("group: kaggle-cd-worker", worker)
        self.assertIn("cancel-in-progress: false", worker)
        self.assertIn("secrets.KAGGLE_USERNAME", worker)
        self.assertIn("secrets.KAGGLE_KEY", worker)
        self.assertIn("7b31fdb492b2050a2f0eba2f035a0955da0c9305", worker)
        for variable in (
            "KAGGLE_KERNEL_SLUG", "KAGGLE_KERNEL_PRIVATE", "KAGGLE_OUTPUT_PART_SIZE_MB",
        ):
            self.assertIn(f"vars.{variable}", worker)


class AssetTests(unittest.TestCase):
    def test_splits_large_release_asset_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.tar.gz"
            path.write_bytes(b"0123456789abcdef")
            parts = kaggle_cd.split_asset(path, max_bytes=6)
            self.assertEqual([part.name for part in parts], [
                "output.tar.gz.part-0001", "output.tar.gz.part-0002", "output.tar.gz.part-0003",
            ])
            self.assertEqual(b"".join(part.read_bytes() for part in parts), path.read_bytes())


if __name__ == "__main__":
    unittest.main()
