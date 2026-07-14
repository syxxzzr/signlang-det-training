import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
WORKER_WORKFLOW = ROOT / ".github" / "workflows" / "kaggle-cd-worker.yml"
ENQUEUE_WORKFLOW = ROOT / ".github" / "workflows" / "kaggle-cd-enqueue.yml"


class WorkflowControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = WORKER_WORKFLOW.read_text(encoding="utf-8")
        cls.enqueue = ENQUEUE_WORKFLOW.read_text(encoding="utf-8")

    def test_conversion_failure_disables_scheduled_worker(self):
        finalize = self.worker.split("  finalize-failure:\n", 1)[1]
        permissions = finalize.split("    steps:\n", 1)[0]
        self.assertIn("      actions: write\n", permissions)

        step_name = "      - name: Disable scheduled worker after model conversion failure\n"
        self.assertIn(step_name, finalize)
        disable_step = finalize.split(step_name, 1)[1]
        self.assertIn(
            "        if: ${{ always() && needs.convert.result != 'success' }}\n",
            disable_step,
        )
        self.assertIn("          GH_TOKEN: ${{ github.token }}\n", disable_step)
        self.assertIn("        run: gh workflow disable kaggle-cd-worker.yml\n", disable_step)

    def test_tag_enqueue_enables_worker_before_immediate_dispatch(self):
        enable = "gh workflow enable kaggle-cd-worker.yml"
        dispatch = "gh workflow run kaggle-cd-worker.yml"
        self.assertIn(enable, self.enqueue)
        self.assertIn(dispatch, self.enqueue)
        self.assertLess(self.enqueue.index(enable), self.enqueue.index(dispatch))

    def test_failure_finalizer_reenables_worker_when_queue_has_work(self):
        finalize = self.worker.split("  finalize-failure:\n", 1)[1]
        disable = "gh workflow disable kaggle-cd-worker.yml"
        probe = "python .github/scripts/kaggle_cd.py probe"
        enable = "gh workflow enable kaggle-cd-worker.yml"
        self.assertIn(disable, finalize)
        self.assertIn(probe, finalize)
        self.assertIn("steps.post_failure_probe.outputs.queue_action == 'start'", finalize)
        self.assertIn(enable, finalize)
        self.assertLess(finalize.index(disable), finalize.index(probe))
        self.assertLess(finalize.index(probe), finalize.index(enable))


if __name__ == "__main__":
    unittest.main()
