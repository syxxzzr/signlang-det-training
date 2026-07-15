import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
SUBMIT_WORKFLOW = ROOT / ".github" / "workflows" / "kaggle-cd-submit.yml"
REGISTER_TAG_WORKFLOW = ROOT / ".github" / "workflows" / "kaggle-cd-register-tag.yml"
ISSUE_WORKFLOW = ROOT / ".github" / "workflows" / "kaggle-cd-issue-convert.yml"


class WorkflowControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.submit = SUBMIT_WORKFLOW.read_text(encoding="utf-8")
        cls.register = REGISTER_TAG_WORKFLOW.read_text(encoding="utf-8")
        cls.issue = ISSUE_WORKFLOW.read_text(encoding="utf-8")

    def test_submit_workflow_has_no_scheduled_polling_or_conversion(self):
        self.assertNotIn("  schedule:\n", self.submit)
        self.assertIn("python .github/scripts/kaggle_cd.py submit", self.submit)
        self.assertIn("      issues: write\n", self.submit)
        self.assertNotIn("convert-handoff", self.submit)
        self.assertNotIn("kernels_output", self.submit)

    def test_tag_registration_can_create_issue_and_dispatch_submission(self):
        permissions = self.register.split("jobs:\n", 1)[0]
        self.assertIn("  issues: write\n", permissions)
        self.assertIn("python .github/scripts/kaggle_cd.py register-tag", self.register)
        self.assertIn("gh workflow run kaggle-cd-submit.yml", self.register)
        self.assertNotIn("gh workflow enable", self.register)

    def test_issue_workflow_requires_a_locked_upload_issue(self):
        self.assertIn("issue_comment:\n", self.issue)
        self.assertIn("github.event.issue.locked", self.issue)
        self.assertIn("github.event.comment.user.type != 'Bot'", self.issue)
        self.assertIn("Kaggle output upload · ", self.issue)
        self.assertNotIn("comment.user.login", self.issue)
        self.assertIn("if: ${{ always() }}", self.issue)
        self.assertIn(
            'gh api --method PUT "repos/$GH_REPO/issues/$ISSUE_NUMBER/lock"',
            self.issue,
        )

    def test_issue_workflow_drains_then_starts_next_tag(self):
        drain = "python .github/scripts/kaggle_cd.py process-issue"
        dispatch = "gh workflow run kaggle-cd-submit.yml"
        self.assertIn(drain, self.issue)
        self.assertIn("steps.drain.outputs.published == 'true'", self.issue)
        self.assertIn(dispatch, self.issue)
        self.assertLess(self.issue.index(drain), self.issue.index(dispatch))

    def test_python_downloads_use_default_package_index(self):
        self.assertNotIn("PIP_INDEX_URL", self.submit)
        self.assertNotIn("PIP_INDEX_URL", self.issue)
        self.assertNotIn("mirrors.nju.edu.cn", self.submit)
        self.assertNotIn("mirrors.nju.edu.cn", self.issue)

    def test_conversion_pins_onnx_compatible_protobuf(self):
        self.assertIn('"onnx==1.16.1"', self.issue)
        self.assertIn('"protobuf==4.25.4"', self.issue)


if __name__ == "__main__":
    unittest.main()
