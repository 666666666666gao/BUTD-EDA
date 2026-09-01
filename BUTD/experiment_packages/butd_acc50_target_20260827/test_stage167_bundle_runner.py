#!/usr/bin/env python3
import os
import unittest


class Stage167BundleRunnerTest(unittest.TestCase):
    def test_preregistered_runner_contains_both_dependency_stacks(self):
        path = os.path.join(
            os.path.dirname(__file__),
            "run_stage167d_final_bundle_reload_if_goal_met.sh",
        )
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
        required = (
            "stage167b_complete_goal_met",
            "stage167_build_final_artifact.py",
            "STAGE154",
            "STAGE164_31",
            "STAGE164_33",
            "STAGE164_PARENT",
            "STAGE165",
            "STAGE167",
            "locks/stage154.json",
            "locks/stage165.json",
            "locks/stage167.json",
            "stage167d_fresh_bundle_reload",
            "'primary_stage150_hits': [5234, 3979]",
            "'primary_inference_required_for_deployment': True",
        )
        for token in required:
            self.assertIn(token, text)
        self.assertIn("sleep 60", text)


if __name__ == "__main__":
    unittest.main()
