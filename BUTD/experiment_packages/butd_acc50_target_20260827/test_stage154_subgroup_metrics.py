import unittest

from stage154_subgroup_metrics import (
    select_fixed_source_ious,
    subgroup_metrics,
    validate_stage154_contract,
)


class Stage154SubgroupMetricsTest(unittest.TestCase):
    def test_reports_unique_multiple_and_overall_from_fixed_selected_ious(self):
        result = subgroup_metrics(
            [True, True, False, False],
            [0.60, 0.30, 0.51, 0.10],
        )

        self.assertEqual(result["counts"], {"unique": 2, "multiple": 2, "total": 4})
        self.assertEqual(result["hits025"], {"unique": 2, "multiple": 1, "overall": 3})
        self.assertEqual(result["hits050"], {"unique": 1, "multiple": 1, "overall": 2})
        self.assertEqual(result["acc025"], {"unique": 1.0, "multiple": 0.5, "overall": 0.75})
        self.assertEqual(result["acc050"], {"unique": 0.5, "multiple": 0.5, "overall": 0.5})

    def test_rejects_a_subgroup_result_that_changes_locked_overall_hits(self):
        metrics = {
            "counts": {"unique": 1419, "multiple": 8089, "total": 9508},
            "hits025": {"unique": 1000, "multiple": 4219, "overall": 5219},
            "hits050": {"unique": 900, "multiple": 3127, "overall": 4027},
        }

        with self.assertRaisesRegex(ValueError, "locked Overall hit mismatch"):
            validate_stage154_contract(metrics, {"acc025": 5220, "acc050": 4027})

    def test_fixed_selector_uses_stage150_at_the_locked_threshold(self):
        selected = select_fixed_source_ious(
            [0.49, 0.50, 0.51],
            [0.10, 0.20, 0.30],
            [0.60, 0.70, 0.80],
            0.50,
        )

        self.assertEqual(selected, [0.10, 0.70, 0.80])


if __name__ == "__main__":
    unittest.main()
