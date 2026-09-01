#!/usr/bin/env python3
import unittest

import stage169_fold_routed_risk_selector as selector


def report(default50, selected50, default25, selected25, fixes, breaks,
           changed_ratio):
    return {
        "default_stage142": {"hits025": default25, "hits050": default50},
        "selected": {"hits025": selected25, "hits050": selected50},
        "fix_break": {
            "fix_050": fixes,
            "break_050": breaks,
        },
        "changed_ratio": changed_ratio,
    }


class FoldRoutedRiskSelectorTest(unittest.TestCase):
    def test_internal_gate_rejects_observed_stage168_precision_ratio(self):
        observed = report(
            default50=4403, selected50=4414,
            default25=4726, selected25=4737,
            fixes=38, breaks=27, changed_ratio=0.09863644412962635,
        )
        self.assertFalse(selector.internal_gate(observed))

    def test_internal_gate_accepts_safe_scene_disjoint_gain(self):
        safe = report(
            default50=4403, selected50=4418,
            default25=4726, selected25=4730,
            fixes=42, breaks=27, changed_ratio=0.10,
        )
        self.assertTrue(selector.internal_gate(safe))

    def test_oof_selection_ignores_candidate_without_precision_gate(self):
        unsafe = {
            "name": "unsafe", "oof": {
                "high_precision_gate": False,
                "fix_break": {"net_050": 999},
                "selected": {"hits025": 999, "mean_iou": 0.99},
                "changed_ratio": 0.01,
            },
        }
        safe = {
            "name": "safe", "oof": {
                "high_precision_gate": True,
                "fix_break": {"net_050": 5},
                "selected": {"hits025": 10, "mean_iou": 0.5},
                "changed_ratio": 0.10,
            },
        }
        self.assertEqual(selector.select_oof_candidate([unsafe, safe]), safe)


if __name__ == "__main__":
    unittest.main()
