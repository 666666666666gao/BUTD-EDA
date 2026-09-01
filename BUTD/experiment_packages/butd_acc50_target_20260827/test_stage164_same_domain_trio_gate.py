#!/usr/bin/env python3
import json
import os
import tempfile
import unittest
from unittest import mock

import stage164_same_domain_trio_gate as gate


def write(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def policy(hits, baseline=(4734, 4346), changed=0.5):
    selected025, selected050 = hits
    return {
        "protocol": "scanrefer_train_only_scene_hash_dev_locked_nested_blend_v1",
        "validation_labels_used_for_selection": False,
        "train_dump_sha256": "train-sha",
        "split_sizes": {"train": 100, "dev": 20, "test": 5647},
        "internal_scene_hash_test": {
            "selected": {
                "count": 5647,
                "hits025": selected025,
                "hits050": selected050,
            },
            "baseline": {
                "hits025": baseline[0],
                "hits050": baseline[1],
            },
            "changed_ratio": changed,
        },
    }


class SameDomainTrioGateTest(unittest.TestCase):
    def test_authorizes_five_hit_gain_with_protected_acc025(self):
        with tempfile.TemporaryDirectory() as root:
            old_path = os.path.join(root, "old.json")
            new_path = os.path.join(root, "new.json")
            out = os.path.join(root, "authorization.json")
            write(old_path, policy((4722, 4396)))
            write(new_path, policy((4711, 4401), changed=0.97))
            with mock.patch.object(gate, "__file__", __file__):
                result = gate.authorize(old_path, new_path, out)
            self.assertTrue(result["internal_gate_pass"])
            self.assertEqual(result["internal_comparison"]["delta_hits"], [-11, 5])

    def test_rejects_four_hit_gain(self):
        with tempfile.TemporaryDirectory() as root:
            old_path = os.path.join(root, "old.json")
            new_path = os.path.join(root, "new.json")
            out = os.path.join(root, "authorization.json")
            write(old_path, policy((4722, 4396)))
            write(new_path, policy((4722, 4400)))
            with mock.patch.object(gate, "__file__", __file__):
                result = gate.authorize(old_path, new_path, out)
            self.assertFalse(result["validation_evaluation_authorized"])

    def test_wraps_only_authorized_fixed_result(self):
        with tempfile.TemporaryDirectory() as root:
            old_path = os.path.join(root, "old.json")
            new_path = os.path.join(root, "new.json")
            auth_path = os.path.join(root, "authorization.json")
            raw_path = os.path.join(root, "raw.json")
            out = os.path.join(root, "wrapped.json")
            write(old_path, policy((4722, 4396)))
            write(new_path, policy((4723, 4402)))
            with mock.patch.object(gate, "__file__", __file__):
                gate.authorize(old_path, new_path, auth_path)
            write(raw_path, {
                "stage": "140_train_only_frozen_nested_blend_on_stage135c",
                "validation_labels_used_for_selection": False,
                "policy_lock_sha256": gate.sha256(new_path),
                "strict_goal_met_offline": True,
                "metrics": {"selected": {"hits025": 5200, "hits050": 4040}},
            })
            result = gate.wrap_validation(auth_path, raw_path, out)
            self.assertEqual(result["stage"], gate.RESULT_STAGE)
            self.assertEqual(result["policy_lock_sha256"], gate.sha256(auth_path))


if __name__ == "__main__":
    unittest.main()
