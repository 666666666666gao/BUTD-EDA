import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class SpacyAugmentationAuditTest(unittest.TestCase):
    def test_evaluator_maps_spacy_rotation_and_profile_ids_to_audit_buckets(self):
        from src.grounding_evaluator import GroundingEvaluator

        evaluator = GroundingEvaluator(prefixes=["last_"])
        end_points = {
            "spacy_rotation_mode_id": torch.tensor([2]),
            "spacy_augmentation_profile_id": torch.tensor([4]),
        }

        self.assertEqual(
            evaluator._spacy_augmentation_bucket(end_points, 0),
            "spacy_aug_yaw_only",
        )
        self.assertEqual(
            evaluator._spacy_augmentation_profile_bucket(end_points, 0),
            "spacy_profile_yaw_relation_free",
        )

    def test_print_stats_includes_spacy_bucket_counts(self):
        from src.grounding_evaluator import GroundingEvaluator

        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        evaluator.spacy_augmentation_dets[
            ("spacy_aug_yaw_only", "last_", 0.25, 1)
        ] = 1
        evaluator.spacy_augmentation_gts[
            ("spacy_aug_yaw_only", "last_", 0.25, 1)
        ] = 2
        evaluator.spacy_augmentation_counts[
            ("spacy_aug_yaw_only", "last_")
        ] = 2

        stdout = StringIO()
        with redirect_stdout(stdout):
            results = evaluator.print_stats()

        self.assertEqual(results["last__spacy_aug_yaw_only_count"], 2)
        self.assertEqual(results["eval_spacy_aug_yaw_only_count"], 2)
        self.assertIn(
            "('spacy_aug_yaw_only', 'last_', 'count') 2",
            stdout.getvalue(),
        )

    def test_parser_accepts_lightweight_spacy_source_audit(self):
        from main_utils import parse_option

        test_args = [
            "prog",
            "--eval_report_spacy_source_scores",
            "--eval_spacy_source_score_sources",
            "base,quality,detector_jointtight",
            "--eval_results_json_path",
            "/tmp/eval_results.json",
        ]
        with mock.patch.object(sys, "argv", test_args):
            args = parse_option()

        self.assertTrue(args.eval_report_spacy_source_scores)
        self.assertEqual(
            args.eval_spacy_source_score_sources,
            "base,quality,detector_jointtight",
        )
        self.assertEqual(args.eval_results_json_path, "/tmp/eval_results.json")

    def test_eval_results_json_writer_creates_parent_and_serializes_results(self):
        from train_dist_mod import write_eval_results_json

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "eval_results.json"

            write_eval_results_json(
                str(path),
                {
                    "eval_primary_score_source": "detector_jointtight",
                    "eval_spacy_source_bucket_quality_acc0.25_top1": 0.5,
                    "eval_spacy_source_bucket_quality_count": 2,
                },
            )

            payload = path.read_text()

        self.assertIn('"eval_primary_score_source": "detector_jointtight"', payload)
        self.assertIn('"eval_spacy_source_bucket_quality_acc0.25_top1": 0.5', payload)

    def test_evaluator_reports_spacy_bucket_score_source_accuracy(self):
        from src.grounding_evaluator import GroundingEvaluator

        evaluator = GroundingEvaluator(
            thresholds=[0.25],
            topks=[1],
            prefixes=["last_"],
        )
        pred_bbox = torch.tensor([[
            [5.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
        ]])
        end_points = {
            "eval_report_spacy_source_scores": True,
            "eval_spacy_source_score_sources": "base,quality",
            "spacy_rotation_mode_id": torch.tensor([2]),
            "spacy_augmentation_profile_id": torch.tensor([4]),
            "positive_map": torch.tensor([[[1.0, 0.0]]]),
            "box_label_mask": torch.tensor([[1.0]]),
            "center_label": torch.tensor([[[0.0, 0.0, 0.0]]]),
            "size_gts": torch.tensor([[[2.0, 2.0, 2.0]]]),
            "last_center": pred_bbox[:, :, :3],
            "last_pred_size": pred_bbox[:, :, 3:],
            "last_sem_cls_scores": torch.tensor([[
                [5.0, -5.0],
                [4.0, -4.0],
            ]]),
            "pred_iou": torch.tensor([[0.10, 0.90]]),
        }

        evaluator.evaluate_bbox_by_span(end_points, "last_")
        stdout = StringIO()
        with redirect_stdout(stdout):
            results = evaluator.print_stats()

        self.assertEqual(
            results[
                "last__spacy_source_spacy_aug_yaw_only_base_acc0.25_top1"
            ],
            0.0,
        )
        self.assertEqual(
            results[
                "last__spacy_source_spacy_aug_yaw_only_quality_acc0.25_top1"
            ],
            1.0,
        )
        self.assertEqual(
            results[
                "eval_spacy_source_spacy_aug_yaw_only_quality_acc0.25_top1"
            ],
            1.0,
        )
        self.assertEqual(
            results["eval_spacy_source_spacy_aug_yaw_only_quality_count"],
            1,
        )
        self.assertIn("Spacy source score audit", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
