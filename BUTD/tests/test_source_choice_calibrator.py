import unittest
import os
import tempfile

import torch

from scripts.new_method_v2.tuning.source_choice_calibrator import (
    build_candidate_feature_matrix,
    build_candidate_groups,
    build_feature_matrix,
    evaluate_candidate_scores,
    evaluate_source_predictions,
    evaluate_source_predictions_with_source_names,
    train_calibrator,
    _load_rows,
    _load_rows_with_metadata,
    train_candidate_listwise_calibrator,
)
from scripts.new_method_v2.tuning.candidate_dump_to_source_rows import (
    _convert_group,
)
from scripts.new_method_v2.tuning.candidate_xgb_ranker import (
    build_ranker_training_data,
)


class TestSourceChoiceCalibrator(unittest.TestCase):
    def test_build_feature_matrix_excludes_oracle_and_iou_leakage(self):
        rows = [
            {
                "base_top_score": 0.9,
                "base_top_margin": 0.2,
                "base_top_iou": 0.1,
                "quality_top_score": 0.8,
                "oracle_source_id": 2.0,
                "oracle_iou": 0.7,
                "oracle_hit025": 1.0,
                "threshold_utility_source_id": 2.0,
                "source_names": ["base", "quality"],
            },
            {
                "base_top_score": 0.1,
                "base_top_margin": 0.0,
                "base_top_iou": 0.9,
                "quality_top_score": 0.2,
                "oracle_source_id": 0.0,
                "oracle_iou": 0.9,
                "oracle_hit025": 1.0,
                "threshold_utility_source_id": 0.0,
                "source_names": ["base", "quality"],
            },
        ]

        x, y, columns = build_feature_matrix(rows)

        self.assertEqual(x.shape, (2, 3))
        self.assertEqual(y.tolist(), [2, 0])
        self.assertEqual(
            columns,
            ["base_top_margin", "base_top_score", "quality_top_score"],
        )

    def test_build_feature_matrix_excludes_dynamic_source_id_labels(self):
        rows = [
            {
                "base_top_score": 0.9,
                "quality_top_score": 0.8,
                "oracle3_source_id": 2.0,
                "fallback_source_id": 1.0,
            },
            {
                "base_top_score": 0.1,
                "quality_top_score": 0.2,
                "oracle3_source_id": 0.0,
                "fallback_source_id": 0.0,
            },
        ]

        x, y, columns = build_feature_matrix(
            rows, label_key="oracle3_source_id"
        )

        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(y.tolist(), [2, 0])
        self.assertEqual(columns, ["base_top_score", "quality_top_score"])

    def test_evaluate_source_predictions_uses_predicted_source_iou(self):
        rows = [
            {
                "base_top_iou": 0.10,
                "fused_top_iou": 0.60,
                "quality_top_iou": 0.20,
                "contrastive_base_top_iou": 0.00,
            },
            {
                "base_top_iou": 0.70,
                "fused_top_iou": 0.20,
                "quality_top_iou": 0.40,
                "contrastive_base_top_iou": 0.00,
            },
        ]

        metrics = evaluate_source_predictions(rows, [1, 0])

        self.assertEqual(metrics["count"], 2)
        self.assertAlmostEqual(metrics["acc025"], 1.0)
        self.assertAlmostEqual(metrics["acc050"], 1.0)
        self.assertAlmostEqual(metrics["mean_iou"], 0.65)

    def test_evaluate_source_predictions_supports_acd_source(self):
        rows = [
            {
                "base_top_iou": 0.10,
                "fused_top_iou": 0.20,
                "quality_top_iou": 0.30,
                "contrastive_base_top_iou": 0.40,
                "acd_top_iou": 0.90,
            },
            {
                "base_top_iou": 0.40,
                "fused_top_iou": 0.20,
                "quality_top_iou": 0.40,
                "contrastive_base_top_iou": 0.00,
                "acd_top_iou": 0.10,
            },
        ]

        metrics = evaluate_source_predictions(rows, [4, 0])

        self.assertEqual(metrics["count"], 2)
        self.assertAlmostEqual(metrics["acc025"], 1.0)
        self.assertAlmostEqual(metrics["acc050"], 0.5)
        self.assertAlmostEqual(metrics["mean_iou"], 0.65)

    def test_evaluate_source_predictions_supports_custom_source_names(self):
        rows = [
            {
                "base_top_iou": 0.10,
                "quality_top_iou": 0.20,
                "detector_jointtight_top_iou": 0.90,
            },
            {
                "base_top_iou": 0.70,
                "quality_top_iou": 0.20,
                "detector_jointtight_top_iou": 0.10,
            },
        ]

        metrics = evaluate_source_predictions_with_source_names(
            rows,
            [2, 0],
            source_names=("base", "quality", "detector_jointtight"),
        )

        self.assertEqual(metrics["count"], 2)
        self.assertAlmostEqual(metrics["acc025"], 1.0)
        self.assertAlmostEqual(metrics["acc050"], 1.0)
        self.assertAlmostEqual(metrics["mean_iou"], 0.80)

    def test_candidate_feature_matrix_excludes_group_and_label_leakage(self):
        rows = [
            {
                "example_id": 0.0,
                "candidate_query": 2.0,
                "base_score": 0.7,
                "candidate_iou": 0.9,
                "oracle_candidate": 1.0,
                "candidate_hit025": 1.0,
                "candidate_hit050": 1.0,
                "threshold_utility": 2.9,
            },
            {
                "example_id": 0.0,
                "candidate_query": 1.0,
                "base_score": 0.8,
                "candidate_iou": 0.1,
                "oracle_candidate": 0.0,
                "candidate_hit025": 0.0,
                "candidate_hit050": 0.0,
                "threshold_utility": 0.1,
            },
        ]

        x, y, columns = build_candidate_feature_matrix(rows)

        self.assertEqual(x.shape, (2, 1))
        self.assertEqual(y.tolist(), [1, 0])
        self.assertEqual(columns, ["base_score"])

    def test_evaluate_candidate_scores_selects_best_candidate_per_example(self):
        rows = [
            {
                "example_id": 0.0,
                "candidate_query": 0.0,
                "candidate_iou": 0.10,
            },
            {
                "example_id": 0.0,
                "candidate_query": 1.0,
                "candidate_iou": 0.70,
            },
            {
                "example_id": 1.0,
                "candidate_query": 0.0,
                "candidate_iou": 0.60,
            },
            {
                "example_id": 1.0,
                "candidate_query": 1.0,
                "candidate_iou": 0.20,
            },
        ]

        metrics = evaluate_candidate_scores(rows, [0.1, 0.9, 0.8, 0.2])

        self.assertEqual(metrics["count"], 2)
        self.assertAlmostEqual(metrics["acc025"], 1.0)
        self.assertAlmostEqual(metrics["acc050"], 1.0)
        self.assertAlmostEqual(metrics["mean_iou"], 0.65)

    def test_candidate_groups_track_target_offset_per_example(self):
        rows = [
            {"example_id": 0.0, "oracle_candidate": 0.0},
            {"example_id": 0.0, "oracle_candidate": 1.0},
            {"example_id": 1.0, "oracle_candidate": 1.0},
            {"example_id": 1.0, "oracle_candidate": 0.0},
        ]

        groups = build_candidate_groups(rows)

        self.assertEqual(
            [
                (group.example_id, group.indices, group.target_offset)
                for group in groups
            ],
            [
                (0.0, [0, 1], 1),
                (1.0, [2, 3], 0),
            ],
        )

    def test_ranker_training_data_uses_iou_labels_and_group_sizes(self):
        rows = [
            {
                "example_id": 0.0,
                "candidate_query": 0.0,
                "base_score": 0.2,
                "candidate_iou": 0.1,
                "oracle_candidate": 0.0,
            },
            {
                "example_id": 0.0,
                "candidate_query": 1.0,
                "base_score": 0.8,
                "candidate_iou": 0.7,
                "oracle_candidate": 1.0,
            },
            {
                "example_id": 1.0,
                "candidate_query": 0.0,
                "base_score": 0.6,
                "candidate_iou": 0.4,
                "oracle_candidate": 1.0,
            },
        ]

        x, y, columns, group_sizes = build_ranker_training_data(
            rows, label_key="candidate_iou"
        )

        self.assertEqual(x.shape, (3, 1))
        self.assertEqual(len(y), 3)
        self.assertAlmostEqual(float(y[0]), 0.1)
        self.assertAlmostEqual(float(y[1]), 0.7)
        self.assertAlmostEqual(float(y[2]), 0.4)
        self.assertEqual(columns, ["base_score"])
        self.assertEqual(group_sizes, [2, 1])

    def test_listwise_candidate_calibrator_learns_group_choice(self):
        train_rows = []
        for example_id in range(12):
            train_rows.extend(
                [
                    {
                        "example_id": float(example_id),
                        "candidate_query": 0.0,
                        "base_score": 0.1,
                        "candidate_iou": 0.1,
                        "candidate_hit025": 0.0,
                        "candidate_hit050": 0.0,
                        "threshold_utility": 0.1,
                        "oracle_candidate": 0.0,
                    },
                    {
                        "example_id": float(example_id),
                        "candidate_query": 1.0,
                        "base_score": 0.9,
                        "candidate_iou": 0.8,
                        "candidate_hit025": 1.0,
                        "candidate_hit050": 1.0,
                        "threshold_utility": 2.8,
                        "oracle_candidate": 1.0,
                    },
                ]
            )

        _, metrics = train_candidate_listwise_calibrator(
            train_rows,
            train_rows,
            hidden_dim=8,
            epochs=80,
            lr=0.05,
            batch_size=4,
            seed=0,
        )

        self.assertEqual(metrics["row_type"], "candidate_listwise")
        self.assertAlmostEqual(metrics["train"]["acc025"], 1.0)
        self.assertAlmostEqual(metrics["train"]["acc050"], 1.0)
        self.assertAlmostEqual(metrics["val"]["acc025"], 1.0)
        self.assertAlmostEqual(metrics["val_oracle"]["acc050"], 1.0)

    def test_load_rows_supports_sharded_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shard1 = os.path.join(tmpdir, "rows_000000.pt")
            shard2 = os.path.join(tmpdir, "rows_000001.pt")
            manifest = os.path.join(tmpdir, "dump.pt")

            torch.save({"rows": [{"a": 1}, {"a": 2}]}, shard1)
            torch.save({"rows": [{"a": 3}]}, shard2)
            torch.save(
                {
                    "format": "source_choice_feature_dump_sharded_v1",
                    "row_count": 3,
                    "shards": [
                        os.path.basename(shard1),
                        os.path.basename(shard2),
                    ],
                },
                manifest,
            )

            rows = _load_rows(manifest)

        self.assertEqual([row["a"] for row in rows], [1, 2, 3])

    def test_sharded_manifest_metadata_drives_source_name_resolution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            shard = os.path.join(tmpdir, "rows_000000.pt")
            manifest = os.path.join(tmpdir, "dump.pt")
            rows = [
                {
                    "base_top_score": 0.1,
                    "quality_top_score": 0.2,
                    "detector_jointtight_top_score": 0.9,
                    "base_top_iou": 0.1,
                    "quality_top_iou": 0.2,
                    "detector_jointtight_top_iou": 0.9,
                    "oracle_source_id": 2.0,
                },
                {
                    "base_top_score": 0.9,
                    "quality_top_score": 0.2,
                    "detector_jointtight_top_score": 0.1,
                    "base_top_iou": 0.8,
                    "quality_top_iou": 0.2,
                    "detector_jointtight_top_iou": 0.1,
                    "oracle_source_id": 0.0,
                },
            ]
            torch.save({"rows": rows}, shard)
            torch.save(
                {
                    "format": "source_choice_feature_dump_sharded_v1",
                    "row_count": 2,
                    "shards": [os.path.basename(shard)],
                    "source_names": [
                        "base",
                        "quality",
                        "detector_jointtight",
                    ],
                },
                manifest,
            )

            loaded_rows, metadata = _load_rows_with_metadata(manifest)
            _, metrics = train_calibrator(
                loaded_rows,
                loaded_rows,
                metadata=metadata,
            )

        self.assertEqual(
            tuple(metadata["source_names"]),
            ("base", "quality", "detector_jointtight"),
        )
        self.assertEqual(
            metrics["source_names"],
            ["base", "quality", "detector_jointtight"],
        )
        self.assertAlmostEqual(metrics["val_oracle"]["acc050"], 1.0)

    def test_convert_group_supports_custom_source_names(self):
        group_rows = [
            {
                "example_id": 0.0,
                "candidate_query": 0.0,
                "candidate_iou": 0.10,
                "base_available": 1.0,
                "base_score": 0.30,
                "base_rank": 2.0,
                "base_delta_to_top": 0.10,
                "quality_available": 1.0,
                "quality_score": 0.40,
                "quality_rank": 1.0,
                "quality_delta_to_top": 0.00,
                "detector_jointtight_available": 1.0,
                "detector_jointtight_score": 0.20,
                "detector_jointtight_rank": 3.0,
                "detector_jointtight_delta_to_top": 0.20,
            },
            {
                "example_id": 0.0,
                "candidate_query": 1.0,
                "candidate_iou": 0.80,
                "base_available": 1.0,
                "base_score": 0.90,
                "base_rank": 1.0,
                "base_delta_to_top": 0.00,
                "quality_available": 1.0,
                "quality_score": 0.10,
                "quality_rank": 2.0,
                "quality_delta_to_top": 0.80,
                "detector_jointtight_available": 1.0,
                "detector_jointtight_score": 0.95,
                "detector_jointtight_rank": 1.0,
                "detector_jointtight_delta_to_top": 0.00,
            },
        ]

        row = _convert_group(
            0.0,
            group_rows,
            ("base", "quality", "detector_jointtight"),
        )

        self.assertAlmostEqual(row["base_top_query"], 1.0)
        self.assertAlmostEqual(row["quality_top_query"], 0.0)
        self.assertAlmostEqual(row["detector_jointtight_top_query"], 1.0)
        self.assertAlmostEqual(row["base_quality_same_query"], 0.0)
        self.assertAlmostEqual(row["base_detector_jointtight_same_query"], 1.0)


if __name__ == "__main__":
    unittest.main()
