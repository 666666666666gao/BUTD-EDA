import sys
import unittest
from unittest import mock

from main_utils import parse_option


class TestQualityTopkRerankConfig(unittest.TestCase):
    def test_parser_accepts_quality_topk_rerank_options(self):
        argv = [
            "train_dist_mod.py",
            "--use_quality_head",
            "--use_structured_slots",
            "--use_sacr",
            "--use_rapf",
            "--quality_topk_rerank_weight",
            "0.05",
            "--quality_topk_rerank_source",
            "fused",
            "--quality_topk_rerank_k",
            "5",
            "--quality_topk_rerank_margin",
            "0.07",
            "--quality_topk_rerank_min_iou_gap",
            "0.03",
            "--quality_topk_rerank_use_logits",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(args.quality_topk_rerank_weight, 0.05)
        self.assertEqual(args.quality_topk_rerank_source, "fused")
        self.assertEqual(args.quality_topk_rerank_k, 5)
        self.assertEqual(args.quality_topk_rerank_margin, 0.07)
        self.assertEqual(args.quality_topk_rerank_min_iou_gap, 0.03)
        self.assertTrue(args.quality_topk_rerank_use_logits)

    def test_parser_accepts_source_pool_quality_topk_rerank_source(self):
        argv = [
            "train_dist_mod.py",
            "--use_quality_head",
            "--quality_topk_rerank_weight",
            "0.05",
            "--quality_topk_rerank_source",
            "source_pool",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(args.quality_topk_rerank_source, "source_pool")

    def test_parser_accepts_external_pool_quality_topk_rerank_source(self):
        argv = [
            "train_dist_mod.py",
            "--use_quality_head",
            "--quality_topk_rerank_weight",
            "0.05",
            "--quality_topk_rerank_source",
            "external_pool",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_option()

        self.assertEqual(args.quality_topk_rerank_source, "external_pool")

    def test_positive_weight_requires_quality_head(self):
        argv = [
            "train_dist_mod.py",
            "--quality_topk_rerank_weight",
            "0.05",
        ]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(ValueError, "requires --use_quality_head"):
                parse_option()


if __name__ == "__main__":
    unittest.main()
