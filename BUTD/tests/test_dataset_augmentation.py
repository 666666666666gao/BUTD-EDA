import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class DatasetAugmentationTest(unittest.TestCase):
    def test_detected_boxes_follow_point_cloud_flip_before_rotation_order(self):
        from data.model_util_scannet import ScannetDatasetConfig
        from src.joint_det_dataset import (
            Joint3DDataset,
            box2points,
            points2box,
            rot_x,
            rot_y,
            rot_z,
        )

        center = np.array([[1.0, 2.0, 3.0]])
        size = np.array([[1.2, 1.4, 1.6]])
        minmax_box = np.concatenate((center - size / 2.0, center + size / 2.0), axis=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            pred_dir = (
                Path(tmpdir)
                / "group_free_pred_bboxes"
                / "group_free_pred_bboxes_train"
            )
            pred_dir.mkdir(parents=True)
            np.save(
                pred_dir / "scene001.npy",
                {
                    "box": minmax_box,
                    "class": ["chair"],
                    "logits": np.zeros((1, 485), dtype=np.float32),
                },
            )

            dataset = Joint3DDataset.__new__(Joint3DDataset)
            dataset.butd = True
            dataset.butd_cls = False
            dataset.augment = True
            dataset.augment_det = False
            dataset.split = "train"
            dataset.data_path = f"{tmpdir}/"
            dataset.label_map = {"chair": next(iter(ScannetDatasetConfig(485).nyu40id2class))}

            augmentations = {
                "theta_z": 37.0,
                "theta_x": -4.0,
                "theta_y": 9.0,
                "yz_flip": True,
                "xz_flip": True,
                "shift": np.array([[0.25, -0.5, 0.75]]),
                "scale": 1.03,
            }

            detected_boxes, detected_mask, _, _, _ = dataset._get_detected_objects(
                "train", "scene001", augmentations
            )

            expected_points = box2points(np.concatenate((center, size), axis=1)).reshape(-1, 3)
            expected_points[:, 0] = -expected_points[:, 0]
            expected_points[:, 1] = -expected_points[:, 1]
            expected_points = rot_z(expected_points, augmentations["theta_z"])
            expected_points = rot_x(expected_points, augmentations["theta_x"])
            expected_points = rot_y(expected_points, augmentations["theta_y"])
            expected_points += augmentations["shift"]
            expected_points *= augmentations["scale"]
            expected_box = points2box(expected_points.reshape(-1, 8, 3))[0]

            self.assertTrue(detected_mask[0])
            np.testing.assert_allclose(detected_boxes[0], expected_box, rtol=1e-6, atol=1e-6)

    def test_detected_box_augmentation_keeps_padding_boxes_zero(self):
        from data.model_util_scannet import ScannetDatasetConfig
        from src.joint_det_dataset import Joint3DDataset

        center = np.array([[1.0, 2.0, 3.0]])
        size = np.array([[1.2, 1.4, 1.6]])
        minmax_box = np.concatenate((center - size / 2.0, center + size / 2.0), axis=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            pred_dir = (
                Path(tmpdir)
                / "group_free_pred_bboxes"
                / "group_free_pred_bboxes_train"
            )
            pred_dir.mkdir(parents=True)
            np.save(
                pred_dir / "scene001.npy",
                {
                    "box": minmax_box,
                    "class": ["chair"],
                    "logits": np.zeros((1, 485), dtype=np.float32),
                },
            )

            dataset = Joint3DDataset.__new__(Joint3DDataset)
            dataset.butd = True
            dataset.butd_cls = False
            dataset.augment = True
            dataset.augment_det = True
            dataset.split = "train"
            dataset.data_path = f"{tmpdir}/"
            dataset.label_map = {"chair": next(iter(ScannetDatasetConfig(485).nyu40id2class))}

            augmentations = {
                "theta_z": 0.0,
                "theta_x": 0.0,
                "theta_y": 0.0,
                "yz_flip": False,
                "xz_flip": False,
                "shift": np.array([[0.25, -0.5, 0.75]]),
                "scale": 1.0,
            }

            def deterministic_random(shape):
                if shape == all_detected_shape:
                    return np.zeros(shape)
                if shape == all_detected_len:
                    return np.ones(shape)
                return np.zeros(shape)

            all_detected_shape = (132, 6)
            all_detected_len = 132
            with mock.patch("numpy.random.random", side_effect=deterministic_random):
                detected_boxes, detected_mask, detected_class_ids, _, _ = dataset._get_detected_objects(
                    "train", "scene001", augmentations
                )

            self.assertTrue(detected_mask[0])
            self.assertFalse(detected_mask[1:].any())
            np.testing.assert_allclose(detected_boxes[1:], 0.0, rtol=0, atol=0)
            np.testing.assert_array_equal(detected_class_ids[1:], 0)

    def test_disable_box_jitter_keeps_training_target_boxes_exact(self):
        from src.joint_det_dataset import Joint3DDataset

        class FakeScan:
            pc = np.zeros((4, 3), dtype=np.float32)
            three_d_objects = [{"points": np.array([0, 1]), "instance_label": "chair"}]

            def get_object_bbox(self, object_id):
                assert object_id == 0
                return np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]], dtype=np.float32)

        dataset = Joint3DDataset.__new__(Joint3DDataset)
        dataset.split = "train"
        dataset.augment = True
        dataset.detect_intermediate = False
        dataset.disable_box_jitter = True

        bboxes, box_mask, point_labels = dataset._get_target_boxes(
            {"target_id": 0, "distractor_ids": [], "anchor_ids": []},
            FakeScan(),
        )

        np.testing.assert_allclose(bboxes[0], np.array([1.0, 2.0, 3.0, 2.0, 4.0, 6.0]))
        self.assertEqual(box_mask[0], 1)
        np.testing.assert_array_equal(point_labels[:2], np.array([0, 0]))

    def test_box_jitter_still_runs_by_default(self):
        from src.joint_det_dataset import Joint3DDataset

        class FakeScan:
            pc = np.zeros((4, 3), dtype=np.float32)
            three_d_objects = [{"points": np.array([0, 1]), "instance_label": "chair"}]

            def get_object_bbox(self, object_id):
                assert object_id == 0
                return np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]], dtype=np.float32)

        dataset = Joint3DDataset.__new__(Joint3DDataset)
        dataset.split = "train"
        dataset.augment = True
        dataset.detect_intermediate = False
        dataset.disable_box_jitter = False

        with mock.patch("numpy.random.random", return_value=np.ones((1, 6))):
            bboxes, _, _ = dataset._get_target_boxes(
                {"target_id": 0, "distractor_ids": [], "anchor_ids": []},
                FakeScan(),
            )

        np.testing.assert_allclose(bboxes[0], np.array([1.05, 2.1, 3.15, 2.1, 4.2, 6.3]))

    def test_height_feature_is_computed_after_point_cloud_augmentation(self):
        from src.joint_det_dataset import Joint3DDataset

        class FakeScan:
            pass

        dataset = Joint3DDataset.__new__(Joint3DDataset)
        dataset.use_color = False
        dataset.use_height = True
        dataset.use_multiview = False
        dataset.split = "train"
        dataset.augment = True

        scan = FakeScan()
        scan.pc = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 2.0],
            ],
            dtype=np.float32,
        )
        scan.color = np.zeros((3, 3), dtype=np.float32)

        def scale_z(pc, color, rotate):
            augmented = pc.copy()
            augmented[:, 2] *= 2.0
            return augmented, color, {"scale": 1.0}

        dataset._augment = scale_z

        point_cloud, _, _ = dataset._get_pc(
            {"scan_id": "scene001", "dataset": "scanrefer", "utterance": "chair"},
            scan,
        )

        expected_height = point_cloud[:, 2] - np.percentile(point_cloud[:, 2], 0.99)
        np.testing.assert_allclose(point_cloud[:, 3], expected_height, rtol=1e-6, atol=1e-6)

    def test_disable_train_augmentation_skips_point_cloud_augmentation(self):
        from src.joint_det_dataset import Joint3DDataset

        class FakeScan:
            pass

        dataset = Joint3DDataset.__new__(Joint3DDataset)
        dataset.use_color = False
        dataset.use_height = False
        dataset.use_multiview = False
        dataset.split = "train"
        dataset.augment = True
        dataset.disable_train_augmentation = True

        scan = FakeScan()
        scan.pc = np.array(
            [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
            dtype=np.float32,
        )
        scan.color = np.zeros((2, 3), dtype=np.float32)

        def fail_if_called(pc, color, rotate):
            raise AssertionError("_augment should not run")

        dataset._augment = fail_if_called

        point_cloud, augmentations, _ = dataset._get_pc(
            {"scan_id": "scene001", "dataset": "scanrefer", "utterance": "chair"},
            scan,
        )

        np.testing.assert_allclose(point_cloud, scan.pc)
        self.assertEqual(augmentations, {})

    def test_disable_train_augmentation_clears_box_jitter_diagnostics(self):
        from src.joint_det_dataset import Joint3DDataset

        dataset = Joint3DDataset.__new__(Joint3DDataset)
        dataset.split = "train"
        dataset.augment = True
        dataset.disable_box_jitter = False
        dataset.disable_train_augmentation = True

        diagnostics = dataset._data_augmentation_diagnostics(
            {
                "dataset": "scanrefer_spacy",
                "utterance": "the chair left of the table",
                "rel_slots": [{"text": "left of"}],
            },
            {},
            augment_det_effective=False,
            augment_det_corrupt_count=0,
        )

        self.assertEqual(diagnostics["dbg_data_aug_box_jitter_active_ratio"], 0.0)
        self.assertEqual(
            diagnostics["dbg_data_spacy_direction_sensitive_box_jitter_ratio"],
            0.0,
        )

    def test_spacy_relation_slots_use_no_rotation_mode(self):
        from src.joint_det_dataset import Joint3DDataset

        class FakeScan:
            pass

        dataset = Joint3DDataset.__new__(Joint3DDataset)
        dataset.use_color = False
        dataset.use_height = False
        dataset.use_multiview = False
        dataset.split = "train"
        dataset.augment = True

        rotate_flags = []

        def record_rotate(pc, color, rotate):
            rotate_flags.append(rotate)
            return pc, color, {"scale": 1.0}

        dataset._augment = record_rotate

        scan = FakeScan()
        scan.pc = np.zeros((3, 3), dtype=np.float32)
        scan.color = np.zeros((3, 3), dtype=np.float32)

        dataset._get_pc(
            {
                "scan_id": "scene001",
                "dataset": "scanrefer_spacy",
                "utterance": "plain wooden chair near table",
                "rel_slots": [{"text": "left of", "start": 19, "end": 23}],
            },
            scan,
        )

        self.assertEqual(rotate_flags, ["none"])

    def test_spacy_non_view_relation_slots_use_yaw_only_mode(self):
        from src.joint_det_dataset import Joint3DDataset

        class FakeScan:
            pass

        dataset = Joint3DDataset.__new__(Joint3DDataset)
        dataset.use_color = False
        dataset.use_height = False
        dataset.use_multiview = False
        dataset.split = "train"
        dataset.augment = True

        rotate_flags = []

        def record_rotate(pc, color, rotate):
            rotate_flags.append(rotate)
            return pc, color, {"scale": 1.0}

        dataset._augment = record_rotate

        scan = FakeScan()
        scan.pc = np.zeros((3, 3), dtype=np.float32)
        scan.color = np.zeros((3, 3), dtype=np.float32)

        dataset._get_pc(
            {
                "scan_id": "scene001",
                "dataset": "scanrefer_spacy",
                "utterance": "the chair is above the table",
                "rel_slots": [{"text": "above", "start": 13, "end": 18}],
            },
            scan,
        )

        self.assertEqual(rotate_flags, ["yaw_only"])

    def test_spacy_relation_free_yaw_only_flag_routes_full_natural_samples(self):
        from src.joint_det_dataset import Joint3DDataset

        class FakeScan:
            pass

        dataset = Joint3DDataset.__new__(Joint3DDataset)
        dataset.use_color = False
        dataset.use_height = False
        dataset.use_multiview = False
        dataset.split = "train"
        dataset.augment = True
        dataset.spacy_relation_free_yaw_only_aug = True

        rotate_flags = []

        def record_rotate(pc, color, rotate):
            rotate_flags.append(rotate)
            return pc, color, {"scale": 1.0}

        dataset._augment = record_rotate

        scan = FakeScan()
        scan.pc = np.zeros((3, 3), dtype=np.float32)
        scan.color = np.zeros((3, 3), dtype=np.float32)

        dataset._get_pc(
            {
                "scan_id": "scene001",
                "dataset": "scanrefer_spacy",
                "utterance": "plain wooden chair near table",
                "rel_slots": [],
            },
            scan,
        )

        self.assertEqual(rotate_flags, ["yaw_only"])

    def test_spacy_augmentation_profile_ids_separate_relation_free_yaw(self):
        from src.joint_det_dataset import Joint3DDataset

        none_anno = {
            "dataset": "scanrefer_spacy",
            "utterance": "the chair is behind the table",
            "rel_slots": [],
        }
        relation_yaw_anno = {
            "dataset": "scanrefer_spacy",
            "utterance": "the chair is above the table",
            "rel_slots": [{"text": "above"}],
        }
        natural_anno = {
            "dataset": "scanrefer_spacy",
            "utterance": "plain wooden chair near table",
            "rel_slots": [],
        }
        relation_free_view_anno = {
            "dataset": "scanrefer_spacy",
            "utterance": "plain wooden chair to the right.",
            "rel_slots": [],
        }

        dataset = Joint3DDataset.__new__(Joint3DDataset)
        dataset.spacy_relation_free_yaw_only_aug = True
        dataset.spacy_relation_free_view_guard_aug = True

        self.assertEqual(Joint3DDataset._spacy_rotation_mode_id(none_anno), 1)
        self.assertEqual(Joint3DDataset._spacy_rotation_mode_id(relation_yaw_anno), 2)
        self.assertEqual(Joint3DDataset._spacy_rotation_mode_id(natural_anno), 3)
        self.assertEqual(dataset._spacy_augmentation_profile_id(none_anno), 1)
        self.assertEqual(dataset._spacy_augmentation_profile_id(relation_yaw_anno), 2)
        self.assertEqual(dataset._spacy_augmentation_profile_id(natural_anno), 4)
        self.assertEqual(dataset._spacy_augmentation_profile_id(relation_free_view_anno), 5)

    def test_spacy_relation_free_rawview_global_only_train_masks_noisy_samples(self):
        from src.joint_det_dataset import Joint3DDataset

        dataset = Joint3DDataset.__new__(Joint3DDataset)
        dataset.split = "train"
        dataset.spacy_relation_free_yaw_only_aug = True
        dataset.spacy_relation_free_rawview_global_only_train = True
        dataset.spacy_relation_free_view_guard_aug = False
        dataset.spacy_relation_free_view_small_yaw_aug = False
        dataset.spacy_relation_free_stable_yaw_aug = False

        noisy_anno = {
            "dataset": "scanrefer_spacy",
            "utterance": "the chair is second from the right.",
            "rel_slots": [],
            "coverage_stats": {"candidate_relation_count": 1},
        }
        plain_anno = {
            "dataset": "scanrefer_spacy",
            "utterance": "plain wooden chair near table",
            "rel_slots": [],
            "coverage_stats": {"candidate_relation_count": 1},
        }

        self.assertTrue(dataset._decomp_global_only_mask_for_sample(noisy_anno))
        self.assertFalse(dataset._decomp_global_only_mask_for_sample(plain_anno))
        self.assertEqual(dataset._spacy_augmentation_profile_id(noisy_anno), 8)

    def test_spacy_direction_sensitive_no_jitter_routes_only_sensitive_samples(self):
        from src.joint_det_dataset import Joint3DDataset

        dataset = Joint3DDataset.__new__(Joint3DDataset)
        dataset.split = "train"
        dataset.spacy_relation_free_yaw_only_aug = True
        dataset.spacy_relation_free_view_guard_aug = True
        dataset.spacy_relation_free_compass_guard_aug = True
        dataset.spacy_relation_free_view_small_yaw_aug = False
        dataset.spacy_relation_free_stable_yaw_aug = False
        dataset.spacy_relation_free_none_aug = False
        dataset.spacy_direction_sensitive_no_jitter_aug = True

        relation_anno = {
            "dataset": "scanrefer_spacy",
            "utterance": "the chair is above the table",
            "rel_slots": [{"text": "above"}],
        }
        raw_view_anno = {
            "dataset": "scanrefer_spacy",
            "utterance": "the chair on the right side",
            "rel_slots": [],
        }
        compass_anno = {
            "dataset": "scanrefer_spacy",
            "utterance": "the chair near the north wall",
            "rel_slots": [],
        }
        plain_relation_free_anno = {
            "dataset": "scanrefer_spacy",
            "utterance": "plain wooden chair near table",
            "rel_slots": [],
        }

        self.assertEqual(
            dataset._spacy_rotation_mode_for_sample(relation_anno),
            "yaw_stable",
        )
        self.assertEqual(
            dataset._spacy_rotation_mode_for_sample(raw_view_anno),
            "none_stable",
        )
        self.assertEqual(
            dataset._spacy_rotation_mode_for_sample(compass_anno),
            "none_stable",
        )
        self.assertEqual(
            dataset._spacy_rotation_mode_for_sample(plain_relation_free_anno),
            "yaw_only",
        )

    def test_spacy_direction_sensitive_no_jitter_parser_flag_is_available(self):
        from main_utils import parse_option

        old_argv = sys.argv[:]
        try:
            sys.argv = [
                "train_dist_mod.py",
                "--spacy_direction_sensitive_no_jitter_aug",
            ]
            args = parse_option()
        finally:
            sys.argv = old_argv

        self.assertTrue(args.spacy_direction_sensitive_no_jitter_aug)

    def test_scanrefer_inject_spacy_decomp_parser_flag_is_available(self):
        from main_utils import parse_option

        old_argv = sys.argv[:]
        try:
            sys.argv = [
                "train_dist_mod.py",
                "--scanrefer_inject_spacy_decomp",
            ]
            args = parse_option()
        finally:
            sys.argv = old_argv

        self.assertTrue(args.scanrefer_inject_spacy_decomp)

    def test_eval_target_cid_source_parser_flag_is_available(self):
        from main_utils import parse_option

        old_argv = sys.argv[:]
        try:
            sys.argv = [
                "train_dist_mod.py",
                "--eval_target_cid_source",
                "text",
            ]
            args = parse_option()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.eval_target_cid_source, "text")

    def test_text_target_alias_policy_parser_flag_is_available(self):
        from main_utils import parse_option

        old_argv = sys.argv[:]
        try:
            sys.argv = [
                "train_dist_mod.py",
                "--text_target_alias_policy",
                "run325_no_computer",
            ]
            args = parse_option()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.text_target_alias_policy, "run325_no_computer")

    def test_scanrefer_inject_spacy_decomp_keeps_raw_identity(self):
        from src.joint_det_dataset import Joint3DDataset

        class FakeScan:
            three_d_objects = [
                {"instance_label": "chair"},
                {"instance_label": "chair"},
                {"instance_label": "table"},
            ]

            def get_object_instance_label(self, object_id):
                return self.three_d_objects[object_id]["instance_label"]

        raw_anno = {
            "scene_id": "scene001",
            "object_id": "0",
            "ann_id": "7",
            "object_name": "office_chair",
            "description": "the office chair left of the table",
            "token": ["the", "office", "chair", "left", "of", "the", "table"],
        }
        refined_anno = {
            **raw_anno,
            "tokens": raw_anno["token"],
            "target_slot": {"start": 4, "end": 16, "text": "office chair"},
            "entities": [{"start": 29, "end": 34, "text": "table"}],
            "attr_slot": {"items": [{"start": 4, "end": 10, "text": "office"}]},
            "rel_slots": [
                {
                    "start": 17,
                    "end": 24,
                    "text": "left of",
                    "head": "chair",
                    "tail": "table",
                }
            ],
            "anchor_slots": [{"start": 29, "end": 34, "text": "table"}],
            "coverage_stats": {"decomposition_status": "ok"},
            "parse_confidence": 0.82,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            scanrefer_dir = Path(tmpdir) / "scanrefer"
            scanrefer_dir.mkdir()
            (scanrefer_dir / "ScanRefer_filtered_train.txt").write_text(
                "scene001\n"
            )
            (scanrefer_dir / "ScanRefer_filtered_train.json").write_text(
                __import__("json").dumps([raw_anno])
            )
            (scanrefer_dir / "ScanRefer_filtered_train_spacy_refined.json").write_text(
                __import__("json").dumps([refined_anno])
            )

            dataset = Joint3DDataset.__new__(Joint3DDataset)
            dataset.data_path = f"{tmpdir}/"
            dataset.split = "train"
            dataset.scans = {"scene001": FakeScan()}
            dataset.label_mapclass = {
                "chair": "chair",
                "table": "table",
            }
            dataset.scanrefer_inject_spacy_decomp = True

            annos = dataset.load_scanrefer_annos(dset="scanrefer")

        self.assertEqual(len(annos), 1)
        sample = annos[0]
        self.assertEqual(sample["dataset"], "scanrefer")
        self.assertEqual(sample["utterance"], "the office chair left of the table")
        self.assertEqual(sample["target"], "office chair")
        self.assertEqual(sample["target_slot"], refined_anno["target_slot"])
        self.assertEqual(sample["rel_slots"], refined_anno["rel_slots"])
        self.assertEqual(sample["coverage_stats"]["decomposition_status"], "ok")
        self.assertEqual(sample["parse_confidence"], 0.82)
        self.assertEqual(sample["entity_spans"][0]["text"], "office chair")
        self.assertEqual(sample["rel_spans"][0]["text"], "left of")
        self.assertEqual(sample["anchor_ids"], [1])
        self.assertEqual(sample["distractor_ids"], [1])
        self.assertTrue(sample["unique"])
        self.assertEqual(sample["dbg_data_scanrefer_spacy_injected_ratio"], 1.0)
        self.assertEqual(sample["dbg_data_scanrefer_spacy_inject_hit_ratio"], 1.0)
        self.assertEqual(sample["dbg_data_scanrefer_spacy_inject_miss_ratio"], 0.0)

    def test_scanrefer_inject_spacy_decomp_is_passed_to_train_and_eval_datasets(self):
        import argparse
        import train_dist_mod

        args = argparse.Namespace(
            dataset=["scanrefer"],
            test_dataset="scanrefer",
            joint_det=False,
            debug=False,
            eval_train=False,
            use_color=False,
            use_height=False,
            data_root="/tmp/data",
            detect_intermediate=False,
            use_multiview=False,
            butd=True,
            butd_gt=False,
            butd_cls=False,
            augment_det=False,
            disable_box_jitter=False,
            spacy_relation_free_yaw_only_aug=False,
            spacy_relation_free_view_guard_aug=False,
            spacy_relation_free_stable_yaw_aug=False,
            spacy_relation_free_view_small_yaw_aug=False,
            spacy_relation_free_rawview_global_only_train=False,
            spacy_relation_free_none_aug=False,
            spacy_relation_free_compass_guard_aug=False,
            spacy_direction_sensitive_no_jitter_aug=False,
            scanrefer_inject_spacy_decomp=True,
            text_target_alias_policy="run325_no_computer",
        )
        calls = []

        def fake_dataset(**kwargs):
            calls.append(kwargs)
            return kwargs

        with mock.patch.object(train_dist_mod, "Joint3DDataset", side_effect=fake_dataset):
            train_dataset, test_dataset = train_dist_mod.TrainTester.get_datasets(args)

        self.assertIs(train_dataset, calls[0])
        self.assertIs(test_dataset, calls[1])
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0]["scanrefer_inject_spacy_decomp"])
        self.assertTrue(calls[1]["scanrefer_inject_spacy_decomp"])
        self.assertEqual(calls[0]["text_target_alias_policy"], "run325_no_computer")
        self.assertEqual(calls[1]["text_target_alias_policy"], "run325_no_computer")

    def test_text_target_cid_infers_scanrefer_target_from_target_slot(self):
        from data.model_util_scannet import ScannetDatasetConfig
        from src.joint_det_dataset import Joint3DDataset

        anno = {
            "target_slot": {"text": "bottom kitchen cabinets"},
            "description": "bottom kitchen cabinets under the counter",
            "utterance": "bottom kitchen cabinets under the counter",
            "object_name": "chair",
        }

        cid = Joint3DDataset._text_target_cid_from_annotation(anno)

        self.assertEqual(
            cid,
            ScannetDatasetConfig(485).type2class["kitchen cabinet"],
        )

    def test_text_target_alias_policy_strict_keeps_existing_mapping(self):
        from data.model_util_scannet import ScannetDatasetConfig
        from src.joint_det_dataset import Joint3DDataset

        anno = {"target_slot": {"text": "a black sofa"}}

        cid = Joint3DDataset._text_target_cid_from_annotation(anno)

        self.assertEqual(cid, ScannetDatasetConfig(485).type2class["sofa"])

    def test_text_target_alias_policy_remaps_only_when_requested(self):
        from data.model_util_scannet import ScannetDatasetConfig
        from src.joint_det_dataset import Joint3DDataset

        anno = {"target_slot": {"text": "a black sofa"}}

        cid = Joint3DDataset._text_target_cid_from_annotation(
            anno,
            alias_policy="run325_no_computer",
        )

        self.assertEqual(cid, ScannetDatasetConfig(485).type2class["couch"])

    def test_text_target_cid_does_not_use_object_name_as_fallback(self):
        from src.joint_det_dataset import Joint3DDataset

        anno = {
            "target_slot": {},
            "description": "the one beside the door",
            "utterance": "the one beside the door",
            "object_name": "office_chair",
        }

        self.assertEqual(
            Joint3DDataset._text_target_cid_from_annotation(anno),
            -1,
        )

    def test_eval_max_samples_limits_eval_dataset_only(self):
        import argparse
        import train_dist_mod

        args = argparse.Namespace(
            dataset=["scanrefer_spacy"],
            test_dataset="scanrefer_spacy",
            joint_det=False,
            debug=False,
            eval_train=True,
            eval_max_samples=3,
            use_color=False,
            use_height=False,
            data_root="/tmp/data",
            detect_intermediate=False,
            use_multiview=False,
            butd=True,
            butd_gt=False,
            butd_cls=False,
            augment_det=False,
            disable_box_jitter=False,
            disable_train_augmentation=True,
            spacy_relation_free_yaw_only_aug=False,
            spacy_relation_free_view_guard_aug=False,
            spacy_relation_free_stable_yaw_aug=False,
            spacy_relation_free_view_small_yaw_aug=False,
            spacy_relation_free_rawview_global_only_train=False,
            spacy_relation_free_none_aug=False,
            spacy_relation_free_compass_guard_aug=False,
            spacy_direction_sensitive_no_jitter_aug=False,
            scanrefer_inject_spacy_decomp=False,
        )

        class FakeDataset:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.annos = list(range(10))

        with mock.patch.object(train_dist_mod, "Joint3DDataset", side_effect=FakeDataset):
            train_dataset, test_dataset = train_dist_mod.TrainTester.get_datasets(args)

        self.assertEqual(train_dataset.annos, list(range(10)))
        self.assertEqual(test_dataset.annos, [0, 1, 2])

    def test_eval_max_samples_parser_flag_is_available(self):
        from main_utils import parse_option

        old_argv = sys.argv[:]
        try:
            sys.argv = ["train_dist_mod.py", "--eval_max_samples", "12000"]
            args = parse_option()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.eval_max_samples, 12000)

    def test_stable_spacy_augmentation_modes_suppress_jitter_terms(self):
        from src.joint_det_dataset import Joint3DDataset

        dataset = Joint3DDataset.__new__(Joint3DDataset)
        dataset.mean_rgb = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        pc = np.zeros((4, 3), dtype=np.float32)
        color = np.ones((4, 3), dtype=np.float32)

        for mode in ("none_stable", "yaw_stable"):
            aug_pc, aug_color, augmentations = dataset._augment(
                pc.copy(), color.copy(), mode
            )

            self.assertAlmostEqual(
                Joint3DDataset._augmentation_shift_l2(augmentations), 0.0
            )
            self.assertAlmostEqual(augmentations["scale"], 1.0)
            self.assertAlmostEqual(
                Joint3DDataset._augmentation_noise_abs_mean(augmentations), 0.0
            )
            self.assertAlmostEqual(
                augmentations["color_scale_delta_abs_mean"], 0.0
            )
            np.testing.assert_allclose(aug_color, color)

    def test_getitem_reports_effective_decomposition_masks(self):
        import src.joint_det_dataset as dataset_module
        from src.joint_det_dataset import Joint3DDataset

        class FakeScan:
            orig_pc = np.zeros((4, 3), dtype=np.float32)
            pc = orig_pc.copy()
            color = np.zeros_like(orig_pc)
            three_d_objects = [{"points": np.array([0, 1]), "instance_label": "chair"}]

            def get_object_instance_label(self, object_id):
                return self.three_d_objects[object_id]["instance_label"]

        dataset = Joint3DDataset.__new__(Joint3DDataset)
        dataset.split = "train"
        dataset.test_dataset = "scanrefer_spacy"
        dataset.scans = {"scene001": FakeScan()}
        dataset.annos = [{
            "scan_id": "scene001",
            "dataset": "scanrefer_spacy",
            "utterance": "the chair is second from the right.",
            "target_id": 0,
            "distractor_ids": [],
            "anchor_ids": [],
            "rel_slots": [],
            "coverage_stats": {"candidate_relation_count": 1},
            "decomposition_status": "ok",
        }]
        dataset.visualize = False
        dataset.butd_gt = False
        dataset.butd_cls = False
        dataset.spacy_relation_free_yaw_only_aug = False
        dataset.spacy_relation_free_rawview_global_only_train = True

        max_num_obj = dataset_module.MAX_NUM_OBJ
        num_classes = dataset_module.NUM_CLASSES
        zeros_6 = np.zeros((max_num_obj, 6), dtype=np.float32)
        zeros_1 = np.zeros(max_num_obj, dtype=np.float32)
        false_mask = np.zeros(max_num_obj, dtype=bool)
        positive = np.zeros((max_num_obj, 256), dtype=np.float32)
        tokens = np.zeros((max_num_obj, 2), dtype=np.int64)
        positive_diag = {
            "positive_map_target_missing": 0.0,
            "positive_map_fallback_used": 0.0,
            "positive_map_global_only_target_empty": 0.0,
            "positive_map_source_explicit_target_slot": 1.0,
            "positive_map_source_entity_exact_match": 0.0,
            "positive_map_source_lexical_exact_match": 0.0,
            "positive_map_source_missing": 0.0,
            "dbg_warn_global_only_target_positive_map": 0.0,
        }

        dataset._get_pc = lambda anno, scan: (
            scan.pc.copy(), {"scale": 1.0}, scan.color.copy()
        )
        dataset._get_target_boxes = lambda anno, scan: (
            zeros_6.copy(), zeros_1.copy(), -np.ones(len(scan.pc), dtype=np.int64)
        )
        dataset._get_scene_objects = lambda scan: (
            zeros_1.copy(), zeros_6.copy(), np.array([True] + [False] * (max_num_obj - 1))
        )
        dataset._get_auxi_boxes = lambda *args: None
        dataset._get_token_positive_map = lambda anno: (
            tokens.copy(), positive.copy(), positive_diag.copy()
        )
        dataset._get_token_positive_map_by_parse = lambda anno, auxi_box: (
            tokens.copy(), positive.copy(), positive.copy(), positive.copy(),
            positive.copy(), positive.copy(), positive.copy()
        )
        dataset._get_detected_objects = lambda split, scan_id, augmentations: (
            zeros_6.copy(), false_mask.copy(), zeros_1.copy(),
            np.zeros((max_num_obj, num_classes), dtype=np.float32)
        )

        sample = dataset.__getitem__(0)

        self.assertTrue(sample["decomp_global_only_mask"])
        self.assertEqual(sample["dbg_data_decomp_global_only_ratio"], 0.0)
        self.assertEqual(
            sample["dbg_data_decomp_global_only_effective_ratio"], 1.0
        )
        self.assertEqual(
            sample["dbg_data_decomp_weak_generic_effective_ratio"], 0.0
        )

    def test_getitem_reports_detector_augmentation_noise(self):
        import src.joint_det_dataset as dataset_module
        from src.joint_det_dataset import Joint3DDataset

        class FakeScan:
            orig_pc = np.zeros((4, 3), dtype=np.float32)
            pc = orig_pc.copy()
            color = np.zeros_like(orig_pc)
            three_d_objects = [{"points": np.array([0, 1]), "instance_label": "chair"}]

            def get_object_instance_label(self, object_id):
                return self.three_d_objects[object_id]["instance_label"]

        dataset = Joint3DDataset.__new__(Joint3DDataset)
        dataset.split = "train"
        dataset.test_dataset = "scanrefer_spacy"
        dataset.scans = {"scene001": FakeScan()}
        dataset.annos = [{
            "scan_id": "scene001",
            "dataset": "scanrefer_spacy",
            "utterance": "the chair is near the table.",
            "target_id": 0,
            "distractor_ids": [],
            "anchor_ids": [],
            "rel_slots": [],
            "decomposition_status": "ok",
        }]
        dataset.visualize = False
        dataset.butd_gt = False
        dataset.butd_cls = False
        dataset.augment_det = True
        dataset.spacy_relation_free_yaw_only_aug = False
        dataset.spacy_relation_free_rawview_global_only_train = False

        max_num_obj = dataset_module.MAX_NUM_OBJ
        num_classes = dataset_module.NUM_CLASSES
        zeros_6 = np.zeros((max_num_obj, 6), dtype=np.float32)
        zeros_1 = np.zeros(max_num_obj, dtype=np.float32)
        false_mask = np.zeros(max_num_obj, dtype=bool)
        positive = np.zeros((max_num_obj, 256), dtype=np.float32)
        tokens = np.zeros((max_num_obj, 2), dtype=np.int64)
        positive_diag = {
            "positive_map_target_missing": 0.0,
            "positive_map_fallback_used": 0.0,
            "positive_map_global_only_target_empty": 0.0,
            "positive_map_source_explicit_target_slot": 1.0,
            "positive_map_source_entity_exact_match": 0.0,
            "positive_map_source_lexical_exact_match": 0.0,
            "positive_map_source_missing": 0.0,
            "dbg_warn_global_only_target_positive_map": 0.0,
        }

        dataset._get_pc = lambda anno, scan: (
            scan.pc.copy(), {"scale": 1.0}, scan.color.copy()
        )
        dataset._get_target_boxes = lambda anno, scan: (
            zeros_6.copy(), zeros_1.copy(), -np.ones(len(scan.pc), dtype=np.int64)
        )
        dataset._get_scene_objects = lambda scan: (
            zeros_1.copy(), zeros_6.copy(), np.array([True] + [False] * (max_num_obj - 1))
        )
        dataset._get_auxi_boxes = lambda *args: None
        dataset._get_token_positive_map = lambda anno: (
            tokens.copy(), positive.copy(), positive_diag.copy()
        )
        dataset._get_token_positive_map_by_parse = lambda anno, auxi_box: (
            tokens.copy(), positive.copy(), positive.copy(), positive.copy(),
            positive.copy(), positive.copy(), positive.copy()
        )
        dataset._get_detected_objects = lambda split, scan_id, augmentations: (
            zeros_6.copy(),
            false_mask.copy(),
            zeros_1.copy(),
            np.zeros((max_num_obj, num_classes), dtype=np.float32),
            {
                "active": 1.0,
                "applied": 1.0,
                "corrupt_obj_count": 3,
                "valid_obj_count": 10,
                "corrupt_obj_ratio": 0.3,
            },
        )

        sample = dataset.__getitem__(0)

        self.assertEqual(sample["dbg_data_augment_det_active_ratio"], 1.0)
        self.assertEqual(sample["dbg_data_augment_det_effective_ratio"], 1.0)
        self.assertEqual(sample["dbg_data_augment_det_corrupt_any_ratio"], 1.0)
        self.assertAlmostEqual(
            sample["dbg_data_augment_det_corrupt_obj_ratio"], 0.3
        )

        dataset.butd_gt = True
        sample = dataset.__getitem__(0)

        self.assertEqual(sample["dbg_data_augment_det_active_ratio"], 1.0)
        self.assertEqual(sample["dbg_data_augment_det_effective_ratio"], 0.0)
        self.assertEqual(sample["dbg_data_augment_det_corrupt_any_ratio"], 0.0)
        self.assertEqual(sample["dbg_data_augment_det_corrupt_obj_ratio"], 0.0)

    def test_getitem_reports_actual_geometry_augmentation_risk(self):
        import src.joint_det_dataset as dataset_module
        from src.joint_det_dataset import Joint3DDataset

        class FakeScan:
            orig_pc = np.zeros((4, 3), dtype=np.float32)
            pc = orig_pc.copy()
            color = np.zeros_like(orig_pc)
            three_d_objects = [{"points": np.array([0, 1]), "instance_label": "chair"}]

            def get_object_instance_label(self, object_id):
                return self.three_d_objects[object_id]["instance_label"]

        dataset = Joint3DDataset.__new__(Joint3DDataset)
        dataset.split = "train"
        dataset.test_dataset = "scanrefer_spacy"
        dataset.scans = {"scene001": FakeScan()}
        dataset.annos = [{
            "scan_id": "scene001",
            "dataset": "scanrefer_spacy",
            "utterance": "the chair is second from the right on the north side.",
            "target_id": 0,
            "distractor_ids": [],
            "anchor_ids": [],
            "rel_slots": [],
            "decomposition_status": "ok",
        }]
        dataset.visualize = False
        dataset.butd_gt = False
        dataset.butd_cls = False
        dataset.spacy_relation_free_yaw_only_aug = False
        dataset.spacy_relation_free_rawview_global_only_train = False

        max_num_obj = dataset_module.MAX_NUM_OBJ
        num_classes = dataset_module.NUM_CLASSES
        zeros_6 = np.zeros((max_num_obj, 6), dtype=np.float32)
        zeros_1 = np.zeros(max_num_obj, dtype=np.float32)
        false_mask = np.zeros(max_num_obj, dtype=bool)
        positive = np.zeros((max_num_obj, 256), dtype=np.float32)
        tokens = np.zeros((max_num_obj, 2), dtype=np.int64)
        positive_diag = {
            "positive_map_target_missing": 0.0,
            "positive_map_fallback_used": 0.0,
            "positive_map_global_only_target_empty": 0.0,
            "positive_map_source_explicit_target_slot": 1.0,
            "positive_map_source_entity_exact_match": 0.0,
            "positive_map_source_lexical_exact_match": 0.0,
            "positive_map_source_missing": 0.0,
            "dbg_warn_global_only_target_positive_map": 0.0,
        }
        augmentations = {
            "theta_z": -93.0,
            "theta_x": 2.5,
            "theta_y": -1.0,
            "yz_flip": True,
            "xz_flip": False,
            "shift": np.array([[0.3, 0.4, 0.0]], dtype=np.float32),
            "scale": 1.04,
            "noise": np.array(
                [[0.0, 0.01, -0.01], [0.02, 0.0, 0.0]], dtype=np.float32
            ),
        }

        dataset._get_pc = lambda anno, scan: (
            scan.pc.copy(), augmentations.copy(), scan.color.copy()
        )
        dataset._get_target_boxes = lambda anno, scan: (
            zeros_6.copy(), zeros_1.copy(), -np.ones(len(scan.pc), dtype=np.int64)
        )
        dataset._get_scene_objects = lambda scan: (
            zeros_1.copy(), zeros_6.copy(), np.array([True] + [False] * (max_num_obj - 1))
        )
        dataset._get_auxi_boxes = lambda *args: None
        dataset._get_token_positive_map = lambda anno: (
            tokens.copy(), positive.copy(), positive_diag.copy()
        )
        dataset._get_token_positive_map_by_parse = lambda anno, auxi_box: (
            tokens.copy(), positive.copy(), positive.copy(), positive.copy(),
            positive.copy(), positive.copy(), positive.copy()
        )
        dataset._get_detected_objects = lambda split, scan_id, augmentations: (
            zeros_6.copy(),
            false_mask.copy(),
            zeros_1.copy(),
            np.zeros((max_num_obj, num_classes), dtype=np.float32),
            {
                "active": 1.0,
                "applied": 1.0,
                "corrupt_obj_count": 2,
                "valid_obj_count": 10,
                "corrupt_obj_ratio": 0.2,
            },
        )

        sample = dataset.__getitem__(0)

        self.assertEqual(sample["dbg_data_aug_pc_active_ratio"], 1.0)
        self.assertAlmostEqual(sample["dbg_data_aug_yaw_abs_deg"], 93.0)
        self.assertAlmostEqual(sample["dbg_data_aug_pitch_abs_deg"], 2.5)
        self.assertAlmostEqual(sample["dbg_data_aug_roll_abs_deg"], 1.0)
        self.assertEqual(sample["dbg_data_aug_flip_any_ratio"], 1.0)
        self.assertAlmostEqual(sample["dbg_data_aug_shift_l2"], 0.5)
        self.assertAlmostEqual(sample["dbg_data_aug_scale_delta_abs"], 0.04)
        self.assertAlmostEqual(
            sample["dbg_data_aug_noise_abs_mean"],
            np.abs(augmentations["noise"]).mean(),
        )
        self.assertEqual(sample["dbg_data_aug_yaw_active_ratio"], 1.0)
        self.assertEqual(sample["dbg_data_aug_pitch_roll_active_ratio"], 1.0)
        self.assertEqual(sample["dbg_data_aug_shift_active_ratio"], 1.0)
        self.assertEqual(sample["dbg_data_aug_scale_active_ratio"], 1.0)
        self.assertEqual(sample["dbg_data_aug_noise_active_ratio"], 1.0)
        self.assertEqual(sample["dbg_data_aug_color_active_ratio"], 0.0)
        self.assertEqual(sample["dbg_data_spacy_relation_free_ratio"], 1.0)
        self.assertEqual(sample["dbg_data_spacy_view_word_ratio"], 1.0)
        self.assertEqual(sample["dbg_data_spacy_compass_word_ratio"], 1.0)
        self.assertEqual(sample["dbg_data_spacy_direction_sensitive_ratio"], 1.0)
        self.assertEqual(
            sample["dbg_data_spacy_direction_sensitive_yaw_aug_ratio"], 1.0
        )
        self.assertEqual(
            sample["dbg_data_spacy_direction_sensitive_pitch_roll_aug_ratio"], 1.0
        )
        self.assertEqual(
            sample["dbg_data_spacy_direction_sensitive_shift_aug_ratio"], 1.0
        )
        self.assertEqual(
            sample["dbg_data_spacy_direction_sensitive_scale_aug_ratio"], 1.0
        )
        self.assertEqual(
            sample["dbg_data_spacy_direction_sensitive_noise_aug_ratio"], 1.0
        )
        self.assertEqual(
            sample["dbg_data_spacy_direction_sensitive_rigid_aug_ratio"], 1.0
        )
        self.assertEqual(
            sample["dbg_data_spacy_direction_sensitive_det_corrupt_ratio"], 1.0
        )

    def test_spacy_augmentation_parser_flags_are_available(self):
        from main_utils import parse_option

        old_argv = sys.argv[:]
        try:
            sys.argv = [
                "train_dist_mod.py",
                "--spacy_relation_free_yaw_only_aug",
                "--spacy_relation_free_view_guard_aug",
                "--spacy_relation_free_stable_yaw_aug",
                "--spacy_relation_free_view_small_yaw_aug",
                "--spacy_relation_free_rawview_global_only_train",
                "--spacy_relation_free_none_aug",
                "--spacy_relation_free_compass_guard_aug",
                "--disable_box_jitter",
                "--disable_train_augmentation",
            ]
            args = parse_option()
        finally:
            sys.argv = old_argv

        self.assertTrue(args.spacy_relation_free_yaw_only_aug)
        self.assertTrue(args.spacy_relation_free_view_guard_aug)
        self.assertTrue(args.spacy_relation_free_stable_yaw_aug)
        self.assertTrue(args.spacy_relation_free_view_small_yaw_aug)
        self.assertTrue(args.spacy_relation_free_rawview_global_only_train)
        self.assertTrue(args.spacy_relation_free_none_aug)
        self.assertTrue(args.spacy_relation_free_compass_guard_aug)
        self.assertTrue(args.disable_box_jitter)
        self.assertTrue(args.disable_train_augmentation)

    def test_getitem_resets_cached_scan_points_before_each_augmentation(self):
        import src.joint_det_dataset as dataset_module
        from src.joint_det_dataset import Joint3DDataset

        class FakeScan:
            def __init__(self):
                self.orig_pc = np.array(
                    [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                    dtype=np.float32,
                )
                self.pc = np.full_like(self.orig_pc, 99.0)
                self.color = np.zeros_like(self.orig_pc)
                self.three_d_objects = [
                    {"points": np.array([0, 1]), "instance_label": "chair"}
                ]

            def get_object_instance_label(self, object_id):
                return self.three_d_objects[object_id]["instance_label"]

        dataset = Joint3DDataset.__new__(Joint3DDataset)
        dataset.split = "train"
        dataset.test_dataset = "scanrefer"
        dataset.scans = {"scene001": FakeScan()}
        dataset.annos = [{
            "scan_id": "scene001",
            "dataset": "scanrefer",
            "utterance": "chair",
            "target_id": 0,
            "distractor_ids": [],
            "anchor_ids": [],
        }]
        dataset.visualize = False
        dataset.butd_gt = False
        dataset.butd_cls = False

        max_num_obj = dataset_module.MAX_NUM_OBJ
        num_classes = dataset_module.NUM_CLASSES
        zeros_6 = np.zeros((max_num_obj, 6), dtype=np.float32)
        zeros_1 = np.zeros(max_num_obj, dtype=np.float32)
        false_mask = np.zeros(max_num_obj, dtype=bool)
        positive = np.zeros((max_num_obj, 256), dtype=np.float32)
        tokens = np.zeros((max_num_obj, 2), dtype=np.int64)
        positive_diag = {
            "positive_map_target_missing": 0.0,
            "positive_map_fallback_used": 0.0,
            "positive_map_global_only_target_empty": 0.0,
            "positive_map_source_explicit_target_slot": 1.0,
            "positive_map_source_entity_exact_match": 0.0,
            "positive_map_source_lexical_exact_match": 0.0,
            "positive_map_source_missing": 0.0,
            "dbg_warn_global_only_target_positive_map": 0.0,
        }
        seen_inputs = []

        def fake_get_pc(anno, scan):
            seen_inputs.append(scan.pc.copy())
            scan.pc[:] = 77.0
            return scan.pc.copy(), {}, scan.color.copy()

        dataset._get_pc = fake_get_pc
        dataset._get_target_boxes = lambda anno, scan: (
            zeros_6.copy(), zeros_1.copy(), -np.ones(len(scan.pc), dtype=np.int64)
        )
        dataset._get_scene_objects = lambda scan: (
            zeros_1.copy(), zeros_6.copy(), np.array([True] + [False] * (max_num_obj - 1))
        )
        dataset._get_auxi_boxes = lambda *args: None
        dataset._get_token_positive_map = lambda anno: (
            tokens.copy(), positive.copy(), positive_diag.copy()
        )
        dataset._get_token_positive_map_by_parse = lambda anno, auxi_box: (
            tokens.copy(), positive.copy(), positive.copy(), positive.copy(),
            positive.copy(), positive.copy(), positive.copy()
        )
        dataset._get_detected_objects = lambda split, scan_id, augmentations: (
            zeros_6.copy(), false_mask.copy(), zeros_1.copy(),
            np.zeros((max_num_obj, num_classes), dtype=np.float32)
        )

        dataset.__getitem__(0)
        dataset.__getitem__(0)

        np.testing.assert_allclose(seen_inputs[0], dataset.scans["scene001"].orig_pc)
        np.testing.assert_allclose(seen_inputs[1], dataset.scans["scene001"].orig_pc)


if __name__ == "__main__":
    unittest.main()
