import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


EDA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDA_ROOT))


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

            detected_boxes, detected_mask, _, _ = dataset._get_detected_objects(
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
                detected_boxes, detected_mask, detected_class_ids, _ = dataset._get_detected_objects(
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

    def test_spacy_natural_language_datasets_use_rotation_augmentation(self):
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

        for dataset_name in ("nr3d", "nr3d_spacy", "scanrefer", "scanrefer_spacy"):
            scan = FakeScan()
            scan.pc = np.zeros((3, 3), dtype=np.float32)
            scan.color = np.zeros((3, 3), dtype=np.float32)

            dataset._get_pc(
                {
                    "scan_id": "scene001",
                    "dataset": dataset_name,
                    "utterance": "plain wooden chair near table",
                },
                scan,
            )

        self.assertEqual(rotate_flags, [True, True, True, True])

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

    def test_spacy_view_words_without_relation_slots_use_no_rotation_mode(self):
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
                "utterance": "the chair is behind the table",
                "rel_slots": [],
            },
            scan,
        )

        self.assertEqual(rotate_flags, ["none"])

    def test_spacy_punctuated_view_words_keep_natural_mode_after_v2_regression(self):
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
                "utterance": "the chair is on the left, near the table.",
                "rel_slots": [],
            },
            scan,
        )

        self.assertEqual(rotate_flags, [True])

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

    def test_spacy_directional_relation_slot_hyphen_terms_use_no_rotation_mode(self):
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
                "utterance": "the picture is above the left-most bed",
                "rel_slots": [{"text": "above the left-most bed"}],
            },
            scan,
        )

        self.assertEqual(rotate_flags, ["none"])

    def test_spacy_compass_relation_slot_uses_no_rotation_mode(self):
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
                "utterance": "the table is closest to the north",
                "rel_slots": [{"text": "closest to the north"}],
            },
            scan,
        )

        self.assertEqual(rotate_flags, ["none"])

    def test_spacy_rotation_mode_ids_match_current_policy(self):
        from src.joint_det_dataset import Joint3DDataset

        none_anno = {
            "dataset": "scanrefer_spacy",
            "utterance": "the chair is behind the table",
            "rel_slots": [],
        }
        yaw_anno = {
            "dataset": "scanrefer_spacy",
            "utterance": "the chair is above the table",
            "rel_slots": [{"text": "above"}],
        }
        natural_anno = {
            "dataset": "scanrefer_spacy",
            "utterance": "plain wooden chair near table",
            "rel_slots": [],
        }

        self.assertEqual(Joint3DDataset._spacy_rotation_mode_id(none_anno), 1)
        self.assertEqual(Joint3DDataset._spacy_rotation_mode_id(yaw_anno), 2)
        self.assertEqual(Joint3DDataset._spacy_rotation_mode_id(natural_anno), 3)
        self.assertEqual(
            Joint3DDataset.SPACY_ROTATION_MODE_NAMES,
            {
                -1: "not_spacy",
                1: "none",
                2: "yaw_only",
                3: "full_natural",
            },
        )

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

        self.assertEqual(
            Joint3DDataset.SPACY_AUGMENTATION_PROFILE_NAMES,
            {
                -1: "not_spacy",
                1: "none",
                2: "yaw_relation",
                3: "full_natural_relation_free",
                4: "yaw_relation_free",
                5: "none_relation_free_view",
                6: "yaw_relation_free_stable",
                7: "small_yaw_relation_free_view",
                8: "rawview_relation_free_global_only",
                9: "none_relation_free",
                10: "none_relation_free_compass",
            },
        )
        self.assertEqual(dataset._spacy_augmentation_profile_id(none_anno), 1)
        self.assertEqual(
            dataset._spacy_augmentation_profile_id(relation_yaw_anno), 2
        )
        self.assertEqual(dataset._spacy_augmentation_profile_id(natural_anno), 4)
        self.assertEqual(
            dataset._spacy_augmentation_profile_id(relation_free_view_anno), 5
        )

        dataset.spacy_relation_free_view_guard_aug = False
        dataset.spacy_relation_free_yaw_only_aug = False
        self.assertEqual(dataset._spacy_augmentation_profile_id(natural_anno), 3)

    def test_spacy_relation_free_none_profile_id_is_separate(self):
        from src.joint_det_dataset import Joint3DDataset

        dataset = Joint3DDataset.__new__(Joint3DDataset)
        dataset.spacy_relation_free_yaw_only_aug = True
        dataset.spacy_relation_free_none_aug = True
        dataset.spacy_relation_free_rawview_global_only_train = False
        dataset.spacy_relation_free_view_guard_aug = False
        dataset.spacy_relation_free_view_small_yaw_aug = False
        dataset.spacy_relation_free_stable_yaw_aug = False

        self.assertEqual(
            dataset._spacy_augmentation_profile_id(
                {
                    "dataset": "scanrefer_spacy",
                    "utterance": "plain wooden chair near table",
                    "rel_slots": [],
                }
            ),
            9,
        )

    def test_spacy_relation_free_rawview_global_only_train_masks_noisy_samples(self):
        from src.joint_det_dataset import Joint3DDataset

        dataset = Joint3DDataset.__new__(Joint3DDataset)
        dataset.split = "train"
        dataset.spacy_relation_free_yaw_only_aug = True
        dataset.spacy_relation_free_view_guard_aug = False
        dataset.spacy_relation_free_view_small_yaw_aug = False
        dataset.spacy_relation_free_stable_yaw_aug = False
        dataset.spacy_relation_free_rawview_global_only_train = True

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
        relation_anno = {
            "dataset": "scanrefer_spacy",
            "utterance": "the chair is right of the table.",
            "rel_slots": [{"text": "right of"}],
            "coverage_stats": {"candidate_relation_count": 1},
        }
        rawview_without_parse_noise = {
            "dataset": "scanrefer_spacy",
            "utterance": "the chair is second from the right.",
            "rel_slots": [],
            "coverage_stats": {},
        }

        self.assertTrue(dataset._decomp_global_only_mask_for_sample(noisy_anno))
        self.assertFalse(dataset._decomp_global_only_mask_for_sample(plain_anno))
        self.assertFalse(dataset._decomp_global_only_mask_for_sample(relation_anno))
        self.assertFalse(
            dataset._decomp_global_only_mask_for_sample(rawview_without_parse_noise)
        )
        self.assertEqual(dataset._spacy_augmentation_profile_id(noisy_anno), 8)

    def test_spacy_relation_free_rawview_global_only_train_does_not_mask_eval(self):
        from src.joint_det_dataset import Joint3DDataset

        dataset = Joint3DDataset.__new__(Joint3DDataset)
        dataset.split = "val"
        dataset.spacy_relation_free_yaw_only_aug = True
        dataset.spacy_relation_free_view_guard_aug = False
        dataset.spacy_relation_free_view_small_yaw_aug = False
        dataset.spacy_relation_free_stable_yaw_aug = False
        dataset.spacy_relation_free_rawview_global_only_train = True

        noisy_anno = {
            "dataset": "scanrefer_spacy",
            "utterance": "the chair is second from the right.",
            "rel_slots": [],
            "coverage_stats": {"candidate_relation_count": 1},
        }

        self.assertFalse(dataset._decomp_global_only_mask_for_sample(noisy_anno))
        self.assertEqual(dataset._spacy_augmentation_profile_id(noisy_anno), 8)

    def test_offline_train_global_only_mask_is_train_only(self):
        from src.joint_det_dataset import Joint3DDataset

        anno = {
            "dataset": "scanrefer_spacy",
            "utterance": "the chair second from the right",
            "rel_slots": [],
            "decomp_train_global_only_mask": True,
            "decomp_train_weak_generic_mask": True,
        }

        train_dataset = Joint3DDataset.__new__(Joint3DDataset)
        train_dataset.split = "train"
        val_dataset = Joint3DDataset.__new__(Joint3DDataset)
        val_dataset.split = "val"

        self.assertTrue(train_dataset._decomp_global_only_mask_for_sample(anno))
        self.assertTrue(train_dataset._decomp_weak_generic_mask_for_sample(anno))
        self.assertFalse(val_dataset._decomp_global_only_mask_for_sample(anno))
        self.assertFalse(val_dataset._decomp_weak_generic_mask_for_sample(anno))

    def test_spacy_relation_free_view_small_yaw_routes_regex_raw_view_words(self):
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
        dataset.spacy_relation_free_view_guard_aug = False
        dataset.spacy_relation_free_stable_yaw_aug = False
        dataset.spacy_relation_free_view_small_yaw_aug = True

        rotate_flags = []

        def record_rotate(pc, color, rotate):
            rotate_flags.append(rotate)
            return pc, color, {"scale": 1.0}

        dataset._augment = record_rotate

        scan = FakeScan()
        scan.pc = np.zeros((3, 3), dtype=np.float32)
        scan.color = np.zeros((3, 3), dtype=np.float32)

        profile_id = dataset._spacy_augmentation_profile_id(
            {
                "dataset": "scanrefer_spacy",
                "utterance": "there is a chair second from the right.",
                "rel_slots": [],
            }
        )
        dataset._get_pc(
            {
                "scan_id": "scene001",
                "dataset": "scanrefer_spacy",
                "utterance": "there is a chair second from the right.",
                "rel_slots": [],
            },
            scan,
        )

        self.assertEqual(rotate_flags, ["small_yaw"])
        self.assertEqual(profile_id, 7)

    def test_spacy_relation_free_view_small_yaw_keeps_plain_samples_yaw_only(self):
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
        dataset.spacy_relation_free_view_guard_aug = False
        dataset.spacy_relation_free_stable_yaw_aug = False
        dataset.spacy_relation_free_view_small_yaw_aug = True

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

    def test_spacy_relation_free_compass_guard_routes_compass_samples_to_none(self):
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
        dataset.spacy_relation_free_compass_guard_aug = True
        dataset.spacy_relation_free_view_guard_aug = False
        dataset.spacy_relation_free_view_small_yaw_aug = False
        dataset.spacy_relation_free_stable_yaw_aug = False
        dataset.spacy_relation_free_none_aug = False
        dataset.spacy_relation_free_rawview_global_only_train = False

        rotate_flags = []

        def record_rotate(pc, color, rotate):
            rotate_flags.append(rotate)
            return pc, color, {"scale": 1.0}

        dataset._augment = record_rotate

        scan = FakeScan()
        scan.pc = np.zeros((3, 3), dtype=np.float32)
        scan.color = np.zeros((3, 3), dtype=np.float32)
        anno = {
            "scan_id": "scene001",
            "dataset": "scanrefer_spacy",
            "utterance": "the chair is in the northeast corner of the room",
            "rel_slots": [],
        }

        dataset._get_pc(anno, scan)

        self.assertEqual(rotate_flags, ["none"])
        self.assertEqual(dataset._spacy_augmentation_profile_id(anno), 10)

    def test_spacy_relation_free_compass_guard_keeps_plain_samples_yaw_only(self):
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
        dataset.spacy_relation_free_compass_guard_aug = True
        dataset.spacy_relation_free_view_guard_aug = False
        dataset.spacy_relation_free_view_small_yaw_aug = False
        dataset.spacy_relation_free_stable_yaw_aug = False
        dataset.spacy_relation_free_none_aug = False
        dataset.spacy_relation_free_rawview_global_only_train = False

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

    def test_spacy_relation_free_compass_guard_is_default_off(self):
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
        dataset.spacy_relation_free_compass_guard_aug = False

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
                "utterance": "the chair is in the northeast corner of the room",
                "rel_slots": [],
            },
            scan,
        )

        self.assertEqual(rotate_flags, ["yaw_only"])

    def test_spacy_relation_free_none_flag_routes_plain_samples_to_none(self):
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
        dataset.spacy_relation_free_none_aug = True

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

        self.assertEqual(rotate_flags, ["none"])

    def test_spacy_relation_free_none_keeps_relation_samples_yaw_only(self):
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
        dataset.spacy_relation_free_none_aug = True

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
                "rel_slots": [{"text": "above"}],
            },
            scan,
        )

        self.assertEqual(rotate_flags, ["yaw_only"])

    def test_spacy_relation_free_view_guard_routes_view_words_to_none(self):
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
        dataset.spacy_relation_free_view_guard_aug = True

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
                "utterance": "there is a chair second from the right.",
                "rel_slots": [],
            },
            scan,
        )

        self.assertEqual(rotate_flags, ["none"])

    def test_spacy_relation_free_view_guard_keeps_plain_samples_yaw_only(self):
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
        dataset.spacy_relation_free_view_guard_aug = True

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

    def test_spacy_relation_free_stable_yaw_flag_routes_plain_samples(self):
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
        dataset.spacy_relation_free_view_guard_aug = False
        dataset.spacy_relation_free_stable_yaw_aug = True

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

        self.assertEqual(rotate_flags, ["yaw_stable"])

    def test_spacy_relation_free_stable_yaw_profile_id_is_separate(self):
        from src.joint_det_dataset import Joint3DDataset

        dataset = Joint3DDataset.__new__(Joint3DDataset)
        dataset.spacy_relation_free_yaw_only_aug = True
        dataset.spacy_relation_free_view_guard_aug = False
        dataset.spacy_relation_free_stable_yaw_aug = True

        anno = {
            "dataset": "scanrefer_spacy",
            "utterance": "plain wooden chair near table",
            "rel_slots": [],
        }

        self.assertEqual(dataset._spacy_augmentation_profile_id(anno), 6)

    def test_stable_yaw_keeps_yaw_flip_but_suppresses_jitter_noise_and_color(self):
        from src.joint_det_dataset import Joint3DDataset

        dataset = Joint3DDataset.__new__(Joint3DDataset)
        dataset.mean_rgb = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        pc = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        )
        color = np.array(
            [
                [0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7],
            ],
            dtype=np.float32,
        )

        with mock.patch("numpy.random.randint", return_value=0), \
                mock.patch("numpy.random.rand", side_effect=[0.5]), \
                mock.patch("numpy.random.random", side_effect=[0.6, 0.4]):
            augmented_pc, augmented_color, augmentations = dataset._augment(
                pc.copy(), color.copy(), "yaw_stable"
            )

        expected_pc = pc.copy()
        expected_pc[:, 0] = -expected_pc[:, 0]

        np.testing.assert_allclose(augmented_pc, expected_pc)
        np.testing.assert_allclose(augmented_color, color)
        self.assertTrue(augmentations["yz_flip"])
        self.assertFalse(augmentations["xz_flip"])
        self.assertEqual(augmentations["theta_z"], 0.0)
        self.assertEqual(augmentations["theta_x"], 0.0)
        self.assertEqual(augmentations["theta_y"], 0.0)
        np.testing.assert_allclose(augmentations["noise"], np.zeros_like(pc))
        np.testing.assert_allclose(augmentations["shift"], np.zeros((1, 3)))
        self.assertEqual(augmentations["scale"], 1.0)

    def test_small_yaw_suppresses_flips_pitch_roll_but_keeps_regular_jitter(self):
        from src.joint_det_dataset import Joint3DDataset, rot_z

        dataset = Joint3DDataset.__new__(Joint3DDataset)
        dataset.mean_rgb = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        pc = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        )

        with mock.patch("numpy.random.randint") as randint_mock, \
                mock.patch(
                    "numpy.random.rand",
                    side_effect=[0.75, np.zeros((len(pc), 3), dtype=np.float32)],
                ), \
                mock.patch(
                    "numpy.random.random",
                    side_effect=[
                        np.array([0.5, 0.5, 0.5], dtype=np.float32),
                        0.5,
                    ],
                ):
            augmented_pc, _, augmentations = dataset._augment(
                pc.copy(), None, "small_yaw"
            )

        expected_pc = rot_z(pc.copy(), 1.5)
        np.testing.assert_allclose(augmented_pc, expected_pc, atol=1e-6)
        self.assertFalse(randint_mock.called)
        self.assertFalse(augmentations["yz_flip"])
        self.assertFalse(augmentations["xz_flip"])
        self.assertEqual(augmentations["theta_z"], 1.5)
        self.assertEqual(augmentations["theta_x"], 0.0)
        self.assertEqual(augmentations["theta_y"], 0.0)
        np.testing.assert_allclose(augmentations["noise"], np.zeros_like(pc))
        np.testing.assert_allclose(augmentations["shift"], np.zeros((1, 3)))
        self.assertEqual(augmentations["scale"], 1.0)

    def test_spacy_frame_cues_keep_relation_aware_yaw_only_mode_after_v2_regression(self):
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
                "utterance": "the chair is near the table from the entrance view",
                "rel_slots": [{
                    "text": "near",
                    "frame_cue_flag": 1,
                    "frame_cue_text": "from the entrance view",
                }],
                "coverage_stats": {"frame_cue_count": 1},
            },
            scan,
        )

        self.assertEqual(rotate_flags, ["yaw_only"])

    def test_spacy_spatial_attributes_keep_natural_mode_after_v2_regression(self):
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
                "utterance": "the pillow is the top pillow on the couch",
                "rel_slots": [],
                "attr_slot": {
                    "items": [{
                        "text": "top",
                        "type": "spatial_attribute",
                    }]
                },
            },
            scan,
        )

        self.assertEqual(rotate_flags, [True])

    def test_no_rotation_mode_has_zero_angles_and_no_flips(self):
        from src.joint_det_dataset import Joint3DDataset

        dataset = Joint3DDataset.__new__(Joint3DDataset)

        point_cloud = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        _, _, augmentations = dataset._augment(point_cloud.copy(), None, "none")

        self.assertEqual(augmentations["theta_z"], 0.0)
        self.assertEqual(augmentations["theta_x"], 0.0)
        self.assertEqual(augmentations["theta_y"], 0.0)
        self.assertNotIn("yz_flip", augmentations)
        self.assertNotIn("xz_flip", augmentations)

    def test_yaw_only_mode_has_zero_pitch_roll_and_keeps_flips(self):
        from src.joint_det_dataset import Joint3DDataset

        dataset = Joint3DDataset.__new__(Joint3DDataset)

        point_cloud = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )

        _, _, augmentations = dataset._augment(point_cloud.copy(), None, "yaw_only")

        self.assertIn("yz_flip", augmentations)
        self.assertIn("xz_flip", augmentations)
        self.assertEqual(augmentations["theta_x"], 0.0)
        self.assertEqual(augmentations["theta_y"], 0.0)

    def test_nr3d_spacy_loader_dispatches_to_spacy_dataset(self):
        import io
        from src.joint_det_dataset import Joint3DDataset

        class FakeScan:
            three_d_objects = [{"instance_label": "chair"}]

            def get_object_instance_label(self, object_id):
                return self.three_d_objects[object_id]["instance_label"]

        dataset = Joint3DDataset.__new__(Joint3DDataset)
        dataset.split = "train"
        dataset.overfit = False
        dataset.data_path = "/tmp/data/"
        dataset.scans = {"scene001": FakeScan()}
        dataset._referit3d_csv_path = lambda dset: f"/tmp/{dset}.csv"
        dataset._parse_spacy_csv_fields = lambda line, headers, dset="": {
            "graph_node": [],
            "graph_edge": [],
            "auxi_entity": None,
            "entity_spans": [],
            "attr_spans": [],
            "rel_spans": [],
            "anchor_span_ids": [],
        }

        csv_text = (
            "scan_id,correct_guess,target_id,instance_type,utterance,anchor_ids\n"
            "scene001,true,0,chair,plain chair,[1]\n"
        )

        def fake_open(path, *args, **kwargs):
            if path == "data/meta_data/nr3d_train_scans.txt":
                return io.StringIO("['scene001']")
            if path == "/tmp/nr3d_spacy.csv":
                return io.StringIO(csv_text)
            raise FileNotFoundError(path)

        with mock.patch("builtins.open", side_effect=fake_open):
            annos = dataset.load_annos("nr3d_spacy")

        self.assertEqual(len(annos), 1)
        self.assertEqual(annos[0]["dataset"], "nr3d_spacy")
        self.assertEqual(annos[0]["anchor_obj_ids"], [1])

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
        dataset._get_token_positive_map = lambda anno: (tokens.copy(), positive.copy(), {})
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

    def test_getitem_reports_annotation_dataset_as_language_dataset(self):
        import src.joint_det_dataset as dataset_module
        from src.joint_det_dataset import Joint3DDataset

        class FakeScan:
            def __init__(self):
                self.orig_pc = np.zeros((2, 3), dtype=np.float32)
                self.pc = self.orig_pc.copy()
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
            "dataset": "scanrefer_spacy",
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

        dataset._get_pc = lambda anno, scan: (scan.pc.copy(), {}, scan.color.copy())
        dataset._get_target_boxes = lambda anno, scan: (
            zeros_6.copy(), zeros_1.copy(), -np.ones(len(scan.pc), dtype=np.int64)
        )
        dataset._get_scene_objects = lambda scan: (
            zeros_1.copy(), zeros_6.copy(), np.array([True] + [False] * (max_num_obj - 1))
        )
        dataset._get_auxi_boxes = lambda *args: None
        dataset._get_token_positive_map_by_parse = lambda anno, auxi_box: (
            tokens.copy(), positive.copy(), positive.copy(), positive.copy(),
            positive.copy(), positive.copy(), positive.copy()
        )
        dataset._get_detected_objects = lambda split, scan_id, augmentations: (
            zeros_6.copy(), false_mask.copy(), zeros_1.copy(),
            np.zeros((max_num_obj, num_classes), dtype=np.float32)
        )

        sample = dataset.__getitem__(0)

        self.assertEqual(sample["language_dataset"], "scanrefer_spacy")


if __name__ == "__main__":
    unittest.main()
