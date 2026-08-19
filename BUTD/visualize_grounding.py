"""
3D Visual Grounding 可视化脚本
从 Nr3D/Sr3D 标注文件读取样本，加载点云，运行模型推理，可视化预测框和GT框。
"""

import csv
import json
import os
import sys
import argparse
from collections import defaultdict

import numpy as np
import open3d as o3d
import torch
from torch.utils.data._utils.collate import default_collate


BBOX_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]

# ============================================================
# 1. 从 Nr3D / Sr3D CSV 标注文件中读取样本
# ============================================================

def load_annotations(csv_path, max_samples=None):
    """读取 Nr3D 或 Sr3D 的 CSV 标注文件，返回样本列表。"""
    annotations = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            annotations.append({
                'scan_id': row['scan_id'],
                'target_id': int(row['target_id']),
                'utterance': row['utterance'],
                'instance_type': row['instance_type'],
                'dataset': row.get('dataset', 'unknown'),
            })
            if max_samples and i + 1 >= max_samples:
                break
    return annotations


# ============================================================
# 2. 加载 ScanNet 点云（PLY）
# ============================================================

def load_axis_alignment_matrix(scan_id, meta_dir='data/meta_data'):
    """加载场景的轴对齐矩阵。"""
    path = os.path.join(meta_dir, 'scans_axis_alignment_matrices.json')
    with open(path) as f:
        matrices = json.load(f)
    return np.array(matrices[scan_id]).reshape(4, 4)


def align_to_axes(pc, alignment_mat):
    """用 4x4 矩阵对齐点云到标准坐标系。"""
    pts = np.ones((pc.shape[0], 4), dtype=pc.dtype)
    pts[:, :3] = pc
    return np.dot(pts, alignment_mat.T)[:, :3]


def get_scannet_scene_paths(scan_id, scans_dir):
    """返回加载一个 ScanNet 场景所需的关键文件路径。"""
    scene_dir = os.path.join(scans_dir, scan_id)
    return {
        'scene_dir': scene_dir,
        'ply': os.path.join(scene_dir, f'{scan_id}_vh_clean_2.ply'),
        'segs': os.path.join(
            scene_dir, f'{scan_id}_vh_clean_2.0.010000.segs.json'
        ),
        'agg': os.path.join(scene_dir, f'{scan_id}.aggregation.json'),
    }


def get_missing_scene_files(scan_id, scans_dir):
    """返回当前场景缺失的关键文件列表。"""
    scene_paths = get_scannet_scene_paths(scan_id, scans_dir)
    missing = []
    for key in ('scene_dir', 'ply', 'segs', 'agg'):
        if not os.path.exists(scene_paths[key]):
            missing.append(scene_paths[key])
    return missing


def load_scannet_scene(scan_id, scans_dir, meta_dir='data/meta_data'):
    """
    加载 ScanNet 场景点云 + 每个物体的点索引。
    返回: pc (N,3), color (N,3), objects [{object_id, points, instance_label}]
    """
    scene_paths = get_scannet_scene_paths(scan_id, scans_dir)
    missing = get_missing_scene_files(scan_id, scans_dir)
    if missing:
        missing_fmt = '\n'.join(f'  - {path}' for path in missing)
        raise FileNotFoundError(
            f'ScanNet 场景 {scan_id} 缺少必要文件:\n{missing_fmt}'
        )

    # --- 点云 ---
    from plyfile import PlyData
    data = PlyData.read(scene_paths['ply'])
    verts = data.elements[0].data
    pc = np.stack([verts['x'], verts['y'], verts['z']], axis=1)
    color = np.stack([verts['red'], verts['green'], verts['blue']], axis=1) / 255.0

    # 轴对齐
    alignment_mat = load_axis_alignment_matrix(scan_id, meta_dir)
    pc = align_to_axes(pc, alignment_mat)

    # --- 物体分割 ---
    with open(scene_paths['segs']) as f:
        seg_indices = json.load(f)['segIndices']
    segments = defaultdict(list)
    for i, s in enumerate(seg_indices):
        segments[s].append(i)

    with open(scene_paths['agg']) as f:
        aggregation = json.load(f)

    objects = []
    for obj_info in aggregation['segGroups']:
        points = []
        for s in obj_info['segments']:
            points.extend(segments[s])
        objects.append({
            'object_id': int(obj_info['objectId']),
            'points': np.array(list(set(points))),
            'instance_label': str(obj_info['label']),
        })

    return pc, color, objects


# ============================================================
# 3. 计算 GT 包围框（轴对齐）
# ============================================================

def compute_aabb(points):
    """
    计算点集的轴对齐包围框。
    返回: center (3,), size (3,)
    """
    min_pt = points.min(axis=0)
    max_pt = points.max(axis=0)
    center = (min_pt + max_pt) / 2.0
    size = max_pt - min_pt
    return center, size


def get_gt_box(pc, objects, target_id):
    """根据 target_id 从物体列表获取 GT 框。"""
    for obj in objects:
        if obj['object_id'] == target_id:
            obj_pc = pc[obj['points']]
            center, size = compute_aabb(obj_pc)
            return center, size, obj['instance_label']
    raise ValueError(f"Object {target_id} not found in scene!")


# ============================================================
# 4. 模型与数据集辅助函数
# ============================================================

REAL_MODEL_DEFAULTS = {
    'num_target': 256,
    'num_decoder_layers': 6,
    'self_position_embedding': 'loc_learned',
    'use_contrastive_align': False,
    'use_soft_token_loss': False,
    'use_color': False,
    'use_height': False,
    'use_multiview': False,
    'dataset': ['sr3d'],
    'test_dataset': 'sr3d',
    'data_root': '/root/autodl-tmp/DATA_ROOT',
    'detect_intermediate': False,
    'joint_det': False,
    'butd': False,
    'butd_gt': False,
    'butd_cls': False,
    'augment_det': False,
    'self_attend': False,
    'use_structured_slots': False,
    'use_late_acd': False,
    'slot_pooling': 'attention',
    'max_rel_anchor_pairs': 3,
    'acd_top_m_targets': 32,
    'acd_top_k_anchors': 16,
    'acd_geo_dim': 16,
    'acd_hidden_dim': 288,
    'acd_global_residual_alpha': 0.5,
    'acd_use_confidence_fusion': False,
    'acd_warmup_steps': 5000,
    'acd_initial_alpha': 0.05,
    'acd_ea_scale': 1.0,
    'acd_pool_ea_multiplier': 1.0,
    'acd_final_ea_multiplier': 1.0,
    'acd_disable_struct_rerank': False,
    'dhc_margin_min': 0.0,
    'dhc_temperature_max': 0.0,
    'structured_debug': False,
    'eval_use_acd_scores': False,
    'debug': False,
    'eval_train': False,
    'pp_checkpoint': None,
}


def _config_to_dict(config_obj):
    """将 checkpoint/config.json 中的配置统一转成 dict。"""
    if config_obj is None:
        return {}
    if isinstance(config_obj, dict):
        return dict(config_obj)
    if isinstance(config_obj, argparse.Namespace):
        return vars(config_obj)
    raise TypeError(f'Unsupported config object type: {type(config_obj)!r}')


def load_checkpoint_config(model_path):
    """优先读取同目录 config.json，其次回退到 checkpoint['config']。"""
    config_path = os.path.join(os.path.dirname(model_path), 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f), config_path

    checkpoint = torch.load(model_path, map_location='cpu')
    return _config_to_dict(checkpoint.get('config')), f'{model_path}:config'


def build_runtime_config(config_dict, data_root_override=None):
    """用默认值补齐训练配置，便于单独做可视化推理。"""
    merged = dict(REAL_MODEL_DEFAULTS)
    merged.update(config_dict or {})
    if data_root_override:
        merged['data_root'] = data_root_override

    dataset = merged.get('dataset', ['sr3d'])
    if isinstance(dataset, str):
        dataset = [dataset]
    merged['dataset'] = dataset

    return argparse.Namespace(**merged)


def strip_module_prefix(state_dict):
    """去掉 DDP 保存时的 module. 前缀。"""
    return {
        key[7:] if key.startswith('module.') else key: value
        for key, value in state_dict.items()
    }


def joint_det_collate(batch):
    """保留 span 嵌套结构，其余字段走默认 collate。"""
    span_keys = {'target_slot', 'entity_spans', 'attr_spans', 'rel_spans'}
    collated = {}
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        if key in span_keys:
            collated[key] = values
        else:
            collated[key] = default_collate(values)
    return collated


def move_tensors_to_device(data_dict, device):
    """仅将张量字段移动到目标设备。"""
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.to(device, non_blocking=(device.type == 'cuda'))
    return data_dict


def build_model_from_config(config):
    """按训练脚本的参数构造 BeaUTyDETR。"""
    from models import BeaUTyDETR

    num_input_channel = int(config.use_color) * 3
    if config.use_height:
        num_input_channel += 1
    if config.use_multiview:
        num_input_channel += 128
    num_class = 256 if config.use_soft_token_loss else 19

    return BeaUTyDETR(
        num_class=num_class,
        num_obj_class=485,
        input_feature_dim=num_input_channel,
        num_queries=config.num_target,
        num_decoder_layers=config.num_decoder_layers,
        self_position_embedding=config.self_position_embedding,
        contrastive_align_loss=config.use_contrastive_align,
        butd=config.butd or config.butd_gt or config.butd_cls,
        pointnet_ckpt=config.pp_checkpoint,
        self_attend=config.self_attend,
        use_structured_slots=config.use_structured_slots,
        use_late_acd=config.use_late_acd,
        slot_pooling=config.slot_pooling,
        max_rel_anchor_pairs=config.max_rel_anchor_pairs,
        acd_top_m_targets=config.acd_top_m_targets,
        acd_top_k_anchors=config.acd_top_k_anchors,
        acd_geo_dim=config.acd_geo_dim,
        acd_hidden_dim=config.acd_hidden_dim,
        acd_global_residual_alpha=config.acd_global_residual_alpha,
        acd_use_confidence_fusion=config.acd_use_confidence_fusion,
        acd_warmup_steps=config.acd_warmup_steps,
        acd_initial_alpha=config.acd_initial_alpha,
        acd_ea_scale=config.acd_ea_scale,
        acd_pool_ea_multiplier=config.acd_pool_ea_multiplier,
        acd_final_ea_multiplier=config.acd_final_ea_multiplier,
        acd_disable_struct_rerank=config.acd_disable_struct_rerank,
        dhc_margin_min=config.dhc_margin_min,
        dhc_temperature_max=config.dhc_temperature_max,
        structured_debug=config.structured_debug
    )


def build_eval_dataset(config):
    """按评测时的参数构建 Joint3DDataset。"""
    from src.joint_det_dataset import Joint3DDataset

    dataset_dict = {name: 1 for name in config.dataset}
    if config.joint_det:
        dataset_dict['scannet'] = 10

    return Joint3DDataset(
        dataset_dict=dataset_dict,
        test_dataset=config.test_dataset,
        split='train' if config.eval_train else 'val',
        use_color=config.use_color,
        use_height=config.use_height,
        overfit=config.debug,
        data_path=config.data_root,
        detect_intermediate=config.detect_intermediate,
        use_multiview=config.use_multiview,
        butd=config.butd,
        butd_gt=config.butd_gt,
        butd_cls=config.butd_cls
    )


def build_model_inputs(batch_data):
    """复用训练/评测时的输入字段定义。"""
    inputs = {
        'point_clouds': batch_data['point_clouds'].float(),
        'text': batch_data['utterances'],
        'det_boxes': batch_data['all_detected_boxes'],
        'det_bbox_label_mask': batch_data['all_detected_bbox_label_mask'],
        'det_class_ids': batch_data['all_detected_class_ids']
    }
    if 'entity_spans' in batch_data:
        inputs['entity_spans'] = batch_data['entity_spans']
    if 'attr_spans' in batch_data:
        inputs['attr_spans'] = batch_data['attr_spans']
    if 'rel_spans' in batch_data:
        inputs['rel_spans'] = batch_data['rel_spans']
    if 'anchor_ids' in batch_data:
        inputs['anchor_ids'] = batch_data['anchor_ids'].long()
    return inputs


def select_query_index(end_points, batch_data, prefix):
    """按评测逻辑选出当前文本对应的最高分 query。"""
    use_acd = bool(end_points.get('eval_use_acd_scores', False))
    if prefix == 'last_' and 'acd_final_scores' in end_points and use_acd:
        scores = end_points['acd_final_scores'][0]
    elif f'{prefix}sem_cls_scores' in end_points:
        sem_scores = end_points[f'{prefix}sem_cls_scores'][0].softmax(-1)
        positive_map = batch_data['positive_map'][0, 0]
        if sem_scores.shape[-1] != positive_map.shape[-1]:
            sem_scores_padded = sem_scores.new_zeros(
                sem_scores.shape[0], positive_map.shape[-1]
            )
            sem_scores_padded[:, :sem_scores.shape[-1]] = sem_scores
            sem_scores = sem_scores_padded
        scores = (sem_scores * positive_map.unsqueeze(0)).sum(-1)
    elif f'{prefix}objectness_scores' in end_points:
        scores = end_points[f'{prefix}objectness_scores'][0]
    else:
        raise KeyError(f'No usable scores found for prefix "{prefix}"')

    query_idx = int(scores.argmax().item())
    score = float(scores[query_idx].detach().cpu())
    return query_idx, score


class RealGroundingModel:
    """使用真实 checkpoint 和 Joint3DDataset 进行单样本推理。"""

    def __init__(self, model_path, data_root=None, device='auto'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Checkpoint not found: {model_path}')

        config_dict, config_source = load_checkpoint_config(model_path)
        self.config = build_runtime_config(config_dict, data_root_override=data_root)
        self.config_source = config_source
        self.device = torch.device(
            'cuda' if device == 'auto' and torch.cuda.is_available()
            else ('cpu' if device == 'auto' else device)
        )

        print(f"[INFO] 读取 checkpoint 配置: {self.config_source}")
        self.dataset = build_eval_dataset(self.config)
        self.model = build_model_from_config(self.config)

        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = strip_module_prefix(checkpoint['model'])
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.model_path = model_path
        self.epoch = checkpoint.get('epoch', None)
        del checkpoint

    def __len__(self):
        return len(self.dataset)

    def get_sample_metadata(self, sample_idx):
        """读取评测集索引对应的元信息，不触发模型前向。"""
        anno = self.dataset.annos[sample_idx]
        target_id = anno['target_id'][0] if isinstance(anno['target_id'], list) else anno['target_id']
        target_name = anno.get('target', '')
        if isinstance(target_name, list):
            target_name = target_name[0] if target_name else ''
        if not target_name:
            target_name = anno.get('instance_type', '')
        return {
            'scan_id': anno['scan_id'],
            'target_id': int(target_id),
            'utterance': anno['utterance'],
            'target_name': target_name,
            'dataset': anno.get('dataset', self.config.test_dataset),
        }

    @torch.no_grad()
    def predict(self, sample_idx):
        """对评测集中的单个样本做推理。"""
        sample = self.dataset[sample_idx]
        batch = joint_det_collate([sample])
        batch = move_tensors_to_device(batch, self.device)

        inputs = build_model_inputs(batch)
        end_points = self.model(inputs)
        end_points['eval_use_acd_scores'] = bool(self.config.eval_use_acd_scores)

        prefix = 'last_' if self.config.num_decoder_layers > 0 else 'proposal_'
        query_idx, score = select_query_index(end_points, batch, prefix)
        pred_center = end_points[f'{prefix}center'][0, query_idx].detach().cpu().numpy()
        pred_size = end_points[f'{prefix}pred_size'][0, query_idx].detach().cpu().numpy()

        meta = self.get_sample_metadata(sample_idx)
        meta.update({
            'pred_center': pred_center,
            'pred_size': pred_size,
            'query_idx': query_idx,
            'score': score,
            'prefix': prefix,
            'model_utterance': batch['utterances'][0],
        })
        return meta


# ============================================================
# 5. Mock 模型（替代真实模型的推理流程）
# ============================================================

class MockGroundingModel:
    """
    模拟 3D Visual Grounding 模型。
    真实模型（如 BUTD-DETR）接口:
        input:  点云 (B, N, 3+), 文本 list[str], 检测框 (B, K, 6)
        output: 预测框 (B, Q, 6) [cxcyczwhd], 分类 logits

    这里用 mock 逻辑：在所有物体中选一个与 GT 接近但加噪的框。
    """

    def __init__(self, model_path=None):
        """如果有真实模型权重，在此加载。"""
        self.model = None
        if model_path and os.path.exists(model_path):
            print(f"[INFO] 加载模型: {model_path}")
            # import torch
            # self.model = torch.load(model_path, map_location='cpu')
            # self.model.eval()
        else:
            print("[INFO] 使用 Mock 模型（加噪GT）进行演示")

    def predict(self, pc, color, utterance, objects):
        """
        模拟推理：
        - 真实场景下: 将点云+文本送入模型，得到预测框
        - Mock: 在物体中随机选一个，对其框加噪声
        """
        if self.model is not None:
            # ---- 真实模型推理示例 ----
            # import torch
            # point_cloud = torch.from_numpy(pc).unsqueeze(0).float()
            # inputs = {
            #     'point_clouds': point_cloud,
            #     'text': [utterance],
            #     'det_boxes': ...,
            #     'det_class_ids': ...,
            #     'det_bbox_label_mask': ...,
            # }
            # with torch.no_grad():
            #     outputs = self.model(inputs)
            # pred_center = outputs['last_center'][0, 0].numpy()  # (3,)
            # pred_size = outputs['last_size'][0, 0].numpy()      # (3,)
            # return pred_center, pred_size
            pass

        # ---- Mock: 随机选一个物体，加噪声模拟预测 ----
        idx = np.random.randint(len(objects))
        obj_pc = pc[objects[idx]['points']]
        center, size = compute_aabb(obj_pc)
        # 加一点噪声模拟预测偏差
        noise_center = np.random.normal(0, 0.05, 3)
        noise_size = np.random.normal(0, 0.03, 3)
        pred_center = center + noise_center
        pred_size = np.clip(size + noise_size, 0.01, None)
        return pred_center, pred_size


# ============================================================
# 5. Open3D 可视化
# ============================================================

def create_bbox_lineset(center, size, color):
    """
    创建一个 Open3D 线段集（3D 包围框）。
    center: (3,), size: (3,), color: (r,g,b) 归一化
    """
    corners = get_bbox_corners(center, size)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(BBOX_EDGES)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(BBOX_EDGES))
    return line_set


def get_bbox_corners(center, size):
    """返回轴对齐包围框的 8 个角点。"""
    cx, cy, cz = center
    dx, dy, dz = size / 2.0
    return np.array([
        [cx - dx, cy - dy, cz - dz],
        [cx + dx, cy - dy, cz - dz],
        [cx + dx, cy + dy, cz - dz],
        [cx - dx, cy + dy, cz - dz],
        [cx - dx, cy - dy, cz + dz],
        [cx + dx, cy - dy, cz + dz],
        [cx + dx, cy + dy, cz + dz],
        [cx - dx, cy + dy, cz + dz],
    ])


def visualize(pc, color, gt_center, gt_size, pred_center, pred_size,
              utterance, target_label, scan_id, sample_idx=None,
              target_id=None, headless=False, save_dir='outputs/grounding_viz',
              max_vis_points=30000):
    """
    可视化结果。
    - 有 DISPLAY 时: 用 Open3D 打开交互窗口
    - 无 DISPLAY 或显式 headless 时: 保存静态 PNG
    """
    # 计算 IoU（轴对齐简化版）
    iou = compute_iou_3d(gt_center, gt_size, pred_center, pred_size)

    title = (
        f"Scene: {scan_id} | Target: {target_label}\n"
        f"Utterance: {utterance[:80]}{'...' if len(utterance) > 80 else ''}\n"
        f"IoU: {iou:.3f} | Blue=GT, Red=Pred"
    )
    print(f"\n{'='*70}")
    print(f"  Scene:     {scan_id}")
    print(f"  Target:    {target_label}")
    print(f"  Utterance: {utterance}")
    print(f"  IoU:       {iou:.3f}")
    print(f"  GT center: {gt_center}")
    print(f"  Pred center: {pred_center}")
    print(f"{'='*70}\n")

    force_headless = headless or not os.environ.get('DISPLAY')
    if force_headless:
        save_path = save_visualization_snapshot(
            pc, color,
            gt_center, gt_size,
            pred_center, pred_size,
            utterance, target_label, scan_id,
            iou,
            sample_idx=sample_idx,
            target_id=target_id,
            save_dir=save_dir,
            max_vis_points=max_vis_points,
        )
        print(f"[INFO] 无可用 DISPLAY，已保存静态图: {save_path}")
        return save_path

    # 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(color)

    # GT 框 - 蓝色
    gt_box = create_bbox_lineset(gt_center, gt_size, color=[0, 0, 1])
    # 预测框 - 红色
    pred_box = create_bbox_lineset(pred_center, pred_size, color=[1, 0, 0])

    # 可视化
    vis = o3d.visualization.Visualizer()
    created = vis.create_window(window_name=title, width=1280, height=720)
    if not created:
        save_path = save_visualization_snapshot(
            pc, color,
            gt_center, gt_size,
            pred_center, pred_size,
            utterance, target_label, scan_id,
            iou,
            sample_idx=sample_idx,
            target_id=target_id,
            save_dir=save_dir,
            max_vis_points=max_vis_points,
        )
        print(f"[WARN] Open3D 窗口创建失败，已回退为静态图: {save_path}")
        return save_path
    vis.add_geometry(pcd)
    vis.add_geometry(gt_box)
    vis.add_geometry(pred_box)

    # 设置渲染参数
    opt = vis.get_render_option()
    if opt is None:
        vis.destroy_window()
        save_path = save_visualization_snapshot(
            pc, color,
            gt_center, gt_size,
            pred_center, pred_size,
            utterance, target_label, scan_id,
            iou,
            sample_idx=sample_idx,
            target_id=target_id,
            save_dir=save_dir,
            max_vis_points=max_vis_points,
        )
        print(f"[WARN] Open3D 渲染初始化失败，已回退为静态图: {save_path}")
        return save_path
    opt.point_size = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.1])  # 深色背景

    vis.run()
    vis.destroy_window()


def save_visualization_snapshot(pc, color, gt_center, gt_size, pred_center,
                                pred_size, utterance, target_label, scan_id,
                                iou, sample_idx=None, target_id=None,
                                save_dir='outputs/grounding_viz',
                                max_vis_points=30000):
    """在无头环境下将点云和包围框导出为静态 PNG。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    if len(pc) > max_vis_points:
        rng = np.random.RandomState(0)
        keep = rng.choice(len(pc), size=max_vis_points, replace=False)
        pc_draw = pc[keep]
        color_draw = color[keep]
    else:
        pc_draw = pc
        color_draw = color

    fig = plt.figure(figsize=(16, 13), facecolor='#111111')
    axes = [
        fig.add_subplot(2, 2, 1, projection='3d'),
        fig.add_subplot(2, 2, 2, projection='3d'),
        fig.add_subplot(2, 2, 3, projection='3d'),
        fig.add_subplot(2, 2, 4, projection='3d'),
    ]
    views = [
        {'elev': 18, 'azim': 35, 'title': 'Global Perspective', 'focus': False},
        {'elev': 85, 'azim': -90, 'title': 'Global Top', 'focus': False},
        {'elev': 20, 'azim': 35, 'title': 'Zoomed Perspective', 'focus': True},
        {'elev': 85, 'azim': -90, 'title': 'Zoomed Top', 'focus': True},
    ]

    corners_gt = get_bbox_corners(gt_center, gt_size)
    corners_pred = get_bbox_corners(pred_center, pred_size)
    global_mins = np.minimum(
        pc_draw.min(axis=0),
        np.minimum(corners_gt.min(axis=0), corners_pred.min(axis=0))
    )
    global_maxs = np.maximum(
        pc_draw.max(axis=0),
        np.maximum(corners_gt.max(axis=0), corners_pred.max(axis=0))
    )
    global_center = (global_mins + global_maxs) / 2.0
    global_radius = max((global_maxs - global_mins).max() / 2.0, 1e-3)

    focus_mins = np.minimum(corners_gt.min(axis=0), corners_pred.min(axis=0))
    focus_maxs = np.maximum(corners_gt.max(axis=0), corners_pred.max(axis=0))
    focus_center = (focus_mins + focus_maxs) / 2.0
    focus_radius = max((focus_maxs - focus_mins).max() * 1.8, 0.8)
    focus_keep = np.all(
        np.abs(pc_draw - focus_center[None, :]) <= focus_radius,
        axis=1
    )
    if focus_keep.any():
        pc_focus = pc_draw[focus_keep]
        color_focus = color_draw[focus_keep]
    else:
        pc_focus = pc_draw
        color_focus = color_draw

    for ax, view in zip(axes, views):
        ax.set_facecolor('#111111')
        if view['focus']:
            pc_cur = pc_focus
            color_cur = color_focus
            center = focus_center
            radius = focus_radius
            point_size = 1.4
            line_width = 4.5
            marker_size = 28
        else:
            pc_cur = pc_draw
            color_cur = color_draw
            center = global_center
            radius = global_radius
            point_size = 0.3
            line_width = 3.0
            marker_size = 18
        ax.scatter(
            pc_cur[:, 0], pc_cur[:, 1], pc_cur[:, 2],
            c=color_cur, s=point_size, alpha=0.9, depthshade=False
        )
        draw_bbox_matplotlib(
            ax, corners_gt, '#3B82F6', 'GT',
            linewidth=line_width, marker_size=marker_size
        )
        draw_bbox_matplotlib(
            ax, corners_pred, '#EF4444', 'Pred',
            linewidth=line_width, marker_size=marker_size
        )
        ax.view_init(elev=view['elev'], azim=view['azim'])
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_title(view['title'], color='white', pad=12)
        ax.set_xlabel('X', color='white')
        ax.set_ylabel('Y', color='white')
        ax.set_zlabel('Z', color='white')
        ax.tick_params(colors='white')
        ax.grid(False)

    sample_tag = f's{sample_idx:05d}_' if sample_idx is not None else ''
    target_tag = f'_obj{target_id}' if target_id is not None else ''
    out_path = os.path.join(save_dir, f'{sample_tag}{scan_id}{target_tag}.png')

    utterance_short = utterance if len(utterance) <= 120 else utterance[:117] + '...'
    fig.suptitle(
        f'{scan_id} | {target_label} | IoU={iou:.3f}\n{utterance_short}',
        color='white',
        fontsize=14,
        y=0.98,
    )
    fig.text(0.01, 0.01, 'Blue=GT, Red=Pred', color='white', fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig.savefig(out_path, dpi=220, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)
    return out_path


def draw_bbox_matplotlib(ax, corners, color, label, linewidth=2.0, marker_size=18):
    """在 Matplotlib 3D 视图中绘制包围框。"""
    for edge_idx, (start, end) in enumerate(BBOX_EDGES):
        pts = corners[[start, end]]
        ax.plot(
            pts[:, 0], pts[:, 1], pts[:, 2],
            color=color,
            linewidth=linewidth,
            label=label if edge_idx == 0 else None,
        )
    ax.scatter(
        corners[:, 0], corners[:, 1], corners[:, 2],
        color=color, s=marker_size, depthshade=False
    )
    if label:
        ax.legend(loc='upper right')


def compute_iou_3d(c1, s1, c2, s2):
    """计算两个轴对齐3D框的 IoU。"""
    min1 = c1 - s1 / 2
    max1 = c1 + s1 / 2
    min2 = c2 - s2 / 2
    max2 = c2 + s2 / 2

    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_size = np.maximum(inter_max - inter_min, 0)
    inter_vol = inter_size.prod()

    vol1 = s1.prod()
    vol2 = s2.prod()
    union_vol = vol1 + vol2 - inter_vol
    return inter_vol / max(union_vol, 1e-8)


# ============================================================
# 6. 主流程
# ============================================================

def run_mock_visualization(args, scans_dir, meta_dir):
    """沿用原始 CSV + Mock 方式做演示。"""
    csv_path = os.path.join(args.data_root, 'refer_it_3d', f'{args.dataset}.csv')

    print(f"[1/4] 读取标注文件: {csv_path}")
    annotations = load_annotations(csv_path)
    print(f"  共 {len(annotations)} 条标注")

    print(f"[2/4] 初始化模型")
    model = MockGroundingModel(args.model_path)

    idx = args.sample_idx
    visualized = 0
    skipped_missing = 0
    skipped_invalid = 0
    while idx < len(annotations) and visualized < args.num_samples:
        anno = annotations[idx]
        scan_id = anno['scan_id']
        target_id = anno['target_id']
        utterance = anno['utterance']

        print(f"\n[3/4] 样本 {idx}: 加载场景 {scan_id}")
        missing = get_missing_scene_files(scan_id, scans_dir)
        if missing:
            if args.strict_missing:
                missing_fmt = '\n'.join(f'  - {path}' for path in missing)
                raise FileNotFoundError(
                    f'样本 {idx} 的场景 {scan_id} 缺少必要文件:\n{missing_fmt}'
                )
            print(
                f"[WARN] 跳过样本 {idx}: 场景 {scan_id} 缺少 "
                f"{len(missing)} 个文件"
            )
            skipped_missing += 1
            idx += 1
            continue

        pc, color, objects = load_scannet_scene(scan_id, scans_dir, meta_dir)

        try:
            gt_center, gt_size, gt_label = get_gt_box(pc, objects, target_id)
        except ValueError as exc:
            if args.strict_missing:
                raise
            print(f"[WARN] 跳过样本 {idx}: {exc}")
            skipped_invalid += 1
            idx += 1
            continue

        print(f"[4/4] 推理: \"{utterance[:60]}...\"")
        pred_center, pred_size = model.predict(pc, color, utterance, objects)

        visualize(
            pc, color,
            gt_center, gt_size,
            pred_center, pred_size,
            utterance, gt_label, scan_id,
            sample_idx=idx,
            target_id=target_id,
            headless=args.headless,
            save_dir=args.save_dir,
            max_vis_points=args.max_vis_points,
        )
        visualized += 1
        idx += 1

    return visualized, skipped_missing, skipped_invalid


def run_real_visualization(args, scans_dir, meta_dir):
    """用真实 checkpoint + 评测集索引做可视化。"""
    print(f"[1/4] 初始化真实模型与评测集")
    model = RealGroundingModel(
        args.model_path,
        data_root=args.data_root,
        device=args.device,
    )
    print(
        f"[INFO] 真实模型模式使用 checkpoint 测试集: "
        f"{model.config.test_dataset} ({model.dataset.split} split)"
    )
    print(f"  共 {len(model)} 条评测样本")
    if model.epoch is not None:
        print(f"  Checkpoint epoch: {model.epoch}")

    idx = args.sample_idx
    visualized = 0
    skipped_missing = 0
    skipped_invalid = 0
    while idx < len(model) and visualized < args.num_samples:
        meta = model.get_sample_metadata(idx)
        scan_id = meta['scan_id']
        target_id = meta['target_id']
        utterance = meta['utterance']

        print(f"\n[2/4] 样本 {idx}: 加载场景 {scan_id}")
        missing = get_missing_scene_files(scan_id, scans_dir)
        if missing:
            if args.strict_missing:
                missing_fmt = '\n'.join(f'  - {path}' for path in missing)
                raise FileNotFoundError(
                    f'样本 {idx} 的场景 {scan_id} 缺少必要文件:\n{missing_fmt}'
                )
            print(
                f"[WARN] 跳过样本 {idx}: 场景 {scan_id} 缺少 "
                f"{len(missing)} 个文件"
            )
            skipped_missing += 1
            idx += 1
            continue

        pc, color, objects = load_scannet_scene(scan_id, scans_dir, meta_dir)

        try:
            gt_center, gt_size, gt_label = get_gt_box(pc, objects, target_id)
        except ValueError as exc:
            if args.strict_missing:
                raise
            print(f"[WARN] 跳过样本 {idx}: {exc}")
            skipped_invalid += 1
            idx += 1
            continue

        print(f"[3/4] 推理: \"{utterance[:60]}...\"")
        pred = model.predict(idx)
        print(
            f"[INFO] 使用 {pred['prefix']} query={pred['query_idx']} "
            f"score={pred['score']:.4f}"
        )

        print(f"[4/4] 可视化")
        visualize(
            pc, color,
            gt_center, gt_size,
            pred['pred_center'], pred['pred_size'],
            utterance, gt_label or pred.get('target_name', ''), scan_id,
            sample_idx=idx,
            target_id=target_id,
            headless=args.headless,
            save_dir=args.save_dir,
            max_vis_points=args.max_vis_points,
        )
        visualized += 1
        idx += 1

    return visualized, skipped_missing, skipped_invalid


def main():
    parser = argparse.ArgumentParser(description='3D Visual Grounding 可视化')
    parser.add_argument(
        '--dataset', type=str, default='nr3d',
        choices=['nr3d', 'sr3d'],
        help='选择 Nr3D 或 Sr3D 数据集'
    )
    parser.add_argument(
        '--data_root', type=str,
        default='/root/autodl-tmp/DATA_ROOT',
        help='数据根目录 (包含 refer_it_3d/, scannet/scans/ 等)'
    )
    parser.add_argument(
        '--sample_idx', type=int, default=0,
        help='要可视化的样本索引'
    )
    parser.add_argument(
        '--model_path', type=str, default=None,
        help='模型权重路径 (不提供则使用 Mock 模型)'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        help='推理设备: auto/cpu/cuda'
    )
    parser.add_argument(
        '--num_samples', type=int, default=5,
        help='连续可视化的样本数（关闭一个窗口后显示下一个）'
    )
    parser.add_argument(
        '--strict_missing', action='store_true',
        help='遇到缺失的 ScanNet 场景文件时立即报错，而不是跳过'
    )
    parser.add_argument(
        '--headless', action='store_true',
        help='强制使用无头模式，将可视化结果保存为 PNG 而不是打开窗口'
    )
    parser.add_argument(
        '--save_dir', type=str, default='outputs/grounding_viz',
        help='无头模式下导出静态图的目录'
    )
    parser.add_argument(
        '--max_vis_points', type=int, default=30000,
        help='无头模式下最多绘制的点数，避免导出过慢'
    )
    args = parser.parse_args()

    # --- 路径 ---
    scans_dir = os.path.join(args.data_root, 'scannet', 'scans')
    meta_dir = 'data/meta_data'
    if args.model_path:
        visualized, skipped_missing, skipped_invalid = run_real_visualization(
            args, scans_dir, meta_dir
        )
    else:
        visualized, skipped_missing, skipped_invalid = run_mock_visualization(
            args, scans_dir, meta_dir
        )

    print("\n可视化完成!")
    print(
        f"  实际可视化样本: {visualized} | "
        f"跳过缺失场景: {skipped_missing} | "
        f"跳过无效标注: {skipped_invalid}"
    )
    if visualized == 0:
        print("  提示: 当前 sample_idx 之后没有找到可用场景，可尝试增大 sample_idx。")


if __name__ == '__main__':
    main()
