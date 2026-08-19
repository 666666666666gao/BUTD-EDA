# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------

from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import torch.nn.functional as F

from .source_pool_selector import compute_soft_token_base_scores
from .source_pool_selector import compute_contrastive_token_base_scores
from .detector_policy_sources import DETECTOR_POLICY_SCORE_KEYS
import torch.distributed as dist
from models.structured_losses import (
    compute_dhc_losses,
    compute_s2s_aux_losses,
    _build_pos_mask,
    _build_target_pos_mask,
    _dataset_not_scannet_mask,
    _structured_valid_mask,
    _write_structured_debug,
    _masked_hard_negative,
)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def box_cxcyczwhd_to_xyzxyz(x):
    x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    x_c, y_c, z_c, w, h, d = x.unbind(-1)
    w = torch.clamp(torch.abs(w), min=1e-6)
    h = torch.clamp(torch.abs(h), min=1e-6)
    d = torch.clamp(torch.abs(d), min=1e-6)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (z_c - 0.5 * d),
         (x_c + 0.5 * w), (y_c + 0.5 * h), (z_c + 0.5 * d)]
    return torch.stack(b, dim=-1)


def _volume_par(box):
    return (
        (box[:, 3] - box[:, 0])
        * (box[:, 4] - box[:, 1])
        * (box[:, 5] - box[:, 2])
    )


def _intersect_par(box_a, box_b):
    xA = torch.max(box_a[:, 0][:, None], box_b[:, 0][None, :])
    yA = torch.max(box_a[:, 1][:, None], box_b[:, 1][None, :])
    zA = torch.max(box_a[:, 2][:, None], box_b[:, 2][None, :])
    xB = torch.min(box_a[:, 3][:, None], box_b[:, 3][None, :])
    yB = torch.min(box_a[:, 4][:, None], box_b[:, 4][None, :])
    zB = torch.min(box_a[:, 5][:, None], box_b[:, 5][None, :])
    return (
        torch.clamp(xB - xA, 0)
        * torch.clamp(yB - yA, 0)
        * torch.clamp(zB - zA, 0)
    )


def _iou3d_par(box_a, box_b):
    intersection = _intersect_par(box_a, box_b)
    vol_a = _volume_par(box_a)
    vol_b = _volume_par(box_b)
    union = vol_a[:, None] + vol_b[None, :] - intersection
    return intersection / union, union


def generalized_box_iou3d(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check

    boxes1 = torch.nan_to_num(boxes1, nan=0.0, posinf=1e4, neginf=-1e4)
    boxes2 = torch.nan_to_num(boxes2, nan=0.0, posinf=1e4, neginf=-1e4)
    boxes1_min = torch.min(boxes1[:, :3], boxes1[:, 3:])
    boxes1_max = torch.max(boxes1[:, :3], boxes1[:, 3:])
    boxes2_min = torch.min(boxes2[:, :3], boxes2[:, 3:])
    boxes2_max = torch.max(boxes2[:, :3], boxes2[:, 3:])
    boxes1 = torch.cat([boxes1_min, boxes1_max], dim=-1)
    boxes2 = torch.cat([boxes2_min, boxes2_max], dim=-1)
    iou, union = _iou3d_par(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :3], boxes2[:, :3])
    rb = torch.max(boxes1[:, None, 3:], boxes2[:, 3:])

    wh = (rb - lt).clamp(min=0)  # [N,M,3]
    volume = wh[:, :, 0] * wh[:, :, 1] * wh[:, :, 2]

    return iou - (volume - union) / volume


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.

    This class is taken from Group-Free code.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        """
        Args:
            gamma: Weighting parameter for hard and easy examples.
            alpha: Weighting parameter for positive and negative examples.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input, target):
        """
        PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
        max(x, 0) - x * z + log(1 + exp(-abs(x))) in

        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #proposals, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = (
            torch.clamp(input, min=0) - input * target
            + torch.log1p(torch.exp(-torch.abs(input)))
        )
        return loss

    def forward(self, input, target, weights):
        """
        Args:
            input: (B, #proposals, #classes) float tensor.
                Predicted logits for each class
            target: (B, #proposals, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #proposals) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #proposals, #classes) float tensor
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss
        loss = loss.squeeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


def compute_points_obj_cls_loss_hard_topk(end_points, topk):
    box_label_mask = end_points['box_label_mask']
    seed_inds = end_points['seed_inds'].long()  # B, K
    seed_xyz = end_points['seed_xyz']  # B, K, 3
    seeds_obj_cls_logits = end_points['seeds_obj_cls_logits']  # B, 1, K
    gt_center = end_points['center_label'][:, :, :3]  # B, G, 3
    gt_size = end_points['size_gts'][:, :, :3]  # B, G, 3
    B = gt_center.shape[0]  # batch size
    K = seed_xyz.shape[1]  # number if points from p++ output
    G = gt_center.shape[1]  # number of gt boxes (with padding)

    # Assign each point to a GT object
    point_instance_label = end_points['point_instance_label']  # B, num_points
    obj_assignment = torch.gather(point_instance_label, 1, seed_inds)  # B, K
    obj_assignment[obj_assignment < 0] = G - 1  # bg points to last gt
    obj_assignment_one_hot = torch.zeros((B, K, G)).to(seed_xyz.device)
    obj_assignment_one_hot.scatter_(2, obj_assignment.unsqueeze(-1), 1)

    # Normalized distances of points and gt centroids
    delta_xyz = seed_xyz.unsqueeze(2) - gt_center.unsqueeze(1)  # (B, K, G, 3)
    delta_xyz = delta_xyz / (gt_size.unsqueeze(1) + 1e-6)  # (B, K, G, 3)
    new_dist = torch.sum(delta_xyz ** 2, dim=-1)
    euclidean_dist1 = torch.sqrt(new_dist + 1e-6)  # BxKxG
    euclidean_dist1 = (
        euclidean_dist1 * obj_assignment_one_hot
        + 100 * (1 - obj_assignment_one_hot)
    )  # BxKxG
    euclidean_dist1 = euclidean_dist1.transpose(1, 2).contiguous()  # BxGxK

    # Find the points that lie closest to each gt centroid
    topk_inds = (
        torch.topk(euclidean_dist1, topk, largest=False)[1]
        * box_label_mask[:, :, None]
        + (box_label_mask[:, :, None] - 1)
    )  # BxGxtopk
    topk_inds = topk_inds.long()  # BxGxtopk
    topk_inds = topk_inds.view(B, -1).contiguous()  # B, Gxtopk
    batch_inds = torch.arange(B)[:, None].repeat(1, G*topk).to(seed_xyz.device)
    batch_topk_inds = torch.stack([
        batch_inds,
        topk_inds
    ], -1).view(-1, 2).contiguous()

    # Topk points closest to each centroid are marked as true objects
    objectness_label = torch.zeros((B, K + 1)).long().to(seed_xyz.device)
    objectness_label[batch_topk_inds[:, 0], batch_topk_inds[:, 1]] = 1
    objectness_label = objectness_label[:, :K]
    objectness_label_mask = torch.gather(point_instance_label, 1, seed_inds)
    objectness_label[objectness_label_mask < 0] = 0

    # Compute objectness loss
    criterion = SigmoidFocalClassificationLoss()
    cls_weights = (objectness_label >= 0).float()
    cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
    cls_weights /= torch.clamp(cls_normalizer, min=1.0)
    cls_loss_src = criterion(
        seeds_obj_cls_logits.view(B, K, 1),
        objectness_label.unsqueeze(-1),
        weights=cls_weights
    )
    objectness_loss = cls_loss_src.sum() / B

    return objectness_loss


class HungarianMatcher(nn.Module):
    """
    Assign targets to predictions.

    This class is taken from MDETR and is modified for our purposes.

    For efficiency reasons, the targets don't include the no_object.
    Because of this, in general, there are more predictions than targets.
    In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class=1, cost_bbox=5, cost_giou=2,
                 soft_token=False):
        """
        Initialize matcher.

        Args:
            cost_class: relative weight of the classification error
            cost_bbox: relative weight of the L1 bounding box regression error
            cost_giou: relative weight of the giou loss of the bounding box
            soft_token: whether to use soft-token prediction
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        self.soft_token = soft_token

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Perform the matching.

        Args:
            outputs: This is a dict that contains at least these entries:
                "pred_logits" (tensor): [batch_size, num_queries, num_classes]
                "pred_boxes" (tensor): [batch_size, num_queries, 6], cxcyczwhd
            targets: list (len(targets) = batch_size) of dict:
                "labels" (tensor): [num_target_boxes]
                    (where num_target_boxes is the no. of ground-truth objects)
                "boxes" (tensor): [num_target_boxes, 6], cxcyczwhd
                "positive_map" (tensor): [num_target_boxes, 256]

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j):
                - index_i is the indices of the selected predictions
                - index_j is the indices of the corresponding selected targets
            For each batch element, it holds:
            len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # Notation: {B: batch_size, Q: num_queries, C: num_classes}
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).float().softmax(-1)  # [B*Q, C]
        out_bbox = outputs["pred_boxes"].flatten(0, 1).float()  # [B*Q, 6]

        # Also concat the target labels and boxes
        positive_map = torch.cat([t["positive_map"] for t in targets]).float()
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets]).float()

        if self.soft_token:
            # pad if necessary
            if out_prob.shape[-1] != positive_map.shape[-1]:
                positive_map = positive_map[..., :out_prob.shape[-1]]
            cost_class = -torch.matmul(out_prob, positive_map.transpose(0, 1))
        else:
            # Compute the classification cost.
            # Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching,
            # it can be ommitted. DETR
            # out_prob = out_prob * out_objectness.view(-1, 1)
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou3d(
            box_cxcyczwhd_to_xyzxyz(out_bbox),
            box_cxcyczwhd_to_xyzxyz(tgt_bbox)
        )

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox
            + self.cost_class * cost_class
            + self.cost_giou * cost_giou
        ).float().view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [
            linear_sum_assignment(c[i])
            for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),  # matched pred boxes
                torch.as_tensor(j, dtype=torch.int64)  # corresponding gt boxes
            )
            for i, j in indices
        ]


class SetCriterion(nn.Module):
    """
    Computes the loss in two steps:
        1) compute hungarian assignment between ground truth and outputs
        2) supervise each pair of matched ground-truth / prediction
    """

    def __init__(self, matcher, losses={}, eos_coef=0.1, temperature=0.07):
        """
        Parameters:
            matcher: module that matches targets and proposals
            losses: list of all the losses to be applied
            eos_coef: weight of the no-object category
            temperature: used to sharpen the contrastive logits
        """
        super().__init__()
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        self.temperature = temperature

    def loss_labels_st(self, outputs, targets, indices, num_boxes):
        """Soft token prediction (with objectness)."""
        logits = outputs["pred_logits"].log_softmax(-1)  # (B, Q, 256)
        positive_map = torch.cat([t["positive_map"] for t in targets])

        # Trick to get target indices across batches
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)

        # Labels, by default lines map to the last element, no_object
        tgt_pos = positive_map[tgt_idx]
        target_sim = torch.zeros_like(logits)
        target_sim[:, :, -1] = 1
        target_sim[src_idx] = tgt_pos

        # Compute entropy
        entropy = torch.log(target_sim + 1e-6) * target_sim
        loss_ce = (entropy - logits * target_sim).sum(-1)

        # Weight less 'no_object'
        eos_coef = torch.full(
            loss_ce.shape, self.eos_coef,
            device=target_sim.device
        )
        eos_coef[src_idx] = 1
        loss_ce = loss_ce * eos_coef
        loss_ce = loss_ce.sum() / num_boxes

        losses = {"loss_ce": loss_ce}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute bbox losses."""
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([
            t['boxes'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)

        loss_bbox = (
            F.l1_loss(
                src_boxes[..., :3], target_boxes[..., :3],
                reduction='none'
            )
            + 0.2 * F.l1_loss(
                src_boxes[..., 3:], target_boxes[..., 3:],
                reduction='none'
            )
        )
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou3d(
            box_cxcyczwhd_to_xyzxyz(src_boxes),
            box_cxcyczwhd_to_xyzxyz(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_contrastive_align(self, outputs, targets, indices, num_boxes):
        """Compute contrastive losses between projected queries and tokens."""
        tokenized = outputs["tokenized"]

        # Contrastive logits
        norm_text_emb = outputs["proj_tokens"]  # B, num_tokens, dim
        norm_img_emb = outputs["proj_queries"]  # B, num_queries, dim
        num_tokens = norm_text_emb.shape[-2]  # Number of text tokens
        logits = (
            torch.matmul(norm_img_emb, norm_text_emb.transpose(-1, -2))
            / self.temperature
        )  # B, num_queries, num_tokens

        # construct a map such that positive_map[k, i, j] = True
        # iff query i is associated to token j in batch item k
        positive_map = torch.zeros(logits.shape, device=logits.device)

        # Handle 'not mentioned' token (always the last valid token in utterance)
        # Use tokenized attention_mask to find the last valid token
        inds = tokenized['attention_mask'].sum(1) - 1
        # Clamp indices to valid range
        inds = torch.clamp(inds, max=logits.shape[-1] - 1)
        positive_map[torch.arange(len(inds)), :, inds] = 0.5
        if logits.shape[-1] > 1:
            positive_map[torch.arange(len(inds)), :, inds - 1] = 0.5

        # Handle true mentions from positive_map
        pmap = torch.cat([
            t['positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)
        idx = self._get_src_permutation_idx(indices)
        # Crop positive_map to match logits shape (in case num_tokens < 256)
        positive_map[idx] = pmap[..., :logits.shape[-1]]
        positive_map = positive_map > 0

        # Mask for matches <> 'not mentioned'
        mask = torch.full(
            logits.shape[:2],
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )
        mask[idx] = 1.0
        # Token mask for matches <> 'not mentioned'
        tmask = torch.full(
            (len(logits), logits.shape[-1]),
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )
        valid_inds = inds.clamp(max=logits.shape[-1] - 1)
        tmask[torch.arange(len(valid_inds)), valid_inds] = 1.0

        # Positive logits are those who correspond to a match
        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits

        # Loss 1: which tokens should each query match?
        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        neg_term = negative_logits.logsumexp(2)
        nb_pos = positive_map.sum(2) + 1e-6
        entropy = -torch.log(nb_pos+1e-6) / nb_pos  # entropy of 1/nb_pos
        box_to_token_loss_ = (
            (entropy + pos_term / nb_pos + neg_term)
        ).masked_fill(~boxes_with_pos, 0)
        box_to_token_loss = (box_to_token_loss_ * mask).sum()

        # Loss 2: which queries should each token match?
        tokens_with_pos = positive_map.any(1)
        pos_term = positive_logits.sum(1)
        neg_term = negative_logits.logsumexp(1)
        nb_pos = positive_map.sum(1) + 1e-6
        entropy = -torch.log(nb_pos+1e-6) / nb_pos
        token_to_box_loss = (
            (entropy + pos_term / nb_pos + neg_term)
        ).masked_fill(~tokens_with_pos, 0)
        token_to_box_loss = (token_to_box_loss * tmask).sum()

        tot_loss = (box_to_token_loss + token_to_box_loss) / 2
        return {"loss_contrastive_align": tot_loss / num_boxes}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([
            torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)
        ])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels_st,
            'boxes': self.loss_boxes,
            'contrastive_align': self.loss_contrastive_align
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """
        Perform the loss computation.

        Parameters:
             outputs: dict of tensors
             targets: list of dicts, such that len(targets) == batch_size.
        """
        # Retrieve the matching between outputs and targets
        indices = self.matcher(outputs, targets)

        num_boxes = sum(len(inds[1]) for inds in indices)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float,
            device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / dist.get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(
                loss, outputs, targets, indices, num_boxes
            ))

        return losses, indices


@torch.no_grad()
def _target_iou_matrix(end_points, prefix='last_'):
    """Return IoU between each predicted query and the primary target GT."""
    pred_center = end_points[f'{prefix}center']
    pred_size = end_points[f'{prefix}pred_size']
    pred_bbox = torch.cat([pred_center, pred_size], dim=-1).detach()
    gt_center = end_points['center_label'][:, :, 0:3]
    gt_size = end_points['size_gts']
    gt_bbox = torch.cat([gt_center, gt_size], dim=-1).detach()
    box_label_mask = end_points['box_label_mask'].bool()
    B, Q = pred_bbox.shape[:2]
    iou_rows = []
    for b in range(B):
        valid_gt = box_label_mask[b]
        if valid_gt.any():
            tgt_box = gt_bbox[b, :1]
            ious, _ = _iou3d_par(
                box_cxcyczwhd_to_xyzxyz(tgt_box),
                box_cxcyczwhd_to_xyzxyz(pred_bbox[b])
            )
            iou_rows.append(torch.nan_to_num(ious[0], nan=0.0, posinf=0.0, neginf=0.0))
        else:
            iou_rows.append(torch.zeros(Q, device=pred_bbox.device))
    return torch.stack(iou_rows, dim=0).clamp(0.0, 1.0)


def _soft_token_base_scores(end_points, prefix='last_'):
    sem_key = f'{prefix}sem_cls_scores'
    if sem_key not in end_points or 'positive_map' not in end_points:
        return None
    try:
        return compute_soft_token_base_scores(
            end_points[sem_key],
            end_points['positive_map'],
            box_label_mask=end_points.get('box_label_mask', None),
        )
    except ValueError:
        return None


def _score_source(end_points, source):
    if source == 'base':
        soft_token_scores = _soft_token_base_scores(end_points, prefix='last_')
        if soft_token_scores is not None:
            return soft_token_scores
        key = 'base_grounding_scores'
    elif source == 'structured':
        key = 'structured_scores'
    elif source == 'quality':
        key = 'pred_iou'
    elif source == 'fused':
        key = 'fused_scores'
    elif source == 'contrastive_base':
        key = 'bbf_base_grounding_scores'
        if key not in end_points:
            required = ('proj_tokens', 'last_proj_queries', 'positive_map')
            if all(field in end_points for field in required):
                return compute_contrastive_token_base_scores(
                    end_points['last_proj_queries'],
                    end_points['proj_tokens'],
                    end_points['positive_map'],
                    box_label_mask=end_points.get('box_label_mask', None),
                ).float()
    elif source == 'detector_policy_adapter':
        key = 'detector_policy_adapter_scores'
    elif source in DETECTOR_POLICY_SCORE_KEYS:
        key = DETECTOR_POLICY_SCORE_KEYS[source]
    else:
        raise ValueError(f"Unknown score source: {source}")
    if key not in end_points:
        raise ValueError(f"Requested {source} scores but {key} is missing")
    return end_points[key].float()


def _selector_choice_source_names(end_points, num_scores):
    source_names = end_points.get('selector_choice_source_names', None)
    if source_names is None:
        return ('base', 'fused', 'quality', 'contrastive_base')[:num_scores]
    source_names = tuple(str(source) for source in source_names)
    return source_names[:num_scores]


def _source_pool_candidate_sources(source, max_sources=None):
    if source == 'source_pool':
        sources = ('base', 'fused', 'quality')
        if max_sources is not None and int(max_sources) > len(sources):
            sources = sources + ('contrastive_base',)
        return sources[:max_sources] if max_sources is not None else sources
    if source == 'external_pool':
        sources = ('base', 'fused')
        return sources[:max_sources] if max_sources is not None else sources
    if source == 'detector_policy_adapter':
        sources = ('detector_policy_adapter',)
        return sources[:max_sources] if max_sources is not None else sources
    raise ValueError(f"Unknown source pool: {source}")


def _quality_topk_candidate_mask(end_points, source, quality_scores, k):
    B, Q = quality_scores.shape
    device = quality_scores.device
    candidate_mask = torch.zeros(B, Q, device=device, dtype=torch.bool)

    if source in ('source_pool', 'external_pool'):
        pool_sources = _source_pool_candidate_sources(source)
        num_sources = 0
        for pool_source in pool_sources:
            try:
                source_scores = _score_source(end_points, pool_source)
            except ValueError:
                continue
            if source_scores.shape != quality_scores.shape:
                raise ValueError(
                    f"{pool_source} scores shape {tuple(source_scores.shape)} "
                    f"does not match quality scores shape "
                    f"{tuple(quality_scores.shape)}"
                )
            source_scores = torch.nan_to_num(
                source_scores.float().detach(),
                nan=0.0, posinf=1e4, neginf=-1e4,
            )
            topk_idx = torch.topk(source_scores, k, dim=1).indices
            candidate_mask.scatter_(1, topk_idx, True)
            num_sources += 1
        if num_sources == 0:
            raise ValueError(
                "Requested source_pool scores but no compatible source scores "
                "are available"
            )
        return candidate_mask, num_sources

    source_scores = _score_source(end_points, source).float().detach()
    source_scores = torch.nan_to_num(
        source_scores, nan=0.0, posinf=1e4, neginf=-1e4
    )
    topk_idx = torch.topk(source_scores, k, dim=1).indices
    candidate_mask.scatter_(1, topk_idx, True)
    return candidate_mask, 1


def _global_only_mask(end_points, B, device):
    mask = end_points.get('global_only_mask', None)
    if mask is None:
        return torch.zeros(B, device=device, dtype=torch.bool)
    return mask.to(device=device).bool()


def _quality_losses(end_points, weight=1.0, iou_threshold=0.25):
    logits = end_points['quality_logits'].float()
    pred_iou = end_points['pred_iou'].float()
    labels = _target_iou_matrix(end_points, prefix='last_').detach()
    valid = _dataset_not_scannet_mask(end_points, logits.shape[0], logits.device)
    valid_q = valid.unsqueeze(1).float()
    reg_raw = F.smooth_l1_loss(
        torch.sigmoid(logits), labels, reduction='none'
    )
    cls_targets = (labels >= 0.5).float()
    cls_raw = F.binary_cross_entropy_with_logits(
        logits, cls_targets, reduction='none'
    )
    denom = valid_q.sum().mul(logits.shape[1]).clamp(min=1.0)
    reg_raw = (reg_raw * valid_q).sum() / denom
    cls_raw = (cls_raw * valid_q).sum() / denom
    loss_raw = reg_raw + cls_raw
    flat_valid = valid.unsqueeze(1).expand_as(labels).bool()
    label_flat = labels[flat_valid]
    pred_flat = pred_iou[flat_valid]
    corr = logits.new_tensor(0.0)
    corr_warning = 1.0
    if label_flat.numel() > 1:
        label_centered = label_flat - label_flat.mean()
        pred_centered = pred_flat - pred_flat.mean()
        corr_denom = (
            label_centered.pow(2).sum().sqrt()
            * pred_centered.pow(2).sum().sqrt()
        )
        if torch.isfinite(corr_denom).item() and corr_denom.item() > 1e-12:
            corr = (label_centered * pred_centered).sum() / corr_denom
            corr_warning = 0.0
    quality_top = pred_iou.argmax(dim=1)
    quality_top_iou = labels[
        torch.arange(labels.shape[0], device=labels.device), quality_top
    ]
    if 'base_grounding_scores' in end_points:
        base_scores = end_points['base_grounding_scores'].float()
        base_top = base_scores.argmax(dim=1)
        base_top_iou = labels[
            torch.arange(labels.shape[0], device=labels.device), base_top
        ]
        top1_improvement = (
            (quality_top_iou - base_top_iou) * valid.float()
        ).sum() / valid.float().sum().clamp(min=1.0)
        top1_improves_ratio = (
            ((quality_top_iou > base_top_iou).float() * valid.float()).sum()
            / valid.float().sum().clamp(min=1.0)
        )
        base_top_iou_mean = (
            base_top_iou * valid.float()
        ).sum() / valid.float().sum().clamp(min=1.0)
    else:
        top1_improvement = logits.new_tensor(0.0)
        top1_improves_ratio = logits.new_tensor(0.0)
        base_top_iou_mean = logits.new_tensor(0.0)
    losses = {
        'loss_quality': loss_raw * weight,
        'dbg_quality_reg_raw': reg_raw,
        'dbg_quality_cls_raw': cls_raw,
        'dbg_quality_total_raw': loss_raw,
        'dbg_quality_label_iou_mean': (
            labels * valid_q
        ).sum() / denom,
        'dbg_quality_target_iou_mean': (
            labels * valid_q
        ).sum() / denom,
        'dbg_quality_target_iou_std': torch.sqrt(
            (((labels - ((labels * valid_q).sum() / denom)) * valid_q).pow(2)).sum()
            / denom
        ),
        'dbg_quality_pred_iou_mean': (
            pred_iou * valid_q
        ).sum() / denom,
        'dbg_quality_pred_iou_std': torch.sqrt(
            (((pred_iou - ((pred_iou * valid_q).sum() / denom)) * valid_q).pow(2)).sum()
            / denom
        ),
        'dbg_quality_positive_ratio': (
            ((labels >= iou_threshold).float() * valid_q).sum()
            / denom
        ),
        'dbg_quality_iou50_positive_ratio': (
            ((labels >= 0.50).float() * valid_q).sum() / denom
        ),
        'dbg_quality_positive50_ratio': (
            ((labels >= 0.50).float() * valid_q).sum() / denom
        ),
        'dbg_quality_pred_positive50_ratio': (
            ((pred_iou >= 0.50).float() * valid_q).sum() / denom
        ),
        'dbg_quality_iou_corr': corr,
        'dbg_quality_pred_target_iou_corr': corr,
        'dbg_quality_corr_pred_iou_target_iou': corr,
        'dbg_quality_top1_iou': (
            quality_top_iou * valid.float()
        ).sum() / valid.float().sum().clamp(min=1.0),
        'dbg_quality_top1_quality_iou': (
            quality_top_iou * valid.float()
        ).sum() / valid.float().sum().clamp(min=1.0),
        'dbg_quality_base_top1_iou': base_top_iou_mean,
        'dbg_quality_top1_base_iou': base_top_iou_mean,
        'dbg_quality_top1_iou_improvement': top1_improvement,
        'dbg_quality_top1_improves_ratio': top1_improves_ratio,
        'dbg_quality_top1_quality_improves_ratio': top1_improves_ratio,
        'dbg_quality_valid_batch_ratio': valid.float().mean(),
        'dbg_warn_quality_corr_unavailable': float(corr_warning),
        'dbg_warn_quality_corr_unavailable_ratio': float(corr_warning),
    }
    return losses


def _quality_topk_rerank_losses(end_points, weight=1.0, source='fused',
                                candidate_k=5, margin=0.05,
                                min_iou_gap=0.02):
    quality_scores = end_points['pred_iou'].float()
    target_iou = _target_iou_matrix(end_points, prefix='last_').detach()
    B, Q = quality_scores.shape
    device = quality_scores.device
    k = max(1, min(int(candidate_k), Q))

    candidate_mask, num_candidate_sources = _quality_topk_candidate_mask(
        end_points, source, quality_scores, k
    )
    masked_iou = target_iou.masked_fill(~candidate_mask, -1.0)
    pos_idx = masked_iou.argmax(dim=1)
    batch_idx = torch.arange(B, device=device)
    pos_score = quality_scores[batch_idx, pos_idx]
    pos_iou = target_iou[batch_idx, pos_idx]

    pos_mask = torch.zeros(B, Q, device=device, dtype=torch.bool)
    pos_mask[batch_idx, pos_idx] = True
    competitor_mask = (
        candidate_mask
        & (~pos_mask)
        & (target_iou <= (pos_iou.unsqueeze(1) - float(min_iou_gap)))
    )

    neg_scores = quality_scores.masked_fill(
        ~competitor_mask, torch.finfo(quality_scores.dtype).min
    )
    neg_score, neg_idx = neg_scores.max(dim=1)
    neg_iou = target_iou[batch_idx, neg_idx]

    valid = competitor_mask.any(dim=1)
    if 'box_label_mask' in end_points:
        target_valid = end_points['box_label_mask'][:, 0].to(
            device=device
        ).bool()
        valid = valid & target_valid
    valid = valid & _dataset_not_scannet_mask(end_points, B, device)
    valid_f = valid.float()
    valid_count = valid_f.sum().clamp(min=1.0)

    margin_t = quality_scores.new_tensor(float(margin))
    per_sample = F.relu(neg_score - pos_score + margin_t) * valid_f
    loss_raw = per_sample.sum() / valid_count
    violation = ((neg_score - pos_score + margin_t) > 0) & valid
    score_gap = pos_score - neg_score
    iou_gap = pos_iou - neg_iou
    total_queries = max(float(B * Q), 1.0)

    return {
        'loss_quality_topk_rerank': loss_raw * weight,
        'loss_quality_topk_rerank_raw': loss_raw,
        'dbg_quality_topk_rerank_loss_raw': loss_raw,
        'dbg_quality_topk_rerank_valid_ratio': valid.float().mean(),
        'dbg_quality_topk_rerank_candidate_query_ratio': (
            candidate_mask.float().sum() / total_queries
        ),
        'dbg_quality_topk_rerank_competitor_query_ratio': (
            competitor_mask.float().sum() / total_queries
        ),
        'dbg_quality_topk_rerank_pos_iou': (
            pos_iou * valid_f
        ).sum() / valid_count,
        'dbg_quality_topk_rerank_neg_iou': (
            neg_iou * valid_f
        ).sum() / valid_count,
        'dbg_quality_topk_rerank_iou_gap': (
            iou_gap * valid_f
        ).sum() / valid_count,
        'dbg_quality_topk_rerank_pos_score': (
            pos_score * valid_f
        ).sum() / valid_count,
        'dbg_quality_topk_rerank_neg_score': (
            neg_score * valid_f
        ).sum() / valid_count,
        'dbg_quality_topk_rerank_score_gap': (
            score_gap * valid_f
        ).sum() / valid_count,
        'dbg_quality_topk_rerank_violation_ratio': (
            violation.float().sum() / valid_count
        ),
        'dbg_quality_topk_rerank_k': float(k),
        'dbg_quality_topk_rerank_margin': float(margin),
        'dbg_quality_topk_rerank_min_iou_gap': float(min_iou_gap),
        'dbg_quality_topk_rerank_source_base': float(source == 'base'),
        'dbg_quality_topk_rerank_source_structured': float(
            source == 'structured'
        ),
        'dbg_quality_topk_rerank_source_quality': float(source == 'quality'),
        'dbg_quality_topk_rerank_source_fused': float(source == 'fused'),
        'dbg_quality_topk_rerank_source_pool': float(
            source == 'source_pool'
        ),
        'dbg_quality_topk_rerank_source_external_pool': float(
            source == 'external_pool'
        ),
        'dbg_quality_topk_rerank_num_candidate_sources': float(
            num_candidate_sources
        ),
        'dbg_warn_quality_topk_rerank_no_valid_samples': float(
            valid.float().sum().detach().item() == 0
        ),
        'dbg_warn_quality_topk_rerank_no_valid_samples_ratio': float(
            valid.float().sum().detach().item() == 0
        ),
    }


def _detector_policy_adapter_losses(
    end_points, weight=1.0, candidate_k=5, margin=0.05, min_iou_gap=0.02
):
    adapter_scores = end_points['detector_policy_adapter_scores'].float()
    target_iou = _target_iou_matrix(end_points, prefix='last_').detach()
    B, Q = adapter_scores.shape
    device = adapter_scores.device
    k = max(1, min(int(candidate_k), Q))
    topk_idx = torch.topk(
        torch.nan_to_num(adapter_scores.detach(), nan=0.0, posinf=1e4, neginf=-1e4),
        k,
        dim=1,
    ).indices
    candidate_mask = torch.zeros(B, Q, device=device, dtype=torch.bool)
    candidate_mask.scatter_(1, topk_idx, True)

    masked_iou = target_iou.masked_fill(~candidate_mask, -1.0)
    pos_idx = masked_iou.argmax(dim=1)
    batch_idx = torch.arange(B, device=device)
    pos_score = adapter_scores[batch_idx, pos_idx]
    pos_iou = target_iou[batch_idx, pos_idx]

    pos_mask = torch.zeros(B, Q, device=device, dtype=torch.bool)
    pos_mask[batch_idx, pos_idx] = True
    competitor_mask = (
        candidate_mask
        & (~pos_mask)
        & (target_iou <= (pos_iou.unsqueeze(1) - float(min_iou_gap)))
    )
    neg_scores = adapter_scores.masked_fill(
        ~competitor_mask, torch.finfo(adapter_scores.dtype).min
    )
    neg_score, neg_idx = neg_scores.max(dim=1)
    neg_iou = target_iou[batch_idx, neg_idx]

    valid = competitor_mask.any(dim=1)
    if 'box_label_mask' in end_points:
        target_valid = end_points['box_label_mask'][:, 0].to(
            device=device
        ).bool()
        valid = valid & target_valid
    valid = valid & _dataset_not_scannet_mask(end_points, B, device)
    valid_f = valid.float()
    valid_count = valid_f.sum().clamp(min=1.0)

    margin_t = adapter_scores.new_tensor(float(margin))
    per_sample = F.relu(neg_score - pos_score + margin_t) * valid_f
    loss_raw = per_sample.sum() / valid_count
    violation = ((neg_score - pos_score + margin_t) > 0) & valid
    score_gap = pos_score - neg_score
    iou_gap = pos_iou - neg_iou

    regularizer = adapter_scores.sum() * 0.0
    weights = end_points.get('detector_policy_adapter_weights', None)
    prior = end_points.get('detector_policy_adapter_prior_weights', None)
    if weights is not None and prior is not None:
        prior = prior.to(device=weights.device, dtype=weights.dtype)
        regularizer = (weights - prior).pow(2).mean()

    return {
        'loss_detector_policy_adapter': loss_raw * weight,
        'loss_detector_policy_adapter_raw': loss_raw,
        'loss_detector_policy_adapter_reg': regularizer,
        'dbg_detector_policy_adapter_loss_raw': loss_raw,
        'dbg_detector_policy_adapter_reg': regularizer,
        'dbg_detector_policy_adapter_valid_ratio': valid.float().mean(),
        'dbg_detector_policy_adapter_candidate_query_ratio': (
            candidate_mask.float().mean()
        ),
        'dbg_detector_policy_adapter_competitor_query_ratio': (
            competitor_mask.float().mean()
        ),
        'dbg_detector_policy_adapter_pos_iou': (
            pos_iou * valid_f
        ).sum() / valid_count,
        'dbg_detector_policy_adapter_neg_iou': (
            neg_iou * valid_f
        ).sum() / valid_count,
        'dbg_detector_policy_adapter_iou_gap': (
            iou_gap * valid_f
        ).sum() / valid_count,
        'dbg_detector_policy_adapter_pos_score': (
            pos_score * valid_f
        ).sum() / valid_count,
        'dbg_detector_policy_adapter_neg_score': (
            neg_score * valid_f
        ).sum() / valid_count,
        'dbg_detector_policy_adapter_score_gap': (
            score_gap * valid_f
        ).sum() / valid_count,
        'dbg_detector_policy_adapter_violation_ratio': (
            violation.float().sum() / valid_count
        ),
        'dbg_detector_policy_adapter_k': float(k),
        'dbg_detector_policy_adapter_margin': float(margin),
        'dbg_detector_policy_adapter_min_iou_gap': float(min_iou_gap),
        'dbg_warn_detector_policy_adapter_no_valid_samples': float(
            valid.float().sum().detach().item() == 0
        ),
        'dbg_warn_detector_policy_adapter_no_valid_samples_ratio': float(
            valid.float().sum().detach().item() == 0
        ),
    }


def _source_pool_selector_source_aware_losses(
    end_points, selector_source_scores, weight=1.0, source='source_pool',
    candidate_k=5, temperature=1.0, min_iou_gap=0.02,
    pairwise_weight=0.5, choice_target='iou'
):
    selector_source_scores = selector_source_scores.float()
    selector_scores = end_points['selector_scores'].float()
    target_iou = _target_iou_matrix(end_points, prefix='last_').detach()
    B, Q = selector_scores.shape
    device = selector_scores.device
    k = max(1, min(int(candidate_k), Q))
    pool_sources = _source_pool_candidate_sources(
        source,
        max_sources=selector_source_scores.shape[1],
    )
    num_sources = min(selector_source_scores.shape[1], len(pool_sources))
    candidate_mask = torch.zeros(
        B, num_sources, Q, device=device, dtype=torch.bool
    )
    num_candidate_sources = 0
    for source_idx, pool_source in enumerate(pool_sources[:num_sources]):
        try:
            source_scores = _score_source(end_points, pool_source)
        except ValueError:
            continue
        source_scores = torch.nan_to_num(
            source_scores.float().detach(),
            nan=0.0, posinf=1e4, neginf=-1e4,
        )
        topk_idx = torch.topk(source_scores, k, dim=1).indices
        candidate_mask[:, source_idx].scatter_(1, topk_idx, True)
        num_candidate_sources += 1
    if num_candidate_sources == 0:
        raise ValueError(
            "Requested source-aware source_pool selector loss but no "
            "compatible source scores are available"
        )

    candidate_logits = selector_source_scores[:, :num_sources, :]
    safe_temperature = max(float(temperature), 1e-6)
    if choice_target == 'source_pool_lse':
        score_mask = candidate_mask.any(dim=1)
        source_masked_logits = candidate_logits.masked_fill(
            ~candidate_mask, torch.finfo(candidate_logits.dtype).min
        )
        score_logits = torch.logsumexp(source_masked_logits, dim=1)
        score_iou = target_iou
    else:
        score_mask = candidate_mask.flatten(1)
        score_logits = candidate_logits.flatten(1)
        score_iou = target_iou.unsqueeze(1).expand_as(
            candidate_logits
        ).flatten(1)
    threshold_utility_target = choice_target in (
        'threshold_utility_softmax',
        'threshold_utility_hard',
        'quality_override',
    )
    if threshold_utility_target:
        score_target_value = (
            score_iou
            + (score_iou >= 0.25).float()
            + (score_iou >= 0.50).float()
        )
    else:
        score_target_value = score_iou
    masked_iou = score_target_value.masked_fill(~score_mask, -1.0)
    batch_idx = torch.arange(B, device=device)
    if choice_target == 'quality_override':
        quality_idx = 2
        if num_sources <= quality_idx:
            raise ValueError(
                "quality_override source_pool target requires quality scores"
            )
        target_values_3d = score_target_value.view(B, num_sources, Q)
        mask_3d = score_mask.view(B, num_sources, Q)
        quality_mask = mask_3d[:, quality_idx]
        nonquality_mask = mask_3d.clone()
        nonquality_mask[:, quality_idx] = False
        min_value = torch.finfo(score_target_value.dtype).min
        quality_values = target_values_3d[:, quality_idx].masked_fill(
            ~quality_mask, min_value
        )
        nonquality_values = target_values_3d.masked_fill(
            ~nonquality_mask, min_value
        ).view(B, -1)
        best_quality_value, best_quality_query = quality_values.max(dim=1)
        best_nonquality_value, best_nonquality_idx = nonquality_values.max(
            dim=1
        )
        quality_flat_idx = quality_idx * Q + best_quality_query
        quality_available = quality_mask.any(dim=1)
        nonquality_available = nonquality_mask.view(B, -1).any(dim=1)
        override = (
            quality_available
            & nonquality_available
            & (
                best_nonquality_value
                >= best_quality_value + float(min_iou_gap)
            )
        )
        fallback_idx = masked_iou.argmax(dim=1)
        pos_idx = torch.where(override, best_nonquality_idx, quality_flat_idx)
        pos_idx = torch.where(quality_available, pos_idx, fallback_idx)
    else:
        pos_idx = masked_iou.argmax(dim=1)
    pos_score = score_logits[batch_idx, pos_idx]
    pos_iou = score_iou[batch_idx, pos_idx]

    pos_mask = torch.zeros_like(score_mask)
    pos_mask[batch_idx, pos_idx] = True
    pos_target_value = score_target_value[batch_idx, pos_idx]
    if choice_target == 'quality_override':
        competitor_mask = score_mask & (~pos_mask)
        acceptable_mask = pos_mask
    elif choice_target == 'threshold_utility_hard':
        competitor_mask = (
            score_mask
            & (~pos_mask)
            & (
                score_target_value
                <= (pos_target_value.unsqueeze(1) - float(min_iou_gap))
            )
        )
    else:
        competitor_mask = (
            score_mask
            & (~pos_mask)
            & (score_iou <= (pos_iou.unsqueeze(1) - float(min_iou_gap)))
        )

    threshold_targets = None
    if choice_target == 'threshold_bucket_bce':
        threshold_targets = (
            (score_iou >= 0.25).float()
            + (score_iou >= 0.50).float()
        ) / 2.0
        threshold_targets = threshold_targets.masked_fill(~score_mask, 0.0)
        positive_candidate = score_mask & (threshold_targets > 0)
        negative_candidate = score_mask & (threshold_targets <= 0)
        valid = (
            score_mask.any(dim=1)
            & positive_candidate.any(dim=1)
            & negative_candidate.any(dim=1)
        )
    elif choice_target == 'threshold_utility_softmax':
        valid = score_mask.float().sum(dim=1) > 1
    elif choice_target == 'threshold_utility_hard':
        valid = competitor_mask.any(dim=1)
    elif choice_target == 'quality_override':
        valid = competitor_mask.any(dim=1)
    else:
        valid = competitor_mask.any(dim=1)
    if 'box_label_mask' in end_points:
        target_valid = end_points['box_label_mask'][:, 0].to(
            device=device
        ).bool()
        valid = valid & target_valid
    valid = valid & _dataset_not_scannet_mask(end_points, B, device)
    valid_f = valid.float()
    valid_count = valid_f.sum().clamp(min=1.0)

    masked_logits = score_logits.masked_fill(
        ~score_mask, torch.finfo(score_logits.dtype).min
    )
    if bool(valid.any().detach().item()) and choice_target == 'threshold_bucket_bce':
        bce_logits = score_logits[valid] / safe_temperature
        bce_targets = threshold_targets[valid]
        bce_weights = score_mask[valid].float()
        bce_loss = F.binary_cross_entropy_with_logits(
            bce_logits,
            bce_targets,
            reduction='none',
        )
        loss_ce = (
            (bce_loss * bce_weights).sum()
            / bce_weights.sum().clamp(min=1e-6)
        )
    elif bool(valid.any().detach().item()) and choice_target == 'threshold_utility_softmax':
        utility_targets = score_target_value.masked_fill(
            ~score_mask, torch.finfo(score_target_value.dtype).min
        )
        target_probs = F.softmax(utility_targets[valid], dim=1).detach()
        log_probs = F.log_softmax(
            masked_logits[valid] / safe_temperature,
            dim=1,
        )
        loss_ce = -(target_probs * log_probs).sum(dim=1).mean()
    elif bool(valid.any().detach().item()):
        loss_ce = F.cross_entropy(
            masked_logits[valid] / safe_temperature,
            pos_idx[valid],
            reduction='mean',
        )
    else:
        loss_ce = score_logits.sum() * 0.0

    selected_idx = masked_logits.argmax(dim=1)
    selected_iou = score_iou[batch_idx, selected_idx]
    neg_scores = score_logits.masked_fill(
        ~competitor_mask, torch.finfo(score_logits.dtype).min
    )
    neg_score, neg_idx = neg_scores.max(dim=1)
    neg_iou = score_iou[batch_idx, neg_idx]
    score_gap = pos_score - neg_score
    iou_gap = pos_iou - neg_iou
    gap_valid = valid & competitor_mask.any(dim=1)
    gap_valid_f = gap_valid.float()
    gap_valid_count = gap_valid_f.sum().clamp(min=1.0)
    safe_neg_score = torch.where(gap_valid, neg_score, pos_score)
    safe_neg_iou = torch.where(gap_valid, neg_iou, pos_iou)
    safe_score_gap = torch.where(
        gap_valid, score_gap, torch.zeros_like(score_gap)
    )
    safe_iou_gap = torch.where(
        gap_valid, iou_gap, torch.zeros_like(iou_gap)
    )
    if choice_target == 'threshold_utility_hard':
        neg_target_value = score_target_value[batch_idx, neg_idx]
        target_value_gap = pos_target_value - neg_target_value
        pairwise_margin = target_value_gap.detach().clamp(
            min=float(min_iou_gap),
            max=1.0,
        )
    else:
        pairwise_margin = float(min_iou_gap)
    pairwise_raw = F.relu(pairwise_margin - score_gap)
    pairwise_loss = (
        (pairwise_raw * valid_f).sum() / valid_count
        if float(pairwise_weight) > 0 else score_logits.sum() * 0.0
    )
    loss_raw = loss_ce + (float(pairwise_weight) * pairwise_loss)
    violation = (selected_idx != pos_idx) & valid
    total_queries = max(float(B * Q), 1.0)

    return {
        'loss_source_pool_selector': loss_raw * weight,
        'loss_source_pool_selector_raw': loss_raw,
        'dbg_source_pool_selector_loss_raw': loss_raw,
        'dbg_source_pool_selector_loss_ce': loss_ce,
        'dbg_source_pool_selector_pairwise_loss': pairwise_loss,
        'dbg_source_pool_selector_pairwise_weight': float(pairwise_weight),
        'dbg_source_pool_selector_valid_ratio': valid.float().mean(),
        'dbg_source_pool_selector_candidate_query_ratio': (
            candidate_mask.any(dim=1).float().sum() / total_queries
        ),
        'dbg_source_pool_selector_competitor_query_ratio': (
            (
                competitor_mask.float().sum()
                if choice_target == 'source_pool_lse'
                else competitor_mask.view(
                    B, num_sources, Q
                ).any(dim=1).float().sum()
            ) / total_queries
        ),
        'dbg_source_pool_selector_pos_iou': (
            pos_iou * valid_f
        ).sum() / valid_count,
        'dbg_source_pool_selector_selected_iou': (
            selected_iou * valid_f
        ).sum() / valid_count,
        'dbg_source_pool_selector_neg_iou': (
            safe_neg_iou * gap_valid_f
        ).sum() / gap_valid_count,
        'dbg_source_pool_selector_iou_gap': (
            safe_iou_gap * gap_valid_f
        ).sum() / gap_valid_count,
        'dbg_source_pool_selector_pos_score': (
            pos_score * gap_valid_f
        ).sum() / gap_valid_count,
        'dbg_source_pool_selector_neg_score': (
            safe_neg_score * gap_valid_f
        ).sum() / gap_valid_count,
        'dbg_source_pool_selector_score_gap': (
            safe_score_gap * gap_valid_f
        ).sum() / gap_valid_count,
        'dbg_source_pool_selector_violation_ratio': (
            violation.float().sum() / valid_count
        ),
        'dbg_source_pool_selector_k': float(k),
        'dbg_source_pool_selector_temperature': float(safe_temperature),
        'dbg_source_pool_selector_min_iou_gap': float(min_iou_gap),
        'dbg_source_pool_selector_source_base': float(source == 'base'),
        'dbg_source_pool_selector_source_structured': float(
            source == 'structured'
        ),
        'dbg_source_pool_selector_source_quality': float(source == 'quality'),
        'dbg_source_pool_selector_source_fused': float(source == 'fused'),
        'dbg_source_pool_selector_source_pool': float(source == 'source_pool'),
        'dbg_source_pool_selector_source_external_pool': float(
            source == 'external_pool'
        ),
        'dbg_source_pool_selector_num_candidate_sources': float(
            num_candidate_sources
        ),
        'dbg_source_pool_selector_choice_target_iou': float(
            choice_target == 'iou'
        ),
        'dbg_source_pool_selector_choice_target_threshold_bucket_bce': float(
            choice_target == 'threshold_bucket_bce'
        ),
        'dbg_source_pool_selector_choice_target_threshold_utility_softmax': float(
            choice_target == 'threshold_utility_softmax'
        ),
        'dbg_source_pool_selector_choice_target_threshold_utility_hard': float(
            choice_target == 'threshold_utility_hard'
        ),
        'dbg_source_pool_selector_choice_target_source_pool_lse': float(
            choice_target == 'source_pool_lse'
        ),
        'dbg_source_pool_selector_choice_target_quality_override': float(
            choice_target == 'quality_override'
        ),
        'dbg_warn_source_pool_selector_no_valid_samples': float(
            valid.float().sum().detach().item() == 0
        ),
        'dbg_warn_source_pool_selector_no_valid_samples_ratio': float(
            valid.float().sum().detach().item() == 0
        ),
    }


def _source_pool_selector_choice_losses(
    end_points, choice_scores, weight=1.0, temperature=1.0,
    min_iou_gap=0.02, choice_balance=False, choice_target='iou',
    pairwise_weight=0.0, choice_balance_power=1.0,
    override_default_source='base',
    oracle_prior_weight=0.0, override_prior_weight=0.0,
    override_source_prior_weight=0.0, override_margin_weight=0.0,
    override_margin=1.0,
    quality_base_margin_weight=0.0, quality_base_margin=0.75,
    quality_default_margin_weight=0.0, quality_default_margin=0.75,
    quality_default_bidirectional_margin_weight=0.0,
    quality_default_bidirectional_margin=0.75,
    iou_aux_weight=0.0, iou_aux_margin=0.5,
    false_base_weight=1.0, false_override_weight=1.0,
    override_utility_gap_weight=0.0,
    sourcewise_negative_weight=1.0,
    source_min_iou_gaps=None,
):
    choice_scores = choice_scores.float()
    if choice_target not in (
        'iou', 'threshold_utility', 'threshold_bucket',
        'threshold_bucket_bce', 'threshold_bucket_argmax',
        'threshold_bucket_unique', 'threshold_bucket_margin',
        'threshold_utility_softmax',
        'threshold_utility_regression',
        'base_threshold_gain', 'base_override_bce',
        'base_override_focal_bce',
        'base_override_sourcewise_focal_bce',
        'threshold_gain_default_sourcewise_focal_bce',
        'threshold_gain_default_diffquery_sourcewise_focal_bce',
        'precision_gain_default_sourcewise_focal_bce',
        'quality_override'
    ):
        raise ValueError(
            "source_choice target must be one of 'iou', "
            "'threshold_utility', 'threshold_bucket', "
            "'threshold_bucket_bce', 'threshold_bucket_argmax', "
            "'threshold_bucket_unique', 'threshold_bucket_margin', "
            "'threshold_utility_softmax', "
            "'threshold_utility_regression', "
            "'base_threshold_gain', 'base_override_bce', "
            "'base_override_focal_bce', "
            "'base_override_sourcewise_focal_bce', "
            "'threshold_gain_default_sourcewise_focal_bce', "
            "'threshold_gain_default_diffquery_sourcewise_focal_bce', "
            "'precision_gain_default_sourcewise_focal_bce', "
            "or 'quality_override'"
        )
    target_iou = _target_iou_matrix(end_points, prefix='last_').detach()
    B = choice_scores.shape[0]
    device = choice_scores.device
    source_names = _selector_choice_source_names(
        end_points, choice_scores.shape[1]
    )
    base_idx = source_names.index('base') if 'base' in source_names else 0
    override_default_source = str(
        end_points.get(
            'source_pool_selector_override_default_source',
            override_default_source,
        )
    )
    override_default_idx = (
        source_names.index(override_default_source)
        if override_default_source in source_names
        else base_idx
    )
    quality_idx = (
        source_names.index('quality')
        if 'quality' in source_names
        else min(2, max(len(source_names) - 1, 0))
    )
    num_sources = min(choice_scores.shape[1], len(source_names))
    if num_sources < 1:
        raise ValueError("selector_choice_scores must contain at least one source")
    source_gap_values = []
    source_min_iou_gaps = source_min_iou_gaps or {}
    for source in source_names[:num_sources]:
        source_gap_values.append(
            float(source_min_iou_gaps.get(source, min_iou_gap))
        )
    source_gap_tensor = choice_scores.new_tensor(source_gap_values).view(
        1, num_sources
    )

    top_iou = torch.zeros(B, num_sources, device=device)
    top_query = torch.full(
        (B, num_sources),
        -1,
        device=device,
        dtype=torch.long,
    )
    available = torch.zeros(B, num_sources, device=device, dtype=torch.bool)
    batch_idx = torch.arange(B, device=device)
    num_candidate_sources = 0
    for source_idx, source in enumerate(source_names[:num_sources]):
        try:
            source_scores = _score_source(end_points, source)
        except ValueError:
            continue
        if source_scores.shape != target_iou.shape:
            raise ValueError(
                f"{source} scores shape {tuple(source_scores.shape)} does "
                f"not match target IoU shape {tuple(target_iou.shape)}"
            )
        source_scores = torch.nan_to_num(
            source_scores.float().detach(),
            nan=0.0, posinf=1e4, neginf=-1e4,
        )
        top_idx = torch.topk(source_scores, 1, dim=1).indices.squeeze(1)
        top_query[:, source_idx] = top_idx
        top_iou[:, source_idx] = target_iou[batch_idx, top_idx]
        available[:, source_idx] = True
        num_candidate_sources += 1
    if num_candidate_sources == 0:
        raise ValueError(
            "Requested source_choice selector loss but no compatible source "
            "scores are available"
        )

    if choice_target in (
        'threshold_utility', 'threshold_utility_softmax',
        'threshold_utility_regression',
        'quality_override'
    ):
        target_values = (
            top_iou
            + (top_iou >= 0.25).float()
            + (top_iou >= 0.50).float()
        )
    elif choice_target in (
        'threshold_bucket', 'threshold_bucket_bce',
        'threshold_bucket_argmax', 'threshold_bucket_unique',
        'threshold_bucket_margin', 'base_threshold_gain',
        'base_override_bce', 'base_override_focal_bce',
        'base_override_sourcewise_focal_bce',
        'threshold_gain_default_sourcewise_focal_bce',
        'threshold_gain_default_diffquery_sourcewise_focal_bce',
        'precision_gain_default_sourcewise_focal_bce'
    ):
        target_values = (
            (top_iou >= 0.25).float()
            + (top_iou >= 0.50).float()
        )
    else:
        target_values = top_iou

    masked_target_values = target_values.masked_fill(~available, -1.0)
    if choice_target == 'quality_override':
        if num_sources <= quality_idx:
            raise ValueError(
                "quality_override source_choice target requires quality scores"
            )
        quality_available = available[:, quality_idx]
        nonquality_mask = available.clone()
        nonquality_mask[:, quality_idx] = False
        nonquality_values = target_values.masked_fill(
            ~nonquality_mask, torch.finfo(target_values.dtype).min
        )
        best_nonquality_value, best_nonquality_source = nonquality_values.max(
            dim=1
        )
        quality_source = torch.full_like(best_nonquality_source, quality_idx)
        quality_value = target_values[:, quality_idx]
        override = (
            quality_available
            & nonquality_mask.any(dim=1)
            & (
                best_nonquality_value
                >= quality_value + float(min_iou_gap)
            )
        )
        fallback_source = masked_target_values.argmax(dim=1)
        pos_source = torch.where(override, best_nonquality_source, quality_source)
        pos_source = torch.where(quality_available, pos_source, fallback_source)
    elif choice_target in (
        'base_threshold_gain', 'base_override_bce',
        'base_override_focal_bce',
        'base_override_sourcewise_focal_bce',
        'threshold_gain_default_sourcewise_focal_bce',
        'threshold_gain_default_diffquery_sourcewise_focal_bce',
        'precision_gain_default_sourcewise_focal_bce'
    ):
        default_idx = (
            override_default_idx
            if choice_target in (
                'base_override_bce', 'base_override_focal_bce',
                'base_override_sourcewise_focal_bce',
                'threshold_gain_default_sourcewise_focal_bce',
                'threshold_gain_default_diffquery_sourcewise_focal_bce',
                'precision_gain_default_sourcewise_focal_bce'
            )
            else base_idx
        )
        default_available = available[:, default_idx]
        base_source = torch.full(
            (B,), default_idx, device=device, dtype=torch.long
        )
        fallback_source = masked_target_values.argmax(dim=1)
        base_or_fallback_source = torch.where(
            default_available, base_source, fallback_source
        )
        nonbase_mask = available.clone()
        nonbase_mask[:, default_idx] = False
        base_value = target_values[:, default_idx]
        base_iou = top_iou[:, default_idx]
        source_gap = source_gap_tensor.expand(B, -1)
        if choice_target in (
            'threshold_gain_default_sourcewise_focal_bce',
            'threshold_gain_default_diffquery_sourcewise_focal_bce',
            'precision_gain_default_sourcewise_focal_bce'
        ):
            threshold_gain_mask = (
                nonbase_mask
                & (target_values > base_value.unsqueeze(1))
                & default_available.unsqueeze(1)
            )
            if choice_target == 'precision_gain_default_sourcewise_focal_bce':
                default_top_query = top_query[:, default_idx]
                default_top_valid = default_top_query >= 0
                threshold_gain_mask = (
                    threshold_gain_mask
                    & default_top_valid.unsqueeze(1)
                    & (top_query >= 0)
                    & (top_query != default_top_query.unsqueeze(1))
                    & (
                        top_iou
                        >= base_iou.unsqueeze(1) + source_gap
                    )
                )
        else:
            threshold_gain_mask = (
                nonbase_mask
                & (
                    (target_values > base_value.unsqueeze(1))
                    | (
                        (target_values == base_value.unsqueeze(1))
                        & (
                            top_iou
                            >= base_iou.unsqueeze(1) + source_gap
                        )
                    )
                )
                & default_available.unsqueeze(1)
            )
        gain_utility = (target_values * 10.0) + top_iou
        gain_utility = gain_utility.masked_fill(
            ~threshold_gain_mask,
            torch.finfo(gain_utility.dtype).min,
        )
        best_gain_value, best_gain_source = gain_utility.max(dim=1)
        override = threshold_gain_mask.any(dim=1)
        pos_source = torch.where(
            override, best_gain_source, base_or_fallback_source
        )
        if (
            choice_target
            == 'threshold_gain_default_diffquery_sourcewise_focal_bce'
        ):
            default_top_query = top_query[:, default_idx]
            default_top_valid = default_top_query >= 0
            diffquery_nonbase_mask = (
                nonbase_mask
                & default_top_valid.unsqueeze(1)
                & (top_query >= 0)
                & (top_query != default_top_query.unsqueeze(1))
            )
            diffquery_valid = diffquery_nonbase_mask.any(dim=1)
    elif choice_target == 'threshold_bucket_unique':
        best_bucket = masked_target_values.max(dim=1).values
        unique_positive_mask = (
            available
            & (target_values == best_bucket.unsqueeze(1))
            & (best_bucket.unsqueeze(1) > 0)
        )
        unique_positive = unique_positive_mask.sum(dim=1) == 1
        unique_source = unique_positive_mask.float().argmax(dim=1)
        fallback_source = masked_target_values.argmax(dim=1)
        pos_source = torch.where(unique_positive, unique_source, fallback_source)
    elif choice_target == 'threshold_bucket_margin':
        best_bucket = masked_target_values.max(dim=1).values
        best_bucket_mask = (
            available
            & (target_values == best_bucket.unsqueeze(1))
            & (best_bucket.unsqueeze(1) > 0)
        )
        bucket_iou = top_iou.masked_fill(~best_bucket_mask, -1.0)
        best_iou, best_source = bucket_iou.max(dim=1)
        if num_sources > 1:
            top2_iou = torch.topk(bucket_iou, 2, dim=1).values[:, 1]
        else:
            top2_iou = torch.full_like(best_iou, -1.0)
        bucket_source_count = best_bucket_mask.sum(dim=1)
        bucket_margin_valid = (
            (bucket_source_count == 1)
            | (
                (bucket_source_count > 1)
                & (best_iou >= top2_iou + float(min_iou_gap))
            )
        )
        bucket_margin_valid = bucket_margin_valid & (bucket_source_count > 0)
        fallback_source = masked_target_values.argmax(dim=1)
        pos_source = torch.where(
            bucket_margin_valid, best_source, fallback_source
        )
    else:
        pos_source = masked_target_values.argmax(dim=1)
    pos_iou = top_iou[batch_idx, pos_source]
    pos_target_value = target_values[batch_idx, pos_source]
    pos_score = choice_scores[batch_idx, pos_source]
    if 'diffquery_valid' not in locals():
        diffquery_valid = torch.ones(B, device=device, dtype=torch.bool)

    pos_mask = torch.zeros_like(available)
    pos_mask[batch_idx, pos_source] = True
    if choice_target == 'quality_override':
        competitor_mask = available & (~pos_mask)
        acceptable_mask = pos_mask
    elif choice_target in (
        'base_threshold_gain', 'base_override_bce',
        'base_override_focal_bce',
        'base_override_sourcewise_focal_bce',
        'threshold_gain_default_sourcewise_focal_bce',
        'threshold_gain_default_diffquery_sourcewise_focal_bce',
        'precision_gain_default_sourcewise_focal_bce'
    ):
        competitor_mask = available & (~pos_mask)
        acceptable_mask = pos_mask
    elif choice_target == 'threshold_bucket_argmax':
        positive_bucket = pos_target_value.unsqueeze(1) > 0
        competitor_mask = available & (~pos_mask) & positive_bucket
        acceptable_mask = pos_mask & positive_bucket
    elif choice_target == 'threshold_bucket_unique':
        competitor_mask = available & (~pos_mask)
        acceptable_mask = pos_mask
    elif choice_target == 'threshold_bucket_margin':
        competitor_mask = available & (~pos_mask)
        acceptable_mask = pos_mask
    elif choice_target in ('threshold_bucket', 'threshold_bucket_bce'):
        competitor_mask = (
            available
            & (
                target_values
                < pos_target_value.unsqueeze(1)
            )
        )
        acceptable_mask = (
            available
            & (
                target_values
                == pos_target_value.unsqueeze(1)
            )
            & (pos_target_value.unsqueeze(1) > 0)
        )
    elif choice_target in (
        'threshold_utility_softmax',
        'threshold_utility_regression',
    ):
        competitor_mask = available & (~pos_mask)
        acceptable_mask = available & (target_values > 0.0)
    else:
        competitor_mask = (
            available
            & (~pos_mask)
            & (
                target_values
                <= (pos_target_value.unsqueeze(1) - float(min_iou_gap))
            )
        )
        acceptable_mask = (
            available
            & (
                target_values
                >= (pos_target_value.unsqueeze(1) - float(min_iou_gap))
            )
        )

    valid = competitor_mask.any(dim=1)
    if choice_target in (
        'threshold_utility_softmax',
        'threshold_utility_regression',
    ):
        valid = available.float().sum(dim=1) > 1
    if choice_target == 'threshold_bucket_unique':
        valid = valid & unique_positive
    elif choice_target == 'threshold_bucket_margin':
        valid = valid & bucket_margin_valid
    elif (
        choice_target
        == 'threshold_gain_default_diffquery_sourcewise_focal_bce'
    ):
        valid = valid & diffquery_valid
    if 'box_label_mask' in end_points:
        target_valid = end_points['box_label_mask'][:, 0].to(
            device=device
        ).bool()
        valid = valid & target_valid
    valid = valid & _dataset_not_scannet_mask(end_points, B, device)
    valid_f = valid.float()
    valid_count = valid_f.sum().clamp(min=1.0)

    safe_temperature = max(float(temperature), 1e-6)
    masked_logits = choice_scores[:, :num_sources].masked_fill(
        ~available, torch.finfo(choice_scores.dtype).min
    )
    class_weights = choice_scores.new_ones(num_sources)
    if bool(valid.any().detach().item()) and bool(choice_balance):
        class_counts = torch.stack([
            (pos_source[valid] == source_idx).float().sum()
            for source_idx in range(num_sources)
        ])
        present_classes = class_counts > 0
        class_weights = choice_scores.new_ones(num_sources)
        class_weights[present_classes] = (
            valid_f.sum()
            / (
                present_classes.float().sum().clamp(min=1.0)
                * class_counts[present_classes].clamp(min=1.0)
            )
        )
        class_weights[present_classes] = class_weights[present_classes].pow(
            float(choice_balance_power)
        )
    oracle_prior_loss = choice_scores.sum() * 0.0
    override_prior_loss = choice_scores.sum() * 0.0
    override_source_prior_loss = choice_scores.sum() * 0.0
    override_margin_loss = choice_scores.sum() * 0.0
    quality_base_margin_loss = choice_scores.sum() * 0.0
    quality_default_margin_loss = choice_scores.sum() * 0.0
    quality_default_bidirectional_margin_loss = choice_scores.sum() * 0.0
    quality_base_margin_valid = torch.zeros(B, device=device, dtype=torch.bool)
    quality_default_margin_valid = torch.zeros(
        B, device=device, dtype=torch.bool
    )
    quality_default_bidirectional_margin_valid = torch.zeros(
        B, device=device, dtype=torch.bool
    )
    quality_base_margin_score_gap = choice_scores.new_zeros(B)
    quality_base_margin_iou_gap = choice_scores.new_zeros(B)
    quality_default_margin_score_gap = choice_scores.new_zeros(B)
    quality_default_margin_iou_gap = choice_scores.new_zeros(B)
    quality_default_bidirectional_margin_score_gap = (
        choice_scores.new_zeros(B)
    )
    quality_default_bidirectional_margin_iou_gap = choice_scores.new_zeros(B)
    iou_aux_loss = choice_scores.sum() * 0.0
    iou_aux_valid = torch.zeros(B, device=device, dtype=torch.bool)
    iou_aux_score_gap = choice_scores.new_zeros(B)
    iou_aux_iou_gap = choice_scores.new_zeros(B)
    override_utility_gap_for_diag = choice_scores.new_zeros(B)
    if float(oracle_prior_weight) > 0.0 and bool(valid.any().detach().item()):
        oracle_targets = F.one_hot(
            pos_source[valid], num_classes=num_sources
        ).to(dtype=choice_scores.dtype)
        pred_probs = torch.softmax(
            masked_logits[valid] / safe_temperature, dim=1
        )
        oracle_hist = oracle_targets.mean(dim=0)
        pred_hist = pred_probs.mean(dim=0)
        eps = torch.finfo(choice_scores.dtype).eps
        oracle_hist = (oracle_hist + eps)
        oracle_hist = oracle_hist / oracle_hist.sum().clamp(min=eps)
        pred_hist = (pred_hist + eps)
        pred_hist = pred_hist / pred_hist.sum().clamp(min=eps)
        mixture = 0.5 * (oracle_hist + pred_hist)
        oracle_prior_loss = 0.5 * (
            F.kl_div(mixture.log(), pred_hist, reduction='sum')
            + F.kl_div(mixture.log(), oracle_hist, reduction='sum')
        )
    if (
        float(override_source_prior_weight) > 0.0
        and choice_target == 'quality_override'
        and num_sources > quality_idx
        and bool(valid.any().detach().item())
    ):
        nonquality_valid = valid & (pos_source != quality_idx)
        if bool(nonquality_valid.any().detach().item()):
            nonquality_logits = masked_logits.clone()
            nonquality_logits[:, quality_idx] = torch.finfo(
                choice_scores.dtype
            ).min
            oracle_nonquality_targets = F.one_hot(
                pos_source[nonquality_valid],
                num_classes=num_sources,
            ).to(dtype=choice_scores.dtype)
            pred_nonquality_probs = torch.softmax(
                nonquality_logits[nonquality_valid] / safe_temperature,
                dim=1,
            )
            oracle_nonquality_hist = oracle_nonquality_targets.mean(dim=0)
            pred_nonquality_hist = pred_nonquality_probs.mean(dim=0)
            eps = torch.finfo(choice_scores.dtype).eps
            oracle_nonquality_hist = oracle_nonquality_hist + eps
            oracle_nonquality_hist = (
                oracle_nonquality_hist
                / oracle_nonquality_hist.sum().clamp(min=eps)
            )
            pred_nonquality_hist = pred_nonquality_hist + eps
            pred_nonquality_hist = (
                pred_nonquality_hist
                / pred_nonquality_hist.sum().clamp(min=eps)
            )
            mixture = 0.5 * (
                oracle_nonquality_hist + pred_nonquality_hist
            )
            override_source_prior_loss = 0.5 * (
                F.kl_div(
                    mixture.log(),
                    pred_nonquality_hist,
                    reduction='sum',
                )
                + F.kl_div(
                    mixture.log(),
                    oracle_nonquality_hist,
                    reduction='sum',
                )
            )
    override_logits_for_selection = None
    override_default_idx_for_selection = base_idx
    if bool(valid.any().detach().item()):
        override_head_used = False
        quality_override_head = (
            choice_target == 'quality_override'
            and end_points.get('selector_choice_override_logit') is not None
        )
        if (
            choice_target in ('base_override_bce', 'base_override_focal_bce')
            or choice_target == 'base_override_sourcewise_focal_bce'
            or choice_target == 'threshold_gain_default_sourcewise_focal_bce'
            or (
                choice_target
                == 'threshold_gain_default_diffquery_sourcewise_focal_bce'
            )
            or choice_target == 'precision_gain_default_sourcewise_focal_bce'
            or quality_override_head
        ):
            default_idx = (
                quality_idx if quality_override_head else override_default_idx
            )
            source_logits = choice_scores[:, :num_sources]
            nondefault_available = available[:, :num_sources].clone()
            nondefault_available[:, default_idx] = False
            nondefault_logits = source_logits.masked_fill(
                ~nondefault_available,
                torch.finfo(choice_scores.dtype).min,
            )
            override_logits = end_points.get(
                'selector_choice_override_logit'
            )
            if override_logits is not None:
                override_logits = override_logits.view(-1).to(
                    device=choice_scores.device,
                    dtype=choice_scores.dtype,
                )
                if override_logits.shape[0] != B:
                    raise ValueError(
                        "selector_choice_override_logit must have shape (B,)"
                    )
                override_head_used = True
                override_logits_for_selection = override_logits
                override_default_idx_for_selection = default_idx
            else:
                override_logits = (
                    torch.logsumexp(nondefault_logits, dim=1)
                    - source_logits[:, default_idx]
                )
            override_targets = (pos_source != default_idx).float()
            nondefault_target_values = target_values.masked_fill(
                ~nondefault_available,
                torch.finfo(target_values.dtype).min,
            )
            best_nondefault_target_value = nondefault_target_values.max(
                dim=1
            ).values
            default_target_value = target_values[:, default_idx]
            override_utility_gap_for_diag = (
                best_nondefault_target_value - default_target_value
            )
            binary_weights = choice_scores.new_ones(B)
            if bool(choice_balance):
                override_valid_targets = override_targets[valid]
                keep_count = (override_valid_targets == 0).float().sum()
                override_count = (override_valid_targets == 1).float().sum()
                keep_present = keep_count > 0
                override_present = override_count > 0
                num_present = (
                    keep_present.float() + override_present.float()
                ).clamp(min=1.0)
                keep_weight = (
                    valid_f.sum() / (num_present * keep_count.clamp(min=1.0))
                ).pow(float(choice_balance_power))
                override_weight = (
                    valid_f.sum()
                    / (num_present * override_count.clamp(min=1.0))
                ).pow(float(choice_balance_power))
                binary_weights = torch.where(
                    override_targets.bool(),
                    override_weight.expand_as(binary_weights),
                    keep_weight.expand_as(binary_weights),
                )
            asym_weights = torch.where(
                override_targets.bool(),
                choice_scores.new_full((B,), float(false_base_weight)),
                choice_scores.new_full((B,), float(false_override_weight)),
            )
            binary_weights = binary_weights * asym_weights
            if float(override_utility_gap_weight) > 0.0:
                utility_gap_weights = (
                    1.0
                    + float(override_utility_gap_weight)
                    * override_utility_gap_for_diag.abs().detach().clamp(
                        max=1.0
                    )
                )
                binary_weights = binary_weights * utility_gap_weights
            binary_logits = override_logits[valid] / safe_temperature
            binary_targets = override_targets[valid]
            binary_weights_valid = binary_weights[valid]
            binary_loss_per_sample = F.binary_cross_entropy_with_logits(
                binary_logits,
                binary_targets,
                reduction='none',
            )
            if choice_target in (
                'base_override_focal_bce',
                'base_override_sourcewise_focal_bce',
                'threshold_gain_default_sourcewise_focal_bce',
                'threshold_gain_default_diffquery_sourcewise_focal_bce',
                'precision_gain_default_sourcewise_focal_bce',
            ):
                binary_prob = torch.sigmoid(binary_logits)
                binary_pt = (
                    binary_prob * binary_targets
                    + (1.0 - binary_prob) * (1.0 - binary_targets)
                )
                binary_loss_per_sample = (
                    binary_loss_per_sample * (1.0 - binary_pt).pow(2.0)
                )
            binary_loss = (
                binary_loss_per_sample * binary_weights_valid
            ).sum() / binary_weights_valid.sum().clamp(min=1e-6)
            if float(override_margin_weight) > 0.0:
                signed_logits = torch.where(
                    binary_targets > 0.5,
                    binary_logits,
                    -binary_logits,
                )
                margin_distance = (
                    override_utility_gap_for_diag[valid]
                    - float(min_iou_gap)
                ).abs().detach().clamp(max=1.0)
                margin_weights = binary_weights_valid * (
                    1.0 + margin_distance
                )
                margin_loss_per_sample = F.relu(
                    float(override_margin) - signed_logits
                )
                override_margin_loss = (
                    margin_loss_per_sample * margin_weights
                ).sum() / margin_weights.sum().clamp(min=1e-6)
            if float(override_prior_weight) > 0.0:
                override_prob = torch.sigmoid(binary_logits).mean()
                override_target_rate = binary_targets.mean()
                override_prior_loss = 0.5 * (
                    override_prob - override_target_rate.detach()
                ).pow(2.0)

            nondefault_valid = valid & (pos_source != default_idx)
            if choice_target in (
                'base_override_sourcewise_focal_bce',
                'threshold_gain_default_sourcewise_focal_bce',
                'threshold_gain_default_diffquery_sourcewise_focal_bce',
                'precision_gain_default_sourcewise_focal_bce',
            ):
                sourcewise_logits = source_logits[valid] / safe_temperature
                sourcewise_targets = F.one_hot(
                    pos_source, num_classes=num_sources
                ).to(dtype=choice_scores.dtype)
                sourcewise_targets[:, default_idx] = 0.0
                sourcewise_targets = sourcewise_targets[valid]
                sourcewise_available = nondefault_available[valid].float()
                pos_weight = choice_scores.new_ones(num_sources)
                if bool(choice_balance):
                    pos_count = (
                        sourcewise_targets * sourcewise_available
                    ).sum(dim=0)
                    neg_count = (
                        (1.0 - sourcewise_targets) * sourcewise_available
                    ).sum(dim=0)
                    has_pos = pos_count > 0
                    balanced_pos_weight = (
                        neg_count.clamp(min=1.0)
                        / pos_count.clamp(min=1.0)
                    ).pow(float(choice_balance_power))
                    pos_weight = torch.where(
                        has_pos,
                        balanced_pos_weight,
                        pos_weight,
                    )
                sourcewise_loss_per_source = (
                    F.binary_cross_entropy_with_logits(
                        sourcewise_logits,
                        sourcewise_targets,
                        pos_weight=pos_weight,
                        reduction='none',
                    )
                )
                sourcewise_prob = torch.sigmoid(sourcewise_logits)
                sourcewise_pt = (
                    sourcewise_prob * sourcewise_targets
                    + (1.0 - sourcewise_prob) * (1.0 - sourcewise_targets)
                )
                sourcewise_loss_per_source = (
                    sourcewise_loss_per_source
                    * (1.0 - sourcewise_pt).pow(2.0)
                )
                sourcewise_label_weights = torch.where(
                    sourcewise_targets > 0.5,
                    torch.ones_like(sourcewise_targets),
                    choice_scores.new_full(
                        sourcewise_targets.shape,
                        float(sourcewise_negative_weight),
                    ),
                )
                sourcewise_weights = (
                    sourcewise_available
                    * binary_weights[valid].view(-1, 1)
                    * sourcewise_label_weights
                )
                nondefault_loss = (
                    (sourcewise_loss_per_source * sourcewise_weights).sum()
                    / sourcewise_weights.sum().clamp(min=1e-6)
                )
            elif bool(nondefault_valid.any().detach().item()):
                nondefault_log_probs = F.log_softmax(
                    nondefault_logits[nondefault_valid] / safe_temperature,
                    dim=1,
                )
                nondefault_targets = torch.zeros_like(nondefault_log_probs)
                nondefault_targets.scatter_(
                    1, pos_source[nondefault_valid].view(-1, 1), 1.0
                )
                nondefault_sample_weights = class_weights[
                    pos_source[nondefault_valid]
                ]
                nondefault_loss_per_sample = -(
                    nondefault_targets * nondefault_log_probs
                ).sum(dim=1)
                nondefault_loss = (
                    nondefault_loss_per_sample * nondefault_sample_weights
                ).sum() / nondefault_sample_weights.sum().clamp(min=1e-6)
            else:
                nondefault_loss = choice_scores.sum() * 0.0
            if (
                bool(nondefault_valid.any().detach().item())
                and float(override_source_prior_weight) > 0.0
            ):
                oracle_nondefault_targets = F.one_hot(
                    pos_source[nondefault_valid],
                    num_classes=num_sources,
                ).to(dtype=choice_scores.dtype)
                pred_nondefault_probs = torch.softmax(
                    nondefault_logits[nondefault_valid] / safe_temperature,
                    dim=1,
                )
                oracle_nondefault_hist = oracle_nondefault_targets.mean(dim=0)
                pred_nondefault_hist = pred_nondefault_probs.mean(dim=0)
                eps = torch.finfo(choice_scores.dtype).eps
                oracle_nondefault_hist = oracle_nondefault_hist + eps
                oracle_nondefault_hist = (
                    oracle_nondefault_hist
                    / oracle_nondefault_hist.sum().clamp(min=eps)
                )
                pred_nondefault_hist = pred_nondefault_hist + eps
                pred_nondefault_hist = (
                    pred_nondefault_hist
                    / pred_nondefault_hist.sum().clamp(min=eps)
                )
                mixture = 0.5 * (
                    oracle_nondefault_hist + pred_nondefault_hist
                )
                override_source_prior_loss = 0.5 * (
                    F.kl_div(
                        mixture.log(),
                        pred_nondefault_hist,
                        reduction='sum',
                    )
                    + F.kl_div(
                        mixture.log(),
                        oracle_nondefault_hist,
                        reduction='sum',
                    )
                )
            loss_ce = binary_loss + nondefault_loss
        elif choice_target == 'threshold_bucket_bce':
            bce_logits = choice_scores[:, :num_sources][valid]
            bce_targets = acceptable_mask[valid].float()
            bce_available = available[:, :num_sources][valid].float()
            bce_weights = bce_available * class_weights.view(1, -1)
            bce_loss = F.binary_cross_entropy_with_logits(
                bce_logits / safe_temperature,
                bce_targets,
                reduction='none',
            )
            loss_ce = (
                (bce_loss * bce_weights).sum()
                / bce_weights.sum().clamp(min=1e-6)
            )
        elif choice_target == 'threshold_utility_regression':
            regression_logits = choice_scores[:, :num_sources][valid]
            regression_targets = target_values[:, :num_sources][valid].detach()
            regression_weights = available[:, :num_sources][valid].float()
            regression_loss = F.smooth_l1_loss(
                regression_logits,
                regression_targets,
                reduction='none',
            )
            loss_ce = (
                (regression_loss * regression_weights).sum()
                / regression_weights.sum().clamp(min=1e-6)
            )
        elif choice_target == 'threshold_utility_softmax':
            utility_targets = target_values.masked_fill(
                ~available, torch.finfo(target_values.dtype).min
            )
            target_probs = F.softmax(utility_targets[valid], dim=1).detach()
            log_probs = F.log_softmax(
                masked_logits[valid] / safe_temperature, dim=1
            )
            loss_per_sample = -(target_probs * log_probs).sum(dim=1)
            loss_ce = loss_per_sample.mean()
        else:
            log_probs = F.log_softmax(
                masked_logits[valid] / safe_temperature, dim=1
            )
            target_probs = acceptable_mask[valid].float()
            target_probs = target_probs / target_probs.sum(
                dim=1, keepdim=True
            ).clamp(min=1e-6)
            loss_per_sample = -(
                target_probs * log_probs
            ).sum(dim=1)
            sample_weights = (
                target_probs * class_weights.view(1, -1)
            ).sum(dim=1)
            loss_ce = (
                loss_per_sample * sample_weights
            ).sum() / sample_weights.sum().clamp(min=1e-6)
    else:
        override_head_used = False
        loss_ce = choice_scores.sum() * 0.0

    if bool(override_head_used) and override_logits_for_selection is not None:
        default_idx = int(override_default_idx_for_selection)
        nondefault_selection_logits = masked_logits.clone()
        nondefault_selection_logits[:, default_idx] = torch.finfo(
            choice_scores.dtype
        ).min
        nondefault_source = nondefault_selection_logits.argmax(dim=1)
        nondefault_available = available[:, :num_sources].clone()
        nondefault_available[:, default_idx] = False
        can_override = nondefault_available.any(dim=1)
        default_source = torch.full_like(nondefault_source, default_idx)
        default_available = available[:, default_idx]
        fallback_source = torch.where(
            default_available,
            default_source,
            nondefault_source,
        )
        selected_source = torch.where(
            (override_logits_for_selection > 0.0) & can_override,
            nondefault_source,
            fallback_source,
        )
    else:
        selected_source = masked_logits.argmax(dim=1)
    selected_iou = top_iou[batch_idx, selected_source]
    neg_scores = choice_scores[:, :num_sources].masked_fill(
        ~competitor_mask, torch.finfo(choice_scores.dtype).min
    )
    neg_score, neg_source = neg_scores.max(dim=1)
    neg_iou = top_iou[batch_idx, neg_source]
    neg_target_value = target_values[batch_idx, neg_source]
    neg_score = torch.where(valid, neg_score, torch.zeros_like(neg_score))
    neg_iou = torch.where(valid, neg_iou, torch.zeros_like(neg_iou))
    neg_target_value = torch.where(
        valid, neg_target_value, torch.zeros_like(neg_target_value)
    )
    score_gap = pos_score - neg_score
    iou_gap = pos_iou - neg_iou
    target_value_gap = pos_target_value - neg_target_value
    pairwise_margin = target_value_gap.detach().clamp(
        min=float(min_iou_gap),
        max=1.0,
    )
    pairwise_raw = F.relu(pairwise_margin - score_gap)
    pairwise_loss = (
        (pairwise_raw * valid_f).sum() / valid_count
        if float(pairwise_weight) > 0 else choice_scores.sum() * 0.0
    )
    if (
        float(quality_base_margin_weight) > 0.0
        and 'base' in source_names
        and 'quality' in source_names
    ):
        qb_base_idx = source_names.index('base')
        qb_quality_idx = source_names.index('quality')
        if qb_base_idx < num_sources and qb_quality_idx < num_sources:
            qb_iou_gap = top_iou[:, qb_quality_idx] - top_iou[:, qb_base_idx]
            qb_score_gap = (
                choice_scores[:, qb_quality_idx] - choice_scores[:, qb_base_idx]
            )
            quality_gap = float(
                source_min_iou_gaps.get('quality', min_iou_gap)
            )
            quality_base_margin_valid = (
                available[:, qb_base_idx]
                & available[:, qb_quality_idx]
                & (qb_iou_gap >= quality_gap)
            )
            if 'box_label_mask' in end_points:
                target_valid = end_points['box_label_mask'][:, 0].to(
                    device=device
                ).bool()
                quality_base_margin_valid = (
                    quality_base_margin_valid & target_valid
                )
            quality_base_margin_valid = (
                quality_base_margin_valid
                & _dataset_not_scannet_mask(end_points, B, device)
            )
            quality_base_margin_score_gap = torch.where(
                quality_base_margin_valid,
                qb_score_gap,
                torch.zeros_like(qb_score_gap),
            )
            quality_base_margin_iou_gap = torch.where(
                quality_base_margin_valid,
                qb_iou_gap,
                torch.zeros_like(qb_iou_gap),
            )
            qb_valid_f = quality_base_margin_valid.float()
            qb_valid_count = qb_valid_f.sum().clamp(min=1.0)
            qb_raw = F.relu(float(quality_base_margin) - qb_score_gap)
            quality_base_margin_loss = (qb_raw * qb_valid_f).sum() / qb_valid_count
    if (
        float(quality_default_margin_weight) > 0.0
        and 'quality' in source_names
        and override_default_source in source_names
        and override_default_source != 'quality'
    ):
        qd_quality_idx = source_names.index('quality')
        qd_default_idx = source_names.index(override_default_source)
        if qd_quality_idx < num_sources and qd_default_idx < num_sources:
            qd_iou_gap = (
                top_iou[:, qd_quality_idx] - top_iou[:, qd_default_idx]
            )
            qd_score_gap = (
                choice_scores[:, qd_quality_idx]
                - choice_scores[:, qd_default_idx]
            )
            quality_gap = float(
                source_min_iou_gaps.get('quality', min_iou_gap)
            )
            quality_default_margin_valid = (
                available[:, qd_quality_idx]
                & available[:, qd_default_idx]
                & (qd_iou_gap >= quality_gap)
            )
            if 'box_label_mask' in end_points:
                target_valid = end_points['box_label_mask'][:, 0].to(
                    device=device
                ).bool()
                quality_default_margin_valid = (
                    quality_default_margin_valid & target_valid
                )
            quality_default_margin_valid = (
                quality_default_margin_valid
                & _dataset_not_scannet_mask(end_points, B, device)
            )
            quality_default_margin_score_gap = torch.where(
                quality_default_margin_valid,
                qd_score_gap,
                torch.zeros_like(qd_score_gap),
            )
            quality_default_margin_iou_gap = torch.where(
                quality_default_margin_valid,
                qd_iou_gap,
                torch.zeros_like(qd_iou_gap),
            )
            qd_valid_f = quality_default_margin_valid.float()
            qd_valid_count = qd_valid_f.sum().clamp(min=1.0)
            qd_raw = F.relu(float(quality_default_margin) - qd_score_gap)
            quality_default_margin_loss = (
                qd_raw * qd_valid_f
            ).sum() / qd_valid_count
    if (
        float(quality_default_bidirectional_margin_weight) > 0.0
        and 'quality' in source_names
        and override_default_source in source_names
        and override_default_source != 'quality'
    ):
        qbidir_quality_idx = source_names.index('quality')
        qbidir_default_idx = source_names.index(override_default_source)
        if qbidir_quality_idx < num_sources and qbidir_default_idx < num_sources:
            qbidir_quality_scores = _score_source(end_points, 'quality')
            qbidir_default_scores = _score_source(
                end_points, override_default_source
            )
            qbidir_quality_top = torch.topk(
                qbidir_quality_scores.detach(), 1, dim=1
            ).indices.squeeze(1)
            qbidir_default_top = torch.topk(
                qbidir_default_scores.detach(), 1, dim=1
            ).indices.squeeze(1)
            qbidir_different_query = qbidir_quality_top != qbidir_default_top
            qbidir_iou_gap = (
                top_iou[:, qbidir_quality_idx]
                - top_iou[:, qbidir_default_idx]
            )
            qbidir_abs_iou_gap = qbidir_iou_gap.abs()
            quality_gap = float(
                source_min_iou_gaps.get('quality', min_iou_gap)
            )
            qbidir_valid = (
                available[:, qbidir_quality_idx]
                & available[:, qbidir_default_idx]
                & qbidir_different_query
                & (qbidir_abs_iou_gap >= quality_gap)
            )
            if 'box_label_mask' in end_points:
                target_valid = end_points['box_label_mask'][:, 0].to(
                    device=device
                ).bool()
                qbidir_valid = qbidir_valid & target_valid
            qbidir_valid = qbidir_valid & _dataset_not_scannet_mask(
                end_points, B, device
            )
            qbidir_quality_score = choice_scores[:, qbidir_quality_idx]
            qbidir_default_score = choice_scores[:, qbidir_default_idx]
            qbidir_signed_score_gap = torch.where(
                qbidir_iou_gap >= 0.0,
                qbidir_quality_score - qbidir_default_score,
                qbidir_default_score - qbidir_quality_score,
            )
            quality_default_bidirectional_margin_valid = qbidir_valid
            quality_default_bidirectional_margin_score_gap = torch.where(
                qbidir_valid,
                qbidir_signed_score_gap,
                torch.zeros_like(qbidir_signed_score_gap),
            )
            quality_default_bidirectional_margin_iou_gap = torch.where(
                qbidir_valid,
                qbidir_iou_gap,
                torch.zeros_like(qbidir_iou_gap),
            )
            qbidir_valid_f = qbidir_valid.float()
            qbidir_valid_count = qbidir_valid_f.sum().clamp(min=1.0)
            qbidir_raw = F.relu(
                float(quality_default_bidirectional_margin)
                - qbidir_signed_score_gap
            )
            quality_default_bidirectional_margin_loss = (
                qbidir_raw * qbidir_valid_f
            ).sum() / qbidir_valid_count
    if (
        float(iou_aux_weight) > 0.0
        and choice_target == 'threshold_gain_default_sourcewise_focal_bce'
        and override_default_idx < num_sources
    ):
        aux_default_idx = override_default_idx
        aux_nondefault_available = available[:, :num_sources].clone()
        aux_nondefault_available[:, aux_default_idx] = False
        aux_default_available = available[:, aux_default_idx]
        aux_default_bucket = target_values[:, aux_default_idx]
        aux_default_iou = top_iou[:, aux_default_idx]
        aux_source_gap = source_gap_tensor.expand(B, -1)
        aux_gain_mask = (
            aux_nondefault_available
            & aux_default_available.unsqueeze(1)
            & (target_values[:, :num_sources] == aux_default_bucket.unsqueeze(1))
            & (
                top_iou[:, :num_sources]
                >= aux_default_iou.unsqueeze(1) + aux_source_gap
            )
        )
        aux_candidate_iou = top_iou[:, :num_sources].masked_fill(
            ~aux_gain_mask,
            -1.0,
        )
        aux_pos_iou, aux_pos_source = aux_candidate_iou.max(dim=1)
        aux_pos_mask = torch.zeros_like(aux_nondefault_available)
        aux_pos_mask.scatter_(1, aux_pos_source.view(-1, 1), True)
        aux_pos_mask = aux_pos_mask & aux_gain_mask
        aux_competitor_mask = aux_nondefault_available & (~aux_pos_mask)
        iou_aux_valid = (
            valid
            & aux_gain_mask.any(dim=1)
            & aux_competitor_mask.any(dim=1)
        )
        aux_competitor_scores = choice_scores[:, :num_sources].masked_fill(
            ~aux_competitor_mask,
            torch.finfo(choice_scores.dtype).min,
        )
        aux_neg_score, aux_neg_source = aux_competitor_scores.max(dim=1)
        aux_pos_score = choice_scores[batch_idx, aux_pos_source]
        aux_neg_iou = top_iou[batch_idx, aux_neg_source]
        iou_aux_score_gap = torch.where(
            iou_aux_valid,
            aux_pos_score - aux_neg_score,
            torch.zeros_like(aux_pos_score),
        )
        iou_aux_iou_gap = torch.where(
            iou_aux_valid,
            aux_pos_iou - aux_neg_iou,
            torch.zeros_like(aux_pos_iou),
        )
        iou_aux_valid_f = iou_aux_valid.float()
        iou_aux_count = iou_aux_valid_f.sum().clamp(min=1.0)
        iou_aux_raw = F.relu(float(iou_aux_margin) - iou_aux_score_gap)
        iou_aux_loss = (
            iou_aux_raw * iou_aux_valid_f
        ).sum() / iou_aux_count
    loss_raw = (
        loss_ce
        + (float(pairwise_weight) * pairwise_loss)
        + (float(oracle_prior_weight) * oracle_prior_loss)
        + (float(override_prior_weight) * override_prior_loss)
        + (float(override_source_prior_weight) * override_source_prior_loss)
        + (float(override_margin_weight) * override_margin_loss)
        + (float(quality_base_margin_weight) * quality_base_margin_loss)
        + (
            float(quality_default_margin_weight)
            * quality_default_margin_loss
        )
        + (
            float(quality_default_bidirectional_margin_weight)
            * quality_default_bidirectional_margin_loss
        )
        + (float(iou_aux_weight) * iou_aux_loss)
    )
    violation = (selected_source != pos_source) & valid

    source_target_ratios = []
    source_acceptable_ratios = []
    source_selected_ratios = []
    for source_idx in range(len(source_names)):
        if source_idx < num_sources:
            ratio = (
                ((pos_source == source_idx) & valid).float().sum()
                / valid_count
            )
            acceptable_ratio = (
                (acceptable_mask[:, source_idx] & valid).float().sum()
                / valid_count
            )
            selected_ratio = (
                ((selected_source == source_idx) & valid).float().sum()
                / valid_count
            )
        else:
            ratio = choice_scores.sum() * 0.0
            acceptable_ratio = choice_scores.sum() * 0.0
            selected_ratio = choice_scores.sum() * 0.0
        source_target_ratios.append(ratio)
        source_acceptable_ratios.append(acceptable_ratio)
        source_selected_ratios.append(selected_ratio)

    selected_acceptable = (
        acceptable_mask[batch_idx, selected_source] & valid
    )
    override_default_idx_for_diag = (
        quality_idx
        if choice_target == 'quality_override' and num_sources > quality_idx
        else override_default_idx
    )
    default_available_for_diag = available[:, override_default_idx_for_diag]
    if num_sources > 1:
        nondefault_available_for_diag = available[:, :num_sources].clone()
        nondefault_available_for_diag[:, override_default_idx_for_diag] = False
        nondefault_available_for_diag = nondefault_available_for_diag.any(dim=1)
    else:
        nondefault_available_for_diag = torch.zeros_like(
            default_available_for_diag
        )
    override_diag_valid = (
        valid & default_available_for_diag & nondefault_available_for_diag
    )
    override_diag_f = override_diag_valid.float()
    override_diag_count = override_diag_f.sum().clamp(min=1.0)
    target_override = (
        (pos_source != override_default_idx_for_diag) & override_diag_valid
    )
    selected_override = (
        (selected_source != override_default_idx_for_diag)
        & override_diag_valid
    )
    false_base = (
        target_override
        & (selected_source == override_default_idx_for_diag)
    )
    false_override = (
        (~target_override)
        & selected_override
        & override_diag_valid
    )
    override_agreement = (
        (
            (pos_source != override_default_idx_for_diag)
            == (selected_source != override_default_idx_for_diag)
        )
        & override_diag_valid
    )

    source_logit_means = []
    source_logit_margin_means = []
    for source_idx in range(len(source_names)):
        if source_idx < num_sources:
            source_logit = choice_scores[:, source_idx]
            source_valid = valid & available[:, source_idx]
            source_valid_f = source_valid.float()
            source_count = source_valid_f.sum().clamp(min=1.0)
            source_logit_mean = (
                source_logit * source_valid_f
            ).sum() / source_count

            other_available = available[:, :num_sources].clone()
            other_available[:, source_idx] = False
            has_other = other_available.any(dim=1)
            other_logits = choice_scores[:, :num_sources].masked_fill(
                ~other_available, torch.finfo(choice_scores.dtype).min
            )
            best_other_logit = other_logits.max(dim=1).values
            margin_valid = source_valid & has_other
            margin_valid_f = margin_valid.float()
            margin_count = margin_valid_f.sum().clamp(min=1.0)
            source_margin_mean = (
                (source_logit - best_other_logit) * margin_valid_f
            ).sum() / margin_count
        else:
            source_logit_mean = choice_scores.sum() * 0.0
            source_margin_mean = choice_scores.sum() * 0.0
        source_logit_means.append(source_logit_mean)
        source_logit_margin_means.append(source_margin_mean)

    def source_ratio(name, values):
        if name not in source_names:
            return choice_scores.sum() * 0.0
        idx = source_names.index(name)
        if idx >= len(values):
            return choice_scores.sum() * 0.0
        return values[idx]

    def source_logit_mean(name):
        if name not in source_names:
            return choice_scores.sum() * 0.0
        idx = source_names.index(name)
        if idx >= len(source_logit_means):
            return choice_scores.sum() * 0.0
        return source_logit_means[idx]

    def source_logit_margin_mean(name):
        if name not in source_names:
            return choice_scores.sum() * 0.0
        idx = source_names.index(name)
        if idx >= len(source_logit_margin_means):
            return choice_scores.sum() * 0.0
        return source_logit_margin_means[idx]

    def source_class_weight(name):
        if name not in source_names:
            return choice_scores.sum() * 0.0
        idx = source_names.index(name)
        if idx >= num_sources:
            return choice_scores.sum() * 0.0
        return class_weights[idx]

    debug = {
        'loss_source_pool_selector': loss_raw * weight,
        'loss_source_pool_selector_raw': loss_raw,
        'dbg_source_pool_selector_loss_raw': loss_raw,
        'dbg_source_pool_selector_loss_ce': loss_ce,
        'dbg_source_pool_selector_pairwise_loss': pairwise_loss,
        'dbg_source_pool_selector_pairwise_weight': float(pairwise_weight),
        'dbg_source_pool_selector_oracle_prior_loss': oracle_prior_loss,
        'dbg_source_pool_selector_oracle_prior_weight': float(
            oracle_prior_weight
        ),
        'dbg_source_pool_selector_override_prior_loss': override_prior_loss,
        'dbg_source_pool_selector_override_prior_weight': float(
            override_prior_weight
        ),
        'dbg_source_pool_selector_override_source_prior_loss': (
            override_source_prior_loss
        ),
        'dbg_source_pool_selector_override_source_prior_weight': float(
            override_source_prior_weight
        ),
        'dbg_source_pool_selector_override_margin_loss': (
            override_margin_loss
        ),
        'dbg_source_pool_selector_override_margin_weight': float(
            override_margin_weight
        ),
        'dbg_source_pool_selector_override_margin': float(override_margin),
        'dbg_source_pool_selector_quality_base_margin_loss': (
            quality_base_margin_loss
        ),
        'dbg_source_pool_selector_quality_base_margin_weight': float(
            quality_base_margin_weight
        ),
        'dbg_source_pool_selector_quality_base_margin': float(
            quality_base_margin
        ),
        'dbg_source_pool_selector_quality_base_margin_valid_ratio': (
            quality_base_margin_valid.float().mean()
        ),
        'dbg_source_pool_selector_quality_base_margin_score_gap': (
            quality_base_margin_score_gap.sum()
            / quality_base_margin_valid.float().sum().clamp(min=1.0)
        ),
        'dbg_source_pool_selector_quality_base_margin_iou_gap': (
            quality_base_margin_iou_gap.sum()
            / quality_base_margin_valid.float().sum().clamp(min=1.0)
        ),
        'dbg_source_pool_selector_quality_default_margin_loss': (
            quality_default_margin_loss
        ),
        'dbg_source_pool_selector_quality_default_margin_weight': float(
            quality_default_margin_weight
        ),
        'dbg_source_pool_selector_quality_default_margin': float(
            quality_default_margin
        ),
        'dbg_source_pool_selector_quality_default_margin_valid_ratio': (
            quality_default_margin_valid.float().mean()
        ),
        'dbg_source_pool_selector_quality_default_margin_score_gap': (
            quality_default_margin_score_gap.sum()
            / quality_default_margin_valid.float().sum().clamp(min=1.0)
        ),
        'dbg_source_pool_selector_quality_default_margin_iou_gap': (
            quality_default_margin_iou_gap.sum()
            / quality_default_margin_valid.float().sum().clamp(min=1.0)
        ),
        'dbg_source_pool_selector_quality_default_bidirectional_margin_loss': (
            quality_default_bidirectional_margin_loss
        ),
        'dbg_source_pool_selector_quality_default_bidirectional_margin_weight': (
            float(quality_default_bidirectional_margin_weight)
        ),
        'dbg_source_pool_selector_quality_default_bidirectional_margin': (
            float(quality_default_bidirectional_margin)
        ),
        'dbg_source_pool_selector_quality_default_bidirectional_margin_valid_ratio': (
            quality_default_bidirectional_margin_valid.float().mean()
        ),
        'dbg_source_pool_selector_quality_default_bidirectional_margin_score_gap': (
            quality_default_bidirectional_margin_score_gap.sum()
            / quality_default_bidirectional_margin_valid.float().sum().clamp(min=1.0)
        ),
        'dbg_source_pool_selector_quality_default_bidirectional_margin_iou_gap': (
            quality_default_bidirectional_margin_iou_gap.sum()
            / quality_default_bidirectional_margin_valid.float().sum().clamp(min=1.0)
        ),
        'dbg_source_pool_selector_iou_aux_loss': iou_aux_loss,
        'dbg_source_pool_selector_iou_aux_weight': float(iou_aux_weight),
        'dbg_source_pool_selector_iou_aux_margin': float(iou_aux_margin),
        'dbg_source_pool_selector_iou_aux_valid_ratio': (
            iou_aux_valid.float().mean()
        ),
        'dbg_source_pool_selector_iou_aux_score_gap': (
            iou_aux_score_gap.sum()
            / iou_aux_valid.float().sum().clamp(min=1.0)
        ),
        'dbg_source_pool_selector_iou_aux_iou_gap': (
            iou_aux_iou_gap.sum()
            / iou_aux_valid.float().sum().clamp(min=1.0)
        ),
        'dbg_source_pool_selector_false_base_weight': float(
            false_base_weight
        ),
        'dbg_source_pool_selector_false_override_weight': float(
            false_override_weight
        ),
        'dbg_source_pool_selector_sourcewise_negative_weight': float(
            sourcewise_negative_weight
        ),
        'dbg_source_pool_selector_override_utility_gap_weight': float(
            override_utility_gap_weight
        ),
        'dbg_source_pool_selector_override_default_source_id': float(
            override_default_idx
        ),
        'dbg_source_pool_selector_override_utility_gap': (
            torch.where(
                valid,
                override_utility_gap_for_diag,
                torch.zeros_like(override_utility_gap_for_diag),
            ).sum() / valid_count
        ),
        'dbg_source_pool_selector_valid_ratio': valid.float().mean(),
        'dbg_source_pool_selector_candidate_query_ratio': (
            available.float().sum() / max(float(B * num_sources), 1.0)
        ),
        'dbg_source_pool_selector_competitor_query_ratio': (
            competitor_mask.float().sum() / max(float(B * num_sources), 1.0)
        ),
        'dbg_source_pool_selector_acceptable_query_ratio': (
            acceptable_mask.float().sum() / max(float(B * num_sources), 1.0)
        ),
        'dbg_source_pool_selector_pos_iou': (
            pos_iou * valid_f
        ).sum() / valid_count,
        'dbg_source_pool_selector_selected_iou': (
            selected_iou * valid_f
        ).sum() / valid_count,
        'dbg_source_pool_selector_neg_iou': (
            neg_iou * valid_f
        ).sum() / valid_count,
        'dbg_source_pool_selector_iou_gap': (
            iou_gap * valid_f
        ).sum() / valid_count,
        'dbg_source_pool_selector_target_value_gap': (
            target_value_gap * valid_f
        ).sum() / valid_count,
        'dbg_source_pool_selector_pos_score': (
            pos_score * valid_f
        ).sum() / valid_count,
        'dbg_source_pool_selector_neg_score': (
            neg_score * valid_f
        ).sum() / valid_count,
        'dbg_source_pool_selector_score_gap': (
            score_gap * valid_f
        ).sum() / valid_count,
        'dbg_source_pool_selector_violation_ratio': (
            violation.float().sum() / valid_count
        ),
        'dbg_source_pool_selector_k': 1.0,
        'dbg_source_pool_selector_temperature': float(safe_temperature),
        'dbg_source_pool_selector_min_iou_gap': float(min_iou_gap),
        'dbg_source_pool_selector_source_base': 0.0,
        'dbg_source_pool_selector_source_structured': 0.0,
        'dbg_source_pool_selector_source_quality': 0.0,
        'dbg_source_pool_selector_source_fused': 0.0,
        'dbg_source_pool_selector_source_pool': 0.0,
        'dbg_source_pool_selector_source_external_pool': 0.0,
        'dbg_source_pool_selector_source_choice': 1.0,
        'dbg_source_pool_selector_target_base_ratio': source_ratio(
            'base', source_target_ratios
        ),
        'dbg_source_pool_selector_target_fused_ratio': source_ratio(
            'fused', source_target_ratios
        ),
        'dbg_source_pool_selector_target_quality_ratio': source_ratio(
            'quality', source_target_ratios
        ),
        'dbg_source_pool_selector_target_contrastive_base_ratio': (
            source_ratio('contrastive_base', source_target_ratios)
        ),
        'dbg_source_pool_selector_target_detector_countboost_ratio': (
            source_ratio('detector_countboost', source_target_ratios)
        ),
        'dbg_source_pool_selector_target_detector_countsplit_ratio': (
            source_ratio('detector_countsplit', source_target_ratios)
        ),
        'dbg_source_pool_selector_target_detector_jointtight_ratio': (
            source_ratio('detector_jointtight', source_target_ratios)
        ),
        'dbg_source_pool_selector_target_detector_strongcoarse_ratio': (
            source_ratio('detector_strongcoarse', source_target_ratios)
        ),
        'dbg_source_pool_selector_acceptable_base_ratio': (
            source_ratio('base', source_acceptable_ratios)
        ),
        'dbg_source_pool_selector_acceptable_fused_ratio': (
            source_ratio('fused', source_acceptable_ratios)
        ),
        'dbg_source_pool_selector_acceptable_quality_ratio': (
            source_ratio('quality', source_acceptable_ratios)
        ),
        'dbg_source_pool_selector_acceptable_contrastive_base_ratio': (
            source_ratio('contrastive_base', source_acceptable_ratios)
        ),
        'dbg_source_pool_selector_acceptable_detector_countboost_ratio': (
            source_ratio('detector_countboost', source_acceptable_ratios)
        ),
        'dbg_source_pool_selector_acceptable_detector_countsplit_ratio': (
            source_ratio('detector_countsplit', source_acceptable_ratios)
        ),
        'dbg_source_pool_selector_acceptable_detector_jointtight_ratio': (
            source_ratio('detector_jointtight', source_acceptable_ratios)
        ),
        'dbg_source_pool_selector_acceptable_detector_strongcoarse_ratio': (
            source_ratio('detector_strongcoarse', source_acceptable_ratios)
        ),
        'dbg_source_pool_selector_selected_base_ratio': (
            source_ratio('base', source_selected_ratios)
        ),
        'dbg_source_pool_selector_selected_fused_ratio': (
            source_ratio('fused', source_selected_ratios)
        ),
        'dbg_source_pool_selector_selected_quality_ratio': (
            source_ratio('quality', source_selected_ratios)
        ),
        'dbg_source_pool_selector_selected_contrastive_base_ratio': (
            source_ratio('contrastive_base', source_selected_ratios)
        ),
        'dbg_source_pool_selector_selected_detector_countboost_ratio': (
            source_ratio('detector_countboost', source_selected_ratios)
        ),
        'dbg_source_pool_selector_selected_detector_countsplit_ratio': (
            source_ratio('detector_countsplit', source_selected_ratios)
        ),
        'dbg_source_pool_selector_selected_detector_jointtight_ratio': (
            source_ratio('detector_jointtight', source_selected_ratios)
        ),
        'dbg_source_pool_selector_selected_detector_strongcoarse_ratio': (
            source_ratio('detector_strongcoarse', source_selected_ratios)
        ),
        'dbg_source_pool_selector_selected_acceptable_ratio': (
            selected_acceptable.float().sum() / valid_count
        ),
        'dbg_source_pool_selector_target_override_ratio': (
            target_override.float().sum() / override_diag_count
        ),
        'dbg_source_pool_selector_selected_override_ratio': (
            selected_override.float().sum() / override_diag_count
        ),
        'dbg_source_pool_selector_false_base_ratio': (
            false_base.float().sum() / override_diag_count
        ),
        'dbg_source_pool_selector_false_override_ratio': (
            false_override.float().sum() / override_diag_count
        ),
        'dbg_source_pool_selector_override_agreement_ratio': (
            override_agreement.float().sum() / override_diag_count
        ),
        'dbg_source_pool_selector_choice_override_head': float(
            bool(override_head_used)
        ),
        'dbg_source_pool_selector_logit_base_mean': (
            source_logit_mean('base')
        ),
        'dbg_source_pool_selector_logit_fused_mean': (
            source_logit_mean('fused')
        ),
        'dbg_source_pool_selector_logit_quality_mean': (
            source_logit_mean('quality')
        ),
        'dbg_source_pool_selector_logit_contrastive_base_mean': (
            source_logit_mean('contrastive_base')
        ),
        'dbg_source_pool_selector_logit_detector_countboost_mean': (
            source_logit_mean('detector_countboost')
        ),
        'dbg_source_pool_selector_logit_detector_countsplit_mean': (
            source_logit_mean('detector_countsplit')
        ),
        'dbg_source_pool_selector_logit_detector_jointtight_mean': (
            source_logit_mean('detector_jointtight')
        ),
        'dbg_source_pool_selector_logit_detector_strongcoarse_mean': (
            source_logit_mean('detector_strongcoarse')
        ),
        'dbg_source_pool_selector_logit_margin_base_mean': (
            source_logit_margin_mean('base')
        ),
        'dbg_source_pool_selector_logit_margin_fused_mean': (
            source_logit_margin_mean('fused')
        ),
        'dbg_source_pool_selector_logit_margin_quality_mean': (
            source_logit_margin_mean('quality')
        ),
        'dbg_source_pool_selector_logit_margin_contrastive_base_mean': (
            source_logit_margin_mean('contrastive_base')
        ),
        'dbg_source_pool_selector_logit_margin_detector_countboost_mean': (
            source_logit_margin_mean('detector_countboost')
        ),
        'dbg_source_pool_selector_logit_margin_detector_countsplit_mean': (
            source_logit_margin_mean('detector_countsplit')
        ),
        'dbg_source_pool_selector_logit_margin_detector_jointtight_mean': (
            source_logit_margin_mean('detector_jointtight')
        ),
        'dbg_source_pool_selector_logit_margin_detector_strongcoarse_mean': (
            source_logit_margin_mean('detector_strongcoarse')
        ),
        'dbg_source_pool_selector_class_weight_base': (
            source_class_weight('base')
        ),
        'dbg_source_pool_selector_class_weight_fused': (
            source_class_weight('fused')
        ),
        'dbg_source_pool_selector_class_weight_quality': (
            source_class_weight('quality')
        ),
        'dbg_source_pool_selector_class_weight_contrastive_base': (
            source_class_weight('contrastive_base')
        ),
        'dbg_source_pool_selector_class_weight_detector_countboost': (
            source_class_weight('detector_countboost')
        ),
        'dbg_source_pool_selector_class_weight_detector_countsplit': (
            source_class_weight('detector_countsplit')
        ),
        'dbg_source_pool_selector_class_weight_detector_jointtight': (
            source_class_weight('detector_jointtight')
        ),
        'dbg_source_pool_selector_class_weight_detector_strongcoarse': (
            source_class_weight('detector_strongcoarse')
        ),
        'dbg_source_pool_selector_choice_balance': float(bool(choice_balance)),
        'dbg_source_pool_selector_choice_balance_power': float(
            choice_balance_power
        ),
        'dbg_source_pool_selector_choice_target_iou': float(
            choice_target == 'iou'
        ),
        'dbg_source_pool_selector_choice_target_threshold_utility': float(
            choice_target == 'threshold_utility'
        ),
        'dbg_source_pool_selector_choice_target_threshold_utility_softmax': (
            float(choice_target == 'threshold_utility_softmax')
        ),
        'dbg_source_pool_selector_choice_target_threshold_utility_regression': (
            float(choice_target == 'threshold_utility_regression')
        ),
        'dbg_source_pool_selector_choice_target_threshold_bucket': float(
            choice_target == 'threshold_bucket'
        ),
        'dbg_source_pool_selector_choice_target_threshold_bucket_bce': float(
            choice_target == 'threshold_bucket_bce'
        ),
        'dbg_source_pool_selector_choice_target_threshold_bucket_argmax': float(
            choice_target == 'threshold_bucket_argmax'
        ),
        'dbg_source_pool_selector_choice_target_threshold_bucket_unique': float(
            choice_target == 'threshold_bucket_unique'
        ),
        'dbg_source_pool_selector_choice_target_threshold_bucket_margin': float(
            choice_target == 'threshold_bucket_margin'
        ),
        'dbg_source_pool_selector_choice_target_base_threshold_gain': float(
            choice_target == 'base_threshold_gain'
        ),
        'dbg_source_pool_selector_choice_target_base_override_bce': float(
            choice_target == 'base_override_bce'
        ),
        'dbg_source_pool_selector_choice_target_base_override_focal_bce': float(
            choice_target == 'base_override_focal_bce'
        ),
        'dbg_source_pool_selector_choice_target_base_override_sourcewise_focal_bce': (
            float(choice_target == 'base_override_sourcewise_focal_bce')
        ),
        'dbg_source_pool_selector_choice_target_threshold_gain_default_sourcewise_focal_bce': (
            float(
                choice_target
                == 'threshold_gain_default_sourcewise_focal_bce'
            )
        ),
        'dbg_source_pool_selector_choice_target_threshold_gain_default_diffquery_sourcewise_focal_bce': (
            float(
                choice_target
                == 'threshold_gain_default_diffquery_sourcewise_focal_bce'
            )
        ),
        'dbg_source_pool_selector_choice_target_precision_gain_default_sourcewise_focal_bce': (
            float(choice_target == 'precision_gain_default_sourcewise_focal_bce')
        ),
        'dbg_source_pool_selector_diffquery_valid_ratio': (
            diffquery_valid.float().mean()
        ),
        'dbg_source_pool_selector_choice_target_quality_override': float(
            choice_target == 'quality_override'
        ),
        'dbg_source_pool_selector_num_candidate_sources': float(
            num_candidate_sources
        ),
        'dbg_warn_source_pool_selector_no_valid_samples': float(
            valid.float().sum().detach().item() == 0
        ),
        'dbg_warn_source_pool_selector_no_valid_samples_ratio': float(
            valid.float().sum().detach().item() == 0
        ),
    }
    for source_idx, source in enumerate(source_names[:num_sources]):
        key = (
            'dbg_source_pool_selector_source_gap_'
            + str(source).replace('-', '_')
        )
        debug[key] = source_gap_values[source_idx]
    return debug


def _source_pool_selector_losses(end_points, weight=1.0, source='source_pool',
                                 candidate_k=5, temperature=1.0,
                                 min_iou_gap=0.02,
                                 pairwise_weight=0.5,
                                 choice_balance=False,
                                 choice_balance_power=1.0,
                                 choice_target='iou',
                                 override_default_source='base',
                                 oracle_prior_weight=0.0,
                                 override_prior_weight=0.0,
                                 override_source_prior_weight=0.0,
                                 override_margin_weight=0.0,
                                 override_margin=1.0,
                                 quality_base_margin_weight=0.0,
                                 quality_base_margin=0.75,
                                 quality_default_margin_weight=0.0,
                                 quality_default_margin=0.75,
                                 quality_default_bidirectional_margin_weight=0.0,
                                 quality_default_bidirectional_margin=0.75,
                                 iou_aux_weight=0.0,
                                 iou_aux_margin=0.5,
                                 false_base_weight=1.0,
                                 false_override_weight=1.0,
                                 override_utility_gap_weight=0.0,
                                 sourcewise_negative_weight=1.0,
                                 source_min_iou_gaps=None):
    selector_scores = end_points['selector_scores'].float()
    if source == 'source_choice':
        if 'selector_choice_scores' not in end_points:
            raise ValueError(
                "source_choice selector loss requires selector_choice_scores"
            )
        return _source_pool_selector_choice_losses(
            end_points,
            end_points['selector_choice_scores'],
            weight=weight,
            temperature=temperature,
            min_iou_gap=min_iou_gap,
            choice_balance=choice_balance,
            choice_balance_power=choice_balance_power,
            choice_target=choice_target,
            pairwise_weight=pairwise_weight,
            override_default_source=override_default_source,
            oracle_prior_weight=oracle_prior_weight,
            override_prior_weight=override_prior_weight,
            override_source_prior_weight=override_source_prior_weight,
            override_margin_weight=override_margin_weight,
            override_margin=override_margin,
            quality_base_margin_weight=quality_base_margin_weight,
            quality_base_margin=quality_base_margin,
            quality_default_margin_weight=quality_default_margin_weight,
            quality_default_margin=quality_default_margin,
            quality_default_bidirectional_margin_weight=(
                quality_default_bidirectional_margin_weight
            ),
            quality_default_bidirectional_margin=(
                quality_default_bidirectional_margin
            ),
            iou_aux_weight=iou_aux_weight,
            iou_aux_margin=iou_aux_margin,
            false_base_weight=false_base_weight,
            false_override_weight=false_override_weight,
            override_utility_gap_weight=override_utility_gap_weight,
            sourcewise_negative_weight=sourcewise_negative_weight,
            source_min_iou_gaps=source_min_iou_gaps,
        )
    if (
        'selector_source_scores' in end_points
        and source in ('source_pool', 'external_pool')
    ):
        return _source_pool_selector_source_aware_losses(
            end_points,
            end_points['selector_source_scores'],
            weight=weight,
            source=source,
            candidate_k=candidate_k,
            temperature=temperature,
            min_iou_gap=min_iou_gap,
            pairwise_weight=pairwise_weight,
            choice_target=choice_target,
        )
    target_iou = _target_iou_matrix(end_points, prefix='last_').detach()
    B, Q = selector_scores.shape
    device = selector_scores.device
    k = max(1, min(int(candidate_k), Q))

    candidate_mask, num_candidate_sources = _quality_topk_candidate_mask(
        end_points, source, selector_scores, k
    )
    masked_iou = target_iou.masked_fill(~candidate_mask, -1.0)
    pos_idx = masked_iou.argmax(dim=1)
    batch_idx = torch.arange(B, device=device)
    pos_score = selector_scores[batch_idx, pos_idx]
    pos_iou = target_iou[batch_idx, pos_idx]

    pos_mask = torch.zeros(B, Q, device=device, dtype=torch.bool)
    pos_mask[batch_idx, pos_idx] = True
    competitor_mask = (
        candidate_mask
        & (~pos_mask)
        & (target_iou <= (pos_iou.unsqueeze(1) - float(min_iou_gap)))
    )

    valid = competitor_mask.any(dim=1)
    if 'box_label_mask' in end_points:
        target_valid = end_points['box_label_mask'][:, 0].to(
            device=device
        ).bool()
        valid = valid & target_valid
    valid = valid & _dataset_not_scannet_mask(end_points, B, device)
    valid_f = valid.float()
    valid_count = valid_f.sum().clamp(min=1.0)

    safe_temperature = max(float(temperature), 1e-6)
    masked_logits = selector_scores.masked_fill(
        ~candidate_mask, torch.finfo(selector_scores.dtype).min
    )
    if bool(valid.any().detach().item()):
        loss_raw = F.cross_entropy(
            masked_logits[valid] / safe_temperature,
            pos_idx[valid],
            reduction='mean',
        )
    else:
        loss_raw = selector_scores.sum() * 0.0

    selected_idx = masked_logits.argmax(dim=1)
    selected_iou = target_iou[batch_idx, selected_idx]
    neg_scores = selector_scores.masked_fill(
        ~competitor_mask, torch.finfo(selector_scores.dtype).min
    )
    neg_score, neg_idx = neg_scores.max(dim=1)
    neg_iou = target_iou[batch_idx, neg_idx]
    score_gap = pos_score - neg_score
    iou_gap = pos_iou - neg_iou
    violation = (selected_idx != pos_idx) & valid
    total_queries = max(float(B * Q), 1.0)

    return {
        'loss_source_pool_selector': loss_raw * weight,
        'loss_source_pool_selector_raw': loss_raw,
        'dbg_source_pool_selector_loss_raw': loss_raw,
        'dbg_source_pool_selector_valid_ratio': valid.float().mean(),
        'dbg_source_pool_selector_candidate_query_ratio': (
            candidate_mask.float().sum() / total_queries
        ),
        'dbg_source_pool_selector_competitor_query_ratio': (
            competitor_mask.float().sum() / total_queries
        ),
        'dbg_source_pool_selector_pos_iou': (
            pos_iou * valid_f
        ).sum() / valid_count,
        'dbg_source_pool_selector_selected_iou': (
            selected_iou * valid_f
        ).sum() / valid_count,
        'dbg_source_pool_selector_neg_iou': (
            neg_iou * valid_f
        ).sum() / valid_count,
        'dbg_source_pool_selector_iou_gap': (
            iou_gap * valid_f
        ).sum() / valid_count,
        'dbg_source_pool_selector_pos_score': (
            pos_score * valid_f
        ).sum() / valid_count,
        'dbg_source_pool_selector_neg_score': (
            neg_score * valid_f
        ).sum() / valid_count,
        'dbg_source_pool_selector_score_gap': (
            score_gap * valid_f
        ).sum() / valid_count,
        'dbg_source_pool_selector_violation_ratio': (
            violation.float().sum() / valid_count
        ),
        'dbg_source_pool_selector_k': float(k),
        'dbg_source_pool_selector_temperature': float(safe_temperature),
        'dbg_source_pool_selector_min_iou_gap': float(min_iou_gap),
        'dbg_source_pool_selector_source_base': float(source == 'base'),
        'dbg_source_pool_selector_source_structured': float(
            source == 'structured'
        ),
        'dbg_source_pool_selector_source_quality': float(source == 'quality'),
        'dbg_source_pool_selector_source_fused': float(source == 'fused'),
        'dbg_source_pool_selector_source_pool': float(source == 'source_pool'),
        'dbg_source_pool_selector_source_external_pool': float(
            source == 'external_pool'
        ),
        'dbg_source_pool_selector_num_candidate_sources': float(
            num_candidate_sources
        ),
        'dbg_warn_source_pool_selector_no_valid_samples': float(
            valid.float().sum().detach().item() == 0
        ),
        'dbg_warn_source_pool_selector_no_valid_samples_ratio': float(
            valid.float().sum().detach().item() == 0
        ),
    }


def _rank_loss_from_scores(scores, indices, end_points, weight, margin,
                           loss_name, debug_prefix):
    B, Q = scores.shape
    device = scores.device
    pos_mask, valid = _build_target_pos_mask(indices, B, Q, device)
    structured_valid = end_points.get('structured_valid_mask', None)
    if structured_valid is not None:
        valid = valid & structured_valid.to(device=device).bool()
    valid = valid & _dataset_not_scannet_mask(end_points, B, device)
    neg_mask = ~pos_mask
    pos_scores = scores.masked_fill(~pos_mask, 0.0).sum(dim=1)
    pos_count = pos_mask.float().sum(dim=1).clamp(min=1.0)
    pos_mean = pos_scores / pos_count
    neg_hard = _masked_hard_negative(scores, neg_mask, dim=1)
    margin_t = scores.new_tensor(float(margin))
    per_batch = F.relu(neg_hard - pos_mean + margin_t) * valid.float()
    valid_count = valid.float().sum().clamp(min=1.0)
    loss_raw = per_batch.sum() / valid_count
    return {
        loss_name: loss_raw * weight,
        f'{loss_name}_raw': loss_raw,
        f'dbg_{debug_prefix}_rank_loss_raw': loss_raw,
        f'dbg_{debug_prefix}_rank_pos_mean': (
            pos_mean * valid.float()
        ).sum() / valid_count,
        f'dbg_{debug_prefix}_rank_neg_hard': (
            neg_hard * valid.float()
        ).sum() / valid_count,
        f'dbg_{debug_prefix}_rank_gap': (
            (pos_mean - neg_hard) * valid.float()
        ).sum() / valid_count,
        f'dbg_{debug_prefix}_rank_valid_ratio': valid.float().mean(),
        f'dbg_{debug_prefix}_rank_positive_query_ratio': pos_mask.float().mean(),
    }


def _rapf_gate_losses(end_points, indices, weight=0.2, iou_margin=0.02):
    gate = end_points['rapf_gate'].float()
    B, Q = gate.shape
    device = gate.device
    target_iou = _target_iou_matrix(end_points, prefix='last_')
    base_scores = _score_source(end_points, 'base')
    structured_scores = _score_source(end_points, 'structured')
    base_top = base_scores.argmax(dim=1)
    structured_top = structured_scores.argmax(dim=1)
    fused_scores = end_points.get('fused_scores', None)
    if fused_scores is not None:
        fused_scores = fused_scores.float()
        fused_top = fused_scores.argmax(dim=1)
    else:
        fused_top = structured_top
    batch_idx = torch.arange(B, device=device)
    base_top_iou = target_iou[batch_idx, base_top]

    # RAPF predicts one reliability value per query.  Supervise those values
    # at query level: positive gates must both move the structured residual in
    # the helpful direction and offer a target-IoU improvement over the
    # official BBS top query.  This avoids diluting a sparse correction signal
    # into the mean of all 256 gate values.
    base_norm = end_points.get('rapf_base_norm', base_scores).float().detach()
    structured_norm = end_points.get(
        'rapf_structured_norm', structured_scores
    ).float().detach()
    helpful_direction = structured_norm > base_norm
    query_labels = (
        (target_iou > base_top_iou.unsqueeze(1) + float(iou_margin))
        & helpful_direction
    ).float()
    sample_valid = end_points.get('structured_valid_mask', None)
    if sample_valid is None:
        sample_valid = torch.ones(B, device=device, dtype=torch.bool)
    else:
        sample_valid = sample_valid.to(device=device).bool()
    sample_valid = sample_valid & (~_global_only_mask(end_points, B, device))
    sample_valid = sample_valid & _dataset_not_scannet_mask(
        end_points, B, device
    )
    query_valid = sample_valid.unsqueeze(1).expand(B, Q)
    query_weights = torch.ones_like(gate)
    positive_count = (query_labels * query_valid.float()).sum()
    negative_count = ((1.0 - query_labels) * query_valid.float()).sum()
    positive_weight = (
        negative_count / positive_count.clamp(min=1.0)
    ).clamp(min=1.0, max=32.0)
    query_weights = torch.where(
        query_labels.bool(), positive_weight, query_weights
    )
    gate_safe = gate.clamp(1e-6, 1.0 - 1e-6)
    with torch.cuda.amp.autocast(enabled=False):
        per_query = F.binary_cross_entropy(
            gate_safe.float(),
            query_labels.float(),
            reduction='none',
        )
    weighted_valid = query_weights * query_valid.float()
    loss_raw = (
        per_query * weighted_valid
    ).sum() / weighted_valid.sum().clamp(min=1.0)
    valid_f = sample_valid.float()
    structured_top_iou = target_iou[batch_idx, structured_top]
    fused_top_iou = target_iou[batch_idx, fused_top]
    base_ok = base_top_iou >= 0.25
    fused_ok = fused_top_iou >= 0.25
    wrong_to_right = (
        (~base_ok) & fused_ok & sample_valid
    ).float().sum() / valid_f.sum().clamp(min=1.0)
    right_to_wrong = (
        base_ok & (~fused_ok) & sample_valid
    ).float().sum() / valid_f.sum().clamp(min=1.0)
    return {
        'loss_rapf_gate': loss_raw * weight,
        'loss_rapf_gate_raw': loss_raw,
        'dbg_rapf_gate_loss_raw': loss_raw,
        'dbg_rapf_gate_label_mean': (
            query_labels * query_valid.float()
        ).sum() / query_valid.float().sum().clamp(min=1.0),
        'dbg_rapf_gate_positive_weight': positive_weight,
        'dbg_rapf_gate_supervision_valid_ratio': sample_valid.float().mean(),
        'dbg_rapf_gate_base_top_iou': (
            base_top_iou * valid_f
        ).sum() / valid_f.sum().clamp(min=1.0),
        'dbg_rapf_gate_structured_top_iou': (
            structured_top_iou * valid_f
        ).sum() / valid_f.sum().clamp(min=1.0),
        'dbg_rapf_gate_fused_top_iou': (
            fused_top_iou * valid_f
        ).sum() / valid_f.sum().clamp(min=1.0),
        'dbg_rapf_top1_correct_ratio': (
            fused_ok.float() * valid_f
        ).sum() / valid_f.sum().clamp(min=1.0),
        'dbg_rapf_iou_delta_mean': (
            (fused_top_iou - base_top_iou) * valid_f
        ).sum() / valid_f.sum().clamp(min=1.0),
        'dbg_rapf_wrong_to_right_ratio': wrong_to_right,
        'dbg_rapf_right_to_wrong_ratio': right_to_wrong,
        'dbg_warn_rapf_right_to_wrong': float(
            right_to_wrong.detach().item() > 0.0
        ),
        'dbg_warn_rapf_right_to_wrong_ratio': right_to_wrong,
        'dbg_warn_rapf_no_gate_supervision': float(
            sample_valid.float().sum().detach().item() == 0
        ),
        'dbg_warn_rapf_no_gate_supervision_ratio': float(
            sample_valid.float().sum().detach().item() == 0
        ),
    }


def _qahnl_losses(end_points, indices, config):
    source = config.get('score_source', 'fused')
    scores = _score_source(end_points, source)
    B, Q = scores.shape
    device = scores.device
    target_iou = _target_iou_matrix(end_points, prefix='last_')
    pos_thr = float(config.get(
        'pos_iou_thresh', config.get('pos_iou_threshold', 0.25)
    ))
    neg_thr = float(config.get(
        'neg_iou_thresh', config.get('neg_iou_threshold', 0.10)
    ))
    topk_iou = max(1, min(int(config.get(
        'topk_iou_pos', config.get('topk_iou', 3)
    )), Q))
    num_hard = max(1, min(int(config.get('num_hard_neg', 16)), Q))
    base_margin = float(config.get('margin_base', config.get('margin', 0.2)))
    margin_iou_lambda = float(config.get('margin_iou_lambda', 0.5))
    margin_min = float(config.get('margin_min', 0.05))
    margin_max = float(config.get('margin_max', 0.5))
    temperature = float(config.get('temperature', 1.0))
    temperature_max = float(config.get('temperature_max', 6.0))
    temperature = max(1e-6, min(temperature, temperature_max))
    weight = float(config.get('loss_weight', 0.2))

    pos_mask = target_iou >= pos_thr
    top_iou_idx = torch.topk(target_iou, topk_iou, dim=1).indices
    pos_mask.scatter_(1, top_iou_idx, True)
    matched_target_mask, matched_valid = _build_target_pos_mask(indices, B, Q, device)
    pos_mask = pos_mask | matched_target_mask

    neg_candidates = (target_iou <= neg_thr) & (~pos_mask)
    hard_scores = scores.masked_fill(~neg_candidates, torch.finfo(scores.dtype).min)
    hard_idx = torch.topk(hard_scores, num_hard, dim=1).indices
    neg_mask = torch.zeros(B, Q, device=device, dtype=torch.bool)
    neg_mask.scatter_(1, hard_idx, True)
    neg_mask = neg_mask & neg_candidates

    global_only = _global_only_mask(end_points, B, device)
    global_only_skipped_structured = (
        global_only if source == 'structured'
        else torch.zeros(B, device=device, dtype=torch.bool)
    )
    valid = matched_valid & _dataset_not_scannet_mask(end_points, B, device)
    if source == 'structured':
        valid = valid & (~global_only)
    valid = valid & pos_mask.any(dim=1) & neg_mask.any(dim=1)
    valid_f = valid.float()
    valid_count = valid_f.sum().clamp(min=1.0)
    no_positive = ~pos_mask.any(dim=1)
    no_hard_negative = ~neg_mask.any(dim=1)

    pos_scores = scores.masked_fill(~pos_mask, torch.finfo(scores.dtype).min)
    pos_score = pos_scores.max(dim=1).values
    neg_selected_scores = scores.masked_fill(~neg_mask, torch.finfo(scores.dtype).min)
    neg_score = neg_selected_scores.max(dim=1).values
    pos_iou = target_iou.masked_fill(~pos_mask, 0.0).max(dim=1).values
    neg_iou = target_iou.masked_fill(~neg_mask, 0.0).max(dim=1).values
    adaptive_margin = scores.new_tensor(base_margin) + (
        scores.new_tensor(margin_iou_lambda) * (pos_iou - neg_iou)
    )
    adaptive_margin = adaptive_margin.clamp(min=margin_min, max=margin_max)
    temp_t = scores.new_tensor(temperature)
    violation = (neg_score - pos_score + adaptive_margin > 0) & valid
    per_sample = temp_t * F.softplus(
        (neg_score - pos_score + adaptive_margin) / temp_t
    ) * valid_f
    loss_raw = per_sample.sum() / valid_count

    ignore_mask = (~pos_mask) & (~neg_mask)
    ambiguous_mask = (target_iou > neg_thr) & (target_iou < pos_thr)
    weak_generic = end_points.get('weak_generic_target_mask', None)
    if weak_generic is None:
        weak_generic_mask = torch.zeros(B, device=device, dtype=torch.bool)
        weak_generic_active = scores.new_tensor(0.0)
    else:
        weak_generic_mask = weak_generic.to(device=device).bool().view(-1)[:B]
        if weak_generic_mask.numel() < B:
            weak_generic_mask = torch.cat([
                weak_generic_mask,
                torch.zeros(
                    B - weak_generic_mask.numel(),
                    device=device,
                    dtype=torch.bool,
                ),
            ], dim=0)
        weak_generic_active = weak_generic_mask.any().float()
    error_count = end_points.get('decomposition_error_flags_count', None)
    if error_count is None:
        error_count = torch.zeros(B, device=device)
    elif torch.is_tensor(error_count):
        error_count = error_count.to(device=device).float().view(-1)[:B]
    elif isinstance(error_count, (list, tuple)):
        error_count = torch.tensor(
            [float(x) for x in error_count[:B]],
            device=device,
        )
    else:
        error_count = torch.full((B,), float(error_count), device=device)
    if error_count.numel() < B:
        pad = torch.zeros(B - error_count.numel(), device=device)
        error_count = torch.cat([error_count, pad], dim=0)
    repaired_mask = (
        (~global_only)
        & (~weak_generic_mask)
        & (error_count > 0)
    )
    ok_mask = (
        (~global_only)
        & (~weak_generic_mask)
        & (error_count <= 0)
    )

    def _status_valid_ratio(status_mask):
        status_f = status_mask.float()
        return ((valid & status_mask).float().sum()
                / status_f.sum().clamp(min=1.0))

    margin_min_value = adaptive_margin[valid].min() if valid.any() else scores.new_tensor(0.0)
    margin_max_value = adaptive_margin[valid].max() if valid.any() else scores.new_tensor(0.0)
    total = float(B * Q)
    positive_query_ratio = pos_mask.float().sum() / total
    negative_query_ratio = neg_mask.float().sum() / total
    iou_gap_mean = ((pos_iou - neg_iou) * valid_f).sum() / valid_count
    score_gap_mean = ((pos_score - neg_score) * valid_f).sum() / valid_count
    losses = {
        'loss_qahnl': loss_raw * weight,
        'loss_qahnl_raw': loss_raw,
        'dbg_qahnl_loss_raw': loss_raw,
        'dbg_qahnl_loss_unweighted': loss_raw,
        'dbg_qahnl_loss_weighted': loss_raw * weight,
        'dbg_qahnl_positive_query_ratio': positive_query_ratio,
        'dbg_qahnl_pos_query_ratio': positive_query_ratio,
        'dbg_qahnl_negative_query_ratio': negative_query_ratio,
        'dbg_qahnl_neg_query_ratio': negative_query_ratio,
        'dbg_qahnl_hard_negative_query_ratio': negative_query_ratio,
        'dbg_qahnl_ignore_query_ratio': ignore_mask.float().sum() / total,
        'dbg_qahnl_ambiguous_query_ratio': ambiguous_mask.float().sum() / total,
        'dbg_qahnl_ambiguous_ignore_ratio': (
            (ambiguous_mask & ignore_mask).float().sum()
            / ambiguous_mask.float().sum().clamp(min=1.0)
        ),
        'dbg_qahnl_ambiguous_as_negative_ratio': (
            (ambiguous_mask & neg_mask).float().sum()
            / ambiguous_mask.float().sum().clamp(min=1.0)
        ),
        'dbg_qahnl_valid_batch_ratio': valid.float().mean(),
        'dbg_qahnl_valid_ok_ratio': _status_valid_ratio(ok_mask),
        'dbg_qahnl_valid_repaired_ratio': _status_valid_ratio(repaired_mask),
        'dbg_qahnl_valid_weak_generic_ratio': _status_valid_ratio(weak_generic_mask),
        'dbg_qahnl_valid_global_only_ratio': _status_valid_ratio(global_only),
        'dbg_qahnl_global_only_used_ratio': (
            (valid & global_only).float().sum() / max(float(B), 1.0)
        ),
        'dbg_qahnl_global_only_skipped_structured_ratio': (
            global_only_skipped_structured.float().sum() / max(float(B), 1.0)
        ),
        'dbg_qahnl_global_only_skipped_ratio': (
            global_only_skipped_structured.float().sum() / max(float(B), 1.0)
        ),
        'dbg_qahnl_weak_generic_used_ratio': (
            (valid & weak_generic_mask).float().sum() / max(float(B), 1.0)
        ),
        'dbg_qahnl_pos_score': (pos_score * valid_f).sum() / valid_count,
        'dbg_qahnl_neg_score': (neg_score * valid_f).sum() / valid_count,
        'dbg_qahnl_pos_iou': (pos_iou * valid_f).sum() / valid_count,
        'dbg_qahnl_neg_iou': (neg_iou * valid_f).sum() / valid_count,
        'dbg_qahnl_hardneg_iou': (neg_iou * valid_f).sum() / valid_count,
        'dbg_qahnl_iou_gap': iou_gap_mean,
        'dbg_qahnl_iou_gap_mean': iou_gap_mean,
        'dbg_qahnl_score_gap': score_gap_mean,
        'dbg_qahnl_score_gap_mean': score_gap_mean,
        'dbg_qahnl_adaptive_margin': (
            adaptive_margin * valid_f
        ).sum() / valid_count,
        'dbg_qahnl_margin_mean': (
            adaptive_margin * valid_f
        ).sum() / valid_count,
        'dbg_qahnl_margin_min': margin_min_value,
        'dbg_qahnl_margin_max': margin_max_value,
        'dbg_qahnl_margin_min_value': margin_min_value,
        'dbg_qahnl_margin_max_value': margin_max_value,
        'dbg_qahnl_margin_base': float(base_margin),
        'dbg_qahnl_margin_iou_lambda': float(margin_iou_lambda),
        'dbg_qahnl_margin_min_config': float(margin_min),
        'dbg_qahnl_margin_max_config': float(margin_max),
        'dbg_qahnl_temperature': float(temperature),
        'dbg_qahnl_violation_ratio': (
            violation.float().sum() / valid_count
        ),
        'dbg_qahnl_score_source_base': float(source == 'base'),
        'dbg_qahnl_score_source_structured': float(source == 'structured'),
        'dbg_qahnl_score_source_quality': float(source == 'quality'),
        'dbg_qahnl_score_source_fused': float(source == 'fused'),
        'dbg_qahnl_num_hard_neg': float(num_hard),
        'dbg_qahnl_entity_hardneg_flag': float(bool(config.get(
            'use_entity_hardneg', config.get('entity_hardneg', False)
        ))),
        'dbg_qahnl_attr_hardneg_flag': float(bool(config.get(
            'use_attr_hardneg', config.get('attr_hardneg', False)
        ))),
        'dbg_qahnl_rel_hardneg_flag': float(bool(config.get(
            'use_relation_hardneg', config.get('rel_hardneg', False)
        ))),
        'dbg_qahnl_entity_target_mask_enabled': (
            scores.new_tensor(float(bool(config.get(
                'use_entity_hardneg', config.get('entity_hardneg', False)
            ))))
            * (1.0 - weak_generic_active)
        ),
        'dbg_warn_qahnl_no_valid_samples': float(
            valid.float().sum().detach().item() == 0
        ),
        'dbg_warn_qahnl_no_valid_samples_ratio': float(
            valid.float().sum().detach().item() == 0
        ),
        'dbg_warn_qahnl_no_positive_ratio': no_positive.float().mean(),
        'dbg_warn_qahnl_no_hard_negative_ratio': no_hard_negative.float().mean(),
        'dbg_warn_qahnl_ambiguous_as_negative': float(
            (
                (ambiguous_mask & neg_mask).float().sum()
                / ambiguous_mask.float().sum().clamp(min=1.0)
            ).detach().item() > 0.0
        ),
        'dbg_warn_qahnl_ambiguous_as_negative_ratio': (
            (ambiguous_mask & neg_mask).float().sum()
            / ambiguous_mask.float().sum().clamp(min=1.0)
        ),
    }
    return losses


def compute_hungarian_loss(end_points, num_decoder_layers, set_criterion,
                           query_points_obj_topk=5,
                           use_s2s_aux_loss=False,
                           s2s_aux_weight=1.0,
                           acd_rank_weight=1.0,
                           use_quality_head=False,
                           quality_loss_weight=1.0,
                           quality_iou_threshold=0.25,
                           use_sacr=False,
                           sacr_rank_loss_weight=0.0,
                           sacr_rank_margin=0.2,
                           use_rapf=False,
                           use_reliability_gate=False,
                           rapf_gate_loss_weight=0.2,
                           rapf_gate_iou_margin=0.02,
                           use_qahnl=False,
                           qahnl_config=None,
                           quality_topk_rerank_weight=0.0,
                           quality_topk_rerank_source='fused',
                           quality_topk_rerank_k=5,
                           quality_topk_rerank_margin=0.05,
                           quality_topk_rerank_min_iou_gap=0.02,
                           source_pool_selector_loss_weight=0.0,
                           source_pool_selector_source='source_pool',
                           source_pool_selector_k=5,
                           source_pool_selector_temperature=1.0,
                           source_pool_selector_min_iou_gap=0.02,
                           source_pool_selector_source_min_iou_gaps=None,
                           source_pool_selector_pairwise_weight=0.5,
                           source_pool_selector_choice_balance=False,
                           source_pool_selector_choice_balance_power=1.0,
                           source_pool_selector_choice_target='iou',
                           source_pool_selector_override_default_source='base',
                           source_pool_selector_oracle_prior_weight=0.0,
                           source_pool_selector_override_prior_weight=0.0,
                           source_pool_selector_override_source_prior_weight=0.0,
                           source_pool_selector_override_margin_weight=0.0,
                           source_pool_selector_override_margin=1.0,
                           source_pool_selector_quality_base_margin_weight=0.0,
                           source_pool_selector_quality_base_margin=0.75,
                           source_pool_selector_quality_default_margin_weight=0.0,
                           source_pool_selector_quality_default_margin=0.75,
                           source_pool_selector_quality_default_bidirectional_margin_weight=0.0,
                           source_pool_selector_quality_default_bidirectional_margin=0.75,
                           source_pool_selector_iou_aux_weight=0.0,
                           source_pool_selector_iou_aux_margin=0.5,
                           source_pool_selector_false_base_weight=1.0,
                           source_pool_selector_false_override_weight=1.0,
                           source_pool_selector_sourcewise_negative_weight=1.0,
                           source_pool_selector_override_utility_gap_weight=0.0,
                           detector_policy_adapter_loss_weight=0.0,
                           detector_policy_adapter_k=5,
                           detector_policy_adapter_margin=0.05,
                           detector_policy_adapter_min_iou_gap=0.02,
                           detector_policy_adapter_reg_weight=0.0,
                           use_dhc=False,
                           dhc_config=None,
                           dhc_module=None):
    """Compute Hungarian matching loss containing CE, bbox and giou."""
    prefixes = ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    prefixes = ['proposal_'] + prefixes

    # Ground-truth
    gt_center = end_points['center_label'][:, :, 0:3]  # B, G, 3
    gt_size = end_points['size_gts']  # (B,G,3)
    gt_labels = end_points['sem_cls_label']  # (B, G)
    gt_bbox = torch.cat([gt_center, gt_size], dim=-1)  # cxcyczwhd
    positive_map = end_points['positive_map']
    box_label_mask = end_points['box_label_mask']
    target = [
        {
            "labels": gt_labels[b, box_label_mask[b].bool()],
            "boxes": gt_bbox[b, box_label_mask[b].bool()],
            "positive_map": positive_map[b, box_label_mask[b].bool()]
        }
        for b in range(gt_labels.shape[0])
    ]

    loss_ce, loss_bbox, loss_giou, loss_contrastive_align = 0, 0, 0, 0
    last_indices = None  # Save 'last_' prefix indices for DHC

    for prefix in prefixes:
        output = {}
        if 'proj_tokens' in end_points:
            output['proj_tokens'] = end_points['proj_tokens'].float()
            output['proj_queries'] = end_points[f'{prefix}proj_queries'].float()
            output['tokenized'] = end_points['tokenized']

        pred_center = end_points[f'{prefix}center']
        pred_size = end_points[f'{prefix}pred_size']
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1).float()

        # Always use sem_cls_scores (B, Q, C) for matcher — ACD scores are
        # (B, Q) and incompatible with the Hungarian matching cost matrix.
        # ACD contributes via DHC losses and evaluation, not via matching.
        pred_logits = end_points[f'{prefix}sem_cls_scores'].float()
        output['pred_logits'] = pred_logits
        output["pred_boxes"] = pred_bbox

        losses, indices = set_criterion(output, target)
        if prefix == 'last_':
            last_indices = indices
        for loss_key in losses.keys():
            end_points[f'{prefix}_{loss_key}'] = losses[loss_key]
        loss_ce += losses.get('loss_ce', 0)
        loss_bbox += losses['loss_bbox']
        loss_giou += losses.get('loss_giou', 0)
        if 'proj_tokens' in end_points:
            loss_contrastive_align += losses['loss_contrastive_align']
    if 'seeds_obj_cls_logits' in end_points.keys():
        query_points_generation_loss = compute_points_obj_cls_loss_hard_topk(
            end_points, query_points_obj_topk
        )
    else:
        query_points_generation_loss = 0.0

    # ACD ranking loss: trains ACD head even without DHC.
    # Matched queries should score higher than the hardest negative, with learned margin.
    loss_acd_rank = torch.tensor(0.0, device=gt_center.device)
    if 'acd_final_scores' in end_points and last_indices is not None and 'dhc_margin_acd_rank' in end_points:
        acd_scores = end_points['acd_final_scores']  # (B, Q)
        margin = end_points['dhc_margin_acd_rank']
        B_acd, Q_acd = acd_scores.shape
        all_pos_mask, all_valid = _build_pos_mask(last_indices, B_acd, Q_acd, acd_scores.device)
        pos_mask, valid = _build_target_pos_mask(last_indices, B_acd, Q_acd, acd_scores.device)
        structured_valid = _structured_valid_mask(end_points, B_acd, acd_scores.device)
        struct_debug = {}
        _write_structured_debug(struct_debug, end_points, B_acd, acd_scores.device)
        end_points.update(struct_debug)
        valid = valid & structured_valid
        all_valid = all_valid & structured_valid
        neg_mask = ~pos_mask
        pos_scores = acd_scores.masked_fill(~pos_mask, 0.0).sum(dim=1)
        pos_count = pos_mask.float().sum(dim=1).clamp(min=1.0)
        pos_mean = pos_scores / pos_count
        neg_for_lse = acd_scores.masked_fill(~neg_mask, torch.finfo(acd_scores.dtype).min)
        neg_lse_raw = neg_for_lse.logsumexp(dim=1)
        neg_count = neg_mask.float().sum(dim=1).clamp(min=1.0)
        neg_logmeanexp = neg_lse_raw - neg_count.log()
        neg_hard = _masked_hard_negative(acd_scores, neg_mask, dim=1)
        per_batch = F.relu(neg_hard - pos_mean + margin) * valid.float()
        valid_count = valid.float().sum().clamp(min=1.0)
        loss_acd_rank_raw = per_batch.sum() / valid_count
        loss_acd_rank = loss_acd_rank_raw * acd_rank_weight
        end_points['loss_acd_rank_raw'] = loss_acd_rank_raw
        end_points['dbg_acd_rank_weight'] = float(acd_rank_weight)
        end_points['dbg_acd_rank_pos_mean'] = (pos_mean * valid.float()).sum() / valid.float().sum().clamp(min=1.0)
        end_points['dbg_acd_rank_neg_hard'] = (neg_hard * valid.float()).sum() / valid.float().sum().clamp(min=1.0)
        end_points['dbg_acd_rank_neg_lse_raw'] = (neg_lse_raw * valid.float()).sum() / valid.float().sum().clamp(min=1.0)
        end_points['dbg_acd_rank_neg_logmeanexp'] = (neg_logmeanexp * valid.float()).sum() / valid.float().sum().clamp(min=1.0)
        end_points['dbg_acd_rank_gap'] = ((pos_mean - neg_hard) * valid.float()).sum() / valid.float().sum().clamp(min=1.0)
        end_points['dbg_acd_rank_margin'] = margin.detach()
        end_points['dbg_acd_rank_structured_valid_ratio'] = structured_valid.float().mean()
        end_points['dbg_acd_rank_valid_ratio'] = valid.float().mean()
        end_points['dbg_acd_rank_target_positive_query_ratio'] = pos_mask.float().mean()
        end_points['dbg_acd_rank_all_positive_query_ratio'] = all_pos_mask.float().mean()
        end_points['dbg_acd_rank_target_valid_ratio'] = valid.float().mean()
        end_points['dbg_acd_rank_all_valid_ratio'] = all_valid.float().mean()
        end_points['dbg_warn_acd_rank_all_invalid'] = float(valid.float().sum().item() == 0)
    end_points['loss_acd_rank'] = loss_acd_rank

    loss_quality = torch.tensor(0.0, device=gt_center.device)
    if use_quality_head:
        if 'quality_logits' not in end_points or 'pred_iou' not in end_points:
            raise ValueError("--use_quality_head is enabled but quality outputs are missing")
        quality_losses = _quality_losses(
            end_points,
            weight=quality_loss_weight,
            iou_threshold=quality_iou_threshold,
        )
        for k, v in quality_losses.items():
            if k == 'loss_quality' and quality_loss_weight <= 0:
                continue
            end_points[k] = v
        if quality_loss_weight > 0:
            loss_quality = quality_losses['loss_quality']

    loss_sacr_rank = torch.tensor(0.0, device=gt_center.device)
    if use_sacr:
        if 'structured_scores' not in end_points:
            raise ValueError("--use_sacr is enabled but structured_scores are missing")
        if last_indices is not None and sacr_rank_loss_weight > 0:
            sacr_losses = _rank_loss_from_scores(
                end_points['structured_scores'].float(),
                last_indices,
                end_points,
                weight=sacr_rank_loss_weight,
                margin=sacr_rank_margin,
                loss_name='loss_sacr_rank',
                debug_prefix='sacr',
            )
            for k, v in sacr_losses.items():
                end_points[k] = v
            loss_sacr_rank = sacr_losses['loss_sacr_rank']

    loss_rapf_gate = torch.tensor(0.0, device=gt_center.device)
    if use_rapf:
        if 'fused_scores' not in end_points or 'rapf_gate' not in end_points:
            raise ValueError("--use_rapf is enabled but RAPF outputs are missing")
        if use_reliability_gate and last_indices is not None and rapf_gate_loss_weight > 0:
            rapf_losses = _rapf_gate_losses(
                end_points,
                last_indices,
                weight=rapf_gate_loss_weight,
                iou_margin=rapf_gate_iou_margin,
            )
            for k, v in rapf_losses.items():
                end_points[k] = v
            loss_rapf_gate = rapf_losses['loss_rapf_gate']

    loss_qahnl = torch.tensor(0.0, device=gt_center.device)
    if use_qahnl:
        if qahnl_config is None:
            raise ValueError("--use_qahnl requires qahnl_config")
        if last_indices is not None:
            qahnl_losses = _qahnl_losses(end_points, last_indices, qahnl_config)
            for k, v in qahnl_losses.items():
                if k == 'loss_qahnl' and qahnl_config.get('loss_weight', 0.2) <= 0:
                    continue
                end_points[k] = v
            if qahnl_config.get('loss_weight', 0.2) > 0:
                loss_qahnl = qahnl_losses['loss_qahnl']

    loss_quality_topk_rerank = torch.tensor(0.0, device=gt_center.device)
    if quality_topk_rerank_weight > 0:
        if 'pred_iou' not in end_points:
            raise ValueError(
                "--quality_topk_rerank_weight requires quality scores"
            )
        quality_topk_losses = _quality_topk_rerank_losses(
            end_points,
            weight=quality_topk_rerank_weight,
            source=quality_topk_rerank_source,
            candidate_k=quality_topk_rerank_k,
            margin=quality_topk_rerank_margin,
            min_iou_gap=quality_topk_rerank_min_iou_gap,
        )
        for k, v in quality_topk_losses.items():
            end_points[k] = v
        loss_quality_topk_rerank = quality_topk_losses[
            'loss_quality_topk_rerank'
        ]

    loss_source_pool_selector = torch.tensor(0.0, device=gt_center.device)
    if source_pool_selector_loss_weight > 0:
        if 'selector_scores' not in end_points:
            raise ValueError(
                "--source_pool_selector_loss_weight requires selector scores"
            )
        selector_losses = _source_pool_selector_losses(
            end_points,
            weight=source_pool_selector_loss_weight,
            source=source_pool_selector_source,
            candidate_k=source_pool_selector_k,
            temperature=source_pool_selector_temperature,
            min_iou_gap=source_pool_selector_min_iou_gap,
            source_min_iou_gaps=source_pool_selector_source_min_iou_gaps,
            pairwise_weight=source_pool_selector_pairwise_weight,
            choice_balance=source_pool_selector_choice_balance,
            choice_balance_power=source_pool_selector_choice_balance_power,
            choice_target=source_pool_selector_choice_target,
            override_default_source=(
                source_pool_selector_override_default_source
            ),
            oracle_prior_weight=source_pool_selector_oracle_prior_weight,
            override_prior_weight=source_pool_selector_override_prior_weight,
            override_source_prior_weight=(
                source_pool_selector_override_source_prior_weight
            ),
            override_margin_weight=(
                source_pool_selector_override_margin_weight
            ),
            override_margin=source_pool_selector_override_margin,
            quality_base_margin_weight=(
                source_pool_selector_quality_base_margin_weight
            ),
            quality_base_margin=source_pool_selector_quality_base_margin,
            quality_default_margin_weight=(
                source_pool_selector_quality_default_margin_weight
            ),
            quality_default_margin=source_pool_selector_quality_default_margin,
            quality_default_bidirectional_margin_weight=(
                source_pool_selector_quality_default_bidirectional_margin_weight
            ),
            quality_default_bidirectional_margin=(
                source_pool_selector_quality_default_bidirectional_margin
            ),
            iou_aux_weight=source_pool_selector_iou_aux_weight,
            iou_aux_margin=source_pool_selector_iou_aux_margin,
            false_base_weight=source_pool_selector_false_base_weight,
            false_override_weight=source_pool_selector_false_override_weight,
            sourcewise_negative_weight=(
                source_pool_selector_sourcewise_negative_weight
            ),
            override_utility_gap_weight=(
                source_pool_selector_override_utility_gap_weight
            ),
        )
        for k, v in selector_losses.items():
            end_points[k] = v
        loss_source_pool_selector = selector_losses[
            'loss_source_pool_selector'
        ]

    loss_detector_policy_adapter = torch.tensor(0.0, device=gt_center.device)
    if detector_policy_adapter_loss_weight > 0:
        if 'detector_policy_adapter_scores' not in end_points:
            raise ValueError(
                "--detector_policy_adapter_loss_weight requires "
                "detector_policy_adapter_scores"
            )
        adapter_losses = _detector_policy_adapter_losses(
            end_points,
            weight=detector_policy_adapter_loss_weight,
            candidate_k=detector_policy_adapter_k,
            margin=detector_policy_adapter_margin,
            min_iou_gap=detector_policy_adapter_min_iou_gap,
        )
        for k, v in adapter_losses.items():
            end_points[k] = v
        loss_detector_policy_adapter = adapter_losses[
            'loss_detector_policy_adapter'
        ]
        if float(detector_policy_adapter_reg_weight) > 0.0:
            loss_detector_policy_adapter = (
                loss_detector_policy_adapter
                + float(detector_policy_adapter_reg_weight)
                * adapter_losses['loss_detector_policy_adapter_reg']
            )
            end_points['loss_detector_policy_adapter'] = (
                loss_detector_policy_adapter
            )

    loss_s2s_aux = torch.tensor(0.0, device=gt_center.device)
    if use_s2s_aux_loss and 'slot_dict' in end_points:
        s2s_losses = compute_s2s_aux_losses(end_points, weight=s2s_aux_weight)
        for k, v in s2s_losses.items():
            end_points[k] = v
        loss_s2s_aux = s2s_losses['loss_s2s_aux']

    # DHC losses (use last_ prefix indices to match last_queries)
    loss_dhc = torch.tensor(0.0, device=gt_center.device)
    if use_dhc and dhc_config is not None and last_indices is not None and 'dhc_margin_entity' in end_points:
        dhc_losses = compute_dhc_losses(end_points, last_indices, dhc_config)
        for k, v in dhc_losses.items():
            end_points[k] = v
            if k.startswith('loss_dhc_'):
                loss_dhc = loss_dhc + v

    # loss
    loss = (
        8 * query_points_generation_loss
        + 1.0 / (num_decoder_layers + 1) * (
            loss_ce
            + 5 * loss_bbox
            + loss_giou
            + loss_contrastive_align
        )
        + loss_s2s_aux
        + loss_acd_rank
        + loss_quality
        + loss_sacr_rank
        + loss_rapf_gate
        + loss_qahnl
        + loss_quality_topk_rerank
        + loss_source_pool_selector
        + loss_detector_policy_adapter
        + loss_dhc
    )
    end_points['loss_ce'] = loss_ce
    end_points['loss_bbox'] = loss_bbox
    end_points['loss_giou'] = loss_giou
    end_points['query_points_generation_loss'] = query_points_generation_loss
    end_points['loss_constrastive_align'] = loss_contrastive_align
    end_points['loss_s2s_aux'] = loss_s2s_aux
    end_points['loss'] = loss
    return loss, end_points
