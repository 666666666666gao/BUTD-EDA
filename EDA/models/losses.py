# ------------------------------------------------------------------------
# Modification: EDA
# Created: 05/21/2022
# Author: Yanmin Wu
# E-mail: wuyanminmax@gmail.com
# https://github.com/yanmin-wu/EDA 
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
import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def box_cxcyczwhd_to_xyzxyz(x):
    x_c, y_c, z_c, w, h, d = x.unbind(-1)
    w = torch.clamp(w, min=1e-6)
    h = torch.clamp(h, min=1e-6)
    d = torch.clamp(d, min=1e-6)
    assert (w < 0).sum() == 0
    assert (h < 0).sum() == 0
    assert (d < 0).sum() == 0
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

# BRIEF 3DIoU loss
def generalized_box_iou3d(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check

    assert (boxes1[:, 3:] >= boxes1[:, :3]).all()
    assert (boxes2[:, 3:] >= boxes2[:, :3]).all()
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
    seed_inds = end_points['seed_inds'].long()      # B, K
    seed_xyz = end_points['seed_xyz']               # B, K, 3
    seeds_obj_cls_logits = end_points['seeds_obj_cls_logits']   # B, 1, K
    gt_center = end_points['center_label'][:, :, :3]            # B, G=132, 3
    gt_size = end_points['size_gts'][:, :, :3]                  # B, G, 3
    B = gt_center.shape[0]  # batch size
    K = seed_xyz.shape[1]   # number if points from p++ output  1024
    G = gt_center.shape[1]  # number of gt boxes (with padding) 132

    # Assign each point to a GT object
    point_instance_label = end_points['point_instance_label']           # B, num_points=5000
    obj_assignment = torch.gather(point_instance_label, 1, seed_inds)   # B, K=1024
    obj_assignment[obj_assignment < 0] = G - 1                          # bg points to last gt
    obj_assignment_one_hot = torch.zeros((B, K, G)).to(seed_xyz.device)
    obj_assignment_one_hot.scatter_(2, obj_assignment.unsqueeze(-1), 1)

    # Normalized distances of points and gt centroids
    delta_xyz = seed_xyz.unsqueeze(2) - gt_center.unsqueeze(1)  # (B, K, G, 3)
    delta_xyz = delta_xyz / (gt_size.unsqueeze(1) + 1e-6)       # (B, K, G, 3)
    new_dist = torch.sum(delta_xyz ** 2, dim=-1)
    euclidean_dist1 = torch.sqrt(new_dist + 1e-6)  # BxKxG
    euclidean_dist1 = (
        euclidean_dist1 * obj_assignment_one_hot
        + 100 * (1 - obj_assignment_one_hot)
    )  # BxKxG
    euclidean_dist1 = euclidean_dist1.transpose(1, 2).contiguous()

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

    For efficiency reasons, the [targets don't include the no_object].
    Because of this, in general, there are [more predictions than targets].
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
        bs, num_queries = outputs["pred_logits"].shape[:2]  # Q: num_queries = 256

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [B*Q, C=256]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [B*Q, 6]

        # Also concat the target labels and boxes
        positive_map = torch.cat([t["positive_map"] for t in targets])  # (B, 256)
        tgt_ids = torch.cat([v["labels"] for v in targets]) # (B)
        tgt_bbox = torch.cat([v["boxes"] for v in targets]) # (B, 6)

        if self.soft_token:
            # pad if necessary
            if out_prob.shape[-1] != positive_map.shape[-1]:
                positive_map = positive_map[..., :out_prob.shape[-1]]
            cost_class = -torch.matmul(out_prob, positive_map.transpose(0, 1))  # (256, 1)
        else:
            # Compute the classification cost.
            # Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching,
            # it can be ommitted. DETR
            # out_prob = out_prob * out_objectness.view(-1, 1)
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)    # ([B*Q, 2])

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou3d(     # ([B*Q, 2])
            box_cxcyczwhd_to_xyzxyz(out_bbox),
            box_cxcyczwhd_to_xyzxyz(tgt_bbox)
        )

        # Final cost matrix
        C = (
            self.cost_bbox * cost_bbox          # 0 * 
            + self.cost_class * cost_class      # 1 * ([B*Q, 2])
            + self.cost_giou * cost_giou        # 2 * ([B*Q, 2])
        ).view(bs, num_queries, -1).cpu()

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

# BRIEF Compute loss
class SetCriterion(nn.Module):
    def __init__(self, matcher, losses={}, eos_coef=0.1, temperature=0.07,
                 sem_align_use_eval_weights=False):
        """
        Parameters:
            matcher: module that matches targets and proposals
            losses: list of all the losses to be applied
            eos_coef: weight of the no-object category
            temperature: used to sharpen the contrastive logits
        """
        super().__init__()
        self.matcher = matcher
        self.eos_coef = eos_coef    # 0.1
        self.losses = losses
        self.temperature = temperature
        self.sem_align_use_eval_weights = sem_align_use_eval_weights

    def _position_component_weights(self, outputs):
        if self.sem_align_use_eval_weights:
            return 1.0, 1.0, 1.0, 1.0
        if outputs["language_dataset"][0] == "sr3d":
            return 0.625, 0.125, 0.125, 0.125
        return 0.6, 0.2, 0.2, 0.1

    def _semantic_component_weights(self):
        if self.sem_align_use_eval_weights:
            return 1.0, 1.0, 1.0, 1.0
        return 1.0, 0.2, 0.2, 0.1
    
    #####################################
    # BRIEF dense position-aligned loss #
    #####################################
    def loss_pos_align(self, outputs, targets, indices, num_boxes, auxi_indices):
        logits = outputs["pred_logits"].log_softmax(-1)
        
        # text position label
        positive_map = torch.cat([t["positive_map"] for t in targets])                  # main object
        modify_positive_map = torch.cat([t["modify_positive_map"] for t in targets])    # attribute(modify)
        pron_positive_map = torch.cat([t["pron_positive_map"] for t in targets])        # pron
        other_entity_map = torch.cat([t["other_entity_map"] for t in targets])          # other(auxi)
        rel_positive_map = torch.cat([t["rel_positive_map"] for t in targets])          # relation

        # Trick to get target indices across batches
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = []
        offset = 0
        for i, (_, tgt) in enumerate(indices):
            tgt_idx.append(tgt + offset)
            offset += len(targets[i]["boxes"])
        tgt_idx = torch.cat(tgt_idx)

        # NOTE constract the position label of the target object
        tgt_pos = positive_map[tgt_idx]
        mod_pos = modify_positive_map[tgt_idx]
        pron_pos = pron_positive_map[tgt_idx]
        other_pos = other_entity_map[tgt_idx]
        rel_pos = rel_positive_map[tgt_idx]
        # TODO ScanRefer & NR3D
        main_w, mod_w, pron_w, rel_w = self._position_component_weights(outputs)
        tgt_weight_pos = (
            tgt_pos * main_w
            + mod_pos * mod_w
            + pron_pos * pron_w
            + rel_pos * rel_w
        )

        # mask, keep the positive term
        pos_mask = tgt_pos + mod_pos + pron_pos + rel_pos + other_pos
        target_mask = torch.zeros_like(logits)
        target_mask[:, :, -1] = 1
        target_mask[src_idx] = pos_mask

        target_sim = torch.zeros_like(logits)
        target_sim[:, :, -1] = 1
        target_sim[src_idx] = tgt_weight_pos

        # STEP Compute entropy
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

    # BRIEF object detection loss.
    def loss_boxes(self, outputs, targets, indices, num_boxes, auxi_indices):
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
        
        loss_giou = 1 - torch.diag(generalized_box_iou3d(
            box_cxcyczwhd_to_xyzxyz(src_boxes),
            box_cxcyczwhd_to_xyzxyz(target_boxes)))

        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    ############################
    # BRIEF semantic alignment #
    ############################
    def loss_sem_align(self, outputs, targets, indices, num_boxes, auxi_indices):
        tokenized = outputs["tokenized"]

        # step 1. Contrastive logits
        norm_text_emb = outputs["proj_tokens"]  # B, num_tokens=L, dim=64
        norm_img_emb = outputs["proj_queries"]  # B, num_queries=256, dim=64
        logits = (
            torch.matmul(norm_img_emb, norm_text_emb.transpose(-1, -2))
            / self.temperature
        )  # [[B, num_queries, num_tokens]

        # step 2. positive map
        # construct a map such that positive_map[k, i, j] = True
        # iff query i is associated to token j in batch item k
        positive_map = torch.zeros(logits.shape, device=logits.device)  # ([B, 256, L])
        # handle 'not mentioned'
        inds = tokenized['attention_mask'].sum(1) - 1
        positive_map[torch.arange(len(inds)), :, inds] = 0.5
        positive_map[torch.arange(len(inds)), :, inds - 1] = 0.5
        # handle true mentions
        pmap = torch.cat([
            t['positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        idx = self._get_src_permutation_idx(indices)
        positive_map[idx] = pmap
        positive_map = positive_map > 0

        modi_positive_map = torch.zeros(logits.shape, device=logits.device)
        pron_positive_map = torch.zeros(logits.shape, device=logits.device)
        other_positive_map = torch.zeros(logits.shape, device=logits.device)
        rel_positive_map = torch.zeros(logits.shape, device=logits.device)
        # [positive, 256] --> [positive, L]
        pmap_modi = torch.cat([
            t['modify_positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]   
        pmap_pron = torch.cat([
            t['pron_positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        pmap_other = torch.cat([
            t['other_entity_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        pmap_rel = torch.cat([
            t['rel_positive_map'][i] for t, (_, i) in zip(targets, indices)
        ], dim=0)[..., :logits.shape[-1]]
        modi_positive_map[idx] = pmap_modi
        pron_positive_map[idx] = pmap_pron
        other_positive_map[idx] = pmap_other
        rel_positive_map[idx] = pmap_rel

        # step object mask
        # Mask for matches <> 'not mentioned'
        mask = torch.full(
            logits.shape[:2],
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )
        mask[idx] = 1.0

        # step text mask
        # Token mask for matches <> 'not mentioned'
        tmask = torch.full(
            (len(logits), logits.shape[-1]),
            self.eos_coef,
            dtype=torch.float32, device=logits.device
        )   # [B, L]
        tmask[torch.arange(len(inds)), inds] = 1.0

        # Positive logits are those who correspond to a match
        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits
        other_entity_neg_term = negative_logits.masked_fill(~(other_positive_map>0), 0)

        modi_positive_logits = -logits.masked_fill(~(modi_positive_map>0), 0)
        pron_positive_logits = -logits.masked_fill(~(pron_positive_map>0), 0)
        rel_positive_logits = -logits.masked_fill(~(rel_positive_map>0), 0)

        pos_modi_term = modi_positive_logits.sum(2)
        pos_pron_term = pron_positive_logits.sum(2)
        pos_rel_term = rel_positive_logits.sum(2)

        # number of the token
        nb_modi_pos_token = (modi_positive_map>0).sum(2) + 1e-6
        nb_pron_pos_token = (pron_positive_map>0).sum(2) + 1e-6
        nb_rel_pos_token = (rel_positive_map>0).sum(2) + 1e-6

        ###############################
        # NOTE loss1: object --> text #
        ###############################
        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        # note negative term
        neg_term = (negative_logits+other_entity_neg_term).logsumexp(2)
        nb_pos_token = positive_map.sum(2) + 1e-6
        entropy = -torch.log(nb_pos_token+1e-6) / nb_pos_token
        _, mod_w, pron_w, rel_w = self._semantic_component_weights()
        box_to_token_loss_ = (
            pos_term/nb_pos_token \
            + mod_w*pos_modi_term/nb_modi_pos_token \
            + pron_w*pos_pron_term/nb_pron_pos_token \
            + rel_w*pos_rel_term/nb_rel_pos_token \
            + neg_term
        ).masked_fill(~boxes_with_pos, 0)
        box_to_token_loss = (box_to_token_loss_ * mask).sum()

        ###############################
        # NOTE loss2: text --> object #
        ###############################
        tokens_with_pos = (positive_map + (modi_positive_map>0) + (pron_positive_map>0) + (rel_positive_map>0)).any(1)
        tmask[positive_map.any(1)] = 1.0
        tmask[(modi_positive_map>0).any(1)] = mod_w
        tmask[(pron_positive_map>0).any(1)] = pron_w
        tmask[(rel_positive_map>0).any(1)] = rel_w
        tmask[torch.arange(len(inds)), inds-1] = 0.1

        pos_term = positive_logits.sum(1)
        pos_modi_term = modi_positive_logits.sum(1)
        pos_pron_term = pron_positive_logits.sum(1)
        pos_rel_term = rel_positive_logits.sum(1)
        # note
        pos_term = pos_term + pos_modi_term + pos_pron_term + pos_rel_term

        neg_term = negative_logits.logsumexp(1)
        nb_pos_obj = positive_map.sum(1) + modi_positive_map.sum(1) + pron_positive_map.sum(1) \
             + rel_positive_map.sum(1) + 1e-6

        entropy = -torch.log(nb_pos_obj+1e-6) / nb_pos_obj
        token_to_box_loss = (
            (entropy + pos_term / nb_pos_obj + neg_term)
        ).masked_fill(~tokens_with_pos, 0)
        token_to_box_loss = (token_to_box_loss * tmask).sum()   

        # total loss
        tot_loss = (box_to_token_loss + token_to_box_loss) / 2
        return {"loss_sem_align": tot_loss / num_boxes}


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
    
    # BRIEF get loss.
    def get_loss(self, loss, outputs, targets, indices, num_boxes, auxi_indices, **kwargs):
        loss_map = {
            'boxes': self.loss_boxes,      # box loss
            'labels': self.loss_pos_align, # position alignment
            'contrastive_align': self.loss_sem_align   # semantic alignment
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, auxi_indices, **kwargs)

    def forward(self, outputs, targets):
        """
        Perform the loss computation.

        Parameters:
             outputs: dict of tensors
             targets: list of dicts, such that len(targets) == batch_size.
        """
        # STEP Retrieve the matching between outputs and targets
        indices = self.matcher(outputs, targets)

        # auxi object
        auxi_target = [
            {
                "labels": targets[b]["labels"],
                "boxes": targets[b]["auxi_box"],
                "positive_map": targets[b]["auxi_entity_positive_map"]
            }
            for b in range(outputs["pred_boxes"].shape[0])
        ]
        auxi_indices = self.matcher(outputs, auxi_target)

        num_boxes = sum(len(inds[1]) for inds in indices)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float,
            device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(
                loss, outputs, targets, indices, num_boxes, auxi_indices
            ))

        return losses, indices


@torch.no_grad()
def _target_iou_matrix(end_points, prefix='last_'):
    pred_center = end_points[f'{prefix}center']
    pred_size = end_points[f'{prefix}pred_size']
    pred_bbox = torch.cat([pred_center, pred_size], dim=-1).detach()
    gt_center = end_points['center_label'][:, :, 0:3]
    gt_size = end_points['size_gts']
    gt_bbox = torch.cat([gt_center, gt_size], dim=-1).detach()
    B, Q = pred_bbox.shape[:2]
    rows = []
    for b in range(B):
        tgt_box = gt_bbox[b, :1]
        ious, _ = _iou3d_par(
            box_cxcyczwhd_to_xyzxyz(tgt_box),
            box_cxcyczwhd_to_xyzxyz(pred_bbox[b])
        )
        rows.append(torch.nan_to_num(ious[0], nan=0.0, posinf=0.0, neginf=0.0))
    return torch.stack(rows, dim=0).clamp(0.0, 1.0)


def _dataset_not_scannet_mask(end_points, batch_size, device):
    datasets = end_points.get('language_dataset', None)
    if datasets is None:
        return torch.ones(batch_size, device=device, dtype=torch.bool)
    if isinstance(datasets, (list, tuple)):
        return torch.tensor(
            [str(item) != 'scannet' for item in datasets[:batch_size]],
            device=device,
            dtype=torch.bool,
        )
    return torch.ones(batch_size, device=device, dtype=torch.bool)


def _crop_or_pad_map(pmap, target_dim):
    if pmap.shape[-1] == target_dim:
        return pmap
    if pmap.shape[-1] > target_dim:
        return pmap[..., :target_dim]
    pad_shape = list(pmap.shape)
    pad_shape[-1] = target_dim - pmap.shape[-1]
    return torch.cat([pmap, pmap.new_zeros(pad_shape)], dim=-1)


def _soft_token_base_scores(end_points, prefix='last_'):
    sem_key = f'{prefix}sem_cls_scores'
    if sem_key not in end_points or 'positive_map' not in end_points:
        return None
    sem_scores = end_points[sem_key].softmax(-1)
    scores = sem_scores.new_zeros(sem_scores.shape[:2])
    terms = (
        ('positive_map', 1.0),
        ('modify_positive_map', 1.0),
        ('pron_positive_map', 1.0),
        ('rel_positive_map', 1.0),
        ('other_entity_map', -1.0),
    )
    for key, weight in terms:
        if key not in end_points:
            continue
        pmap = end_points[key].to(device=sem_scores.device, dtype=sem_scores.dtype)
        if pmap.dim() == 3:
            pmap = pmap[:, :1]
        elif pmap.dim() == 2:
            pmap = pmap.unsqueeze(1)
        pmap = _crop_or_pad_map(pmap, sem_scores.shape[-1])
        term = (sem_scores.unsqueeze(1) * pmap.unsqueeze(2)).sum(-1)
        scores = scores + float(weight) * term[:, 0]
    return scores


def _normalize_query_scores(scores):
    scores = torch.nan_to_num(scores.float(), nan=0.0, posinf=0.0, neginf=0.0)
    centered = scores - scores.mean(dim=1, keepdim=True)
    scale = centered.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
    return centered / scale


def _soft_token_component_scores(end_points, keys, prefix='last_'):
    sem_key = f'{prefix}sem_cls_scores'
    if sem_key not in end_points:
        return None
    sem_scores = end_points[sem_key].softmax(-1)
    scores = sem_scores.new_zeros(sem_scores.shape[:2])
    found = False
    for key in keys:
        if key not in end_points:
            continue
        pmap = end_points[key].to(device=sem_scores.device, dtype=sem_scores.dtype)
        if pmap.dim() == 3:
            pmap = pmap[:, :1]
        elif pmap.dim() == 2:
            pmap = pmap.unsqueeze(1)
        pmap = _crop_or_pad_map(pmap, sem_scores.shape[-1])
        scores = scores + (sem_scores.unsqueeze(1) * pmap.unsqueeze(2)).sum(-1)[:, 0]
        found = True
    return scores if found else None


def _qahnl_hardneg_mining_scores(end_points, scores, config):
    component_keys = []
    if config.get('use_entity_hardneg', False):
        component_keys.extend((
            'positive_map',
            'other_entity_map',
            'auxi_entity_positive_map',
        ))
    if config.get('use_attr_hardneg', False):
        component_keys.extend(('modify_positive_map', 'pron_positive_map'))
    if config.get('use_relation_hardneg', False):
        component_keys.append('rel_positive_map')
    if not component_keys:
        return scores

    component_scores = _soft_token_component_scores(
        end_points, component_keys, prefix='last_'
    )
    if component_scores is None:
        return scores
    weight = float(config.get('hardneg_semantic_weight', 2.0))
    return _normalize_query_scores(scores) + (
        scores.new_tensor(weight) * _normalize_query_scores(component_scores)
    )


def _semantic_eval_base_scores(end_points, prefix='last_'):
    query_key = f'{prefix}proj_queries'
    if query_key not in end_points or 'proj_tokens' not in end_points:
        raise ValueError(
            "Semantic-IoU rank loss requires proj_tokens and "
            f"{query_key}"
        )
    proj_queries = end_points[query_key]
    proj_tokens = end_points['proj_tokens']
    token_scores = torch.matmul(
        proj_queries, proj_tokens.transpose(-1, -2)
    ).div(0.07).softmax(-1)
    scores = token_scores.new_zeros(token_scores.shape[:2])
    terms = (
        ('positive_map', 1.0),
        ('modify_positive_map', 1.0),
        ('pron_positive_map', 1.0),
        ('rel_positive_map', 1.0),
        ('other_entity_map', -1.0),
    )
    for key, weight in terms:
        if key not in end_points:
            continue
        pmap = end_points[key].to(
            device=token_scores.device,
            dtype=token_scores.dtype,
        )
        if pmap.dim() == 3:
            pmap = pmap[:, :1]
        elif pmap.dim() == 2:
            pmap = pmap.unsqueeze(1)
        pmap = _crop_or_pad_map(pmap, token_scores.shape[-1])
        term_scores = (token_scores.unsqueeze(1) * pmap.unsqueeze(2)).sum(-1)
        scores = scores + float(weight) * term_scores[:, 0]
    return scores


def _score_source(end_points, source):
    if source == 'base':
        scores = _soft_token_base_scores(end_points, prefix='last_')
        if scores is not None:
            return scores.float()
        key = 'base_grounding_scores'
    elif source == 'structured':
        key = 'structured_scores'
    elif source == 'quality':
        key = 'pred_iou'
    elif source == 'fused':
        key = 'fused_scores'
    elif source == 'semantic_rerank':
        key = 'semantic_rerank_scores'
    elif source == 'semantic_support':
        key = 'semantic_support_scores'
    else:
        raise ValueError(f"Unknown score source: {source}")
    if key not in end_points:
        raise ValueError(f"Requested {source} scores but {key} is missing")
    return end_points[key].float()


def _build_target_pos_mask(indices, batch_size, num_queries, device):
    mask = torch.zeros(batch_size, num_queries, device=device, dtype=torch.bool)
    valid = torch.zeros(batch_size, device=device, dtype=torch.bool)
    for b, (src, tgt) in enumerate(indices):
        src = src.to(device=device)
        tgt = tgt.to(device=device)
        selected = src[tgt == 0]
        if selected.numel() == 0 and src.numel() > 0:
            selected = src[:1]
        if selected.numel() > 0:
            mask[b, selected] = True
            valid[b] = True
    return mask, valid


def _masked_hard_negative(scores, mask, dim=1):
    masked = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    values = masked.max(dim=dim).values
    return torch.where(
        torch.isfinite(values),
        values,
        torch.zeros_like(values),
    )


def _batch_bool_tensor(end_points, key, batch_size, device):
    value = end_points.get(key, None)
    if value is None:
        return None
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, device=device)
    else:
        value = value.to(device=device)
    return value.bool().view(-1)[:batch_size]


def _rank_loss_from_scores(scores, indices, end_points, weight, margin,
                           loss_name, debug_prefix):
    B, Q = scores.shape
    device = scores.device
    pos_mask, valid = _build_target_pos_mask(indices, B, Q, device)
    structured_valid = end_points.get('structured_valid_mask', None)
    if structured_valid is not None:
        valid = valid & structured_valid.to(device=device).bool().view(-1)[:B]
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
        loss_name: loss_raw * float(weight),
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


def _global_only_mask(end_points, batch_size, device):
    mask = end_points.get('global_only_mask', None)
    if mask is None:
        return torch.zeros(batch_size, device=device, dtype=torch.bool)
    return mask.to(device=device).bool().view(-1)[:batch_size]


def _quality_losses(end_points, weight=1.0, iou_threshold=0.25):
    logits = end_points['quality_logits'].float()
    pred_iou = end_points['pred_iou'].float()
    labels = _target_iou_matrix(end_points, prefix='last_').detach()
    valid = _dataset_not_scannet_mask(end_points, logits.shape[0], logits.device)
    valid_q = valid.unsqueeze(1).float()
    denom = valid_q.sum().mul(logits.shape[1]).clamp(min=1.0)
    reg_raw = F.smooth_l1_loss(torch.sigmoid(logits), labels, reduction='none')
    cls_targets = (labels >= 0.5).float()
    cls_raw = F.binary_cross_entropy_with_logits(
        logits, cls_targets, reduction='none'
    )
    reg_raw = (reg_raw * valid_q).sum() / denom
    cls_raw = (cls_raw * valid_q).sum() / denom
    loss_raw = reg_raw + cls_raw
    quality_top = pred_iou.argmax(dim=1)
    quality_top_iou = labels[
        torch.arange(labels.shape[0], device=labels.device), quality_top
    ]
    return {
        'loss_quality': loss_raw * weight,
        'dbg_quality_reg_raw': reg_raw,
        'dbg_quality_cls_raw': cls_raw,
        'dbg_quality_total_raw': loss_raw,
        'dbg_quality_target_iou_mean': (labels * valid_q).sum() / denom,
        'dbg_quality_pred_iou_mean': (pred_iou * valid_q).sum() / denom,
        'dbg_quality_positive_ratio': (
            ((labels >= iou_threshold).float() * valid_q).sum() / denom
        ),
        'dbg_quality_positive50_ratio': (
            ((labels >= 0.50).float() * valid_q).sum() / denom
        ),
        'dbg_quality_top1_iou': (
            quality_top_iou * valid.float()
        ).sum() / valid.float().sum().clamp(min=1.0),
    }


def _rapf_gate_losses(end_points, weight=0.2, iou_margin=0.02):
    gate = end_points['rapf_gate'].float()
    B, Q = gate.shape
    device = gate.device
    target_iou = _target_iou_matrix(end_points, prefix='last_')
    base_scores = _score_source(end_points, 'base')
    structured_scores = _score_source(end_points, 'structured')
    base_top = base_scores.argmax(dim=1)
    structured_top = structured_scores.argmax(dim=1)
    labels = (
        target_iou[torch.arange(B, device=device), structured_top]
        > target_iou[torch.arange(B, device=device), base_top] + float(iou_margin)
    ).float()
    valid = end_points.get('structured_valid_mask', None)
    if valid is None:
        valid = torch.ones(B, device=device, dtype=torch.bool)
    else:
        valid = valid.to(device=device).bool().view(-1)[:B]
    valid = valid & (~_global_only_mask(end_points, B, device))
    valid = valid & _dataset_not_scannet_mask(end_points, B, device)
    sample_gate = gate.mean(dim=1).clamp(1e-6, 1.0 - 1e-6)
    per_sample = F.binary_cross_entropy(
        sample_gate.float(), labels.float(), reduction='none'
    )
    valid_f = valid.float()
    loss_raw = (per_sample * valid_f).sum() / valid_f.sum().clamp(min=1.0)
    return {
        'loss_rapf_gate': loss_raw * weight,
        'loss_rapf_gate_raw': loss_raw,
        'dbg_rapf_gate_label_mean': (
            labels * valid_f
        ).sum() / valid_f.sum().clamp(min=1.0),
        'dbg_rapf_gate_supervision_valid_ratio': valid.float().mean(),
    }


def _qahnl_losses(end_points, indices, config):
    source = config.get('score_source', 'fused')
    scores = _score_source(end_points, source)
    B, Q = scores.shape
    device = scores.device
    target_iou = _target_iou_matrix(end_points, prefix='last_')
    pos_thr = float(config.get('pos_iou_thresh', 0.25))
    neg_thr = float(config.get('neg_iou_thresh', 0.10))
    topk_iou = max(1, min(int(config.get('topk_iou_pos', 3)), Q))
    num_hard = max(1, min(int(config.get('num_hard_neg', 16)), Q))
    base_margin = float(config.get('margin_base', 0.2))
    margin_iou_lambda = float(config.get('margin_iou_lambda', 0.5))
    margin_min = float(config.get('margin_min', 0.05))
    margin_max = float(config.get('margin_max', 0.5))
    temperature = max(
        1e-6,
        min(float(config.get('temperature', 1.0)),
            float(config.get('temperature_max', 6.0))),
    )
    weight = float(config.get('loss_weight', 0.2))

    pos_mask = target_iou >= pos_thr
    pos_mask.scatter_(1, torch.topk(target_iou, topk_iou, dim=1).indices, True)
    matched_target_mask, matched_valid = _build_target_pos_mask(
        indices, B, Q, device
    )
    pos_mask = pos_mask | matched_target_mask
    neg_candidates = (target_iou <= neg_thr) & (~pos_mask)
    mining_scores = _qahnl_hardneg_mining_scores(end_points, scores, config)
    hard_scores = mining_scores.masked_fill(
        ~neg_candidates, torch.finfo(scores.dtype).min
    )
    hard_idx = torch.topk(hard_scores, num_hard, dim=1).indices
    neg_mask = torch.zeros(B, Q, device=device, dtype=torch.bool)
    neg_mask.scatter_(1, hard_idx, True)
    neg_mask = neg_mask & neg_candidates
    valid = (
        matched_valid
        & _dataset_not_scannet_mask(end_points, B, device)
        & pos_mask.any(dim=1)
        & neg_mask.any(dim=1)
    )
    if source == 'structured':
        valid = valid & (~_global_only_mask(end_points, B, device))
    valid_f = valid.float()
    valid_count = valid_f.sum().clamp(min=1.0)

    pos_score = scores.masked_fill(~pos_mask, torch.finfo(scores.dtype).min).max(dim=1).values
    neg_score = scores.masked_fill(~neg_mask, torch.finfo(scores.dtype).min).max(dim=1).values
    pos_iou = target_iou.masked_fill(~pos_mask, 0.0).max(dim=1).values
    neg_iou = target_iou.masked_fill(~neg_mask, 0.0).max(dim=1).values
    adaptive_margin = scores.new_tensor(base_margin) + (
        scores.new_tensor(margin_iou_lambda) * (pos_iou - neg_iou)
    )
    adaptive_margin = adaptive_margin.clamp(min=margin_min, max=margin_max)
    temp_t = scores.new_tensor(temperature)
    loss_raw = (
        temp_t
        * F.softplus((neg_score - pos_score + adaptive_margin) / temp_t)
        * valid_f
    ).sum() / valid_count
    violation = (neg_score - pos_score + adaptive_margin > 0) & valid
    total = float(B * Q)
    return {
        'loss_qahnl': loss_raw * weight,
        'loss_qahnl_raw': loss_raw,
        'dbg_qahnl_positive_query_ratio': pos_mask.float().sum() / total,
        'dbg_qahnl_negative_query_ratio': neg_mask.float().sum() / total,
        'dbg_qahnl_valid_batch_ratio': valid.float().mean(),
        'dbg_qahnl_pos_score': (pos_score * valid_f).sum() / valid_count,
        'dbg_qahnl_neg_score': (neg_score * valid_f).sum() / valid_count,
        'dbg_qahnl_pos_iou': (pos_iou * valid_f).sum() / valid_count,
        'dbg_qahnl_neg_iou': (neg_iou * valid_f).sum() / valid_count,
        'dbg_qahnl_score_gap_mean': (
            (pos_score - neg_score) * valid_f
        ).sum() / valid_count,
        'dbg_qahnl_iou_gap_mean': (
            (pos_iou - neg_iou) * valid_f
        ).sum() / valid_count,
        'dbg_qahnl_margin_mean': (
            adaptive_margin * valid_f
        ).sum() / valid_count,
        'dbg_qahnl_violation_ratio': violation.float().sum() / valid_count,
        'dbg_qahnl_score_source_base': float(source == 'base'),
        'dbg_qahnl_score_source_structured': float(source == 'structured'),
        'dbg_qahnl_score_source_quality': float(source == 'quality'),
        'dbg_qahnl_score_source_fused': float(source == 'fused'),
        'dbg_qahnl_score_source_semantic_rerank': float(
            source == 'semantic_rerank'
        ),
        'dbg_qahnl_score_source_semantic_support': float(
            source == 'semantic_support'
        ),
    }


def _semantic_iou_rank_losses(end_points, config):
    scores = _semantic_eval_base_scores(end_points, prefix='last_').float()
    B, Q = scores.shape
    device = scores.device
    target_iou = _target_iou_matrix(end_points, prefix='last_')
    pos_thr = float(config.get('pos_iou_thresh', 0.50))
    neg_thr = float(config.get('neg_iou_thresh', 0.25))
    topk_iou = max(1, min(int(config.get('topk_iou_pos', 1)), Q))
    num_hard = max(1, min(int(config.get('num_hard_neg', 16)), Q))
    margin = float(config.get('margin', 0.1))
    temperature = max(1e-6, float(config.get('temperature', 1.0)))
    weight = float(config.get('loss_weight', 0.05))

    pos_mask = target_iou >= pos_thr
    pos_mask.scatter_(1, torch.topk(target_iou, topk_iou, dim=1).indices, True)
    neg_candidates = (target_iou <= neg_thr) & (~pos_mask)
    hard_scores = scores.masked_fill(
        ~neg_candidates, torch.finfo(scores.dtype).min
    )
    hard_idx = torch.topk(hard_scores, num_hard, dim=1).indices
    neg_mask = torch.zeros(B, Q, device=device, dtype=torch.bool)
    neg_mask.scatter_(1, hard_idx, True)
    neg_mask = neg_mask & neg_candidates
    valid = (
        _dataset_not_scannet_mask(end_points, B, device)
        & pos_mask.any(dim=1)
        & neg_mask.any(dim=1)
    )
    valid_f = valid.float()
    valid_count = valid_f.sum().clamp(min=1.0)

    pos_score = _masked_hard_negative(scores, pos_mask, dim=1)
    neg_score = _masked_hard_negative(scores, neg_mask, dim=1)
    pos_iou = _masked_hard_negative(target_iou, pos_mask, dim=1)
    neg_iou = _masked_hard_negative(target_iou, neg_mask, dim=1)
    margin_t = scores.new_tensor(margin)
    temp_t = scores.new_tensor(temperature)
    loss_raw = (
        temp_t
        * F.softplus((neg_score - pos_score + margin_t) / temp_t)
        * valid_f
    ).sum() / valid_count
    violation = (neg_score - pos_score + margin_t > 0) & valid
    total = float(B * Q)
    return {
        'loss_sem_iou_rank': loss_raw * weight,
        'loss_sem_iou_rank_raw': loss_raw,
        'dbg_sem_iou_rank_positive_query_ratio': (
            pos_mask.float().sum() / total
        ),
        'dbg_sem_iou_rank_negative_query_ratio': (
            neg_mask.float().sum() / total
        ),
        'dbg_sem_iou_rank_valid_batch_ratio': valid.float().mean(),
        'dbg_sem_iou_rank_pos_score': (
            pos_score * valid_f
        ).sum() / valid_count,
        'dbg_sem_iou_rank_neg_score': (
            neg_score * valid_f
        ).sum() / valid_count,
        'dbg_sem_iou_rank_pos_iou': (
            pos_iou * valid_f
        ).sum() / valid_count,
        'dbg_sem_iou_rank_neg_iou': (
            neg_iou * valid_f
        ).sum() / valid_count,
        'dbg_sem_iou_rank_score_gap_mean': (
            (pos_score - neg_score) * valid_f
        ).sum() / valid_count,
        'dbg_sem_iou_rank_iou_gap_mean': (
            (pos_iou - neg_iou) * valid_f
        ).sum() / valid_count,
        'dbg_sem_iou_rank_margin_mean': (
            margin_t.expand_as(pos_score) * valid_f
        ).sum() / valid_count,
        'dbg_sem_iou_rank_violation_ratio': (
            violation.float().sum() / valid_count
        ),
    }


def _semantic_iou_listwise_losses(end_points, config):
    scores = _semantic_eval_base_scores(end_points, prefix='last_').float()
    B, Q = scores.shape
    device = scores.device
    target_iou = _target_iou_matrix(
        end_points, prefix='last_'
    ).to(device=device, dtype=scores.dtype)
    topk = max(1, min(int(config.get('topk', 32)), Q))
    score_temperature = max(
        1e-6, float(config.get('score_temperature', 0.25))
    )
    target_iou_power = max(
        0.0, float(config.get('target_iou_power', 2.0))
    )
    high_iou_threshold = float(config.get('high_iou_threshold', 0.50))
    high_iou_weight = max(
        0.0, float(config.get('high_iou_weight', 2.0))
    )
    min_target_iou = max(
        0.0, float(config.get('min_target_iou', 0.01))
    )
    weight = float(config.get('loss_weight', 0.05))

    candidate_mask = torch.zeros(B, Q, device=device, dtype=torch.bool)
    candidate_mask.scatter_(1, torch.topk(scores.detach(), topk, dim=1).indices, True)
    candidate_mask.scatter_(1, torch.topk(target_iou, topk, dim=1).indices, True)

    target_weights = target_iou.clamp(min=0.0)
    if min_target_iou > 0:
        target_weights = torch.where(
            target_weights >= min_target_iou,
            target_weights,
            torch.zeros_like(target_weights),
        )
    if target_iou_power == 0.0:
        target_weights = (target_weights > 0).to(dtype=target_weights.dtype)
    elif target_iou_power != 1.0:
        target_weights = target_weights.pow(target_iou_power)
    if high_iou_weight != 1.0:
        target_weights = target_weights * torch.where(
            target_iou >= high_iou_threshold,
            target_iou.new_tensor(high_iou_weight),
            target_iou.new_tensor(1.0),
        )
    target_weights = target_weights * candidate_mask.float()

    target_top_iou, target_top_idx = target_iou.max(dim=1)
    needs_fallback = (
        (target_weights.sum(dim=1) <= 0)
        & (target_top_iou >= min_target_iou)
    )
    if needs_fallback.any():
        target_weights = target_weights.clone()
        target_weights[needs_fallback, target_top_idx[needs_fallback]] = 1.0

    valid = (
        _dataset_not_scannet_mask(end_points, B, device)
        & (target_weights.sum(dim=1) > 0)
    )
    valid_f = valid.float()
    valid_count = valid_f.sum().clamp(min=1.0)
    target_probs = target_weights / target_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

    masked_scores = scores.masked_fill(
        ~candidate_mask, scores.new_tensor(-1e4)
    )
    log_probs = F.log_softmax(
        masked_scores / scores.new_tensor(score_temperature), dim=1
    )
    per_batch = -(
        target_probs * log_probs.masked_fill(~candidate_mask, 0.0)
    ).sum(dim=1)
    loss_raw = (per_batch * valid_f).sum() / valid_count

    pred_top = masked_scores.argmax(dim=1)
    pred_top_iou = target_iou[
        torch.arange(B, device=device), pred_top
    ]
    total = float(B * Q)
    return {
        'loss_sem_iou_listwise': loss_raw * weight,
        'loss_sem_iou_listwise_raw': loss_raw,
        'dbg_sem_iou_listwise_candidate_query_ratio': (
            candidate_mask.float().sum() / total
        ),
        'dbg_sem_iou_listwise_valid_batch_ratio': valid.float().mean(),
        'dbg_sem_iou_listwise_target_top_iou': (
            target_top_iou * valid_f
        ).sum() / valid_count,
        'dbg_sem_iou_listwise_pred_top_iou': (
            pred_top_iou * valid_f
        ).sum() / valid_count,
        'dbg_sem_iou_listwise_target_entropy': (
            -(target_probs * target_probs.clamp(min=1e-6).log()).sum(dim=1)
            * valid_f
        ).sum() / valid_count,
    }


def _semantic_iou_top1_losses(end_points, config):
    scores = _semantic_eval_base_scores(end_points, prefix='last_').float()
    B, Q = scores.shape
    device = scores.device
    target_iou = _target_iou_matrix(
        end_points, prefix='last_'
    ).to(device=device, dtype=scores.dtype)
    weight = float(config.get('loss_weight', 0.02))
    pos_iou_thresh = float(config.get('pos_iou_thresh', 0.50))
    iou_gap = max(0.0, float(config.get('iou_gap', 0.05)))
    margin = float(config.get('margin', 0.1))
    temperature = max(1e-6, float(config.get('temperature', 0.5)))

    batch_idx = torch.arange(B, device=device)
    pred_top_idx = scores.detach().argmax(dim=1)
    target_top_iou, target_top_idx = target_iou.max(dim=1)
    pred_top_iou = target_iou[batch_idx, pred_top_idx]
    pos_score = scores[batch_idx, target_top_idx]
    neg_score = scores[batch_idx, pred_top_idx]
    top1_iou_gap = target_top_iou - pred_top_iou
    valid = (
        _dataset_not_scannet_mask(end_points, B, device)
        & (target_top_iou >= scores.new_tensor(pos_iou_thresh))
        & (top1_iou_gap >= scores.new_tensor(iou_gap))
    )
    valid_f = valid.float()
    valid_count = valid_f.sum().clamp(min=1.0)
    margin_t = scores.new_tensor(margin)
    temp_t = scores.new_tensor(temperature)
    loss_raw = (
        temp_t
        * F.softplus((neg_score - pos_score + margin_t) / temp_t)
        * valid_f
    ).sum() / valid_count
    violation = (neg_score - pos_score + margin_t > 0) & valid
    return {
        'loss_sem_iou_top1': loss_raw * weight,
        'loss_sem_iou_top1_raw': loss_raw,
        'dbg_sem_iou_top1_valid_batch_ratio': valid.float().mean(),
        'dbg_sem_iou_top1_target_top_iou': (
            target_top_iou * valid_f
        ).sum() / valid_count,
        'dbg_sem_iou_top1_pred_top_iou': (
            pred_top_iou * valid_f
        ).sum() / valid_count,
        'dbg_sem_iou_top1_iou_gap': (
            top1_iou_gap * valid_f
        ).sum() / valid_count,
        'dbg_sem_iou_top1_score_gap': (
            (pos_score - neg_score) * valid_f
        ).sum() / valid_count,
        'dbg_sem_iou_top1_violation_ratio': (
            violation.float().sum() / valid_count
        ),
    }


def _semantic_eval_margin_losses(end_points, indices, config):
    scores = _semantic_eval_base_scores(end_points, prefix='last_').float()
    B, Q = scores.shape
    device = scores.device
    target_iou = _target_iou_matrix(
        end_points, prefix='last_'
    ).to(device=device, dtype=scores.dtype)
    weight = float(config.get('loss_weight', 0.02))
    min_pos_iou = float(config.get('min_pos_iou', 0.50))
    neg_iou_thresh = float(config.get('neg_iou_thresh', 0.45))
    num_hard = max(1, min(int(config.get('num_hard_neg', 8)), Q))
    margin = float(config.get('margin', 0.08))
    temperature = max(1e-6, float(config.get('temperature', 0.5)))

    pos_mask, matched_valid = _build_target_pos_mask(indices, B, Q, device)
    neg_candidates = (~pos_mask) & (
        target_iou <= scores.new_tensor(neg_iou_thresh)
    )
    hard_scores = scores.masked_fill(
        ~neg_candidates, torch.finfo(scores.dtype).min
    )
    hard_idx = torch.topk(hard_scores, num_hard, dim=1).indices
    neg_mask = torch.zeros(B, Q, device=device, dtype=torch.bool)
    neg_mask.scatter_(1, hard_idx, True)
    neg_mask = neg_mask & neg_candidates

    pos_score = _masked_hard_negative(scores, pos_mask, dim=1)
    neg_score = _masked_hard_negative(scores, neg_mask, dim=1)
    pos_iou = _masked_hard_negative(target_iou, pos_mask, dim=1)
    neg_iou = _masked_hard_negative(target_iou, neg_mask, dim=1)
    valid = (
        matched_valid
        & _dataset_not_scannet_mask(end_points, B, device)
        & pos_mask.any(dim=1)
        & neg_mask.any(dim=1)
        & (pos_iou >= scores.new_tensor(min_pos_iou))
    )
    valid_f = valid.float()
    valid_count = valid_f.sum().clamp(min=1.0)
    margin_t = scores.new_tensor(margin)
    temp_t = scores.new_tensor(temperature)
    loss_raw = (
        temp_t
        * F.softplus((neg_score - pos_score + margin_t) / temp_t)
        * valid_f
    ).sum() / valid_count
    violation = (neg_score - pos_score + margin_t > 0) & valid
    total = float(B * Q)
    return {
        'loss_sem_eval_margin': loss_raw * weight,
        'loss_sem_eval_margin_raw': loss_raw,
        'dbg_sem_eval_margin_valid_batch_ratio': valid.float().mean(),
        'dbg_sem_eval_margin_positive_query_ratio': (
            pos_mask.float().sum() / total
        ),
        'dbg_sem_eval_margin_negative_query_ratio': (
            neg_mask.float().sum() / total
        ),
        'dbg_sem_eval_margin_pos_score': (
            pos_score * valid_f
        ).sum() / valid_count,
        'dbg_sem_eval_margin_neg_score': (
            neg_score * valid_f
        ).sum() / valid_count,
        'dbg_sem_eval_margin_pos_iou': (
            pos_iou * valid_f
        ).sum() / valid_count,
        'dbg_sem_eval_margin_neg_iou': (
            neg_iou * valid_f
        ).sum() / valid_count,
        'dbg_sem_eval_margin_score_gap_mean': (
            (pos_score - neg_score) * valid_f
        ).sum() / valid_count,
        'dbg_sem_eval_margin_violation_ratio': (
            violation.float().sum() / valid_count
        ),
    }


def _semantic_rerank_losses(end_points, config):
    train_use_support_scores = bool(
        config.get('train_use_support_scores', False)
    )
    score_key = (
        'semantic_support_scores'
        if train_use_support_scores else 'semantic_rerank_scores'
    )
    if score_key not in end_points:
        raise ValueError(
            "--use_semantic_rerank_head is enabled but "
            f"{score_key} are missing"
        )
    scores = end_points[score_key].float()
    B, Q = scores.shape
    device = scores.device
    target_iou = _target_iou_matrix(
        end_points, prefix='last_'
    ).to(device=device, dtype=scores.dtype)
    topk = max(1, min(int(config.get('topk', 16)), Q))
    temperature = max(1e-6, float(config.get('temperature', 0.5)))
    target_iou_power = max(
        0.0, float(config.get('target_iou_power', 2.0))
    )
    min_target_iou = max(
        0.0, float(config.get('min_target_iou', 0.01))
    )
    hard_sample_weight = max(
        0.0, float(config.get('hard_sample_weight', 0.0))
    )
    multi_sample_weight = max(
        0.0, float(config.get('multi_sample_weight', 0.0))
    )
    high_iou_threshold = max(
        0.0, float(config.get('high_iou_threshold', 0.5))
    )
    high_iou_weight = max(
        1.0, float(config.get('high_iou_weight', 1.0))
    )
    top1_margin_weight = max(
        0.0, float(config.get('top1_margin_weight', 0.0))
    )
    top1_margin = max(0.0, float(config.get('top1_margin', 0.1)))
    top1_neg_iou_threshold = max(
        0.0, float(config.get('top1_neg_iou_threshold', 0.25))
    )
    listwise_weight = max(
        0.0, float(config.get('listwise_weight', 1.0))
    )
    threshold_mass_weight = max(
        0.0, float(config.get('threshold_mass_weight', 0.0))
    )
    failure_margin_weight = max(
        0.0, float(config.get('failure_margin_weight', 0.0))
    )
    failure_margin = max(
        0.0, float(config.get('failure_margin', 0.1))
    )
    weight = float(config.get('loss_weight', 0.02))

    candidate_mask = torch.zeros(B, Q, device=device, dtype=torch.bool)
    candidate_mask.scatter_(1, torch.topk(scores.detach(), topk, dim=1).indices, True)
    base_scores = end_points.get(
        'semantic_rerank_base_scores',
        end_points.get('semantic_eval_base_scores', None),
    )
    if base_scores is not None:
        base_scores = base_scores.to(device=device, dtype=scores.dtype)
        candidate_mask.scatter_(
            1, torch.topk(base_scores.detach(), topk, dim=1).indices, True
        )
    candidate_mask.scatter_(1, torch.topk(target_iou, topk, dim=1).indices, True)

    target_weights = target_iou.clamp(min=0.0)
    if min_target_iou > 0:
        target_weights = torch.where(
            target_weights >= min_target_iou,
            target_weights,
            torch.zeros_like(target_weights),
        )
    if target_iou_power == 0.0:
        target_weights = (target_weights > 0).to(dtype=target_weights.dtype)
    elif target_iou_power != 1.0:
        target_weights = target_weights.pow(target_iou_power)
    if high_iou_weight > 1.0:
        target_weights = target_weights * torch.where(
            target_iou >= high_iou_threshold,
            target_weights.new_tensor(high_iou_weight),
            target_weights.new_tensor(1.0),
        )
    target_weights = target_weights * candidate_mask.float()

    target_top_iou, target_top_idx = target_iou.max(dim=1)
    needs_fallback = (
        (target_weights.sum(dim=1) <= 0)
        & (target_top_iou >= min_target_iou)
    )
    if needs_fallback.any():
        target_weights = target_weights.clone()
        target_weights[needs_fallback, target_top_idx[needs_fallback]] = 1.0

    valid = (
        _dataset_not_scannet_mask(end_points, B, device)
        & (target_weights.sum(dim=1) > 0)
    )
    valid_f = valid.float()
    valid_count = valid_f.sum().clamp(min=1.0)
    target_probs = target_weights / target_weights.sum(
        dim=1, keepdim=True
    ).clamp(min=1e-6)

    sample_weights = torch.ones(B, device=device, dtype=scores.dtype)
    hard_mask = _batch_bool_tensor(end_points, 'is_hard', B, device)
    if hard_mask is not None and hard_sample_weight > 0:
        sample_weights = sample_weights + (
            hard_mask.float() * scores.new_tensor(hard_sample_weight)
        )
    unique_mask = _batch_bool_tensor(end_points, 'is_unique', B, device)
    if unique_mask is not None and multi_sample_weight > 0:
        sample_weights = sample_weights + (
            (~unique_mask).float() * scores.new_tensor(multi_sample_weight)
        )
    sample_weights = sample_weights * valid_f

    masked_scores = scores.masked_fill(
        ~candidate_mask, scores.new_tensor(-1e4)
    )
    log_probs = F.log_softmax(
        masked_scores / scores.new_tensor(temperature), dim=1
    )
    per_batch = -(
        target_probs * log_probs.masked_fill(~candidate_mask, 0.0)
    ).sum(dim=1)
    listwise_loss = (
        per_batch * sample_weights
    ).sum() / sample_weights.sum().clamp(min=1.0)

    threshold_mass_terms = []
    threshold_valid_ratios = []
    candidate_logits = masked_scores / scores.new_tensor(temperature)
    candidate_log_mass = torch.logsumexp(candidate_logits, dim=1)
    for threshold in (0.25, 0.50):
        positive_mask = candidate_mask & (target_iou > threshold)
        threshold_valid = (
            _dataset_not_scannet_mask(end_points, B, device)
            & positive_mask.any(dim=1)
        )
        positive_logits = candidate_logits.masked_fill(
            ~positive_mask, torch.finfo(scores.dtype).min
        )
        positive_log_mass = torch.logsumexp(positive_logits, dim=1)
        threshold_weights = sample_weights * threshold_valid.float()
        threshold_loss = (
            (candidate_log_mass - positive_log_mass)
            * threshold_weights
        ).sum() / threshold_weights.sum().clamp(min=1.0)
        threshold_mass_terms.append(threshold_loss)
        threshold_valid_ratios.append(threshold_valid.float().mean())
    threshold_mass_loss = torch.stack(threshold_mass_terms).mean()

    # Target only correctable top-1 failures. The discrete top-1/failure gate
    # is computed from detached scores so this term cannot move an already
    # correct sample merely to increase probability mass elsewhere.
    detached_top = masked_scores.detach().argmax(dim=1)
    detached_top_iou = target_iou[
        torch.arange(B, device=device), detached_top
    ]
    detached_top_scores = scores[
        torch.arange(B, device=device), detached_top
    ]
    failure_margin_terms = []
    failure_valid_ratios = []
    for threshold in (0.25, 0.50):
        positive_mask = candidate_mask & (target_iou > threshold)
        positive_scores = scores.masked_fill(
            ~positive_mask, torch.finfo(scores.dtype).min
        ).max(dim=1).values
        failure_valid = (
            _dataset_not_scannet_mask(end_points, B, device)
            & (detached_top_iou <= threshold)
            & positive_mask.any(dim=1)
        )
        failure_valid_f = failure_valid.float()
        failure_weights = sample_weights * failure_valid_f
        safe_positive_scores = torch.where(
            failure_valid, positive_scores, torch.zeros_like(positive_scores)
        )
        safe_top_scores = torch.where(
            failure_valid,
            detached_top_scores,
            torch.zeros_like(detached_top_scores),
        )
        failure_loss = (
            scores.new_tensor(temperature)
            * F.softplus(
                (
                    safe_top_scores - safe_positive_scores
                    + scores.new_tensor(failure_margin)
                ) / scores.new_tensor(temperature)
            )
            * failure_weights
        ).sum() / failure_weights.sum().clamp(min=1.0)
        failure_margin_terms.append(failure_loss)
        failure_valid_ratios.append(failure_valid_f.mean())
    failure_margin_loss = torch.stack(failure_margin_terms).mean()

    high_iou_mask = candidate_mask & (target_iou >= high_iou_threshold)
    low_iou_mask = candidate_mask & (
        target_iou <= top1_neg_iou_threshold
    )
    positive_scores = scores.masked_fill(
        ~high_iou_mask, torch.finfo(scores.dtype).min
    ).max(dim=1).values
    negative_scores = scores.masked_fill(
        ~low_iou_mask, torch.finfo(scores.dtype).min
    ).max(dim=1).values
    margin_valid = valid & high_iou_mask.any(dim=1) & low_iou_mask.any(dim=1)
    margin_valid_f = margin_valid.float()
    margin_sample_weights = sample_weights * margin_valid_f
    positive_scores = torch.where(
        margin_valid, positive_scores, torch.zeros_like(positive_scores)
    )
    negative_scores = torch.where(
        margin_valid, negative_scores, torch.zeros_like(negative_scores)
    )
    margin_loss = (
        scores.new_tensor(temperature)
        * F.softplus(
            (
                negative_scores - positive_scores
                + scores.new_tensor(top1_margin)
            ) / scores.new_tensor(temperature)
        )
        * margin_sample_weights
    ).sum() / margin_sample_weights.sum().clamp(min=1.0)
    loss_raw = (
        scores.new_tensor(listwise_weight) * listwise_loss
        + scores.new_tensor(threshold_mass_weight) * threshold_mass_loss
        + scores.new_tensor(failure_margin_weight) * failure_margin_loss
        + scores.new_tensor(top1_margin_weight) * margin_loss
    )

    pred_top = masked_scores.argmax(dim=1)
    pred_top_iou = target_iou[
        torch.arange(B, device=device), pred_top
    ]
    total = float(B * Q)
    return {
        'loss_semantic_rerank': loss_raw * weight,
        'loss_semantic_rerank_raw': loss_raw,
        'dbg_semantic_rerank_listwise_loss': listwise_loss.detach(),
        'dbg_semantic_rerank_threshold_mass_loss': (
            threshold_mass_loss.detach()
        ),
        'dbg_semantic_rerank_threshold25_valid_ratio': (
            threshold_valid_ratios[0].detach()
        ),
        'dbg_semantic_rerank_threshold50_valid_ratio': (
            threshold_valid_ratios[1].detach()
        ),
        'dbg_semantic_rerank_failure_margin_loss': (
            failure_margin_loss.detach()
        ),
        'dbg_semantic_rerank_failure25_valid_ratio': (
            failure_valid_ratios[0].detach()
        ),
        'dbg_semantic_rerank_failure50_valid_ratio': (
            failure_valid_ratios[1].detach()
        ),
        'dbg_semantic_rerank_train_uses_support': scores.new_tensor(
            float(train_use_support_scores)
        ),
        'dbg_semantic_rerank_top1_margin_loss': margin_loss.detach(),
        'dbg_semantic_rerank_top1_margin_valid_ratio': (
            margin_valid.float().mean()
        ),
        'dbg_semantic_rerank_candidate_query_ratio': (
            candidate_mask.float().sum() / total
        ),
        'dbg_semantic_rerank_valid_batch_ratio': valid.float().mean(),
        'dbg_semantic_rerank_target_top_iou': (
            target_top_iou * valid_f
        ).sum() / valid_count,
        'dbg_semantic_rerank_pred_top_iou': (
            pred_top_iou * valid_f
        ).sum() / valid_count,
    }


def _semantic_component_losses(end_points, config):
    if 'semantic_component_scores' not in end_points:
        raise ValueError(
            "--use_semantic_component_calibration is enabled but "
            "semantic_component_scores are missing"
        )
    scores = end_points['semantic_component_scores'].float()
    B, Q = scores.shape
    device = scores.device
    target_iou = _target_iou_matrix(
        end_points, prefix='last_'
    ).to(device=device, dtype=scores.dtype)
    topk = max(1, min(int(config.get('topk', 16)), Q))
    temperature = max(1e-6, float(config.get('temperature', 1.0)))
    target_iou_power = max(
        0.0, float(config.get('target_iou_power', 2.0))
    )
    min_target_iou = max(
        0.0, float(config.get('min_target_iou', 0.01))
    )
    hard_sample_weight = max(
        0.0, float(config.get('hard_sample_weight', 0.0))
    )
    multi_sample_weight = max(
        0.0, float(config.get('multi_sample_weight', 0.0))
    )
    weight = float(config.get('loss_weight', 0.02))

    candidate_mask = torch.zeros(B, Q, device=device, dtype=torch.bool)
    candidate_mask.scatter_(1, torch.topk(scores.detach(), topk, dim=1).indices, True)
    base_scores = end_points.get('semantic_eval_base_scores', None)
    if base_scores is not None:
        base_scores = base_scores.to(device=device, dtype=scores.dtype)
        candidate_mask.scatter_(
            1, torch.topk(base_scores.detach(), topk, dim=1).indices, True
        )
    candidate_mask.scatter_(1, torch.topk(target_iou, topk, dim=1).indices, True)

    target_weights = target_iou.clamp(min=0.0)
    if min_target_iou > 0:
        target_weights = torch.where(
            target_weights >= min_target_iou,
            target_weights,
            torch.zeros_like(target_weights),
        )
    if target_iou_power == 0.0:
        target_weights = (target_weights > 0).to(dtype=target_weights.dtype)
    elif target_iou_power != 1.0:
        target_weights = target_weights.pow(target_iou_power)
    target_weights = target_weights * candidate_mask.float()

    target_top_iou, target_top_idx = target_iou.max(dim=1)
    needs_fallback = (
        (target_weights.sum(dim=1) <= 0)
        & (target_top_iou >= min_target_iou)
    )
    if needs_fallback.any():
        target_weights = target_weights.clone()
        target_weights[needs_fallback, target_top_idx[needs_fallback]] = 1.0

    valid = (
        _dataset_not_scannet_mask(end_points, B, device)
        & (target_weights.sum(dim=1) > 0)
    )
    valid_f = valid.float()
    valid_count = valid_f.sum().clamp(min=1.0)
    target_probs = target_weights / target_weights.sum(
        dim=1, keepdim=True
    ).clamp(min=1e-6)

    sample_weights = torch.ones(B, device=device, dtype=scores.dtype)
    hard_mask = _batch_bool_tensor(end_points, 'is_hard', B, device)
    if hard_mask is not None and hard_sample_weight > 0:
        sample_weights = sample_weights + (
            hard_mask.float() * scores.new_tensor(hard_sample_weight)
        )
    unique_mask = _batch_bool_tensor(end_points, 'is_unique', B, device)
    if unique_mask is not None and multi_sample_weight > 0:
        sample_weights = sample_weights + (
            (~unique_mask).float() * scores.new_tensor(multi_sample_weight)
        )
    sample_weights = sample_weights * valid_f

    masked_scores = scores.masked_fill(
        ~candidate_mask, scores.new_tensor(-1e4)
    )
    log_probs = F.log_softmax(
        masked_scores / scores.new_tensor(temperature), dim=1
    )
    per_batch = -(
        target_probs * log_probs.masked_fill(~candidate_mask, 0.0)
    ).sum(dim=1)
    loss_raw = (
        per_batch * sample_weights
    ).sum() / sample_weights.sum().clamp(min=1.0)

    pred_top = masked_scores.argmax(dim=1)
    pred_top_iou = target_iou[
        torch.arange(B, device=device), pred_top
    ]
    total = float(B * Q)
    return {
        'loss_semantic_component': loss_raw * weight,
        'loss_semantic_component_raw': loss_raw,
        'dbg_semantic_component_candidate_query_ratio': (
            candidate_mask.float().sum() / total
        ),
        'dbg_semantic_component_valid_batch_ratio': valid.float().mean(),
        'dbg_semantic_component_sample_weight_mean': (
            sample_weights.sum() / valid_f.sum().clamp(min=1.0)
        ),
        'dbg_semantic_component_hard_sample_ratio': (
            hard_mask.float().mean()
            if hard_mask is not None
            else torch.tensor(0.0, device=device)
        ),
        'dbg_semantic_component_multi_sample_ratio': (
            (~unique_mask).float().mean()
            if unique_mask is not None
            else torch.tensor(0.0, device=device)
        ),
        'dbg_semantic_component_target_top_iou': (
            target_top_iou * valid_f
        ).sum() / valid_count,
        'dbg_semantic_component_pred_top_iou': (
            pred_top_iou * valid_f
        ).sum() / valid_count,
    }


def _semantic_support_gate_losses(end_points, config):
    """Supervise whether detector support should preserve or defer ranking.

    The target is derived only from training-time 3D IoU.  A gate target of
    one keeps the fixed RAPF support residual, while zero defers to the raw
    semantic ranking.  Samples on which both rankings have identical utility
    at the two evaluation thresholds are deliberately ignored.
    """
    required = (
        'semantic_support_gate',
        'semantic_support_raw_scores',
        'semantic_support_fixed_scores',
    )
    missing = [key for key in required if key not in end_points]
    if missing:
        raise ValueError(
            "Semantic support gate loss requires: " + ", ".join(missing)
        )

    gate = end_points['semantic_support_gate'].float().squeeze(-1)
    raw_scores = end_points['semantic_support_raw_scores'].float().detach()
    fixed_scores = end_points['semantic_support_fixed_scores'].float().detach()
    if raw_scores.shape != fixed_scores.shape:
        raise ValueError("raw and fixed semantic support scores must match")
    if gate.dim() != 1 or gate.shape[0] != raw_scores.shape[0]:
        raise ValueError("semantic_support_gate must have shape [B, 1]")

    low = float(config.get('low_iou_threshold', 0.25))
    high = float(config.get('high_iou_threshold', 0.5))
    beta = float(config.get('beta', 0.25))
    weight = max(0.0, float(config.get('loss_weight', 0.0)))
    if not (0.0 <= low < high <= 1.0):
        raise ValueError("support gate IoU thresholds must satisfy 0 <= low < high <= 1")
    if beta <= 0.0:
        raise ValueError("semantic support gate loss beta must be positive")

    target_iou = _target_iou_matrix(
        end_points, prefix='last_'
    ).to(device=gate.device, dtype=raw_scores.dtype)
    raw_index = raw_scores.argmax(dim=1, keepdim=True)
    fixed_index = fixed_scores.argmax(dim=1, keepdim=True)
    raw_iou = target_iou.gather(1, raw_index).squeeze(1)
    fixed_iou = target_iou.gather(1, fixed_index).squeeze(1)

    raw_utility = (
        raw_iou.ge(low).float() + raw_iou.ge(high).float()
    )
    fixed_utility = (
        fixed_iou.ge(low).float() + fixed_iou.ge(high).float()
    )
    utility_delta = fixed_utility - raw_utility
    valid = (
        utility_delta.ne(0)
        & raw_index.squeeze(1).ne(fixed_index.squeeze(1))
        & _dataset_not_scannet_mask(
            end_points, gate.shape[0], gate.device
        )
    )
    target_gate = utility_delta.gt(0).to(dtype=gate.dtype)
    error = (gate - target_gate).abs()
    per_sample = torch.where(
        error < beta,
        0.5 * error.square() / beta,
        error - 0.5 * beta,
    )
    valid_f = valid.float()
    valid_count = valid_f.sum().clamp(min=1.0)
    loss_raw = (per_sample * valid_f).sum() / valid_count

    return {
        'loss_semantic_support_gate': loss_raw * weight,
        'loss_semantic_support_gate_raw': loss_raw,
        'dbg_semantic_support_gate_valid_batch_ratio': valid_f.mean(),
        'dbg_semantic_support_gate_keep_target_ratio': (
            target_gate * valid_f
        ).sum() / valid_count,
        'dbg_semantic_support_gate_pred_mean': (
            gate * valid_f
        ).sum() / valid_count,
        'dbg_semantic_support_gate_raw_iou': (
            raw_iou * valid_f
        ).sum() / valid_count,
        'dbg_semantic_support_gate_fixed_iou': (
            fixed_iou * valid_f
        ).sum() / valid_count,
    }


def _semantic_threshold_losses(end_points, config):
    if 'semantic_threshold_logits' not in end_points:
        raise ValueError(
            "Semantic threshold loss requires semantic_threshold_logits"
        )
    logits = end_points['semantic_threshold_logits'].float()
    if logits.dim() != 3 or logits.shape[-1] != 2:
        raise ValueError("semantic_threshold_logits must have shape [B, Q, 2]")
    target_iou = _target_iou_matrix(
        end_points, prefix='last_'
    ).to(device=logits.device, dtype=logits.dtype)
    targets = torch.stack([
        target_iou.ge(0.25),
        target_iou.ge(0.50),
    ], dim=-1).float()
    valid = _dataset_not_scannet_mask(
        end_points, logits.shape[0], logits.device
    )
    valid_q = valid.view(-1, 1, 1).float()
    denom = valid_q.sum().mul(logits.shape[1] * 2).clamp(min=1.0)

    # Per-batch positive balancing keeps the scarce IoU>=0.5 queries visible
    # without injecting dataset-specific class or scene metadata.
    positive_count = (targets * valid_q).sum(dim=(0, 1))
    total_count = valid_q.sum().mul(logits.shape[1]).clamp(min=1.0)
    negative_count = total_count - positive_count
    positive_weight = (
        negative_count / positive_count.clamp(min=1.0)
    ).clamp(min=1.0, max=32.0)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=positive_weight,
        reduction='none',
    )
    probability = torch.sigmoid(logits)
    pt = probability * targets + (1.0 - probability) * (1.0 - targets)
    gamma = max(0.0, float(config.get('focal_gamma', 2.0)))
    if gamma > 0:
        bce = bce * (1.0 - pt).pow(gamma)
    bce_loss = (bce * valid_q).sum() / denom
    bce_weight = max(0.0, float(config.get('bce_weight', 1.0)))

    # Supervise the exact local decision made at inference: should the fixed
    # support scorer keep its top query or swap to its runner-up?  Restricting
    # the target to clear one-good/one-bad pairs prevents gradients from
    # disturbing samples for which either choice has the same threshold
    # outcome.
    fixed_scores = end_points.get(
        'semantic_support_scores_without_threshold', None
    )
    pairwise_weight = max(
        0.0, float(config.get('pairwise_weight', 0.0))
    )
    pairwise_margin = max(
        0.0, float(config.get('pairwise_margin', 0.25))
    )
    pairwise_temperature = max(
        1e-6, float(config.get('pairwise_temperature', 0.5))
    )
    pairwise_high_weight = max(
        0.0, float(config.get('pairwise_high_weight', 1.0))
    )
    pairwise_terms = []
    pairwise_valid_ratios = []
    if fixed_scores is not None:
        fixed_scores = fixed_scores.to(
            device=logits.device, dtype=logits.dtype
        )
        top_pair = torch.topk(
            fixed_scores.detach(), k=min(2, logits.shape[1]), dim=1
        ).indices
        if top_pair.shape[1] == 2:
            pair_iou = torch.gather(target_iou, 1, top_pair)
            for threshold_idx, threshold in enumerate((0.25, 0.50)):
                pair_good = pair_iou.ge(threshold)
                pair_valid = (
                    valid
                    & pair_good[:, 0].ne(pair_good[:, 1])
                )
                pair_logits = torch.gather(
                    logits[..., threshold_idx], 1, top_pair
                )
                good_idx = pair_good.long().argmax(dim=1, keepdim=True)
                bad_idx = 1 - good_idx
                good_logits = torch.gather(
                    pair_logits, 1, good_idx
                ).squeeze(1)
                bad_logits = torch.gather(
                    pair_logits, 1, bad_idx
                ).squeeze(1)
                pair_valid_f = pair_valid.float()
                pair_loss = (
                    logits.new_tensor(pairwise_temperature)
                    * F.softplus(
                        (
                            bad_logits - good_logits
                            + logits.new_tensor(pairwise_margin)
                        ) / logits.new_tensor(pairwise_temperature)
                    )
                    * pair_valid_f
                ).sum() / pair_valid_f.sum().clamp(min=1.0)
                if threshold_idx == 1:
                    pair_loss = pair_loss * logits.new_tensor(
                        pairwise_high_weight
                    )
                pairwise_terms.append(pair_loss)
                pairwise_valid_ratios.append(pair_valid_f.mean())
    if pairwise_terms:
        pairwise_loss = torch.stack(pairwise_terms).mean()
    else:
        pairwise_loss = logits.sum() * 0.0
        pairwise_valid_ratios = [
            logits.new_tensor(0.0), logits.new_tensor(0.0)
        ]

    loss_raw = (
        logits.new_tensor(bce_weight) * bce_loss
        + logits.new_tensor(pairwise_weight) * pairwise_loss
    )
    weight = max(0.0, float(config.get('loss_weight', 1.0)))
    predicted = probability.detach()
    return {
        'loss_semantic_threshold': loss_raw * weight,
        'loss_semantic_threshold_raw': loss_raw,
        'dbg_semantic_threshold_bce_loss': bce_loss.detach(),
        'dbg_semantic_threshold_pairwise_loss': pairwise_loss.detach(),
        'dbg_semantic_threshold_pairwise25_valid_ratio': (
            pairwise_valid_ratios[0].detach()
        ),
        'dbg_semantic_threshold_pairwise50_valid_ratio': (
            pairwise_valid_ratios[1].detach()
        ),
        'dbg_semantic_threshold_positive25_ratio': (
            targets[..., 0] * valid_q.squeeze(-1)
        ).sum() / total_count,
        'dbg_semantic_threshold_positive50_ratio': (
            targets[..., 1] * valid_q.squeeze(-1)
        ).sum() / total_count,
        'dbg_semantic_threshold_pred25_mean': (
            predicted[..., 0] * valid_q.squeeze(-1)
        ).sum() / total_count,
        'dbg_semantic_threshold_pred50_mean': (
            predicted[..., 1] * valid_q.squeeze(-1)
        ).sum() / total_count,
    }


# BRIEF loss
def compute_hungarian_loss(end_points, num_decoder_layers, set_criterion,
                           query_points_obj_topk=5,
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
                           use_sem_iou_rank=False,
                           sem_iou_rank_config=None,
                           use_sem_iou_listwise=False,
                           sem_iou_listwise_config=None,
                           use_sem_iou_top1=False,
                           sem_iou_top1_config=None,
                           use_sem_eval_margin=False,
                           sem_eval_margin_config=None,
                           use_semantic_rerank_head=False,
                           semantic_rerank_config=None,
                           semantic_threshold_config=None,
                           semantic_support_gate_config=None,
                           use_semantic_component_calibration=False,
                           semantic_component_config=None):
    """Compute Hungarian matching loss containing CE, bbox and giou."""
    prefixes = ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    prefixes = ['proposal_'] + prefixes     # 6+1: 'proposal_'  'last_' '0head_'  '1head_'  '2head_'  '3head_'  '4head_'

    # STEP target GT box
    gt_center = end_points['center_label'][:, :, 0:3]
    gt_size = end_points['size_gts']
    gt_labels = end_points['sem_cls_label']
    gt_bbox = torch.cat([gt_center, gt_size], dim=-1)
    # text
    positive_map = end_points['positive_map']               # main obj.
    modify_positive_map = end_points['modify_positive_map'] # attribute(modify)
    pron_positive_map = end_points['pron_positive_map']     # pron
    other_entity_map = end_points['other_entity_map']       # other(auxi)
    rel_positive_map = end_points['rel_positive_map']       # relation
    box_label_mask = end_points['box_label_mask']           # (132,) target object mask
    auxi_entity_positive_map = end_points['auxi_entity_positive_map']
    auxi_box = end_points['auxi_box']

    target = [
        {
            "labels": gt_labels[b, box_label_mask[b].bool()],
            "boxes": gt_bbox[b, box_label_mask[b].bool()],
            "positive_map": positive_map[b, box_label_mask[b].bool()],
            "modify_positive_map": modify_positive_map[b, box_label_mask[b].bool()],
            "pron_positive_map": pron_positive_map[b, box_label_mask[b].bool()],
            "other_entity_map": other_entity_map[b, box_label_mask[b].bool()],
            "rel_positive_map": rel_positive_map[b, box_label_mask[b].bool()],
            "auxi_entity_positive_map": auxi_entity_positive_map[b, 0].unsqueeze(0),
            "auxi_box": auxi_box[b]
        }
        for b in range(gt_labels.shape[0])
    ]

    loss_ce, loss_bbox, loss_giou, loss_sem_align = 0, 0, 0, 0
    last_indices = None
    for prefix in prefixes:
        output = {}
        if 'proj_tokens' in end_points:
            output['proj_tokens'] = end_points['proj_tokens']           
            output['proj_queries'] = end_points[f'{prefix}proj_queries']
            output['tokenized'] = end_points['tokenized']

        # STEP Get predicted boxes and labels
        pred_center = end_points[f'{prefix}center']     # B, K, 3
        pred_size = end_points[f'{prefix}pred_size']    # (B,K,3) (l,w,h)
        pred_bbox = torch.cat([pred_center, pred_size], dim=-1)
        pred_logits = end_points[f'{prefix}sem_cls_scores']     # (B, Q, n_class)
        output['pred_logits'] = pred_logits
        output["pred_boxes"] = pred_bbox
        output["language_dataset"] = end_points["language_dataset"] # dataset

        # NOTE Compute all the requested losses, forward
        losses, indices = set_criterion(output, target)
        if prefix == 'last_':
            last_indices = indices
        for loss_key in losses.keys():
            end_points[f'{prefix}_{loss_key}'] = losses[loss_key]
        loss_ce += losses.get('loss_ce', 0)
        loss_bbox += losses['loss_bbox']
        loss_giou += losses.get('loss_giou', 0)
        if 'proj_tokens' in end_points:
            loss_sem_align += losses['loss_sem_align']

    if 'seeds_obj_cls_logits' in end_points.keys():
        query_points_generation_loss = compute_points_obj_cls_loss_hard_topk(
            end_points, query_points_obj_topk
        )
    else:
        query_points_generation_loss = 0.0

    loss_quality = torch.tensor(0.0, device=gt_center.device)
    if use_quality_head:
        if 'quality_logits' not in end_points or 'pred_iou' not in end_points:
            raise ValueError(
                "--use_quality_head is enabled but quality outputs are missing"
            )
        quality_losses = _quality_losses(
            end_points,
            weight=quality_loss_weight,
            iou_threshold=quality_iou_threshold,
        )
        for key, value in quality_losses.items():
            end_points[key] = value
        if quality_loss_weight > 0:
            loss_quality = quality_losses['loss_quality']

    loss_sacr_rank = torch.tensor(0.0, device=gt_center.device)
    if use_sacr:
        if 'structured_scores' not in end_points:
            raise ValueError(
                "--use_sacr is enabled but structured_scores are missing"
            )
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
            for key, value in sacr_losses.items():
                end_points[key] = value
            loss_sacr_rank = sacr_losses['loss_sacr_rank']
    end_points['loss_sacr_rank'] = loss_sacr_rank

    loss_rapf_gate = torch.tensor(0.0, device=gt_center.device)
    if use_rapf:
        if 'fused_scores' not in end_points or 'rapf_gate' not in end_points:
            raise ValueError(
                "--use_rapf is enabled but RAPF outputs are missing"
            )
        if use_reliability_gate and rapf_gate_loss_weight > 0:
            rapf_losses = _rapf_gate_losses(
                end_points,
                weight=rapf_gate_loss_weight,
                iou_margin=rapf_gate_iou_margin,
            )
            for key, value in rapf_losses.items():
                end_points[key] = value
            loss_rapf_gate = rapf_losses['loss_rapf_gate']

    loss_qahnl = torch.tensor(0.0, device=gt_center.device)
    if use_qahnl:
        if qahnl_config is None:
            raise ValueError("--use_qahnl requires qahnl_config")
        if last_indices is not None:
            qahnl_losses = _qahnl_losses(end_points, last_indices, qahnl_config)
            for key, value in qahnl_losses.items():
                end_points[key] = value
            if qahnl_config.get('loss_weight', 0.2) > 0:
                loss_qahnl = qahnl_losses['loss_qahnl']

    loss_sem_iou_rank = torch.tensor(0.0, device=gt_center.device)
    if use_sem_iou_rank:
        sem_iou_rank_config = sem_iou_rank_config or {}
        sem_iou_losses = _semantic_iou_rank_losses(
            end_points,
            sem_iou_rank_config,
        )
        for key, value in sem_iou_losses.items():
            end_points[key] = value
        if sem_iou_rank_config.get('loss_weight', 0.05) > 0:
            loss_sem_iou_rank = sem_iou_losses['loss_sem_iou_rank']
    end_points['loss_sem_iou_rank'] = loss_sem_iou_rank

    loss_sem_iou_listwise = torch.tensor(0.0, device=gt_center.device)
    if use_sem_iou_listwise:
        sem_iou_listwise_config = sem_iou_listwise_config or {}
        sem_iou_listwise_losses = _semantic_iou_listwise_losses(
            end_points,
            sem_iou_listwise_config,
        )
        for key, value in sem_iou_listwise_losses.items():
            end_points[key] = value
        if sem_iou_listwise_config.get('loss_weight', 0.05) > 0:
            loss_sem_iou_listwise = sem_iou_listwise_losses[
                'loss_sem_iou_listwise'
            ]
    end_points['loss_sem_iou_listwise'] = loss_sem_iou_listwise

    loss_sem_iou_top1 = torch.tensor(0.0, device=gt_center.device)
    if use_sem_iou_top1:
        sem_iou_top1_config = sem_iou_top1_config or {}
        sem_iou_top1_losses = _semantic_iou_top1_losses(
            end_points,
            sem_iou_top1_config,
        )
        for key, value in sem_iou_top1_losses.items():
            end_points[key] = value
        if sem_iou_top1_config.get('loss_weight', 0.02) > 0:
            loss_sem_iou_top1 = sem_iou_top1_losses['loss_sem_iou_top1']
    end_points['loss_sem_iou_top1'] = loss_sem_iou_top1

    loss_sem_eval_margin = torch.tensor(0.0, device=gt_center.device)
    if use_sem_eval_margin:
        sem_eval_margin_config = sem_eval_margin_config or {}
        if last_indices is not None:
            sem_eval_margin_losses = _semantic_eval_margin_losses(
                end_points,
                last_indices,
                sem_eval_margin_config,
            )
            for key, value in sem_eval_margin_losses.items():
                end_points[key] = value
            if sem_eval_margin_config.get('loss_weight', 0.02) > 0:
                loss_sem_eval_margin = sem_eval_margin_losses[
                    'loss_sem_eval_margin'
                ]
    end_points['loss_sem_eval_margin'] = loss_sem_eval_margin

    loss_semantic_rerank = torch.tensor(0.0, device=gt_center.device)
    if use_semantic_rerank_head:
        semantic_rerank_config = semantic_rerank_config or {}
        semantic_rerank_losses = _semantic_rerank_losses(
            end_points,
            semantic_rerank_config,
        )
        for key, value in semantic_rerank_losses.items():
            end_points[key] = value
        if semantic_rerank_config.get('loss_weight', 0.02) > 0:
            loss_semantic_rerank = semantic_rerank_losses[
                'loss_semantic_rerank'
            ]
    end_points['loss_semantic_rerank'] = loss_semantic_rerank

    loss_semantic_threshold = torch.tensor(0.0, device=gt_center.device)
    if semantic_threshold_config is not None:
        semantic_threshold_losses = _semantic_threshold_losses(
            end_points,
            semantic_threshold_config,
        )
        for key, value in semantic_threshold_losses.items():
            end_points[key] = value
        if semantic_threshold_config.get('loss_weight', 1.0) > 0:
            loss_semantic_threshold = semantic_threshold_losses[
                'loss_semantic_threshold'
            ]
    end_points['loss_semantic_threshold'] = loss_semantic_threshold

    loss_semantic_support_gate = torch.tensor(0.0, device=gt_center.device)
    if semantic_support_gate_config is not None:
        semantic_support_gate_losses = _semantic_support_gate_losses(
            end_points,
            semantic_support_gate_config,
        )
        for key, value in semantic_support_gate_losses.items():
            end_points[key] = value
        if semantic_support_gate_config.get('loss_weight', 0.0) > 0:
            loss_semantic_support_gate = semantic_support_gate_losses[
                'loss_semantic_support_gate'
            ]
    end_points['loss_semantic_support_gate'] = loss_semantic_support_gate

    loss_semantic_component = torch.tensor(0.0, device=gt_center.device)
    if use_semantic_component_calibration:
        semantic_component_config = semantic_component_config or {}
        semantic_component_losses = _semantic_component_losses(
            end_points,
            semantic_component_config,
        )
        for key, value in semantic_component_losses.items():
            end_points[key] = value
        if semantic_component_config.get('loss_weight', 0.02) > 0:
            loss_semantic_component = semantic_component_losses[
                'loss_semantic_component'
            ]
    end_points['loss_semantic_component'] = loss_semantic_component

    # total loss
    weight = 1
    if str(end_points["language_dataset"][0]).startswith("scanrefer"):
        weight = 0.5
    loss = (
        8 * query_points_generation_loss
        + 1.0 / (num_decoder_layers + 1) * (
            weight * loss_ce
            + 5 * loss_bbox
            + loss_giou
            + weight * loss_sem_align
        )
        + loss_quality
        + loss_sacr_rank
        + loss_rapf_gate
        + loss_qahnl
        + loss_sem_iou_rank
        + loss_sem_iou_listwise
        + loss_sem_iou_top1
        + loss_sem_eval_margin
        + loss_semantic_rerank
        + loss_semantic_threshold
        + loss_semantic_support_gate
        + loss_semantic_component
    )
    end_points['loss_ce'] = loss_ce
    end_points['loss_bbox'] = loss_bbox
    end_points['loss_giou'] = loss_giou
    end_points['query_points_generation_loss'] = query_points_generation_loss
    end_points['loss_sem_align'] = loss_sem_align
    end_points['loss'] = loss
    return loss, end_points
