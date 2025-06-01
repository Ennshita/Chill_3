import copy
import math 
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from datetime import datetime 

from ...core import register
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .dfine_utils import bbox2distance
from .iou_utils import bbox_distillation_iou_loss 


@register()
class DFINECriterion(nn.Module):
    def __init__(
        self,
        matcher,
        weight_dict,
        losses, 
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        use_distillation: bool = False, 
        distill_decay_type: str = 'linear_epoch',
        distill_stop_epoch_ratio: float = 1.0,
        distill_cls_loss_factor: float = 1.0,
        distill_l1_loss_factor: float = 5.0,
        distill_iou_loss_factor: float = 2.0,
        distill_iou_ratio: float = 1.25,
        distill_power_transform_p: float = 2.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict 
        self.original_losses = copy.deepcopy(losses) 
        self.losses_to_compute_for_main_output = copy.deepcopy(losses)

        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None 
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.enable_distillation_processing = use_distillation
        self.distill_decay_type = distill_decay_type
        self.distill_stop_epoch_ratio = distill_stop_epoch_ratio
        self.distill_cls_w_factor = distill_cls_loss_factor
        self.distill_l1_w_factor = distill_l1_loss_factor
        self.distill_iou_w_factor = distill_iou_loss_factor
        self.distill_iou_expanded_ratio = distill_iou_ratio
        self.distill_power_p_for_transform = distill_power_transform_p
        
        if self.enable_distillation_processing and 'distill' not in self.losses_to_compute_for_main_output:
            self.losses_to_compute_for_main_output.append('distill')
            if 'loss_distill' not in self.weight_dict:
                print(f"[{datetime.now().isoformat()}] WARNING: DFINECriterion: 'loss_distill' not found in weight_dict. Using default weight 1.0 for distillation loss.")
                self.weight_dict['loss_distill'] = 1.0 
        
        print(f"[{datetime.now().isoformat()}] INFO: DFINECriterion initialized. Losses for main output: {self.losses_to_compute_for_main_output}")
        if self.enable_distillation_processing:
            print(f"  Distillation enabled: decay='{self.distill_decay_type}', stop_ratio={self.distill_stop_epoch_ratio}")
            print(f"  Distill sub-factors: cls={self.distill_cls_w_factor}, l1={self.distill_l1_w_factor}, iou={self.distill_iou_w_factor}")
            print(f"  Distill IoU params: ratio={self.distill_iou_expanded_ratio}, power_p={self.distill_power_p_for_transform}")

    def loss_labels_focal(self, outputs, targets, indices, num_boxes, meta=None, **kwargs_ignored):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0: 
            return {"loss_focal": torch.tensor(0.0, device=src_logits.device)}

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :self.num_classes].float()
        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits, target, alpha=self.alpha, gamma=self.gamma, reduction="sum"
        )
        loss = loss / (num_boxes + 1e-9)
        return {"loss_focal": loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None, meta=None, **kwargs_ignored):
        assert "pred_logits" in outputs and "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)

        # Ưu tiên `values` từ `meta` nếu `values` trực tiếp không được cung cấp
        if values is None and meta and 'values' in meta:
            values = meta['values']
        
        if idx[0].numel() == 0:
            return {"loss_vfl": torch.tensor(0.0, device=outputs["pred_logits"].device)}

        if values is None:
            src_boxes_matched = outputs["pred_boxes"][idx]
            target_boxes_matched = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            if src_boxes_matched.numel() == 0 or target_boxes_matched.numel() == 0 or src_boxes_matched.shape[0] != target_boxes_matched.shape[0]:
                 return {"loss_vfl": torch.tensor(0.0, device=outputs["pred_logits"].device)}
            ious_matrix, _ = box_iou(box_cxcywh_to_xyxy(src_boxes_matched), box_cxcywh_to_xyxy(target_boxes_matched))
            ious = torch.diag(ious_matrix).detach()
        else:
            ious = values.detach()

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        target_classes_one_hot = F.one_hot(target_classes_o, num_classes=self.num_classes).to(src_logits.dtype)

        target_score_vfl = torch.zeros_like(src_logits, device=src_logits.device)
        target_score_vfl[idx[0], idx[1], target_classes_o] = ious.to(target_score_vfl.dtype)

        pred_score_sigmoid_detached = src_logits.sigmoid().detach()
        
        p_t = target_score_vfl
        weight = self.alpha * pred_score_sigmoid_detached.pow(self.gamma) * (1 - p_t) + p_t
        weight = weight.detach()

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score_vfl, weight=weight, reduction="sum")
        loss = loss / (num_boxes + 1e-9)
        return {"loss_vfl": loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None, meta=None, **kwargs_ignored):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
             return {"loss_bbox": torch.tensor(0.0, device=outputs["pred_boxes"].device), 
                     "loss_giou": torch.tensor(0.0, device=outputs["pred_boxes"].device)}

        if boxes_weight is None and meta and 'boxes_weight' in meta:
            boxes_weight = meta['boxes_weight']
            
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox_sum = loss_bbox.sum() / (num_boxes + 1e-9)

        loss_giou_values = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        if boxes_weight is not None: 
            loss_giou_values = loss_giou_values * boxes_weight
        loss_giou_sum = loss_giou_values.sum() / (num_boxes + 1e-9)

        return {"loss_bbox": loss_bbox_sum, "loss_giou": loss_giou_sum}
        
    def loss_local(self, outputs, targets, indices, num_boxes, T=5, meta=None, **kwargs_ignored):
        losses = {}
        if "pred_corners" in outputs:
            idx = self._get_src_permutation_idx(indices)
            if idx[0].numel() == 0:
                losses["loss_fgl"] = torch.tensor(0.0, device=outputs["pred_corners"].device)
                if "teacher_corners" in outputs:
                    losses["loss_ddf"] = torch.tensor(0.0, device=outputs["pred_corners"].device)
                return losses
                
            target_boxes_matched = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            pred_corners_matched = outputs["pred_corners"][idx].reshape(-1, (self.reg_max + 1)) 
            ref_points_matched = outputs["ref_points"][idx].detach()

            with torch.no_grad(): 
                target_key = "fgl_targets_dn" if "is_dn" in outputs else "fgl_targets"
                if getattr(self, target_key, None) is None:
                    calculated_fgl_targets = bbox2distance(
                        ref_points_matched,
                        box_cxcywh_to_xyxy(target_boxes_matched),
                        self.reg_max,
                        outputs["reg_scale"],
                        outputs["up"],
                    )
                    setattr(self, target_key, calculated_fgl_targets)
            
            target_corners, weight_right, weight_left = getattr(self, target_key)

            # Tính IoUs cho FGL weight
            src_boxes_for_fgl_iou = outputs["pred_boxes"][idx]
            if src_boxes_for_fgl_iou.shape[0] == target_boxes_matched.shape[0] and target_boxes_matched.numel() > 0:
                 ious_matrix_fgl, _ = box_iou(
                     box_cxcywh_to_xyxy(src_boxes_for_fgl_iou), box_cxcywh_to_xyxy(target_boxes_matched)
                 )
                 ious_for_fgl_weight = torch.diag(ious_matrix_fgl).detach()
            else: 
                 ious_for_fgl_weight = torch.ones(pred_corners_matched.shape[0] // 4, 
                                                  device=pred_corners_matched.device, 
                                                  dtype=pred_corners_matched.dtype) 
            
            if ious_for_fgl_weight.numel() * 4 == pred_corners_matched.shape[0]:
                weight_targets_fgl = ious_for_fgl_weight.unsqueeze(-1).repeat(1, 4).reshape(-1)
            else: 
                weight_targets_fgl = torch.ones(pred_corners_matched.shape[0], 
                                                device=pred_corners_matched.device,
                                                dtype=pred_corners_matched.dtype)

            losses["loss_fgl"] = self.unimodal_distribution_focal_loss(
                pred_corners_matched, target_corners,
                weight_right, weight_left,
                weight=weight_targets_fgl, avg_factor=num_boxes
            )
            
            # DDF Loss
            if "teacher_corners" in outputs and "teacher_logits" in outputs:
                pred_corners_all = outputs["pred_corners"].reshape(-1, (self.reg_max + 1)) 
                teacher_corners_all = outputs["teacher_corners"].reshape(-1, (self.reg_max + 1))
                if torch.equal(pred_corners_all, teacher_corners_all):
                    losses["loss_ddf"] = pred_corners_all.sum() * 0
                else:
                    weight_targets_local_ddf = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]
                    
                    mask_ddf = torch.zeros_like(weight_targets_local_ddf, dtype=torch.bool)
                    mask_ddf[idx] = True
                    mask_ddf = mask_ddf.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_all_queries = outputs["teacher_logits"].sigmoid().max(dim=-1)[0].detach()
                    weight_all_queries[idx] = ious_for_fgl_weight.to(weight_all_queries.dtype)
                    weight_targets_local_final = weight_all_queries.unsqueeze(-1).repeat(1,1,4).reshape(-1)

                    loss_kl_ddf = nn.KLDivLoss(reduction='none')(
                                    F.log_softmax(pred_corners_all / T, dim=1),
                                    F.softmax(teacher_corners_all.detach() / T, dim=1)
                                ).sum(-1)
                    
                    loss_match_local_ddf = weight_targets_local_final * (T**2) * loss_kl_ddf
                    
                    if "is_dn" not in outputs:
                        bs_ddf = outputs["pred_corners"].size(0)
                        batch_scale_ddf = (bs_ddf / outputs["pred_boxes"].size(0)) if outputs["pred_boxes"].size(0) > 0 else 1.0
                        self.num_pos = (mask_ddf.sum() * batch_scale_ddf) ** 0.5 + 1e-9
                        self.num_neg = ((~mask_ddf).sum() * batch_scale_ddf) ** 0.5 + 1e-9
                    
                    loss_match_local1_ddf = loss_match_local_ddf[mask_ddf].mean() if mask_ddf.any() else torch.tensor(0.0, device=self.device)
                    loss_match_local2_ddf = loss_match_local_ddf[~mask_ddf].mean() if (~mask_ddf).any() else torch.tensor(0.0, device=self.device)
                    
                    losses["loss_ddf"] = (loss_match_local1_ddf * self.num_pos + loss_match_local2_ddf * self.num_neg) / \
                                         (self.num_pos + self.num_neg + 1e-9)

        return losses

    def power_transform(self, predictions, power_p=2.0):
        """Apply power transformation to predictions"""
        if power_p <= 0:
            return predictions
        return torch.sign(predictions) * torch.abs(predictions) ** power_p

    def loss_distill(self, outputs, targets, indices, num_boxes, teacher_outputs=None, 
                     epoch=None, step=None, total_epochs=None, meta=None, **kwargs_ignored):
        """Distillation loss between student and teacher outputs"""
        device = outputs["pred_logits"].device
        
        if not self.enable_distillation_processing:
            return {"loss_distill": torch.tensor(0.0, device=device)}
        
        if teacher_outputs is None or not teacher_outputs:
            return {"loss_distill": torch.tensor(0.0, device=device)}
        
        # Calculate distillation weight based on decay type
        distill_weight = 1.0
        if epoch is not None and total_epochs is not None and total_epochs > 0:
            if self.distill_decay_type == 'linear_epoch':
                stop_epoch = self.distill_stop_epoch_ratio * total_epochs
                if epoch >= stop_epoch:
                    distill_weight = 0.0
                else:
                    distill_weight = 1.0 - (epoch / stop_epoch)
            elif self.distill_decay_type == 'exponential':
                decay_rate = -math.log(0.01) / (self.distill_stop_epoch_ratio * total_epochs)
                distill_weight = math.exp(-decay_rate * epoch)
            elif self.distill_decay_type == 'constant':
                distill_weight = 1.0
        
        if distill_weight <= 0.0:
            return {"loss_distill": torch.tensor(0.0, device=device)}
        
        total_distill_loss = torch.tensor(0.0, device=device)
        num_gt_objects_in_batch = sum(len(t["labels"]) for t in targets)
        
        # Classification distillation
        if "pred_logits" in outputs and "pred_logits" in teacher_outputs:
            student_logits = outputs["pred_logits"]
            teacher_logits = teacher_outputs["pred_logits"]
            
            if student_logits.shape == teacher_logits.shape:
                # Apply power transform to teacher predictions
                teacher_probs = F.softmax(teacher_logits.detach(), dim=-1)
                teacher_probs_transformed = self.power_transform(teacher_probs, self.distill_power_p_for_transform)
                teacher_probs_transformed = F.normalize(teacher_probs_transformed, p=1, dim=-1)
                
                cls_distill_loss = F.kl_div(
                    F.log_softmax(student_logits, dim=-1),
                    teacher_probs_transformed,
                    reduction='batchmean'
                )
                total_distill_loss += self.distill_cls_w_factor * cls_distill_loss
        
        # Box regression distillation (L1)
        if "pred_boxes" in outputs and "pred_boxes" in teacher_outputs:
            student_boxes = outputs["pred_boxes"]
            teacher_boxes = teacher_outputs["pred_boxes"]
            
            if student_boxes.shape == teacher_boxes.shape:
                l1_distill_loss = F.l1_loss(student_boxes, teacher_boxes.detach(), reduction='mean')
                total_distill_loss += self.distill_l1_w_factor * l1_distill_loss
        
        # IoU-based distillation with expanded boxes
        if ("pred_boxes" in outputs and "pred_boxes" in teacher_outputs and 
            "pred_logits" in outputs and "pred_logits" in teacher_outputs):
            
            student_boxes = outputs["pred_boxes"]
            teacher_boxes = teacher_outputs["pred_boxes"]
            student_logits = outputs["pred_logits"]
            teacher_logits = teacher_outputs["pred_logits"]
            
            if (student_boxes.shape == teacher_boxes.shape and 
                student_logits.shape == teacher_logits.shape):
                
                # Get confidence scores
                student_conf = F.softmax(student_logits, dim=-1).max(dim=-1)[0]
                teacher_conf = F.softmax(teacher_logits.detach(), dim=-1).max(dim=-1)[0]
                
                # Expand boxes by ratio
                def expand_boxes(boxes, ratio):
                    cx, cy, w, h = boxes.unbind(-1)
                    w_expanded = w * ratio
                    h_expanded = h * ratio
                    return torch.stack([cx, cy, w_expanded, h_expanded], dim=-1)
                
                student_boxes_expanded = expand_boxes(student_boxes, self.distill_iou_expanded_ratio)
                teacher_boxes_expanded = expand_boxes(teacher_boxes, self.distill_iou_expanded_ratio)
                
                # Compute IoU distillation loss
                iou_distill_loss = bbox_distillation_iou_loss(
                    student_boxes_expanded,
                    teacher_boxes_expanded.detach(),
                    student_conf,
                    teacher_conf
                )
                
                total_distill_loss += self.distill_iou_w_factor * iou_distill_loss
        
        # Apply distillation weight and normalize
        final_distill_loss = distill_weight * total_distill_loss
        if num_gt_objects_in_batch > 0:
            final_distill_loss = final_distill_loss / num_gt_objects_in_batch
        
        return {"loss_distill": final_distill_loss}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, main_indices, indices_aux_list):
        results = []
        current_merged_indices_per_batch = copy.deepcopy(main_indices) 

        for aux_indices_per_batch in indices_aux_list:
            if len(current_merged_indices_per_batch) != len(aux_indices_per_batch):
                continue

            temp_merged_next = []
            for idx_main_pair, idx_aux_pair in zip(current_merged_indices_per_batch, aux_indices_per_batch):
                merged_queries = torch.cat([idx_main_pair[0], idx_aux_pair[0]])
                merged_targets = torch.cat([idx_main_pair[1], idx_aux_pair[1]])
                temp_merged_next.append((merged_queries, merged_targets))
            current_merged_indices_per_batch = temp_merged_next
        
        for ind_tuple_per_batch in current_merged_indices_per_batch:
            if ind_tuple_per_batch[0].numel() == 0:
                results.append((torch.empty(0, dtype=torch.long, device=ind_tuple_per_batch[0].device),
                                torch.empty(0, dtype=torch.long, device=ind_tuple_per_batch[0].device)))
                continue

            ind_tensor_per_batch = torch.stack(ind_tuple_per_batch, dim=1)
            
            unique_pairs, counts = torch.unique(ind_tensor_per_batch, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted_pairs = unique_pairs[count_sort_indices]
            
            column_to_row_map = {} 
            for unique_pair_item in unique_sorted_pairs:
                query_idx_item = unique_pair_item[0].item()
                target_idx_item = unique_pair_item[1].item()
                if query_idx_item not in column_to_row_map:
                    column_to_row_map[query_idx_item] = target_idx_item
            
            if column_to_row_map:
                final_rows = torch.tensor(list(column_to_row_map.keys()), device=ind_tensor_per_batch.device, dtype=torch.long)
                final_cols = torch.tensor(list(column_to_row_map.values()), device=ind_tensor_per_batch.device, dtype=torch.long)
            else:
                final_rows = torch.empty(0, dtype=torch.long, device=ind_tensor_per_batch.device)
                final_cols = torch.empty(0, dtype=torch.long, device=ind_tensor_per_batch.device)
            results.append((final_rows, final_cols))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    def get_loss(self, loss_name, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "boxes": self.loss_boxes,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
            "local": self.loss_local,
            "distill": self.loss_distill,
        }
        assert loss_name in loss_map, f"Loss type '{loss_name}' is not supported."
        
        return loss_map[loss_name](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, teacher_outputs=None, **kwargs_runtime_metas):
        device_for_tensors = self.device
        if outputs:
            for key in ['pred_logits', 'pred_boxes', 'pred_corners']:
                if key in outputs and isinstance(outputs[key], torch.Tensor):
                    device_for_tensors = outputs[key].device
                    break
            else:
                first_tensor_val = next((v for v in outputs.values() if isinstance(v, torch.Tensor)), None)
                if first_tensor_val is not None:
                    device_for_tensors = first_tensor_val.device
                elif outputs.get("aux_outputs"):
                    try:
                        first_tensor_val = next((v for aux_layer_out in outputs["aux_outputs"] 
                                                 for v in aux_layer_out.values() if isinstance(v, torch.Tensor)), None)
                        if first_tensor_val is not None: 
                            device_for_tensors = first_tensor_val.device
                    except: 
                        pass
        
        main_student_outputs = {
            k: v for k, v in outputs.items() 
            if "aux" not in k and "dn" not in k and "enc" not in k and "pre" not in k
        }
        
        main_output_has_preds_for_matching = ('pred_logits' in main_student_outputs and 
                                             'pred_boxes' in main_student_outputs and 
                                             main_student_outputs['pred_logits'] is not None and 
                                             main_student_outputs['pred_boxes'] is not None)
        
        if main_output_has_preds_for_matching:
            indices = self.matcher(main_student_outputs, targets)["indices"]
        else: 
            indices = [(torch.empty(0, dtype=torch.long, device=device_for_tensors), 
                        torch.empty(0, dtype=torch.long, device=device_for_tensors)) 
                       for _ in range(len(targets))]
        
        self._clear_cache()

        cached_indices_for_all_dec_layers = [] 
        if main_output_has_preds_for_matching:
            cached_indices_for_all_dec_layers.append(indices)
        
        pre_outputs_data = outputs.get("pre_outputs")
        if (pre_outputs_data and 'pred_logits' in pre_outputs_data and 
            'pred_boxes' in pre_outputs_data):
            cached_indices_for_all_dec_layers.append(self.matcher(pre_outputs_data, targets)["indices"])

        if "aux_outputs" in outputs and isinstance(outputs["aux_outputs"], list):
            for aux_out_item in outputs["aux_outputs"]:
                if (aux_out_item and 'pred_logits' in aux_out_item and 
                    'pred_boxes' in aux_out_item):
                    cached_indices_for_all_dec_layers.append(self.matcher(aux_out_item, targets)["indices"])
        
        num_boxes_go = 0.0
        indices_go = indices
        if cached_indices_for_all_dec_layers:
            if len(cached_indices_for_all_dec_layers) > 1:
                indices_go = self._get_go_indices(cached_indices_for_all_dec_layers[0], cached_indices_for_all_dec_layers[1:])
            else:
                indices_go = cached_indices_for_all_dec_layers[0]
            
            num_boxes_go = sum(len(idx_pair[0]) for idx_pair in indices_go if idx_pair and len(idx_pair) == 2)
            num_boxes_go_tensor = torch.as_tensor([num_boxes_go], dtype=torch.float, device=device_for_tensors)
            if is_dist_available_and_initialized(): 
                torch.distributed.all_reduce(num_boxes_go_tensor)
            num_boxes_go = torch.clamp(num_boxes_go_tensor / get_world_size(), min=1.0).item()
        else:
            num_boxes_go = 0.0
            indices_go = indices

        num_boxes_for_cls_main = sum(len(t["labels"]) for t in targets)
        num_boxes_for_cls_main_tensor = torch.as_tensor([num_boxes_for_cls_main], dtype=torch.float, device=device_for_tensors)
        if is_dist_available_and_initialized(): 
            torch.distributed.all_reduce(num_boxes_for_cls_main_tensor)
        num_boxes_for_cls_main = torch.clamp(num_boxes_for_cls_main_tensor / get_world_size(), min=1.0).item()
        if not main_output_has_preds_for_matching: 
            num_boxes_for_cls_main = 0.0

        calculated_losses = {}
        
        # Loss cho OUTPUT CHÍNH (student)
        if main_output_has_preds_for_matching:
            for loss_name in self.losses_to_compute_for_main_output:
                current_indices_for_loss = indices_go if loss_name in ["boxes", "local"] else indices
                current_num_boxes_to_norm = num_boxes_go if loss_name in ["boxes", "local"] else num_boxes_for_cls_main
                
                has_matches = any(idx_pair[0].numel() > 0 for idx_pair in current_indices_for_loss)
                if not has_matches and current_num_boxes_to_norm == 0 and loss_name != 'distill':
                    pass

                call_kwargs_for_loss = {}
                if loss_name == 'distill':
                    call_kwargs_for_loss.update(kwargs_runtime_metas) 
                    call_kwargs_for_loss['teacher_outputs'] = teacher_outputs
                    loss_dict_component = self.get_loss(loss_name, main_student_outputs, targets, indices, num_boxes_for_cls_main, **call_kwargs_for_loss)
                else: 
                    meta_info = self.get_loss_meta_info(loss_name, main_student_outputs, targets, current_indices_for_loss)
                    call_kwargs_for_loss['meta'] = meta_info
                    relevant_runtime_metas = {k:v for k,v in kwargs_runtime_metas.items() if k not in ['teacher_outputs']}
                    call_kwargs_for_loss.update(relevant_runtime_metas)
                    loss_dict_component = self.get_loss(loss_name, main_student_outputs, targets, current_indices_for_loss, current_num_boxes_to_norm, **call_kwargs_for_loss)
                
                if loss_dict_component: 
                     weighted_loss_dict_component = {k: loss_dict_component[k] * self.weight_dict[k] for k in loss_dict_component if k in self.weight_dict and k in loss_dict_component}
                     calculated_losses.update(weighted_loss_dict_component)
        
        # Auxiliary Losses
        def _compute_losses_for_single_aux_output(aux_output_data_item, output_name_prefix, specific_matcher_indices):
            if "up" not in aux_output_data_item and "up" in outputs: 
                aux_output_data_item["up"] = outputs["up"]
            if "reg_scale" not in aux_output_data_item and "reg_scale" in outputs: 
                aux_output_data_item["reg_scale"] = outputs["reg_scale"]

            for loss_type_original in self.original_losses:
                indices_for_aux_loss = indices_go if loss_type_original in ["boxes", "local"] else specific_matcher_indices
                num_boxes_for_aux_loss_norm = num_boxes_go if loss_type_original in ["boxes", "local"] else num_boxes_for_cls_main

                has_aux_matches = any(idx_pair[0].numel() > 0 for idx_pair in indices_for_aux_loss)
                if not has_aux_matches and num_boxes_for_aux_loss_norm == 0:
                    loss_keys_for_type_aux = [] 
                    if loss_type_original == "boxes": 
                        loss_keys_for_type_aux = ["loss_bbox", "loss_giou"]
                    elif loss_type_original == "focal":
                        loss_keys_for_type_aux = ["loss_focal"]
                    elif loss_type_original == "vfl":
                        loss_keys_for_type_aux = ["loss_vfl"]
                    elif loss_type_original == "local":
                        loss_keys_for_type_aux = ["loss_fgl", "loss_ddf"]
                    
                    for k_loss_aux_type in loss_keys_for_type_aux:
                        if k_loss_aux_type in self.weight_dict:
                             calculated_losses[k_loss_aux_type + output_name_prefix] = torch.tensor(0.0, device=device_for_tensors)
                    continue

                meta_info_for_aux = self.get_loss_meta_info(loss_type_original, aux_output_data_item, targets, indices_for_aux_loss)
                
                call_kwargs_for_aux_loss = {'meta': meta_info_for_aux}
                relevant_runtime_metas_for_aux = {k:v for k,v in kwargs_runtime_metas.items() if k not in ['teacher_outputs']}
                call_kwargs_for_aux_loss.update(relevant_runtime_metas_for_aux)

                loss_dict_component_aux = self.get_loss(loss_type_original, aux_output_data_item, targets, 
                                                        indices_for_aux_loss, num_boxes_for_aux_loss_norm, 
                                                        **call_kwargs_for_aux_loss)
                
                if loss_dict_component_aux:
                    weighted_l_dict_aux = {k: loss_dict_component_aux[k] * self.weight_dict[k] for k in loss_dict_component_aux if k in self.weight_dict and k in loss_dict_component_aux}
                    final_l_dict_aux = {k + output_name_prefix: v for k, v in weighted_l_dict_aux.items()}
                    calculated_losses.update(final_l_dict_aux)

        # Xử lý pre_outputs
        if "pre_outputs" in outputs and outputs["pre_outputs"] and 'pred_logits' in outputs["pre_outputs"]:
            pre_match_indices = cached_indices_for_all_dec_layers[1] if len(cached_indices_for_all_dec_layers) > 1 else self.matcher(outputs["pre_outputs"], targets)["indices"]
            _compute_losses_for_single_aux_output(outputs["pre_outputs"], "_pre", pre_match_indices)

        # Xử lý aux_outputs
        if "aux_outputs" in outputs and isinstance(outputs["aux_outputs"], list):
            aux_start_idx_in_cache = (1 if main_output_has_preds_for_matching else 0) + \
                                     (1 if "pre_outputs" in outputs and outputs["pre_outputs"] and 'pred_logits' in outputs["pre_outputs"] else 0)
            for i, aux_out_item_data in enumerate(outputs["aux_outputs"]):
                if aux_out_item_data and 'pred_logits' in aux_out_item_data:
                    current_aux_match_indices = cached_indices_for_all_dec_layers[aux_start_idx_in_cache + i] if aux_start_idx_in_cache + i < len(cached_indices_for_all_dec_layers) else self.matcher(aux_out_item_data, targets)["indices"]
                    _compute_losses_for_single_aux_output(aux_out_item_data, f"_aux_{i}", current_aux_match_indices)
        
        # Xử lý enc_aux_outputs
        if "enc_aux_outputs" in outputs and outputs["enc_aux_outputs"]:
            assert "enc_meta" in outputs, "'enc_meta' must be in outputs for 'enc_aux_outputs'"
            class_agnostic = outputs["enc_meta"]["class_agnostic"]
            current_enc_targets = targets
            orig_num_classes_backup = self.num_classes
            if class_agnostic:
                self.num_classes = 1
                current_enc_targets = copy.deepcopy(targets)
                for t_enc in current_enc_targets: 
                    t_enc["labels"] = torch.zeros_like(t_enc["labels"])
            
            for i, enc_aux_item_data in enumerate(outputs["enc_aux_outputs"]):
                if enc_aux_item_data and 'pred_logits' in enc_aux_item_data:
                    enc_match_indices = self.matcher(enc_aux_item_data, current_enc_targets)["indices"] 
                    # Need to pass current_enc_targets to the aux function
                    _compute_losses_for_single_aux_output_enc(enc_aux_item_data, f"_enc_{i}", enc_match_indices, current_enc_targets)

            if class_agnostic: 
                self.num_classes = orig_num_classes_backup

        # Helper function for encoder outputs with different targets
        def _compute_losses_for_single_aux_output_enc(aux_output_data_item, output_name_prefix, specific_matcher_indices, enc_targets):
            if "up" not in aux_output_data_item and "up" in outputs: 
                aux_output_data_item["up"] = outputs["up"]
            if "reg_scale" not in aux_output_data_item and "reg_scale" in outputs: 
                aux_output_data_item["reg_scale"] = outputs["reg_scale"]

            for loss_type_original in self.original_losses:
                indices_for_aux_loss = indices_go if loss_type_original in ["boxes", "local"] else specific_matcher_indices
                num_boxes_for_aux_loss_norm = num_boxes_go if loss_type_original in ["boxes", "local"] else num_boxes_for_cls_main

                has_aux_matches = any(idx_pair[0].numel() > 0 for idx_pair in indices_for_aux_loss)
                if not has_aux_matches and num_boxes_for_aux_loss_norm == 0:
                    loss_keys_for_type_aux = [] 
                    if loss_type_original == "boxes": 
                        loss_keys_for_type_aux = ["loss_bbox", "loss_giou"]
                    elif loss_type_original == "focal":
                        loss_keys_for_type_aux = ["loss_focal"]
                    elif loss_type_original == "vfl":
                        loss_keys_for_type_aux = ["loss_vfl"]
                    elif loss_type_original == "local":
                        loss_keys_for_type_aux = ["loss_fgl", "loss_ddf"]
                    
                    for k_loss_aux_type in loss_keys_for_type_aux:
                        if k_loss_aux_type in self.weight_dict:
                             calculated_losses[k_loss_aux_type + output_name_prefix] = torch.tensor(0.0, device=device_for_tensors)
                    continue

                meta_info_for_aux = self.get_loss_meta_info(loss_type_original, aux_output_data_item, enc_targets, indices_for_aux_loss)
                
                call_kwargs_for_aux_loss = {'meta': meta_info_for_aux}
                relevant_runtime_metas_for_aux = {k:v for k,v in kwargs_runtime_metas.items() if k not in ['teacher_outputs']}
                call_kwargs_for_aux_loss.update(relevant_runtime_metas_for_aux)

                loss_dict_component_aux = self.get_loss(loss_type_original, aux_output_data_item, enc_targets, 
                                                        indices_for_aux_loss, num_boxes_for_aux_loss_norm, 
                                                        **call_kwargs_for_aux_loss)
                
                if loss_dict_component_aux:
                    weighted_l_dict_aux = {k: loss_dict_component_aux[k] * self.weight_dict[k] for k in loss_dict_component_aux if k in self.weight_dict and k in loss_dict_component_aux}
                    final_l_dict_aux = {k + output_name_prefix: v for k, v in weighted_l_dict_aux.items()}
                    calculated_losses.update(final_l_dict_aux)

        # Xử lý dn_outputs
        if "dn_outputs" in outputs and outputs["dn_outputs"]:
            assert "dn_meta" in outputs, "'dn_meta' must be in outputs for 'dn_outputs'"
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_boxes_norm = num_boxes_for_cls_main * outputs["dn_meta"]["dn_num_group"]
            dn_num_boxes_norm = dn_num_boxes_norm if dn_num_boxes_norm > 0 else 1.0

            def _compute_losses_for_dn_set(dn_outputs_list_data, dn_prefix_str_base):
                 for i_dn, dn_out_data_item in enumerate(dn_outputs_list_data):
                    if not (dn_out_data_item and 'pred_logits' in dn_out_data_item): 
                        continue
                    dn_out_data_item["is_dn"] = True 
                    if "up" not in dn_out_data_item and "up" in outputs : 
                        dn_out_data_item["up"] = outputs["up"]
                    if "reg_scale" not in dn_out_data_item and "reg_scale" in outputs : 
                        dn_out_data_item["reg_scale"] = outputs["reg_scale"]

                    for loss_name_dn_orig in self.original_losses:
                        has_dn_matches = any(idx_pair[0].numel() > 0 for idx_pair in indices_dn)
                        if not has_dn_matches and dn_num_boxes_norm == 0:
                            loss_keys_dn = []
                            if loss_name_dn_orig == "boxes": 
                                loss_keys_dn = ["loss_bbox", "loss_giou"]
                            elif loss_name_dn_orig == "focal":
                                loss_keys_dn = ["loss_focal"]
                            elif loss_name_dn_orig == "vfl":
                                loss_keys_dn = ["loss_vfl"]
                            elif loss_name_dn_orig == "local":
                                loss_keys_dn = ["loss_fgl", "loss_ddf"]
                            
                            for k_dn in loss_keys_dn:
                                if k_dn in self.weight_dict:
                                    calculated_losses[k_dn + (f"_{dn_prefix_str_base}_{i_dn}" if len(dn_outputs_list_data)>1 else f"_{dn_prefix_str_base}")] = torch.tensor(0.0, device=device_for_tensors)
                            continue
                        
                        meta_dn = self.get_loss_meta_info(loss_name_dn_orig, dn_out_data_item, targets, indices_dn)
                        dn_call_kwargs = {'meta': meta_dn}
                        dn_call_kwargs.update({k:v for k,v in kwargs_runtime_metas.items() if k not in ['teacher_outputs']})

                        l_dict_dn_comp = self.get_loss(loss_name_dn_orig, dn_out_data_item, targets, 
                                                       indices_dn, dn_num_boxes_norm, **dn_call_kwargs)
                        if l_dict_dn_comp:
                            weighted_l_dict_dn = {k: l_dict_dn_comp[k] * self.weight_dict[k] for k in l_dict_dn_comp if k in self.weight_dict and k in l_dict_dn_comp}
                            final_l_dict_dn = {k + (f"_{dn_prefix_str_base}_{i_dn}" if len(dn_outputs_list_data)>1 else f"_{dn_prefix_str_base}") : v for k, v in weighted_l_dict_dn.items()}
                            calculated_losses.update(final_l_dict_dn)
            
            _compute_losses_for_dn_set(outputs["dn_outputs"], "dn")
            if "dn_pre_outputs" in outputs and outputs["dn_pre_outputs"]: 
                _compute_losses_for_dn_set([outputs["dn_pre_outputs"]], "dn_pre")

        calculated_losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in calculated_losses.items()}
        return calculated_losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            return {}

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0)

        if src_boxes.numel() == 0 or target_boxes.numel() == 0:
            return {}

        if self.boxes_weight_format == "iou":
            iou, _ = box_iou(
                box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)
            )
            iou = torch.diag(iou)
        elif self.boxes_weight_format == "giou":
            iou = torch.diag(
                generalized_box_iou(
                    box_cxcywh_to_xyxy(src_boxes.detach()), box_cxcywh_to_xyxy(target_boxes)
                )
            )
        else:
            raise AttributeError(f"Unknown boxes_weight_format: {self.boxes_weight_format}")

        if loss in ("boxes",):
            meta = {"boxes_weight": iou}
        elif loss in ("vfl",):
            meta = {"values": iou}
        else:
            meta = {}

        return meta

    @staticmethod
    def get_cdn_matched_indices(dn_meta, targets):
        """get_cdn_matched_indices"""
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t["labels"]) for t in targets]
        device = targets[0]["labels"].device

        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append(
                    (
                        torch.zeros(0, dtype=torch.int64, device=device),
                        torch.zeros(0, dtype=torch.int64, device=device),
                    )
                )

        return dn_match_indices

    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea) ** 2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss)

    def unimodal_distribution_focal_loss(
        self, pred, label, weight_right, weight_left, weight=None, reduction="sum", avg_factor=None
    ):
        dis_left = label.long()
        dis_right = dis_left + 1

        loss = F.cross_entropy(pred, dis_left, reduction="none") * weight_left.reshape(
            -1
        ) + F.cross_entropy(pred, dis_right, reduction="none") * weight_right.reshape(-1)

        if weight is not None:
            weight = weight.float()
            loss = loss * weight

        if avg_factor is not None:
            loss = loss.sum() / avg_factor
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

    def get_gradual_steps(self, outputs):
        num_layers = len(outputs["aux_outputs"]) + 1 if "aux_outputs" in outputs else 1
        step = 0.5 / (num_layers - 1)
        opt_list = [0.5 + step * i for i in range(num_layers)] if num_layers > 1 else [1]
        return opt_list