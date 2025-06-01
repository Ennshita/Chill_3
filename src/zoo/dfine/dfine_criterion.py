import copy
import math # THÊM IMPORT NÀY
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from datetime import datetime # THÊM IMPORT NÀY (để logging nếu cần)

from ...core import register
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .dfine_utils import bbox2distance
# GIẢ SỬ BẠN ĐÃ TẠO FILE NÀY VÀ HÀM BÊN TRONG
from .iou_utils import bbox_distillation_iou_loss 


@register()
class DFINECriterion(nn.Module):
    """This class computes the loss for D-FINE, now with distillation capabilities."""

    __share__ = [
        "num_classes",
    ]
    __inject__ = [
        "matcher",
    ]

    def __init__(
        self,
        matcher,
        weight_dict,
        losses, # Danh sách loss gốc từ config student
        alpha=0.2,
        gamma=2.0,
        num_classes=80,
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False,
        # --- THAM SỐ MỚI CHO DISTILLATION (sẽ được inject từ YAML) ---
        use_distillation: bool = False, 
        distill_decay_type: str = 'linear_epoch',
        distill_stop_epoch_ratio: float = 1.0,
        distill_cls_loss_factor: float = 1.0,
        distill_l1_loss_factor: float = 5.0,
        distill_iou_loss_factor: float = 2.0,
        distill_iou_ratio: float = 1.25,       # Ratio cho expanded/inner IoU
        distill_power_transform_p: float = 2.0 # Power cho power_transform
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
        self.own_targets, self.own_targets_dn = None, None # Giữ lại nếu D-FINE gốc dùng
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- Lưu các tham số distillation ---
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

    # --- CÁC HÀM LOSS GỐC (loss_labels_focal, loss_labels_vfl, loss_boxes, loss_local) GIỮ NGUYÊN ---
    # Thêm **kwargs_ignored vào cuối danh sách tham số của các hàm loss gốc để chúng có thể nhận
    # các tham số thừa (như teacher_outputs, epoch, etc.) mà không báo lỗi khi được gọi từ get_loss.
    def loss_labels_focal(self, outputs, targets, indices, num_boxes, meta=None, **kwargs_extra_just_in_case):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]
        loss = torchvision.ops.sigmoid_focal_loss(
            src_logits, target, self.alpha, self.gamma, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes

        return {"loss_focal": loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None, meta=None):
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        if values is None:
            src_boxes = outputs["pred_boxes"][idx]
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
            ious, _ = box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
            ious = torch.diag(ious).detach()
        else:
            ious = values

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = F.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(
            src_logits, target_score, weight=weight, reduction="none"
        )
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None, meta=None, **kwargs_extra_just_in_case):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        loss_giou = loss_giou if boxes_weight is None else loss_giou * boxes_weight
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses

    def loss_local(self, outputs, targets, indices, num_boxes, T=5, meta=None, **kwargs_extra_just_in_case):
        """Compute Fine-Grained Localization (FGL) Loss
        and Decoupled Distillation Focal (DDF) Loss."""

        losses = {}
        if "pred_corners" in outputs:
            idx = self._get_src_permutation_idx(indices)
            target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

            pred_corners = outputs["pred_corners"][idx].reshape(-1, (self.reg_max + 1))
            ref_points = outputs["ref_points"][idx].detach()
            with torch.no_grad():
                if self.fgl_targets_dn is None and "is_dn" in outputs:
                    self.fgl_targets_dn = bbox2distance(
                        ref_points,
                        box_cxcywh_to_xyxy(target_boxes),
                        self.reg_max,
                        outputs["reg_scale"],
                        outputs["up"],
                    )
                if self.fgl_targets is None and "is_dn" not in outputs:
                    self.fgl_targets = bbox2distance(
                        ref_points,
                        box_cxcywh_to_xyxy(target_boxes),
                        self.reg_max,
                        outputs["reg_scale"],
                        outputs["up"],
                    )

            target_corners, weight_right, weight_left = (
                self.fgl_targets_dn if "is_dn" in outputs else self.fgl_targets
            )

            ious = torch.diag(
                box_iou(
                    box_cxcywh_to_xyxy(outputs["pred_boxes"][idx]), box_cxcywh_to_xyxy(target_boxes)
                )[0]
            )
            weight_targets = ious.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()

            losses["loss_fgl"] = self.unimodal_distribution_focal_loss(
                pred_corners,
                target_corners,
                weight_right,
                weight_left,
                weight_targets,
                avg_factor=num_boxes,
            )

            if "teacher_corners" in outputs:
                pred_corners = outputs["pred_corners"].reshape(-1, (self.reg_max + 1))
                target_corners = outputs["teacher_corners"].reshape(-1, (self.reg_max + 1))
                if torch.equal(pred_corners, target_corners):
                    losses["loss_ddf"] = pred_corners.sum() * 0
                else:
                    weight_targets_local = outputs["teacher_logits"].sigmoid().max(dim=-1)[0]

                    mask = torch.zeros_like(weight_targets_local, dtype=torch.bool)
                    mask[idx] = True
                    mask = mask.unsqueeze(-1).repeat(1, 1, 4).reshape(-1)

                    weight_targets_local[idx] = ious.reshape_as(weight_targets_local[idx]).to(
                        weight_targets_local.dtype
                    )
                    weight_targets_local = (
                        weight_targets_local.unsqueeze(-1).repeat(1, 1, 4).reshape(-1).detach()
                    )

                    loss_match_local = (
                        weight_targets_local
                        * (T**2)
                        * (
                            nn.KLDivLoss(reduction="none")(
                                F.log_softmax(pred_corners / T, dim=1),
                                F.softmax(target_corners.detach() / T, dim=1),
                            )
                        ).sum(-1)
                    )
                    if "is_dn" not in outputs:
                        batch_scale = (
                            8 / outputs["pred_boxes"].shape[0]
                        )  # Avoid the influence of batch size per GPU
                        self.num_pos, self.num_neg = (
                            (mask.sum() * batch_scale) ** 0.5,
                            ((~mask).sum() * batch_scale) ** 0.5,
                        )
                    loss_match_local1 = loss_match_local[mask].mean() if mask.any() else 0
                    loss_match_local2 = loss_match_local[~mask].mean() if (~mask).any() else 0
                    losses["loss_ddf"] = (
                        loss_match_local1 * self.num_pos + loss_match_local2 * self.num_neg
                    ) / (self.num_pos + self.num_neg)

        return losses


    # --- HÀM MỚI HOẶC ĐÃ SỬA ---
    def power_transform(self, array: torch.Tensor, power: float) -> torch.Tensor:
        return torch.where(array < 0.5, torch.pow(array, power), torch.pow(array, 1.0 / power))

    def loss_distill(self, student_outputs, targets_for_norm, student_indices_not_used, num_student_boxes_for_norm,
                     teacher_outputs=None, epoch=0, total_epochs=1, current_iter=0, total_iters_epoch=1,
                     **kwargs_meta_info_not_used):
        
        if not self.enable_distillation_processing or teacher_outputs is None:
            return {"loss_distill": torch.tensor(0.0, device=self.device)}

        effective_total_epochs = total_epochs if total_epochs > 0 else 1
        # Distill cho đến hết epoch được chỉ định (không phải <)
        stop_kd_epoch_num = int(effective_total_epochs * self.distill_stop_epoch_ratio)
        if epoch >= stop_kd_epoch_num and stop_kd_epoch_num < effective_total_epochs : # Chỉ dừng nếu stop_epoch < total_epochs
            return {"loss_distill": torch.tensor(0.0, device=self.device)}

        distill_decay = 1.0
        if self.distill_decay_type == 'linear_epoch':
            current_global_iter = current_iter + total_iters_epoch * epoch
            # Decay trong khoảng thời gian distillation được áp dụng
            total_distill_iterations = stop_kd_epoch_num * total_iters_epoch 
            if total_distill_iterations > 0:
                progress_ratio = min(max(current_global_iter / total_distill_iterations, 0.0), 1.0)
                distill_decay = 1.0 - (1.0 - 0.01) * progress_ratio 
        elif self.distill_decay_type == 'cosine_epoch':
            current_global_iter = current_iter + total_iters_epoch * epoch
            total_distill_iterations = stop_kd_epoch_num * total_iters_epoch
            if total_distill_iterations > 0:
                progress_ratio = min(max(current_global_iter / total_distill_iterations, 0.0), 1.0)
                eta_min, base_ratio = 0.01, 1.0
                distill_decay = eta_min + (base_ratio - eta_min) * (1 + math.cos(math.pi * progress_ratio)) / 2
        
        distill_decay = torch.tensor(distill_decay, device=self.device, dtype=torch.float32)

        s_logits = student_outputs.get('pred_logits')
        s_bboxes = student_outputs.get('pred_boxes') # cxcywh
        
        if s_logits is None or s_bboxes is None:
            print("WARNING (loss_distill): Student outputs missing 'pred_logits' or 'pred_boxes'. Skipping.")
            return {"loss_distill": torch.tensor(0.0, device=self.device)}

        t_logits = teacher_outputs.get('pred_logits') # Đã detach
        t_bboxes = teacher_outputs.get('pred_boxes')   # cxcywh, đã detach

        if t_logits is None or t_bboxes is None:
            print("WARNING (loss_distill): Teacher outputs missing 'pred_logits' or 'pred_boxes'. Skipping.")
            return {"loss_distill": torch.tensor(0.0, device=self.device)}

        # --- Xử lý DeNoising (DN) Queries ---
        s_dn_meta = student_outputs.get('dn_meta')
        t_dn_meta = teacher_outputs.get('dn_meta') # Teacher cũng có thể có dn_meta

        s_logits_obj, s_bboxes_obj = s_logits, s_bboxes
        t_logits_obj, t_bboxes_obj = t_logits, t_bboxes

        # Ưu tiên tách dựa trên num_dn_queries nếu có
        if s_dn_meta and 'num_dn_queries' in s_dn_meta:
            num_dn_s = s_dn_meta['num_dn_queries']
            s_logits_obj = s_logits[:, num_dn_s:]
            s_bboxes_obj = s_bboxes[:, num_dn_s:]
        elif s_dn_meta and 'dn_num_split' in s_dn_meta and isinstance(s_dn_meta['dn_num_split'], (list, tuple)) and len(s_dn_meta['dn_num_split']) == 2:
            s_logits_list = torch.split(s_logits, s_dn_meta['dn_num_split'], dim=1)
            s_bboxes_list = torch.split(s_bboxes, s_dn_meta['dn_num_split'], dim=1)
            s_logits_obj, s_bboxes_obj = s_logits_list[1], s_bboxes_list[1]
        
        if t_dn_meta and 'num_dn_queries' in t_dn_meta:
            num_dn_t = t_dn_meta['num_dn_queries']
            t_logits_obj = t_logits[:, num_dn_t:]
            t_bboxes_obj = t_bboxes[:, num_dn_t:]
        elif t_dn_meta and 'dn_num_split' in t_dn_meta and isinstance(t_dn_meta['dn_num_split'], (list, tuple)) and len(t_dn_meta['dn_num_split']) == 2:
            t_logits_list = torch.split(t_logits, t_dn_meta['dn_num_split'], dim=1)
            t_bboxes_list = torch.split(t_bboxes, t_dn_meta['dn_num_split'], dim=1)
            t_logits_obj, t_bboxes_obj = t_logits_list[1], t_bboxes_list[1]

        # Đồng bộ số lượng queries (lấy min)
        min_queries = min(s_logits_obj.size(1), t_logits_obj.size(1))
        if min_queries == 0:
             return {"loss_distill": torch.tensor(0.0, device=self.device)}
        s_logits_obj = s_logits_obj[:, :min_queries]
        s_bboxes_obj = s_bboxes_obj[:, :min_queries]
        t_logits_obj = t_logits_obj[:, :min_queries]
        t_bboxes_obj = t_bboxes_obj[:, :min_queries]
        
        # --- Tính toán các thành phần loss ---
        t_obj_scale = t_logits_obj.sigmoid().max(dim=-1, keepdim=True)[0] 

        loss_cls_distill_elementwise = F.binary_cross_entropy_with_logits(s_logits_obj, t_logits_obj.sigmoid(), reduction='none')
        lcls = loss_cls_distill_elementwise.sum() # Sum over all elements (batch, queries, classes)

        loss_l1_distill_raw = F.l1_loss(s_bboxes_obj, t_bboxes_obj, reduction='none')
        lbox_l1 = (loss_l1_distill_raw * t_obj_scale.expand_as(s_bboxes_obj)).sum()

        # Sử dụng hàm bbox_distillation_iou_loss bạn đã tạo/import
        iou_loss_terms = bbox_distillation_iou_loss(s_bboxes_obj, t_bboxes_obj, 
                                                    ratio=self.distill_iou_expanded_ratio, 
                                                    use_siou_penalty=True, eps=1e-7) # loss = 1 - metric_iou
        # t_obj_scale.squeeze(-1) có shape [bs, num_obj_queries]
        # iou_loss_terms cũng có shape [bs, num_obj_queries]
        lbox_iou = (iou_loss_terms * self.power_transform(t_obj_scale.squeeze(-1), power=self.distill_power_p_for_transform)).sum()
        
        # --- Normalization ---
        # num_gt_objects_in_batch được tính từ `targets_for_norm` (là `targets` gốc của batch)
        num_gt_objects_in_batch = sum(len(t_dict.get("labels", [])) for t_dict in targets_for_norm)
        # Nếu batch không có GT nào (ví dụ ảnh toàn background), dùng batch_size để tránh chia cho 0
        # Hoặc dùng số lượng queries để normalize.
        # RTDETRLogicLoss dùng batch['bboxes'].size(0) - số lượng ảnh có GT trong batch.
        # Ở đây, num_gt_objects_in_batch là tổng số GT objects trên toàn batch.
        # Sử dụng num_predictions_to_normalize = bs * num_obj_queries có vẻ ổn định hơn
        
        bs_actual = s_logits_obj.size(0)
        num_obj_queries_actual = s_logits_obj.size(1)
        
        # Normalizer cho L1 và IoU có thể là tổng số scalar values trong s_bboxes_obj (bs * nq * 4) hoặc chỉ (bs * nq)
        # Normalizer cho Cls có thể là tổng số scalar values trong s_logits_obj (bs * nq * nc) hoặc (bs * nq)
        # RTDETRLogicLoss chuẩn hóa các loss đã sum bằng num_gt_objects_in_batch.
        # Hãy thử theo cách đó trước.
        normalizer_gt_count = float(num_gt_objects_in_batch if num_gt_objects_in_batch > 0 else bs_actual) # Tránh chia cho 0

        lcls_norm = lcls / (normalizer_gt_count * self.num_classes + 1e-9) # Chia cho số GT * số class
        lbox_l1_norm = lbox_l1 / (normalizer_gt_count * 4 + 1e-9)    # Chia cho số GT * 4 tọa độ
        lbox_iou_norm = lbox_iou / (normalizer_gt_count + 1e-9)      # Chia cho số GT

        final_cls_loss = lcls_norm * self.distill_cls_w_factor
        final_l1_loss = lbox_l1_norm * self.distill_l1_w_factor
        final_iou_loss = lbox_iou_norm * self.distill_iou_w_factor
        
        total_distill_loss_components = final_cls_loss + final_l1_loss + final_iou_loss
        final_distill_loss = total_distill_loss_components * distill_decay

        return {"loss_distill": final_distill_loss}
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_go_indices(self, indices, indices_aux_list):
        """Get a matching union set across all decoder layers."""
        results = []
        for indices_aux in indices_aux_list:
            indices = [
                (torch.cat([idx1[0], idx2[0]]), torch.cat([idx1[1], idx2[1]]))
                for idx1, idx2 in zip(indices.copy(), indices_aux.copy())
            ]

        for ind in [torch.cat([idx[0][:, None], idx[1][:, None]], 1) for idx in indices]:
            unique, counts = torch.unique(ind, return_counts=True, dim=0)
            count_sort_indices = torch.argsort(counts, descending=True)
            unique_sorted = unique[count_sort_indices]
            column_to_row = {}
            for idx in unique_sorted:
                row_idx, col_idx = idx[0].item(), idx[1].item()
                if row_idx not in column_to_row:
                    column_to_row[row_idx] = col_idx
            final_rows = torch.tensor(list(column_to_row.keys()), device=ind.device)
            final_cols = torch.tensor(list(column_to_row.values()), device=ind.device)
            results.append((final_rows.long(), final_cols.long()))
        return results

    def _clear_cache(self):
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.num_pos, self.num_neg = None, None

    # --- HÀM FORWARD ĐÃ ĐƯỢC SỬA ĐỔI ---
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
        # Xác định device từ một tensor bất kỳ để tạo tensor mới nếu cần
        # Cố gắng lấy từ output chính trước
        main_output_keys = [k for k in outputs if "aux" not in k and "dn" not in k and "enc" not in k and "pre" not in k and isinstance(outputs[k], torch.Tensor)]
        if main_output_keys:
            device_for_tensors = outputs[main_output_keys[0]].device
        elif outputs: # Nếu không có output chính rõ ràng, lấy từ bất kỳ tensor nào
            first_tensor_val = next((v for v in outputs.values() if isinstance(v, torch.Tensor)), None)
            if first_tensor_val is None and outputs.get("aux_outputs"): # Thử aux_outputs
                first_tensor_val = next((v for v_list in outputs["aux_outputs"] for v in v_list.values() if isinstance(v, torch.Tensor)), None)

            if first_tensor_val is not None:
                device_for_tensors = first_tensor_val.device
            else: # Fallback nếu không tìm thấy tensor nào
                device_for_tensors = self.device 
        else: # outputs rỗng
            return {} 


        outputs_without_aux_or_dn = {k: v for k, v in outputs.items() if "aux" not in k and "dn" not in k and "enc" not in k and "pre" not in k}
        
        main_output_has_preds = 'pred_logits' in outputs_without_aux_or_dn and 'pred_boxes' in outputs_without_aux_or_dn
        
        if main_output_has_preds:
            indices = self.matcher(outputs_without_aux_or_dn, targets)["indices"]
        else: 
            indices = [(torch.empty(0, dtype=torch.long, device=device_for_tensors), 
                        torch.empty(0, dtype=torch.long, device=device_for_tensors)) 
                       for _ in range(len(targets))]
        
        self._clear_cache()

        # --- Xử lý indices_go và num_boxes ---
        # Logic này cần phải rất cẩn thận. Dưới đây là một cách tiếp cận.
        # cached_indices sẽ lưu matching cho từng lớp decoder (main, pre, aux)
        cached_indices = []
        if main_output_has_preds:
            cached_indices.append(indices) # Main output's indices
        if "pre_outputs" in outputs and outputs["pre_outputs"]:
            cached_indices.append(self.matcher(outputs["pre_outputs"], targets)["indices"])
        if "aux_outputs" in outputs:
            for aux_out in outputs["aux_outputs"]:
                cached_indices.append(self.matcher(aux_out, targets)["indices"])
        
        if cached_indices: # Nếu có ít nhất một bộ indices
            indices_go = self._get_go_indices(cached_indices[0], cached_indices[1:]) if len(cached_indices) > 1 else cached_indices[0]
            num_boxes_go = sum(len(x[0]) for x in indices_go)
            num_boxes_go_tensor = torch.as_tensor([num_boxes_go], dtype=torch.float, device=device_for_tensors)
            if is_dist_available_and_initialized(): torch.distributed.all_reduce(num_boxes_go_tensor)
            num_boxes_go = torch.clamp(num_boxes_go_tensor / get_world_size(), min=1).item()
        else: # Không có output nào để matching
            indices_go = indices # Sẽ là rỗng
            num_boxes_go = 0.0

        num_boxes_for_cls = sum(len(t["labels"]) for t in targets)
        num_boxes_for_cls_tensor = torch.as_tensor([num_boxes_for_cls], dtype=torch.float, device=device_for_tensors)
        if is_dist_available_and_initialized(): torch.distributed.all_reduce(num_boxes_for_cls_tensor)
        num_boxes_for_cls = torch.clamp(num_boxes_for_cls_tensor / get_world_size(), min=1).item()
        if not main_output_has_preds: num_boxes_for_cls = 0.0


        calculated_losses = {}
        
        # --- Loss cho OUTPUT CHÍNH (student) ---
        if main_output_has_preds:
            for loss_name in self.losses_to_compute_for_main_output:
                current_indices = indices_go if loss_name in ["boxes", "local"] else indices
                current_num_boxes = num_boxes_go if loss_name in ["boxes", "local"] else num_boxes_for_cls

                if current_num_boxes == 0 and current_indices[0][0].numel() == 0 and loss_name != 'distill':
                    calculated_losses[loss_name] = torch.tensor(0.0, device=device_for_tensors)
                    continue

                call_kwargs = {}
                if loss_name == 'distill':
                    call_kwargs.update(kwargs_runtime_metas) 
                    call_kwargs['teacher_outputs'] = teacher_outputs
                    # `targets` được truyền cho `loss_distill` để lấy `num_gt_objects_in_batch`
                    # `indices` và `num_boxes` ở đây là của student, `loss_distill` có thể không dùng chúng
                    # hoặc dùng `num_boxes` (là `num_boxes_for_cls`) cho normalization.
                    loss_dict_component = self.get_loss(loss_name, outputs_without_aux_or_dn, targets, indices, num_boxes_for_cls, **call_kwargs)
                else: 
                    meta_info = self.get_loss_meta_info(loss_name, outputs, targets, current_indices)
                    call_kwargs_for_get_loss = {'meta': meta_info} # Chỉ truyền meta
                    # Truyền runtime metas nếu loss gốc CẦN (ví dụ T cho loss_local)
                    if loss_name == 'local' and 'T' in kwargs_runtime_metas: # Ví dụ
                        call_kwargs_for_get_loss['T'] = kwargs_runtime_metas['T']
                    elif loss_name == 'local' and 'T' not in kwargs_runtime_metas: # Nếu T không có trong runtime_metas thì dùng giá trị mặc định của loss_local
                        pass


                    # Xóa các runtime metas không cần thiết cho loss gốc để tránh lỗi
                    filtered_runtime_metas = {k:v for k,v in kwargs_runtime_metas.items() 
                                              if k not in ['teacher_outputs', 'epoch', 'step', 'global_step', 'epoch_step', 'total_epochs'] or (loss_name=='local' and k=='T')}
                    call_kwargs_for_get_loss.update(filtered_runtime_metas)


                    loss_dict_component = self.get_loss(loss_name, outputs, targets, current_indices, current_num_boxes, **call_kwargs_for_get_loss)
                
                if loss_dict_component: # Kiểm tra None
                     weighted_loss_dict_component = {k: loss_dict_component[k] * self.weight_dict[k] for k in loss_dict_component if k in self.weight_dict}
                     calculated_losses.update(weighted_loss_dict_component)

        # --- Auxiliary Losses (CHỈ TÍNH LOSS GỐC) ---
        # Helper để tránh lặp code
        def _compute_single_aux_loss_set(aux_output_data, prefix_str, aux_idx_in_cache_list_or_none):
            # aux_output_data cần có 'up', 'reg_scale'
            if "up" not in aux_output_data: aux_output_data["up"] = outputs.get("up")
            if "reg_scale" not in aux_output_data: aux_output_data["reg_scale"] = outputs.get("reg_scale")

            # Matching cho lớp aux này
            if aux_idx_in_cache_list_or_none is not None and aux_idx_in_cache_list_or_none < len(cached_indices): # Sử dụng cache nếu có
                current_aux_match_indices = cached_indices[aux_idx_in_cache_list_or_none]
            else: # Matching lại
                current_aux_match_indices = self.matcher(aux_output_data, targets)["indices"]


            for loss_name_aux in self.original_losses:
                indices_in_aux = indices_go if loss_name_aux in ["boxes", "local"] else current_aux_match_indices
                num_boxes_in_aux = num_boxes_go if loss_name_aux in ["boxes", "local"] else num_boxes_for_cls
                
                if num_boxes_in_aux == 0 and indices_in_aux[0][0].numel() == 0:
                    calculated_losses[loss_name_aux + prefix_str] = torch.tensor(0.0, device=device_for_tensors)
                    continue

                meta_aux = self.get_loss_meta_info(loss_name_aux, aux_output_data, targets, indices_in_aux)
                aux_call_kwargs = {'meta': meta_aux}
                aux_call_kwargs.update({k:v for k,v in kwargs_runtime_metas.items() if k not in ['teacher_outputs']})

                l_dict_aux = self.get_loss(loss_name_aux, aux_output_data, targets, 
                                           indices_in_aux, num_boxes_in_aux, **aux_call_kwargs)
                
                if l_dict_aux:
                    l_dict_aux_weighted = {k: l_dict_aux[k] * self.weight_dict[k] for k in l_dict_aux if k in self.weight_dict}
                    l_dict_aux_final = {k + prefix_str: v for k, v in l_dict_aux_weighted.items()}
                    calculated_losses.update(l_dict_aux_final)

        # Xử lý pre_outputs
        if "pre_outputs" in outputs and outputs["pre_outputs"]:
            # Giả sử pre_outputs là phần tử thứ 2 trong cached_indices nếu main output tồn tại
            pre_idx_in_cache = (1 if main_output_has_preds else 0) if cached_indices else None
            _compute_single_aux_loss_set(outputs["pre_outputs"], "_pre", pre_idx_in_cache)

        # Xử lý aux_outputs (decoder)
        if "aux_outputs" in outputs:
            offset_for_aux = (1 if main_output_has_preds else 0) + (1 if "pre_outputs" in outputs and outputs["pre_outputs"] else 0)
            for i, aux_out_item in enumerate(outputs["aux_outputs"]):
                aux_idx_in_cache = offset_for_aux + i if cached_indices else None
                _compute_single_aux_loss_set(aux_out_item, f"_aux_{i}", aux_idx_in_cache)

        # Xử lý enc_aux_outputs
        if "enc_aux_outputs" in outputs:
            assert "enc_meta" in outputs, "'enc_meta' must be in outputs for 'enc_aux_outputs'"
            class_agnostic = outputs["enc_meta"]["class_agnostic"]
            current_enc_targets = targets
            orig_num_classes_backup = self.num_classes
            if class_agnostic:
                self.num_classes = 1
                current_enc_targets = copy.deepcopy(targets)
                for t in current_enc_targets: t["labels"] = torch.zeros_like(t["labels"])
            
            # enc_aux_outputs có thể cần matching riêng và không tham gia vào indices_go
            enc_cached_indices = [self.matcher(enc_out, current_enc_targets)["indices"] for enc_out in outputs["enc_aux_outputs"]]
            for i, enc_aux_out_item in enumerate(outputs["enc_aux_outputs"]):
                 _compute_single_aux_loss_set(enc_aux_out_item, f"_enc_{i}", i, base_cached_indices=enc_cached_indices) # Truyền enc_targets

            if class_agnostic: self.num_classes = orig_num_classes_backup
        
        # Xử lý dn_outputs
        if "dn_outputs" in outputs:
            assert "dn_meta" in outputs, "'dn_meta' must be in outputs for 'dn_outputs'"
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            # dn_num_boxes dùng num_boxes_for_cls thay vì num_boxes_go vì DN là về classification/regression chính xác
            dn_num_boxes_norm = num_boxes_for_cls * outputs["dn_meta"]["dn_num_group"]
            dn_num_boxes_norm = dn_num_boxes_norm if dn_num_boxes_norm > 0 else 1.0

            def _compute_single_dn_loss_set(dn_outputs_list, dn_name_prefix_base):
                for i, aux_out_data in enumerate(dn_outputs_list):
                    aux_out_data["is_dn"] = True 
                    if "up" not in aux_out_data: aux_out_data["up"] = outputs.get("up")
                    if "reg_scale" not in aux_out_data: aux_out_data["reg_scale"] = outputs.get("reg_scale")

                    for loss_name_dn in self.original_losses:
                        # DN losses luôn dùng indices_dn
                        if dn_num_boxes_norm == 0 and indices_dn[0][0].numel() == 0 :
                             calculated_losses[loss_name_dn + f"_{dn_name_prefix_base}_{i}"] = torch.tensor(0.0, device=device_for_tensors)
                             continue
                        
                        meta_dn = self.get_loss_meta_info(loss_name_dn, aux_out_data, targets, indices_dn)
                        dn_call_kwargs = {'meta': meta_dn}
                        dn_call_kwargs.update({k:v for k,v in kwargs_runtime_metas.items() if k not in ['teacher_outputs']})

                        l_dict_dn = self.get_loss(loss_name_dn, aux_out_data, targets, 
                                                  indices_dn, dn_num_boxes_norm, **dn_call_kwargs)
                        if l_dict_dn:
                            l_dict_dn_weighted = {k: l_dict_dn[k] * self.weight_dict[k] for k in l_dict_dn if k in self.weight_dict}
                            l_dict_dn_final = {k + f"_{dn_name_prefix_base}_{i}" if len(dn_outputs_list)>1 else k + f"_{dn_name_prefix_base}" : v for k, v in l_dict_dn_weighted.items()}
                            calculated_losses.update(l_dict_dn_final)
            
            _compute_single_dn_loss_set(outputs["dn_outputs"], "dn")
            if "dn_pre_outputs" in outputs: # dn_pre_outputs thường là 1 dict, không phải list
                _compute_single_dn_loss_set([outputs["dn_pre_outputs"]], "dn_pre")


        calculated_losses = {k: torch.nan_to_num(v, nan=0.0) for k, v in calculated_losses.items()}
        return calculated_losses

    def get_loss_meta_info(self, loss, outputs, targets, indices):
        if self.boxes_weight_format is None:
            return {}

        src_boxes = outputs["pred_boxes"][self._get_src_permutation_idx(indices)]
        target_boxes = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)], dim=0)

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
            raise AttributeError()

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
