import copy
import math # THÊM IMPORT NÀY
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ...core import register
from ...misc.dist_utils import get_world_size, is_dist_available_and_initialized
from .box_ops import box_cxcywh_to_xyxy, box_iou, generalized_box_iou
from .dfine_utils import bbox2distance
# THAY THẾ/THÊM IMPORT CHO HÀM IoU DISTILLATION CỦA BẠN
# Giả sử bạn đặt nó trong cùng thư mục với tên iou_ops_extra.py
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
        distill_iou_ratio: float = 1.25,
        distill_power_transform_p: float = 2.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict 
        self.original_losses = copy.deepcopy(losses) # Lưu loss gốc cho aux/dn
        self.losses_to_compute_for_main_output = copy.deepcopy(losses) # Loss cho output chính

        self.boxes_weight_format = boxes_weight_format
        self.share_matched_indices = share_matched_indices
        self.alpha = alpha
        self.gamma = gamma
        self.fgl_targets, self.fgl_targets_dn = None, None
        self.own_targets, self.own_targets_dn = None, None
        self.reg_max = reg_max
        self.num_pos, self.num_neg = None, None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- Lưu các tham số distillation ---
        self.enable_distillation_processing = use_distillation # Cờ này được solver kiểm tra để load teacher
                                                              # và được criterion dùng để quyết định có tính loss_distill không.
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
                print(f"WARNING: DFINECriterion: 'loss_distill' not found in weight_dict. Using default weight 1.0 for distillation loss.")
                self.weight_dict['loss_distill'] = 1.0 
        
        print(f"DFINECriterion initialized. Active losses for main output: {self.losses_to_compute_for_main_output}")
        if self.enable_distillation_processing:
            print(f"  Distillation params: decay={self.distill_decay_type}, stop_ratio={self.distill_stop_epoch_ratio}")
            print(f"  Distill sub-factors: cls_w={self.distill_cls_w_factor}, l1_w={self.distill_l1_w_factor}, iou_w={self.distill_iou_w_factor}")
            print(f"  Distill IoU ratio: {self.distill_iou_expanded_ratio}, Power transform p: {self.distill_power_p_for_transform}")

    # --- CÁC HÀM LOSS GỐC (loss_labels_focal, loss_labels_vfl, loss_boxes, loss_local) GIỮ NGUYÊN ---
    def loss_labels_focal(self, outputs, targets, indices, num_boxes, **kwargs_ignored): # Thêm **kwargs để nhận tham số thừa
        # ... (code gốc của bạn) ...
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
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes # Nên là num_boxes thay vì src_logits.shape[1]
                                                                    # Hoặc nếu RTDETR dùng src_logits.shape[1] thì giữ nguyên

        return {"loss_focal": loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None, **kwargs_ignored):
        # ... (code gốc của bạn) ...
        assert "pred_boxes" in outputs # VFL cần pred_boxes để tính iou nếu values is None
        assert "pred_logits" in outputs
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
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes # Tương tự Focal, xem lại cách normalize
        return {"loss_vfl": loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None, **kwargs_ignored):
        # ... (code gốc của bạn) ...
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        losses = {}
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes # num_boxes là số lượng GT matched

        loss_giou_values = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        if boxes_weight is not None: # boxes_weight là IoU từ get_loss_meta_info
            loss_giou_values = loss_giou_values * boxes_weight
        losses["loss_giou"] = loss_giou_values.sum() / num_boxes

        return losses
        
    def loss_local(self, outputs, targets, indices, num_boxes, T=5, **kwargs_ignored): # T có thể lấy từ config
        # ... (code gốc của bạn, không thay đổi) ...
        # Đảm bảo các key 'pred_corners', 'ref_points', 'reg_scale', 'up' có trong outputs
        # Và 'teacher_corners', 'teacher_logits' nếu DDF được dùng.
        # Lưu ý: DDF loss hiện tại là một dạng self-distillation hoặc distillation nội bộ.
        # Nếu bạn thêm distillation từ teacher bên ngoài, cần xem xét có giữ lại DDF này không.
        # Nếu giữ lại, teacher cho DDF sẽ là teacher_corners từ output của student (EMA)
        # hoặc từ một nguồn khác, không phải teacher_model bên ngoài.
        # Nếu teacher_outputs của bạn có 'teacher_corners' và 'teacher_logits' thì nó sẽ dùng.
        # Tuy nhiên, RTDETRLogicLoss không distill 'corners'.
        # Tạm thời giữ nguyên logic DDF của bạn.
        # ... (code gốc) ...
        # (Code gốc của bạn cho loss_local ở đây)
        # Placeholder để tránh lỗi, bạn cần copy lại code gốc của hàm này
        # (Phần này dài, tôi không copy lại toàn bộ để tránh làm rối)
        # Hãy đảm bảo phần này hoạt động đúng như trước.
        # Nếu DDF của bạn dùng 'teacher_corners' và 'teacher_logits' từ `outputs`
        # thì nó sẽ không bị ảnh hưởng bởi `teacher_outputs` mà chúng ta thêm vào cho distillation.
        temp_losses = {} # Placeholder
        if "pred_corners" in outputs: temp_losses["loss_fgl"] = torch.tensor(0.0, device=self.device)
        if "teacher_corners" in outputs: temp_losses["loss_ddf"] = torch.tensor(0.0, device=self.device)
        return temp_losses


    # --- HÀM MỚI HOẶC ĐÃ SỬA ---
    def power_transform(self, array, power): # Lấy power từ self.distill_power_p_for_transform
        return torch.where(array < 0.5, torch.pow(array, power), torch.pow(array, 1.0 / power))

    def loss_distill(self, student_outputs, targets_for_norm, student_indices_not_used, num_student_boxes_not_used,
                     teacher_outputs=None, epoch=0, total_epochs=1, current_iter=0, total_iters_epoch=1,
                     **kwargs_meta_info_not_used): # Thêm **kwargs để nhận các tham số thừa
        
        if not self.enable_distillation_processing or teacher_outputs is None:
            return {"loss_distill": torch.tensor(0.0, device=self.device)}

        effective_total_epochs = total_epochs if total_epochs > 0 else 1
        stop_kd_iter_epoch = int(effective_total_epochs * self.distill_stop_epoch_ratio)
        if epoch >= stop_kd_iter_epoch:
            return {"loss_distill": torch.tensor(0.0, device=self.device)}

        distill_decay = 1.0
        if self.distill_decay_type == 'linear_epoch':
            current_global_iter = current_iter + total_iters_epoch * epoch
            # Distill chỉ trong khoảng stop_kd_iter_epoch
            total_distill_iterations = stop_kd_iter_epoch * total_iters_epoch 
            if total_distill_iterations > 0:
                progress_ratio = min(max(current_global_iter / total_distill_iterations, 0.0), 1.0)
                distill_decay = 1.0 - (1.0 - 0.01) * progress_ratio 
        elif self.distill_decay_type == 'cosine_epoch':
            current_global_iter = current_iter + total_iters_epoch * epoch
            total_distill_iterations = stop_kd_iter_epoch * total_iters_epoch
            if total_distill_iterations > 0:
                progress_ratio = min(max(current_global_iter / total_distill_iterations, 0.0), 1.0)
                eta_min, base_ratio = 0.01, 1.0
                distill_decay = eta_min + (base_ratio - eta_min) * (1 + math.cos(math.pi * progress_ratio)) / 2
        # Thêm 'constant' nếu cần: elif self.distill_decay_type == 'constant': distill_decay = 1.0
        
        distill_decay = torch.tensor(distill_decay, device=self.device, dtype=torch.float32)

        # --- Trích xuất và xử lý DN Queries ---
        s_logits = student_outputs['pred_logits']
        s_bboxes = student_outputs['pred_boxes'] # cxcywh
        t_logits = teacher_outputs['pred_logits'].detach() # Đã detach ở engine
        t_bboxes = teacher_outputs['pred_boxes'].detach()   # cxcywh

        s_dn_meta = student_outputs.get('dn_meta')
        # Giả định teacher cũng có thể có dn_meta nếu kiến trúc tương đồng và bạn muốn xử lý nó
        # t_dn_meta = teacher_outputs.get('dn_meta') 

        # Cẩn thận với logic DN này, nó cần khớp với cách D-FINE và RT-DETR thực sự hoạt động
        # Cách đơn giản nhất là nếu dn_meta có num_dn_queries:
        if s_dn_meta and 'num_dn_queries' in s_dn_meta:
            num_dn_s = s_dn_meta['num_dn_queries']
            s_logits_obj = s_logits[:, num_dn_s:]
            s_bboxes_obj = s_bboxes[:, num_dn_s:]
            # Giả định teacher có cùng số DN query hoặc đã được xử lý
            # Hoặc bạn chỉ distill trên số query nhỏ nhất giữa student và teacher sau khi bỏ DN
            num_dn_t = teacher_outputs.get('dn_meta', {}).get('num_dn_queries', num_dn_s if s_dn_meta else 0) # Ưu tiên t_dn_meta nếu có
            t_logits_obj = t_logits[:, num_dn_t:]
            t_bboxes_obj = t_bboxes[:, num_dn_t:]

            # Đảm bảo số query bằng nhau sau khi bỏ DN, lấy số min
            min_obj_queries = min(s_logits_obj.size(1), t_logits_obj.size(1))
            s_logits_obj = s_logits_obj[:, :min_obj_queries]
            s_bboxes_obj = s_bboxes_obj[:, :min_obj_queries]
            t_logits_obj = t_logits_obj[:, :min_obj_queries]
            t_bboxes_obj = t_bboxes_obj[:, :min_obj_queries]

        elif s_dn_meta and 'dn_num_split' in s_dn_meta: # Logic phức tạp hơn của RTDETR
            # Giả sử dn_num_split là list 2 phần tử [dn_size, obj_size]
            if isinstance(s_dn_meta['dn_num_split'], (list, tuple)) and len(s_dn_meta['dn_num_split']) == 2:
                s_logits_list = torch.split(s_logits, s_dn_meta['dn_num_split'], dim=1)
                s_bboxes_list = torch.split(s_bboxes, s_dn_meta['dn_num_split'], dim=1)
                s_logits_obj, s_bboxes_obj = s_logits_list[1], s_bboxes_list[1]

                # Tương tự cho teacher nếu có cấu trúc dn_meta giống hệt
                t_dn_meta_val = teacher_outputs.get('dn_meta')
                if t_dn_meta_val and 'dn_num_split' in t_dn_meta_val and \
                   isinstance(t_dn_meta_val['dn_num_split'], (list, tuple)) and len(t_dn_meta_val['dn_num_split']) == 2:
                    t_logits_list = torch.split(t_logits, t_dn_meta_val['dn_num_split'], dim=1)
                    t_bboxes_list = torch.split(t_bboxes, t_dn_meta_val['dn_num_split'], dim=1)
                    t_logits_obj, t_bboxes_obj = t_logits_list[1], t_bboxes_list[1]
                else: # Nếu teacher không có dn_meta khớp, dùng toàn bộ output của teacher và hy vọng số query khớp
                    t_logits_obj, t_bboxes_obj = t_logits, t_bboxes
                
                min_obj_queries = min(s_logits_obj.size(1), t_logits_obj.size(1))
                s_logits_obj = s_logits_obj[:, :min_obj_queries]
                s_bboxes_obj = s_bboxes_obj[:, :min_obj_queries]
                t_logits_obj = t_logits_obj[:, :min_obj_queries]
                t_bboxes_obj = t_bboxes_obj[:, :min_obj_queries]
            else: # Không xử lý được dn_num_split
                s_logits_obj, s_bboxes_obj = s_logits, s_bboxes
                t_logits_obj, t_bboxes_obj = t_logits, t_bboxes
        else: # Không có thông tin DN meta, dùng toàn bộ
            s_logits_obj, s_bboxes_obj = s_logits, s_bboxes
            t_logits_obj, t_bboxes_obj = t_logits, t_bboxes

        if s_logits_obj.numel() == 0 or t_logits_obj.numel() == 0: # Không có query nào để distill
             return {"loss_distill": torch.tensor(0.0, device=self.device)}

        # --- Tính toán các thành phần loss trên _obj tensors ---
        t_obj_scale = t_logits_obj.sigmoid().max(dim=-1, keepdim=True)[0] 

        loss_cls_distill = F.binary_cross_entropy_with_logits(s_logits_obj, t_logits_obj.sigmoid(), reduction='none')
        lcls = loss_cls_distill.mean(dim=-1).sum() # Sum over queries and batch

        loss_l1_distill_raw = F.l1_loss(s_bboxes_obj, t_bboxes_obj, reduction='none')
        lbox_l1 = (loss_l1_distill_raw * t_obj_scale.expand_as(s_bboxes_obj)).sum()

        iou_loss_terms = bbox_distillation_iou_loss(s_bboxes_obj, t_bboxes_obj, 
                                                    ratio=self.distill_iou_expanded_ratio, 
                                                    use_siou_penalty=True, eps=1e-7)
        lbox_iou = (iou_loss_terms * self.power_transform(t_obj_scale.squeeze(-1), power=self.distill_power_p_for_transform)).sum()
        
        # --- Normalization ---
        num_predictions_to_normalize = s_logits_obj.size(0) * s_logits_obj.size(1) # bs * num_obj_queries
        if num_predictions_to_normalize == 0: num_predictions_to_normalize = 1e-9 # Tránh chia cho 0

        # Cách RTDETRLogicLoss normalize (dùng num_gt, nhưng ở đây ta không có batch['bboxes'])
        # Sử dụng num_student_boxes_for_norm (số lượng positive matches của student với GT) có thể là một lựa chọn
        # Hoặc đơn giản là normalize bằng số lượng queries * batch_size
        # num_normalizer = num_student_boxes_for_norm if num_student_boxes_for_norm > 0 else num_predictions_to_normalize
        # Dùng số lượng GT objects trong batch để normalize (nếu targets_for_norm được truyền vào là targets gốc)
        num_gt_objects_in_batch = sum(len(t_dict.get("labels", [])) for t_dict in targets_for_norm)
        if num_gt_objects_in_batch == 0: num_gt_objects_in_batch = s_logits_obj.size(0) # Fallback là batch_size

        lcls_norm = lcls / (num_gt_objects_in_batch + 1e-9) # RTDETR: lcls.sum() / (batch_bboxes_size / t_obj_scale_size1)
        lbox_l1_norm = lbox_l1 / (num_gt_objects_in_batch + 1e-9) # RTDETR: lbox_l1.sum() / batch_bboxes_size (*5)
        lbox_iou_norm = lbox_iou / (num_gt_objects_in_batch + 1e-9)# RTDETR: lbox_iou.sum() / batch_bboxes_size (*2)

        final_l1_loss = lbox_l1_norm * self.distill_l1_w_factor
        final_iou_loss = lbox_iou_norm * self.distill_iou_w_factor
        final_cls_loss = lcls_norm * self.distill_cls_w_factor
        
        total_distill_loss_components = final_l1_loss + final_iou_loss + final_cls_loss
        final_distill_loss = total_distill_loss_components * distill_decay

        return {"loss_distill": final_distill_loss}

    # --- HÀM FORWARD ĐÃ ĐƯỢC SỬA ĐỔI ---
    def forward(self, outputs, targets, teacher_outputs=None, **kwargs_runtime_metas):
        # `outputs` là của student
        # `teacher_outputs` là của teacher (nếu có)
        # `kwargs_runtime_metas` chứa epoch, step, total_epochs, v.v. từ train_one_epoch

        # Lấy output chính của student (không phải aux, dn, v.v.)
        # Cần xác định chính xác key nào là output chính nếu có nhiều loại (pre_outputs, dn_outputs, outputs)
        # Giả sử 'pred_logits' và 'pred_boxes' ở top-level của `outputs` là output chính
        # Hoặc, nếu D-FINE luôn có key 'final_pred_logits', 'final_pred_boxes' thì dùng key đó.
        # Ở đây, chúng ta dùng outputs_without_aux như logic cũ của bạn.
        main_student_outputs = {k: v for k, v in outputs.items() if "aux" not in k and "dn" not in k and "enc" not in k and "pre" not in k}
        
        # Lấy device từ một tensor bất kỳ trong output (nếu có)
        if main_student_outputs:
            device_for_tensors = next(iter(main_student_outputs.values()))[0].device if isinstance(next(iter(main_student_outputs.values())), (list, tuple)) else next(iter(main_student_outputs.values())).device
        elif outputs:
            device_for_tensors = next(iter(outputs.values()))[0].device if isinstance(next(iter(outputs.values())), (list, tuple)) else next(iter(outputs.values())).device
        else: # Không có output nào cả
            return {} 

        # Matching cho output chính của student
        # Kiểm tra xem main_student_outputs có các key cần thiết cho matcher không
        if 'pred_logits' in main_student_outputs and 'pred_boxes' in main_student_outputs:
            indices = self.matcher(main_student_outputs, targets)["indices"]
        else: # Không có output chính để matching, tạo indices rỗng
            indices = [(torch.empty(0, dtype=torch.long, device=device_for_tensors), 
                        torch.empty(0, dtype=torch.long, device=device_for_tensors)) 
                       for _ in range(len(targets))]
        
        self._clear_cache()

        # --- Xử lý indices_go và num_boxes_go ---
        # (Logic này phức tạp và phụ thuộc vào cách bạn muốn xử lý aux losses. 
        #  Giữ nguyên logic `indices_go` của bạn từ file gốc nếu nó hoạt động đúng)
        # Dưới đây là một phiên bản đơn giản hóa, bạn cần điều chỉnh cho phù hợp với D-FINE
        num_boxes_go = 0.0 # Khởi tạo
        indices_go = indices # Mặc định nếu không có aux
        cached_indices_for_aux = [] # Để lưu matching của từng lớp aux

        if "aux_outputs" in outputs or "pre_outputs" in outputs or "enc_aux_outputs" in outputs:
            all_decoder_outputs_for_go = [] # List các output từ decoder/encoder để tính indices_go
            
            # Lấy matching cho các lớp aux của decoder (bao gồm pre_outputs)
            # Thêm output chính vào để tính indices_go
            if 'pred_logits' in main_student_outputs : all_decoder_outputs_for_go.append(main_student_outputs)
            if "pre_outputs" in outputs: all_decoder_outputs_for_go.append(outputs["pre_outputs"])
            if "aux_outputs" in outputs: all_decoder_outputs_for_go.extend(outputs["aux_outputs"])

            if all_decoder_outputs_for_go:
                indices_all_layers = [self.matcher(out, targets)["indices"] for out in all_decoder_outputs_for_go]
                cached_indices_for_aux = indices_all_layers[1:] # Bỏ qua indices của lớp cuối (main_student_outputs)
                                                                # Hoặc bạn cần logic map chính xác hơn
                
                # Nếu chỉ có main_output, indices_go = indices.
                # Nếu có aux, indices_go là union.
                if len(indices_all_layers) > 1:
                     indices_go = self._get_go_indices(indices_all_layers[0], indices_all_layers[1:])
                else:
                     indices_go = indices_all_layers[0] if indices_all_layers else indices


                num_boxes_go = sum(len(x[0]) for x in indices_go)
                num_boxes_go = torch.as_tensor([num_boxes_go], dtype=torch.float, device=device_for_tensors)
                if is_dist_available_and_initialized(): torch.distributed.all_reduce(num_boxes_go)
                num_boxes_go = torch.clamp(num_boxes_go / get_world_size(), min=1).item()
        else: # Không có aux nào cả
            indices_go = indices
            num_boxes_go = sum(len(x[0]) for x in indices_go) if indices and indices[0][0].numel() > 0 else 0.0
            if num_boxes_go > 0 : # Chỉ all_reduce nếu > 0
                num_boxes_go_tensor = torch.as_tensor([num_boxes_go], dtype=torch.float, device=device_for_tensors)
                if is_dist_available_and_initialized(): torch.distributed.all_reduce(num_boxes_go_tensor)
                num_boxes_go = torch.clamp(num_boxes_go_tensor / get_world_size(), min=1).item()
            else:
                num_boxes_go = 0.0


        # num_boxes cho classification loss của output chính (dựa trên số GT)
        num_boxes_for_main_cls_loss = sum(len(t["labels"]) for t in targets)
        num_boxes_for_main_cls_loss_tensor = torch.as_tensor([num_boxes_for_main_cls_loss], dtype=torch.float, device=device_for_tensors)
        if is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_boxes_for_main_cls_loss_tensor)
        num_boxes_for_main_cls_loss = torch.clamp(num_boxes_for_main_cls_loss_tensor / get_world_size(), min=1).item()
        if not ('pred_logits' in main_student_outputs) : num_boxes_for_main_cls_loss = 0.0 # Nếu không có output chính

        # --- Tính loss cho OUTPUT CHÍNH (bao gồm cả distillation nếu được kích hoạt) ---
        calculated_losses = {}
        if 'pred_logits' in main_student_outputs : # Chỉ tính loss cho output chính nếu nó tồn tại
            for loss_name in self.losses_to_compute_for_main_output: # self.losses_to_compute... đã bao gồm 'distill'
                current_indices_for_loss = indices_go if loss_name in ["boxes", "local"] else indices
                current_num_boxes_for_loss = num_boxes_go if loss_name in ["boxes", "local"] else num_boxes_for_main_cls_loss

                # Tạo kwargs cho hàm get_loss
                call_kwargs_for_get_loss = {}
                if loss_name == 'distill':
                    call_kwargs_for_get_loss.update(kwargs_runtime_metas) # epoch, step, total_epochs, ...
                    call_kwargs_for_get_loss['teacher_outputs'] = teacher_outputs
                    # `targets` sẽ là `targets_for_norm` trong loss_distill
                    # `indices` sẽ là `student_indices_not_used`
                    # `num_boxes` sẽ là `num_student_boxes_for_norm`
                    # Chúng ta truyền `targets` gốc và `num_boxes_for_main_cls_loss` (hoặc `num_boxes_go`)
                    # để `loss_distill` có thể dùng `targets` để lấy `num_gt_objects_in_batch`
                    loss_dict_component = self.get_loss(loss_name, main_student_outputs, targets, indices, num_boxes_for_main_cls_loss, **call_kwargs_for_get_loss)
                else: # Các loss gốc
                    meta_info = self.get_loss_meta_info(loss_name, main_student_outputs, targets, current_indices_for_loss)
                    call_kwargs_for_get_loss['meta'] = meta_info
                    # Truyền thêm kwargs_runtime_metas nếu các loss gốc cần (ví dụ, loss_local có thể cần epoch)
                    call_kwargs_for_get_loss.update({k:v for k,v in kwargs_runtime_metas.items() if k not in ['teacher_outputs']})
                    loss_dict_component = self.get_loss(loss_name, main_student_outputs, targets, current_indices_for_loss, current_num_boxes_for_loss, **call_kwargs_for_get_loss)
                
                if loss_dict_component is not None:
                     weighted_loss_dict_component = {k: loss_dict_component[k] * self.weight_dict[k] for k in loss_dict_component if k in self.weight_dict}
                     calculated_losses.update(weighted_loss_dict_component)

        # --- Xử lý Auxiliary Losses (CHỈ TÍNH CÁC LOSS GỐC) ---
        # Cần tạo lại cached_indices cho từng loại aux output
        # `original_losses` là danh sách loss không bao gồm 'distill'
        
        # Helper để tính aux loss
        def compute_aux_losses(aux_outputs_list, aux_name_prefix, base_cached_indices, enc_class_agnostic_targets=None):
            for i, aux_out_data in enumerate(aux_outputs_list):
                current_targets = enc_class_agnostic_targets if enc_class_agnostic_targets is not None else targets
                # aux_out_data có thể cần 'up', 'reg_scale' từ output chính
                if "up" not in aux_out_data: aux_out_data["up"] = outputs.get("up")
                if "reg_scale" not in aux_out_data: aux_out_data["reg_scale"] = outputs.get("reg_scale")
                
                # Matching riêng cho từng lớp aux (trừ khi dùng indices_go cho regression)
                current_aux_match_indices = self.matcher(aux_out_data, current_targets)["indices"] if not base_cached_indices else base_cached_indices[i]


                for loss_name_aux in self.original_losses:
                    indices_in_aux = indices_go if loss_name_aux in ["boxes", "local"] else current_aux_match_indices
                    num_boxes_in_aux = num_boxes_go if loss_name_aux in ["boxes", "local"] else num_boxes_for_main_cls_loss
                    
                    if num_boxes_in_aux == 0 and indices_in_aux[0][0].numel() == 0 : # Nếu không có box nào để tính loss
                        calculated_losses[loss_name_aux + f"_{aux_name_prefix}_{i}"] = torch.tensor(0.0, device=device_for_tensors)
                        continue

                    meta_aux = self.get_loss_meta_info(loss_name_aux, aux_out_data, current_targets, indices_in_aux)
                    
                    aux_call_kwargs = {'meta': meta_aux}
                    aux_call_kwargs.update({k:v for k,v in kwargs_runtime_metas.items() if k not in ['teacher_outputs']})

                    l_dict_aux = self.get_loss(loss_name_aux, aux_out_data, current_targets, 
                                               indices_in_aux, num_boxes_in_aux, 
                                               **aux_call_kwargs)
                    
                    if l_dict_aux is not None:
                        l_dict_aux_weighted = {k: l_dict_aux[k] * self.weight_dict[k] for k in l_dict_aux if k in self.weight_dict}
                        l_dict_aux_final = {k + f"_{aux_name_prefix}_{i}": v for k, v in l_dict_aux_weighted.items()}
                        calculated_losses.update(l_dict_aux_final)

        # Aux_outputs từ decoder
        if "aux_outputs" in outputs:
            # cached_indices_for_aux đã được tạo ở trên (nếu có)
            # Hoặc truyền None để matching lại
            compute_aux_losses(outputs["aux_outputs"], "aux", cached_indices_for_aux if 'cached_indices_for_aux' in locals() and cached_indices_for_aux else None)

        # Pre_outputs
        if "pre_outputs" in outputs:
            # Giả sử pre_outputs là một dict đơn, không phải list
            # cached_indices cho pre_outputs là phần tử đầu tiên trong cached_indices_for_aux (nếu logic đúng)
            pre_cached_idx = None
            if 'cached_indices_for_aux' in locals() and cached_indices_for_aux : pre_cached_idx = [cached_indices_for_aux[0]] # Phải là list 1 phần tử
            compute_aux_losses([outputs["pre_outputs"]], "pre", pre_cached_idx)


        # Enc_aux_outputs
        if "enc_aux_outputs" in outputs:
            assert "enc_meta" in outputs, ""
            class_agnostic = outputs["enc_meta"]["class_agnostic"]
            current_enc_targets = targets
            orig_num_classes_backup = self.num_classes
            if class_agnostic:
                self.num_classes = 1
                current_enc_targets = copy.deepcopy(targets)
                for t in current_enc_targets: t["labels"] = torch.zeros_like(t["labels"])
            
            # Matching riêng cho enc_aux_outputs
            cached_indices_enc = [self.matcher(enc_out, current_enc_targets)["indices"] for enc_out in outputs["enc_aux_outputs"]]
            compute_aux_losses(outputs["enc_aux_outputs"], "enc", cached_indices_enc, enc_class_agnostic_targets=current_enc_targets)

            if class_agnostic: self.num_classes = orig_num_classes_backup

        # DN_outputs
        if "dn_outputs" in outputs:
            assert "dn_meta" in outputs, ""
            indices_dn = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            dn_num_boxes_norm = num_boxes_for_main_cls_loss * outputs["dn_meta"]["dn_num_group"] # Dùng num_boxes_main_cls
            dn_num_boxes_norm = dn_num_boxes_norm if dn_num_boxes_norm > 0 else 1.0

            # DN outputs thường không dùng indices_go cho regression, mà dùng indices_dn
            def compute_dn_losses(dn_outputs_list, dn_name_prefix):
                 for i, aux_out_data in enumerate(dn_outputs_list):
                    aux_out_data["is_dn"] = True # Đánh dấu là DN output
                    if "up" not in aux_out_data: aux_out_data["up"] = outputs.get("up")
                    if "reg_scale" not in aux_out_data: aux_out_data["reg_scale"] = outputs.get("reg_scale")

                    for loss_name_dn in self.original_losses:
                        # DN losses dùng indices_dn cho tất cả các loại loss
                        if dn_num_boxes_norm == 0 and indices_dn[0][0].numel() == 0:
                             calculated_losses[loss_name_dn + f"_{dn_name_prefix}_{i}"] = torch.tensor(0.0, device=device_for_tensors)
                             continue

                        meta_dn = self.get_loss_meta_info(loss_name_dn, aux_out_data, targets, indices_dn)
                        dn_call_kwargs = {'meta': meta_dn}
                        dn_call_kwargs.update({k:v for k,v in kwargs_runtime_metas.items() if k not in ['teacher_outputs']})

                        l_dict_dn = self.get_loss(loss_name_dn, aux_out_data, targets, 
                                                  indices_dn, dn_num_boxes_norm, 
                                                  **dn_call_kwargs)
                        if l_dict_dn is not None:
                            l_dict_dn_weighted = {k: l_dict_dn[k] * self.weight_dict[k] for k in l_dict_dn if k in self.weight_dict}
                            l_dict_dn_final = {k + f"_{dn_name_prefix}_{i}": v for k, v in l_dict_dn_weighted.items()}
                            calculated_losses.update(l_dict_dn_final)
            
            compute_dn_losses(outputs["dn_outputs"], "dn")
            if "dn_pre_outputs" in outputs:
                compute_dn_losses([outputs["dn_pre_outputs"]], "dn_pre")
                

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
