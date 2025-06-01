import torch
import math

# Hàm tiện ích chuyển đổi format nếu cần (D-FINE đã có box_cxcywh_to_xyxy)
# Nếu box_ops.py của bạn chưa có, bạn có thể thêm vào đó hoặc ở đây.
# from .box_ops import box_cxcywh_to_xyxy # Giả sử đã có
# Nếu chưa có, đây là một bản triển khai đơn giản:
def box_cxcywh_to_xyxy_elementwise(boxes_cxcywh):
    # boxes_cxcywh: [..., 4]
    cx, cy, w, h = boxes_cxcywh.unbind(-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack((x1, y1, x2, y2), dim=-1)

def get_expanded_iou_elementwise(boxes1_xywh: torch.Tensor, 
                                 boxes2_xywh: torch.Tensor, 
                                 ratio: float = 1.25, 
                                 eps: float = 1e-7) -> torch.Tensor:
    """
    Calculates element-wise expanded Intersection over Union (IoU).
    Boxes are expected in (center_x, center_y, width, height) format.
    Args:
        boxes1_xywh (torch.Tensor): Tensor of shape [B, N, 4] or [N, 4].
        boxes2_xywh (torch.Tensor): Tensor of shape [B, N, 4] or [N, 4].
        ratio (float): Expansion ratio.
        eps (float): Small epsilon to prevent division by zero.
    Returns:
        torch.Tensor: Element-wise expanded IoU, shape [B, N] or [N,].
    """
    if boxes1_xywh.shape != boxes2_xywh.shape:
        raise ValueError("Input boxes should have the same shape.")

    # Unbind coordinates
    cx1, cy1, w1, h1 = boxes1_xywh.unbind(-1)
    cx2, cy2, w2, h2 = boxes2_xywh.unbind(-1)

    # Calculate expanded box coordinates (xyxy format)
    exp_w1, exp_h1 = w1 * ratio, h1 * ratio
    b1_x1_exp, b1_y1_exp = cx1 - exp_w1 / 2, cy1 - exp_h1 / 2
    b1_x2_exp, b1_y2_exp = cx1 + exp_w1 / 2, cy1 + exp_h1 / 2

    exp_w2, exp_h2 = w2 * ratio, h2 * ratio
    b2_x1_exp, b2_y1_exp = cx2 - exp_w2 / 2, cy2 - exp_h2 / 2
    b2_x2_exp, b2_y2_exp = cx2 + exp_w2 / 2, cy2 + exp_h2 / 2

    # Intersection area of expanded boxes
    inter_x1 = torch.maximum(b1_x1_exp, b2_x1_exp)
    inter_y1 = torch.maximum(b1_y1_exp, b2_y1_exp)
    inter_x2 = torch.minimum(b1_x2_exp, b2_x2_exp)
    inter_y2 = torch.minimum(b1_y2_exp, b2_y2_exp)
    
    inter_w = (inter_x2 - inter_x1).clamp_(min=0)
    inter_h = (inter_y2 - inter_y1).clamp_(min=0)
    inter_area_exp = inter_w * inter_h

    # Union Area of expanded boxes
    area1_exp = exp_w1 * exp_h1
    area2_exp = exp_w2 * exp_h2
    union_area_exp = area1_exp + area2_exp - inter_area_exp + eps
    
    return inter_area_exp / union_area_exp

def bbox_distillation_iou_loss(s_boxes_xywh: torch.Tensor, 
                               t_boxes_xywh: torch.Tensor, 
                               ratio: float = 1.25, 
                               use_siou_penalty: bool = True,
                               eps: float = 1e-7) -> torch.Tensor:
    """
    Calculates the distillation IoU loss (1 - (ExpandedIoU - SIoUPenalty_if_any)).
    Inputs are element-wise, in (center_x, center_y, width, height) format.
    Shapes: [B, N, 4] for s_boxes_xywh and t_boxes_xywh.
    Returns tensor of shape [B, N].
    """
    if s_boxes_xywh.shape != t_boxes_xywh.shape:
        raise ValueError("Student and Teacher boxes should have the same shape.")
    if s_boxes_xywh.dim() < 2 or s_boxes_xywh.shape[-1] != 4:
        raise ValueError("Input boxes should have at least 2 dimensions with the last one being 4.")

    # 1. Calculate Expanded IoU (element-wise)
    expanded_iou = get_expanded_iou_elementwise(s_boxes_xywh, t_boxes_xywh, ratio=ratio, eps=eps)

    siou_penalty = torch.zeros_like(expanded_iou) # Default no penalty

    if use_siou_penalty:
        # For SIoU penalty, we use the original (non-expanded) boxes
        # Convert original boxes to xyxy for SIoU penalty calculations
        s_boxes_xyxy = box_cxcywh_to_xyxy_elementwise(s_boxes_xywh) # Use your framework's function or the one above
        t_boxes_xyxy = box_cxcywh_to_xyxy_elementwise(t_boxes_xywh) # Use your framework's function or the one above

        b1_x1, b1_y1, b1_x2, b1_y2 = s_boxes_xyxy.unbind(-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = t_boxes_xyxy.unbind(-1)
        
        # Original widths and heights (from xywh input)
        _, _, w1, h1 = s_boxes_xywh.unbind(-1)
        _, _, w2, h2 = t_boxes_xywh.unbind(-1)

        # Add small epsilon to avoid division by zero for w, h, cw, ch, sigma
        w1, h1 = w1 + eps, h1 + eps
        w2, h2 = w2 + eps, h2 + eps

        # Convex (smallest enclosing box) width
        cw = torch.maximum(b1_x2, b2_x2) - torch.minimum(b1_x1, b2_x1) + eps
        # Convex height
        ch = torch.maximum(b1_y2, b2_y2) - torch.minimum(b1_y1, b2_y1) + eps

        # Center points distance components (s_cw and s_ch in Ultralytics code)
        # (b2_center_x - b1_center_x), (b2_center_y - b1_center_y)
        delta_center_x = (b2_x1 + b2_x2)/2 - (b1_x1 + b1_x2)/2
        delta_center_y = (b2_y1 + b2_y2)/2 - (b1_y1 + b1_y2)/2
        
        # --- SIoU Penalty Calculation (Logic from Ultralytics bbox_iou when SIoU=True) ---
        # Distance Angle Cost
        sigma = torch.sqrt(delta_center_x**2 + delta_center_y**2) + eps 
        
        sin_alpha_1 = torch.abs(delta_center_x) / sigma
        sin_alpha_2 = torch.abs(delta_center_y) / sigma
        # arcsin argument must be in [-1, 1]
        # Threshold is sin(pi/4)
        threshold = pow(2, 0.5) / 2 
        sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
        
        # Clamp sin_alpha to avoid domain errors for arcsin due to numerical instability
        sin_alpha = torch.clamp(sin_alpha, -1.0 + eps, 1.0 - eps)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2) # Should be between -1 (bad) and 1 (good)

        # Distance Cost
        # rho_x and rho_y are (delta_center_coord / convex_hull_dim) ** 2
        rho_x_sq = (delta_center_x / cw)**2
        rho_y_sq = (delta_center_y / ch)**2
        
        # gamma in Ultralytics' SIoU is angle_cost - 2 (making it <= -1)
        # penalty_factor_gamma = angle_cost - 2
        # Original SIoU paper uses gamma = 2 - angle_cost. Let's follow Ultralytics' for consistency with RTDETRLogicLoss.
        # However, the formula in paper is Loss = ... + (distance_cost + shape_cost)/2
        # where distance_cost = exp(gamma * rho_x) + exp(gamma * rho_y) with gamma = 2 - angle_cost
        # Ultralytics code returns iou - 0.5 * (distance_term + shape_term)
        # where distance_term = 2 - exp( (angle_cost - 2) * rho_x ) - exp( (angle_cost - 2) * rho_y )
        # This distance_term increases as angle_cost decreases (more misalignment)
        # or as rho_x/rho_y increases (larger normalized distance).
        
        gamma_for_distance_penalty = angle_cost - 2 
        distance_penalty_component = (2 - torch.exp(gamma_for_distance_penalty * rho_x_sq) - 
                                     torch.exp(gamma_for_distance_penalty * rho_y_sq))

        # Shape Cost (Omega)
        # delta_w = torch.abs(w1 - w2)
        # delta_h = torch.abs(h1 - h2)
        # max_w = torch.maximum(w1, w2)
        # max_h = torch.maximum(h1, h2)
        # omega_w = delta_w / (max_w + eps)
        # omega_h = delta_h / (max_h + eps)
        # Original SIoU paper uses theta for pow, default is 4 in some implementations.
        # shape_penalty_component = torch.pow(1 - torch.exp(-omega_w), 4) + \
        #                           torch.pow(1 - torch.exp(-omega_h), 4)
        
        # Ultralytics YOLOv8 SIoU shape cost (omiga is used, often theta=4)
        # It seems Ultralytics' `bbox_iou` (which `bbox_inner_iou` calls)
        # calculates omiga_w, omiga_h slightly differently if you trace it.
        # Let's use the direct w1, h1, w2, h2.
        dw = torch.abs(w1 - w2)
        dh = torch.abs(h1 - h2)
        # According to Ultralytics' code, it seems to be:
        # omiga_w = dw / cw if cw > 0 else 0 # Not cw, but max(w1,w2)
        # omiga_h = dh / ch if ch > 0 else 0 # Not ch, but max(h1,h2)
        # Let's stick to the more common definition for shape cost from papers like EIoU/AlphaIoU
        # which RTDETRLogicLoss's bbox_inner_iou seems to follow (from Ultralytics utils)

        omega_w = dw / (torch.maximum(w1, w2) + eps)
        omega_h = dh / (torch.maximum(h1, h2) + eps)
        # Default theta = 4 from common SIoU implementations
        theta_shape = 4 
        shape_penalty_component = torch.pow(1 - torch.exp(-omega_w), theta_shape) + \
                                  torch.pow(1 - torch.exp(-omega_h), theta_shape)
        
        siou_penalty = 0.5 * (distance_penalty_component + shape_penalty_component)
        # Ensure penalty is not negative, though terms are designed to be positive or zero
        siou_penalty = torch.clamp(siou_penalty, min=0.0)

    # Final value is expanded_iou minus the penalty
    final_iou_metric = expanded_iou - siou_penalty
    
    # The loss is 1.0 - this metric
    return 1.0 - final_iou_metric