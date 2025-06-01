# det_engine.py

import math
import sys
from typing import Dict, Iterable, List
from pathlib import Path # Thêm Path nếu output_dir là string

import numpy as np
import torch
import torch.amp
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

# Giả sử các import này đúng với cấu trúc project của bạn
from ..data import CocoEvaluator 
from ..data.dataset import mscoco_category2label # Được dùng trong evaluate
from ..misc import MetricLogger, SmoothedValue, dist_utils, save_samples
from ..optim import ModelEMA, Warmup # ModelEMA và Warmup được lấy từ kwargs
from .validator import Validator, scale_boxes # Được dùng trong evaluate

import torch.nn.functional as F
from datetime import datetime


def train_one_epoch(
    model: torch.nn.Module,    # Student model
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    # Các tham số sau sẽ được truyền qua **kwargs từ DetSolver.fit()
    # use_wandb: bool, 
    # max_norm: float = 0,
    teacher_model: torch.nn.Module = None,
    cfg_solver = None, # Instance YAMLConfig của student
    **kwargs, 
):
    # Lấy use_wandb_flag từ kwargs, với giá trị mặc định là False
    use_wandb_flag = kwargs.get('use_wandb', False)
    if use_wandb_flag:
        try:
            import wandb
        except ImportError:
            print(f"[{datetime.now().isoformat()}] WARNING (train_one_epoch): wandb is enabled but not installed. Disabling wandb.")
            use_wandb_flag = False


    model.train()
    criterion.train() # Đặt criterion ở train mode (nếu nó có các lớp như Dropout, BatchNorm)
    
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))

    total_epochs_for_training = kwargs.get("epochs", 1) # Lấy tổng số epoch từ kwargs
    header = "Epoch: [{}]".format(epoch) if total_epochs_for_training == 1 else "Epoch: [{}/{}]".format(epoch, total_epochs_for_training)

    # --- Khởi tạo và kiểm tra các biến distillation ---
    use_distillation_active_for_epoch = False
    distill_stop_epoch_val_for_epoch = total_epochs_for_training 

    if cfg_solver is not None and hasattr(cfg_solver, 'yaml_cfg'):
        _use_distill_from_cfg_in_yaml = cfg_solver.yaml_cfg.get('use_distillation', False)
        
        if _use_distill_from_cfg_in_yaml:
            if teacher_model is not None:
                use_distillation_active_for_epoch = True
                _distill_stop_epoch_ratio = cfg_solver.yaml_cfg.get('distill_stop_epoch_ratio', 1.0)
                if total_epochs_for_training > 0:
                    distill_stop_epoch_val_for_epoch = int(total_epochs_for_training * _distill_stop_epoch_ratio)
                else: 
                    use_distillation_active_for_epoch = False 
                    if dist_utils.is_main_process():
                        print(f"[{datetime.now().isoformat()}] WARNING (train_one_epoch): Total epochs ({total_epochs_for_training}) is not valid. Distillation set to False.")
            # Trường hợp teacher_model is None đã được xử lý ở BaseSolver (in log) và ở đây use_distillation_active_for_epoch sẽ là False
    elif dist_utils.is_main_process() and kwargs.get('use_distillation', False): # Nếu cờ use_distillation được truyền trực tiếp qua kwargs
         print(f"[{datetime.now().isoformat()}] WARNING (train_one_epoch): cfg_solver is None, but 'use_distillation' found in kwargs. Distillation logic might be based on this if teacher_model is present.")
         if teacher_model is not None and kwargs.get('use_distillation'):
             use_distillation_active_for_epoch = True # Kích hoạt nếu teacher có và cờ là True
             # Cần thêm logic lấy distill_stop_epoch_ratio từ kwargs nếu có
    
    # --- In thông tin distillation một lần ở epoch đầu tiên ---
    if epoch == 0 and dist_utils.is_main_process():
        if use_distillation_active_for_epoch: 
            print(f"[{datetime.now().isoformat()}] INFO (train_one_epoch): Distillation is ACTIVE. Teacher model is present. Distillation will stop after epoch {distill_stop_epoch_val_for_epoch -1}.")
        elif cfg_solver is not None and hasattr(cfg_solver, 'yaml_cfg') and \
             cfg_solver.yaml_cfg.get('use_distillation', False) and \
             teacher_model is None:
            print(f"[{datetime.now().isoformat()}] INFO (train_one_epoch @ epoch 0 print): 'use_distillation' is True in config, but teacher_model is None (setup failed or not provided). Distillation was SKIPPED.")
        elif cfg_solver is not None and hasattr(cfg_solver, 'yaml_cfg') and \
             not cfg_solver.yaml_cfg.get('use_distillation', False):
            print(f"[{datetime.now().isoformat()}] INFO (train_one_epoch @ epoch 0 print): Distillation is NOT active ('use_distillation' is False in config).")
        else: 
             print(f"[{datetime.now().isoformat()}] INFO (train_one_epoch @ epoch 0 print): Distillation is NOT active (cfg_solver issue or teacher_model missing and use_distillation not explicitly True in config).")
    # --- Kết thúc phần khởi tạo và in log distillation ---
            
    print_freq = kwargs.get("print_freq", 10)
    writer: SummaryWriter = kwargs.get("writer", None)
    ema: ModelEMA = kwargs.get("ema", None) 
    scaler: GradScaler = kwargs.get("scaler", None) 
    lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)
    max_norm_from_kwargs = kwargs.get("max_norm", 0.0) # Lấy max_norm từ kwargs
    
    losses_log_accumulator = [] 

    output_dir_str = kwargs.get("output_dir")
    output_dir_path = Path(output_dir_str) if output_dir_str else None
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    for i, (batch_samples_cpu, batch_targets_cpu) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        global_step = epoch * len(data_loader) + i
        
        samples_on_device = batch_samples_cpu.to(device)
        targets_on_device = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in batch_targets_cpu]
        
        # `total_epochs_for_training` đã được lấy ở đầu hàm
        metas_for_criterion = dict(epoch=epoch, step=i, global_step=global_step, 
                                   epoch_step=len(data_loader), 
                                   total_epochs=total_epochs_for_training)
        
        teacher_outputs_for_criterion = None 
        if use_distillation_active_for_epoch and teacher_model is not None and epoch < distill_stop_epoch_val_for_epoch:
            run_teacher_forward_debug_print = (i < 2 and epoch == 0 and dist_utils.is_main_process())
            
            if run_teacher_forward_debug_print:
                print(f"[{datetime.now().isoformat()}] DEBUG (train_one_epoch): Attempting teacher forward pass for epoch {epoch}, batch {i}...")
                print(f"    Input samples_on_device.device: {samples_on_device.device}")
                if list(teacher_model.parameters()):
                     print(f"    Teacher model (a parameter's device): {next(teacher_model.parameters()).device}")
                else:
                     print(f"    Teacher model has no parameters to check device.")

            with torch.no_grad():
                try:
                    # **QUAN TRỌNG**: Xác nhận D-FINE teacher có cần targets khi eval() không.
                    raw_teacher_outputs = teacher_model(samples_on_device) 
                    # HOẶC: raw_teacher_outputs = teacher_model(samples_on_device, targets=targets_on_device) # Nếu cần
                    
                    if run_teacher_forward_debug_print:
                        print(f"[{datetime.now().isoformat()}] DEBUG: Teacher forward pass completed for epoch {epoch}, batch {i}.")

                    if isinstance(raw_teacher_outputs, dict) and \
                       'pred_logits' in raw_teacher_outputs and \
                       'pred_boxes' in raw_teacher_outputs:
                        
                        teacher_outputs_for_criterion = {
                            'pred_logits': raw_teacher_outputs['pred_logits'].detach(),
                            'pred_boxes': raw_teacher_outputs['pred_boxes'].detach()
                            # Thêm 'pred_corners' nếu DFINECriterion.loss_distill của bạn cần
                        }
                        if run_teacher_forward_debug_print:
                            print(f"    Teacher output pred_logits - shape: {teacher_outputs_for_criterion['pred_logits'].shape}, dtype: {teacher_outputs_for_criterion['pred_logits'].dtype}, device: {teacher_outputs_for_criterion['pred_logits'].device}")
                            print(f"    Teacher output pred_boxes - shape: {teacher_outputs_for_criterion['pred_boxes'].shape}, dtype: {teacher_outputs_for_criterion['pred_boxes'].dtype}, device: {teacher_outputs_for_criterion['pred_boxes'].device}")
                    else:
                        if run_teacher_forward_debug_print or (i == 0 and epoch == 0 and dist_utils.is_main_process()):
                            print(f"[{datetime.now().isoformat()}] WARNING: Teacher output has unexpected structure or missing keys ('pred_logits', 'pred_boxes').")
                            print(f"    Type of raw_teacher_outputs: {type(raw_teacher_outputs)}")
                            if isinstance(raw_teacher_outputs, dict): print(f"    Available keys in teacher output: {list(raw_teacher_outputs.keys())}")
                            
                except Exception as e_teacher_fwd:
                    if run_teacher_forward_debug_print or (i == 0 and epoch == 0 and dist_utils.is_main_process()):
                        print(f"[{datetime.now().isoformat()}] ERROR during teacher forward pass for epoch {epoch}, batch {i}:")
                        import traceback
                        traceback.print_exc()
        
        if global_step < num_visualization_sample_batch and output_dir_path is not None and dist_utils.is_main_process():
            save_samples(batch_samples_cpu, batch_targets_cpu, output_dir_path, "train", normalized=True, box_fmt="cxcywh")

        # Student forward pass và tính loss
        if scaler is not None: 
            with torch.autocast(device_type=str(device), cache_enabled=True):
                student_outputs = model(samples_on_device, targets=targets_on_device)
            
            if torch.isnan(student_outputs["pred_boxes"]).any() or torch.isinf(student_outputs["pred_boxes"]).any():
                print(f"NaN/Inf in student_outputs['pred_boxes'] at epoch {epoch}, batch {i}: {student_outputs['pred_boxes']}")
                # ... (logic lưu NaN.pth) ...
                if output_dir_path and dist_utils.is_main_process():
                    nan_ckpt_path = output_dir_path / f"NaN_checkpoint_epoch{epoch}_batch{i}.pth"
                    dist_utils.save_on_master(model.state_dict(), nan_ckpt_path) # Chỉ lưu model state
                    print(f"Saved NaN checkpoint to {nan_ckpt_path}")
                # sys.exit(1) # Cân nhắc có nên dừng hẳn không

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(student_outputs, targets_on_device, 
                                      teacher_outputs=teacher_outputs_for_criterion, 
                                      **metas_for_criterion) # Truyền metas vào đây
        else: 
            student_outputs = model(samples_on_device, targets=targets_on_device)
            if torch.isnan(student_outputs["pred_boxes"]).any() or torch.isinf(student_outputs["pred_boxes"]).any():
                print(f"NaN/Inf in student_outputs['pred_boxes'] at epoch {epoch}, batch {i}: {student_outputs['pred_boxes']}")
                # ... (logic lưu NaN.pth) ...
                if output_dir_path and dist_utils.is_main_process():
                    nan_ckpt_path = output_dir_path / f"NaN_checkpoint_epoch{epoch}_batch{i}.pth"
                    dist_utils.save_on_master(model.state_dict(), nan_ckpt_path)
                    print(f"Saved NaN checkpoint to {nan_ckpt_path}")
                # sys.exit(1)

            loss_dict = criterion(student_outputs, targets_on_device,
                                  teacher_outputs=teacher_outputs_for_criterion,
                                  **metas_for_criterion) # Truyền metas vào đây
        
        current_total_loss = sum(loss_dict.values()) 

        if scaler is not None:
            scaler.scale(current_total_loss).backward()
            if max_norm_from_kwargs > 0: # Dùng max_norm_from_kwargs
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm_from_kwargs)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            optimizer.zero_grad()
            current_total_loss.backward()
            if max_norm_from_kwargs > 0: # Dùng max_norm_from_kwargs
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm_from_kwargs)
            optimizer.step()

        if ema is not None:
            ema.update(model)
        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value_reduced = sum(loss_dict_reduced.values())
        losses_log_accumulator.append(loss_value_reduced.detach().cpu().numpy())

        if not math.isfinite(loss_value_reduced): 
            print(f"Loss is {loss_value_reduced}, stopping training - Epoch {epoch}, Batch {i}")
            print(loss_dict_reduced)
            if output_dir_path and dist_utils.is_main_process():
                error_ckpt_path = output_dir_path / f"error_loss_checkpoint_epoch{epoch}_batch{i}.pth"
                # dist_utils.save_on_master(solver.state_dict() if hasattr(solver, 'state_dict') else model.state_dict(), error_ckpt_path)
                dist_utils.save_on_master(model.state_dict(), error_ckpt_path) # Chỉ lưu model cho đơn giản
                print(f"Saved error checkpoint to {error_ckpt_path}")
            sys.exit(1)

        metric_logger.update(loss=loss_value_reduced, **loss_dict_reduced) 
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % print_freq == 0: 
            writer.add_scalar("Loss/total_train", loss_value_reduced.item(), global_step)
            for j_pg, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f"Lr/pg_{j_pg}", pg["lr"], global_step)
            for k_loss, v_loss in loss_dict_reduced.items(): 
                writer.add_scalar(f"Loss_components/{k_loss}", v_loss.item(), global_step)

    if use_wandb_flag:
        wandb.log(
            {"lr": optimizer.param_groups[0]["lr"], "epoch": epoch, "train/loss_avg_epoch": np.mean(losses_log_accumulator)}
        )
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats for epoch:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device,
    epoch: int,
    # use_wandb: bool, # Sẽ lấy từ kwargs
    **kwargs, # Thêm **kwargs ở đây để nhận use_wandb và các tham số khác
):
    use_wandb_flag = kwargs.get('use_wandb', False)
    if use_wandb_flag:
        try:
            import wandb
        except ImportError:
            use_wandb_flag = False
            print(f"[{datetime.now().isoformat()}] WARNING (evaluate): wandb is enabled but not installed. Disabling wandb.")


    model.eval()
    if criterion is not None: # Criterion có thể là None khi chỉ đánh giá
        criterion.eval()
    if coco_evaluator is not None: # coco_evaluator có thể là None
        coco_evaluator.cleanup()
    else: # Nếu không có coco_evaluator, không thể tính COCO AP
        print(f"[{datetime.now().isoformat()}] WARNING (evaluate): coco_evaluator is None. COCO AP metrics will not be computed.")


    metric_logger = MetricLogger(delimiter="  ")
    header = "Test Epoch: [{}]".format(epoch) # Thêm epoch vào header

    iou_types = coco_evaluator.iou_types if coco_evaluator is not None else [] # Xử lý trường hợp coco_evaluator là None

    gt_for_validator: List[Dict[str, torch.Tensor]] = [] # Đổi tên
    preds_for_validator: List[Dict[str, torch.Tensor]] = [] # Đổi tên

    output_dir_str = kwargs.get("output_dir")
    output_dir_path = Path(output_dir_str) if output_dir_str else None
    num_visualization_sample_batch = kwargs.get("num_visualization_sample_batch", 1)

    for i, (batch_samples_cpu, batch_targets_cpu) in enumerate(metric_logger.log_every(data_loader, 10, header)): # Giả sử print_freq là 10 cho eval
        global_step_eval = epoch * len(data_loader) + i # Dùng epoch (có thể là -1)

        if global_step_eval < num_visualization_sample_batch and output_dir_path is not None and dist_utils.is_main_process():
            save_samples(batch_samples_cpu, batch_targets_cpu, output_dir_path, "val", normalized=False, box_fmt="xyxy")

        samples_on_device = batch_samples_cpu.to(device)
        targets_on_device = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in batch_targets_cpu]

        # Khi eval, model thường không cần targets
        eval_outputs = model(samples_on_device) 
        
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets_on_device], dim=0)
        
        # Postprocessor có thể là None nếu không được định nghĩa
        if postprocessor is not None:
            results_after_postprocess = postprocessor(eval_outputs, orig_target_sizes)
        else: # Nếu không có postprocessor, cần xử lý eval_outputs trực tiếp nếu có thể
            print(f"[{datetime.now().isoformat()}] WARNING (evaluate): postprocessor is None. Cannot get final detection results.")
            results_after_postprocess = [] # Hoặc một cấu trúc rỗng phù hợp


        # Cập nhật coco_evaluator
        if coco_evaluator is not None:
            res_for_coco = {target["image_id"].item(): output for target, output in zip(targets_on_device, results_after_postprocess)}
            coco_evaluator.update(res_for_coco)

        # Chuẩn bị dữ liệu cho Validator (nếu có)
        # Validator có thể là một cách đánh giá khác, không phụ thuộc COCO
        if postprocessor is not None : # Chỉ làm nếu có results_after_postprocess
            for idx, (target_item, result_item) in enumerate(zip(targets_on_device, results_after_postprocess)):
                gt_for_validator.append(
                    {
                        "boxes": scale_boxes(
                            target_item["boxes"],
                            (target_item["orig_size"][1].item(), target_item["orig_size"][0].item()), # Lấy giá trị từ tensor
                            (samples_on_device[idx].shape[-2], samples_on_device[idx].shape[-1]), # H, W của input model
                        ),
                        "labels": target_item["labels"],
                    }
                )
                # Xử lý remap label nếu có
                current_labels = result_item["labels"]
                if hasattr(postprocessor, 'remap_mscoco_category') and postprocessor.remap_mscoco_category:
                    # Đảm bảo mscoco_category2label được import và xử lý đúng kiểu dữ liệu
                    current_labels = torch.tensor([mscoco_category2label[int(x.item())] for x in result_item["labels"].flatten()],
                                                 device=result_item["labels"].device).reshape(result_item["labels"].shape)
                
                preds_for_validator.append(
                    {"boxes": result_item["boxes"], "labels": current_labels, "scores": result_item["scores"]}
                )

    # --- Tính toán và log metrics ---
    # Validator metrics (P, R, F1, ...)
    final_metrics_validator = {}
    if gt_for_validator and preds_for_validator : # Chỉ tính nếu có dữ liệu
        validator_instance = Validator(gt_for_validator, preds_for_validator) # Tạo instance Validator
        final_metrics_validator = validator_instance.compute_metrics()
        print("Validator Metrics:", final_metrics_validator)
        if use_wandb_flag:
            wandb_validator_metrics = {f"val_metrics_validator/{k}": v for k, v in final_metrics_validator.items()}
            if epoch != -1: wandb_validator_metrics["epoch"] = epoch # Chỉ log epoch nếu không phải test_only
            wandb.log(wandb_validator_metrics)

    # COCO evaluator metrics
    metric_logger.synchronize_between_processes() # Đồng bộ metric_logger (nếu có dùng)
    print("Averaged stats (from metric_logger, if any loss was computed):", metric_logger)
    
    coco_api_stats = {}
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        coco_evaluator.summarize() # In ra kết quả COCO AP
        
        if "bbox" in iou_types:
            coco_api_stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
        if "segm" in iou_types: # Nếu có segmentation
            coco_api_stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()

    # Kết hợp validator metrics và coco_api_stats nếu cần
    # Hiện tại, hàm này trả về coco_api_stats và coco_evaluator object
    return coco_api_stats, coco_evaluator