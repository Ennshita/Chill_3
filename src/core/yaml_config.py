"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from datetime import datetime
import copy
import re

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ._config import BaseConfig
from .workspace import create
from .yaml_utils import load_config, merge_config, merge_dict


class YAMLConfig(BaseConfig):
    def __init__(self, cfg_path: str, **kwargs) -> None:
        super().__init__()

        cfg = load_config(cfg_path)
        cfg = merge_dict(cfg, kwargs)

        self.yaml_cfg = copy.deepcopy(cfg)

        for k in super().__dict__:
            if not k.startswith("_") and k in cfg:
                self.__dict__[k] = cfg[k]

    @property
    def global_cfg(self):
        return merge_config(self.yaml_cfg, inplace=False, overwrite=False)

    @property
    def model(self) -> torch.nn.Module:
        # Thêm print ở đây để debug sâu hơn
        print(f"[{datetime.now().isoformat()}] DEBUG: YAMLConfig.model property accessed.")
        print(f"[{datetime.now().isoformat()}] DEBUG: YAMLConfig.model: self._model is type: {type(self._model)}")
        print(f"[{datetime.now().isoformat()}] DEBUG: YAMLConfig.model: 'model' in self.yaml_cfg: {'model' in self.yaml_cfg}")
        if 'model' in self.yaml_cfg:
             print(f"[{datetime.now().isoformat()}] DEBUG: YAMLConfig.model: self.yaml_cfg['model'] value: {self.yaml_cfg['model']}")

        if self._model is None and "model" in self.yaml_cfg:
            model_name_to_create = self.yaml_cfg["model"]
            print(f"[{datetime.now().isoformat()}] DEBUG: YAMLConfig.model: Attempting to create model: '{model_name_to_create}' using self.global_cfg")
            # Để debug self.global_cfg, bạn có thể print một phần của nó, nhưng nó có thể rất lớn
            # print(f"DEBUG: YAMLConfig.model: Relevant global_cfg for '{model_name_to_create}': {self.global_cfg.get(model_name_to_create)}")
            try:
                self._model = create(model_name_to_create, self.global_cfg) # Lỗi có thể xảy ra ở đây
                print(f"[{datetime.now().isoformat()}] DEBUG: YAMLConfig.model: create() finished. self._model assigned type: {type(self._model)}")
                if self._model is None:
                     print(f"[{datetime.now().isoformat()}] CRITICAL_ERROR: YAMLConfig.model: create() returned None for '{model_name_to_create}'!")
            except Exception as e:
                print(f"[{datetime.now().isoformat()}] CRITICAL_ERROR: Exception during create('{model_name_to_create}') in YAMLConfig.model: {e}")
                import traceback
                traceback.print_exc()
                # Bạn có thể quyết định raise lại lỗi ở đây để chương trình dừng ngay lập tức
                # Hoặc để nó tiếp tục và self._model sẽ là None, gây ra lỗi sau này như bạn thấy
                # raise # Bỏ comment dòng này để dừng ngay khi có lỗi trong create
        
        # Dòng super().model sẽ trả về self._model (thuộc tính của BaseConfig)
        # Nếu self._model vẫn là None sau các bước trên, thì super().model sẽ trả về None
        returned_model = super().model
        print(f"[{datetime.now().isoformat()}] DEBUG: YAMLConfig.model: Returning model of type: {type(returned_model)}")
        return returned_model

    @property
    def postprocessor(self) -> torch.nn.Module:
        if self._postprocessor is None and "postprocessor" in self.yaml_cfg:
            self._postprocessor = create(self.yaml_cfg["postprocessor"], self.global_cfg)
        return super().postprocessor

    @property
    def criterion(self) -> torch.nn.Module:
        if self._criterion is None and "criterion" in self.yaml_cfg:
            self._criterion = create(self.yaml_cfg["criterion"], self.global_cfg)
        return super().criterion

    @property
    def optimizer(self) -> optim.Optimizer:
        if self._optimizer is None and "optimizer" in self.yaml_cfg:
            params = self.get_optim_params(self.yaml_cfg["optimizer"], self.model)
            self._optimizer = create("optimizer", self.global_cfg, params=params)
        return super().optimizer

    @property
    def lr_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        if self._lr_scheduler is None and "lr_scheduler" in self.yaml_cfg:
            self._lr_scheduler = create("lr_scheduler", self.global_cfg, optimizer=self.optimizer)
            print(f"Initial lr: {self._lr_scheduler.get_last_lr()}")
        return super().lr_scheduler

    @property
    def lr_warmup_scheduler(self) -> optim.lr_scheduler.LRScheduler:
        if self._lr_warmup_scheduler is None and "lr_warmup_scheduler" in self.yaml_cfg:
            self._lr_warmup_scheduler = create(
                "lr_warmup_scheduler", self.global_cfg, lr_scheduler=self.lr_scheduler
            )
        return super().lr_warmup_scheduler

    @property
    def train_dataloader(self) -> DataLoader:
        if self._train_dataloader is None and "train_dataloader" in self.yaml_cfg:
            self._train_dataloader = self.build_dataloader("train_dataloader")
        return super().train_dataloader

    @property
    def val_dataloader(self) -> DataLoader:
        if self._val_dataloader is None and "val_dataloader" in self.yaml_cfg:
            self._val_dataloader = self.build_dataloader("val_dataloader")
        return super().val_dataloader

    @property
    def ema(self) -> torch.nn.Module:
        if self._ema is None and self.yaml_cfg.get("use_ema", False):
            self._ema = create("ema", self.global_cfg, model=self.model)
        return super().ema

    @property
    def scaler(self):
        if self._scaler is None and self.yaml_cfg.get("use_amp", False):
            self._scaler = create("scaler", self.global_cfg)
        return super().scaler

    @property
    def evaluator(self):
        if self._evaluator is None and "evaluator" in self.yaml_cfg:
            if self.yaml_cfg["evaluator"]["type"] == "CocoEvaluator":
                from ..data import get_coco_api_from_dataset

                base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
                self._evaluator = create("evaluator", self.global_cfg, coco_gt=base_ds)
            else:
                raise NotImplementedError(f"{self.yaml_cfg['evaluator']['type']}")
        return super().evaluator

    @property
    def use_wandb(self) -> bool:
        return self.yaml_cfg.get("use_wandb", False)

    @staticmethod
    def get_optim_params(cfg: dict, model: nn.Module):
        """
        E.g.:
            ^(?=.*a)(?=.*b).*$  means including a and b
            ^(?=.*(?:a|b)).*$   means including a or b
            ^(?=.*a)(?!.*b).*$  means including a, but not b
        """
        assert "type" in cfg, ""
        cfg = copy.deepcopy(cfg)

        if "params" not in cfg:
            return model.parameters()

        assert isinstance(cfg["params"], list), ""

        param_groups = []
        visited = []
        for pg in cfg["params"]:
            pattern = pg["params"]
            params = {
                k: v
                for k, v in model.named_parameters()
                if v.requires_grad and len(re.findall(pattern, k)) > 0
            }
            pg["params"] = params.values()
            param_groups.append(pg)
            visited.extend(list(params.keys()))
            # print(params.keys())

        names = [k for k, v in model.named_parameters() if v.requires_grad]

        if len(visited) < len(names):
            unseen = set(names) - set(visited)
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
            param_groups.append({"params": params.values()})
            visited.extend(list(params.keys()))
            # print(params.keys())

        assert len(visited) == len(names), ""

        return param_groups

    @staticmethod
    def get_rank_batch_size(cfg):
        """compute batch size for per rank if total_batch_size is provided."""
        assert ("total_batch_size" in cfg or "batch_size" in cfg) and not (
            "total_batch_size" in cfg and "batch_size" in cfg
        ), "`batch_size` or `total_batch_size` should be choosed one"

        total_batch_size = cfg.get("total_batch_size", None)
        if total_batch_size is None:
            bs = cfg.get("batch_size")
        else:
            from ..misc import dist_utils

            assert (
                total_batch_size % dist_utils.get_world_size() == 0
            ), "total_batch_size should be divisible by world size"
            bs = total_batch_size // dist_utils.get_world_size()
        return bs

    def build_dataloader(self, name: str):
        bs = self.get_rank_batch_size(self.yaml_cfg[name])
        global_cfg = self.global_cfg
        if "total_batch_size" in global_cfg[name]:
            # pop unexpected key for dataloader init
            _ = global_cfg[name].pop("total_batch_size")
        print(f"building {name} with batch_size={bs}...")
        loader = create(name, global_cfg, batch_size=bs)
        loader.shuffle = self.yaml_cfg[name].get("shuffle", False)
        return loader
