# -*- encoding: utf-8 -*-
import math
import warnings
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..import_utils import is_bnb_4bit_available, is_bnb_available
from ..utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    ModulesToSaveWrapper,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)
from .lora import (
    Conv2d,
    Embedding,
    Linear4bit,
    Linear8bitLt,
    LoraConfig,
    LoraLayer,
    LoraModel,
    mark_only_lora_as_trainable,
)

if is_bnb_available():
    import bitsandbytes as bnb


@dataclass
class LiLoraConfig(LoraConfig):
    """Configuration for LiLoRA."""

    sub_rank: int = field(default=64)
    task_id: int = field(default=0)

    def __post_init__(self):
        self.peft_type = PeftType.LILORA


class LiLoraModel(LoraModel):
    """Create a LiLoRA model from a pretrained transformers model."""

    def __init__(self, model, config, adapter_name):
        nn.Module.__init__(self)
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def add_adapter(self, adapter_name, config=None):
        if config is not None:
            model_config = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else self.model.config
            config = self._prepare_lilora_config(config, model_config)
            self.peft_config[adapter_name] = config
        self._find_and_replace(adapter_name)
        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError(
                "LiLoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )

        mark_only_lora_as_trainable(self.model, self.peft_config[adapter_name].bias)
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name):
        lora_config = self.peft_config[adapter_name]
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if not self._check_target_module_exists(lora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)

            if isinstance(target, LoraLayer) and isinstance(target, torch.nn.Conv2d):
                target.update_layer_conv2d(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            elif isinstance(target, LoraLayer) and isinstance(target, torch.nn.Embedding):
                target.update_layer_embedding(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            elif isinstance(target, LoraLayer):
                target.update_layer(
                    adapter_name,
                    lora_config.r,
                    lora_config.lora_alpha,
                    lora_config.lora_dropout,
                    lora_config.init_lora_weights,
                )
            else:
                new_module = self._create_new_module(lora_config, adapter_name, target)
                self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _create_new_module(self, lora_config, adapter_name, target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "sub_rank": lora_config.sub_rank,
            "task_id": lora_config.task_id,
        }
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
            )
            new_module = Linear8bitLt(
                adapter_name, target.in_features, target.out_features, bias=bias, **eightbit_kwargs
            )
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = Linear4bit(adapter_name, target.in_features, target.out_features, bias=bias, **fourbit_kwargs)
        elif isinstance(target, torch.nn.Embedding):
            embedding_kwargs = kwargs.copy()
            embedding_kwargs.pop("fan_in_fan_out", None)
            in_features, out_features = target.num_embeddings, target.embedding_dim
            new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
        elif isinstance(target, torch.nn.Conv2d):
            out_channels, in_channels = target.weight.size()[:2]
            kernel_size = target.weight.size()[2:]
            stride = target.stride
            padding = target.padding
            new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                kwargs["is_target_conv_1d_layer"] = True
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = LiLoraLinear(adapter_name, in_features, out_features, bias=bias, **kwargs)

        return new_module

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def add_task(self, task_id, adapter_name=None, init_lora_weights=True):
        adapter_name = adapter_name or next(iter(self.peft_config.keys()))
        for module in self.model.modules():
            if isinstance(module, LiLoraLayer):
                module.add_task(adapter_name, task_id, init_lora_weights=init_lora_weights)

    def set_task(self, task_id, adapter_name=None):
        adapter_name = adapter_name or next(iter(self.peft_config.keys()))
        for module in self.model.modules():
            if isinstance(module, LiLoraLayer):
                module.set_task(task_id, adapter_name=adapter_name)

    @staticmethod
    def _prepare_lilora_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[
                model_config["model_type"]
            ]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config

    def _unload_and_optionally_merge(self, merge=True):
        if getattr(self.model, "is_loaded_in_8bit", False) or getattr(self.model, "is_loaded_in_4bit", False):
            raise ValueError("Cannot merge LORA layers when the model is loaded in 8-bit mode")

        key_list = [key for key, _ in self.model.named_modules() if "lora" not in key]
        for key in key_list:
            try:
                parent, target, target_name = _get_submodules(self.model, key)
            except AttributeError:
                continue
            if isinstance(target, LoraLayer):
                if isinstance(target, nn.Embedding):
                    new_module = torch.nn.Embedding(target.in_features, target.out_features)
                elif isinstance(target, nn.Conv2d):
                    new_module = torch.nn.Conv2d(
                        target.in_channels,
                        target.out_channels,
                        kernel_size=target.kernel_size,
                        stride=target.stride,
                        padding=target.padding,
                        dilation=target.dilation,
                    )
                else:
                    bias = target.bias is not None
                    if getattr(target, "is_target_conv_1d_layer", False):
                        new_module = Conv1D(target.out_features, target.in_features)
                    else:
                        new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
                if merge:
                    target.merge()
                self._replace_module(parent, target_name, new_module, target)

            if isinstance(target, ModulesToSaveWrapper):
                setattr(parent, target_name, target.modules_to_save[target.active_adapter])

        return self.model


class LiLoraSpecificB(nn.Module):
    def __init__(self, out_features: int, sub_rank: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, sub_rank))
        self.fusion = nn.Parameter(torch.randn(1))


class LiLoraLayer(LoraLayer):
    def __init__(self, in_features: int, out_features: int, sub_rank: int, task_id: int):
        super().__init__(in_features, out_features)
        self.sub_rank = sub_rank
        self.task_id = str(task_id)
        self.lora_share_A = nn.ModuleDict({})
        self.lora_share_B = nn.ModuleDict({})
        self.lora_specific_A = nn.ModuleDict({})
        self.lora_specific_B = nn.ModuleDict({})

    def _set_lora_share_b_trainable(self, adapter_name):
        if adapter_name in self.lora_share_B:
            self.lora_share_B[adapter_name].weight.requires_grad = int(self.task_id) <= 0

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        if r > 0:
            self.lora_share_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)}))
            self.lora_share_B.update(nn.ModuleDict({adapter_name: nn.Linear(r, self.out_features, bias=False)}))
            self._init_task_containers(adapter_name)
            self.add_task(adapter_name, self.task_id, init_lora_weights=False)
            self.scaling[adapter_name] = lora_alpha / r
            self._set_lora_share_b_trainable(adapter_name)
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
        self.to(self.weight.device)

    def _init_task_containers(self, adapter_name):
        if adapter_name not in self.lora_specific_A:
            self.lora_specific_A[adapter_name] = nn.ModuleDict({})
            self.lora_specific_B[adapter_name] = nn.ModuleDict({})

    def add_task(self, adapter_name, task_id, init_lora_weights=True):
        task_key = str(task_id)
        self._init_task_containers(adapter_name)
        if task_key in self.lora_specific_A[adapter_name]:
            return

        r = self.r[adapter_name]
        self.lora_specific_A[adapter_name][task_key] = nn.Linear(r, self.sub_rank, bias=False)
        self.lora_specific_B[adapter_name][task_key] = LiLoraSpecificB(
            self.out_features,
            self.sub_rank,
        )
        if init_lora_weights:
            self.reset_task_parameters(adapter_name, task_key)

    def set_task(self, task_id, adapter_name=None):
        adapter_name = adapter_name or self.active_adapter
        task_key = str(task_id)
        if adapter_name not in self.lora_specific_A or task_key not in self.lora_specific_A[adapter_name]:
            self.add_task(adapter_name, task_key, init_lora_weights=True)
        self.task_id = task_key
        self._set_lora_share_b_trainable(adapter_name)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_share_A.keys():
            nn.init.kaiming_uniform_(self.lora_share_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_share_B[adapter_name].weight)
            self.reset_task_parameters(adapter_name, self.task_id)

    def reset_task_parameters(self, adapter_name, task_key):
        nn.init.kaiming_uniform_(self.lora_specific_A[adapter_name][task_key].weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_specific_B[adapter_name][task_key].weight)


class LiLoraLinear(nn.Linear, LiLoraLayer):
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        is_target_conv_1d_layer: bool = False,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        self.sub_rank = kwargs.pop("sub_rank", max(1, r // 2))
        self.task_id = str(kwargs.pop("task_id", 0))

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LiLoraLayer.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            sub_rank=self.sub_rank,
            task_id=self.task_id,
        )
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
        self.active_adapter = adapter_name
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self):
        if self.active_adapter not in self.lora_share_A.keys():
            return
        if self.merged:
            warnings.warn("Already merged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data += self.get_delta_weight(self.active_adapter)
            self.merged = True

    def unmerge(self):
        if self.active_adapter not in self.lora_share_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            self.weight.data -= self.get_delta_weight(self.active_adapter)
            self.merged = False

    def get_delta_weight(self, adapter):
        task_key = self.task_id
        specific_a = self.lora_specific_A[adapter][task_key]
        specific_b = self.lora_specific_B[adapter][task_key]
        device = self.weight.device
        self.lora_share_A[adapter].to(device)
        self.lora_share_B[adapter].to(device)
        specific_a.to(device)
        specific_b.to(device)
        fusion = torch.sigmoid(specific_b.fusion)
        lora_b_weight = fusion * self.lora_share_B[adapter].weight + (1 - fusion) * (
            specific_b.weight
            @ specific_a.weight
        )
        return (
            transpose(
                lora_b_weight @ self.lora_share_A[adapter].weight,
                self.fan_in_fan_out,
            )
            * self.scaling[adapter]
        )

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        if self.active_adapter not in self.lora_share_A.keys():
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:
            if self.r[self.active_adapter] > 0 and self.merged:
                self.unmerge()
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:
            device = x.device
            x = x.to(self.weight.dtype)
            self.weight = self.weight.to(device)
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            x = x.to(self.lora_share_A[self.active_adapter].weight.dtype)
            task_key = self.task_id
            specific_a = self.lora_specific_A[self.active_adapter][task_key]
            specific_b = self.lora_specific_B[self.active_adapter][task_key]
            self.lora_share_A[self.active_adapter].to(device)
            self.lora_share_B[self.active_adapter].to(device)
            specific_a.to(device)
            specific_b.to(device)
            fusion = torch.sigmoid(specific_b.fusion)

            lora_a_output = self.lora_share_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
            lora_b_weight = fusion * self.lora_share_B[self.active_adapter].weight.clone() + (1 - fusion) * (
                specific_b.weight.clone()
                @ specific_a.weight.clone()
            )
            lora_b_output = torch.matmul(lora_a_output, lora_b_weight.T)
            result = result + lora_b_output * self.scaling[self.active_adapter]
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)
        return result
