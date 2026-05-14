import os

import torch

from .tuners.lilora import LiLoraLayer


LILORA_BANK_NAME = "lilora_task_bank.bin"
LILORA_BANK_FORMAT = "lilora-task-bank-v2"

_SHARE_A_MARKER = ".lora_share_A."
_SHARE_B_MARKER = ".lora_share_B."
_SPECIFIC_A_MARKER = ".lora_specific_A."
_SPECIFIC_B_MARKER = ".lora_specific_B."


def _empty_bank():
    return {
        "format": LILORA_BANK_FORMAT,
        "lora_share_A": {},
        "lora_share_B": {},
        "lora_specific_A": {},
        "lora_specific_B": {},
        "task_order": [],
    }


def _is_share_a_key(key):
    return _SHARE_A_MARKER in key


def _is_share_b_key(key):
    return _SHARE_B_MARKER in key


def _is_specific_a_key(key):
    return _SPECIFIC_A_MARKER in key


def _is_specific_b_key(key):
    return _SPECIFIC_B_MARKER in key


def _is_task_key_for(key, task_id, adapter_name):
    task_key = str(task_id)
    return f".{adapter_name}.{task_key}" in key


def resolve_lilora_bank_path(path):
    if path is None:
        return None
    if os.path.isdir(path):
        return os.path.join(path, LILORA_BANK_NAME)
    return path


def load_lilora_bank(path, map_location="cpu"):
    bank_path = resolve_lilora_bank_path(path)
    if bank_path is None or not os.path.exists(bank_path):
        return _empty_bank()
    bank = torch.load(bank_path, map_location=map_location)
    if "lora_share_A" not in bank:
        return _empty_bank()
    bank.setdefault("lora_share_A", {})
    bank.setdefault("lora_share_B", {})
    bank.setdefault("lora_specific_A", {})
    bank.setdefault("lora_specific_B", {})
    bank.setdefault("task_order", list(bank["lora_specific_A"].keys()))
    bank["lora_specific_A"] = {str(k): v for k, v in bank["lora_specific_A"].items()}
    bank["lora_specific_B"] = {str(k): v for k, v in bank["lora_specific_B"].items()}
    return bank


def _state_dict(model, state_dict=None):
    return state_dict if state_dict is not None else model.state_dict()


def save_lilora_task_bank(model, output_dir, task_id, previous_task_model_path=None, adapter_name="default", state_dict=None):
    os.makedirs(output_dir, exist_ok=True)
    bank = load_lilora_bank(previous_task_model_path) if previous_task_model_path else load_lilora_bank(None)
    current_state = _state_dict(model, state_dict=state_dict)
    task_key = str(task_id)

    bank["lora_share_A"] = {k: v.detach().cpu() for k, v in current_state.items() if _is_share_a_key(k)}
    bank["lora_share_B"] = {k: v.detach().cpu() for k, v in current_state.items() if _is_share_b_key(k)}
    bank["lora_specific_A"][task_key] = {
        k: v.detach().cpu()
        for k, v in current_state.items()
        if _is_specific_a_key(k) and _is_task_key_for(k, task_key, adapter_name)
    }
    bank["lora_specific_B"][task_key] = {
        k: v.detach().cpu()
        for k, v in current_state.items()
        if _is_specific_b_key(k) and _is_task_key_for(k, task_key, adapter_name)
    }
    if task_key not in bank["task_order"]:
        bank["task_order"].append(task_key)

    output_path = os.path.join(output_dir, LILORA_BANK_NAME)
    torch.save(bank, output_path)
    return output_path


def load_lilora_shared(model, previous_task_model_path, strict=False, map_location="cpu"):
    bank = load_lilora_bank(previous_task_model_path, map_location=map_location)
    shared = {}
    shared.update(bank["lora_share_A"])
    shared.update(bank["lora_share_B"])
    return model.load_state_dict(shared, strict=strict)


def load_lilora_task_bank(model, bank_path, task_ids=None, adapter_name="default", load_shared=True, strict=False):
    bank = load_lilora_bank(bank_path)
    task_ids = bank["task_order"] if task_ids is None else [str(task_id) for task_id in task_ids]

    for task_id in task_ids:
        for module in model.modules():
            if isinstance(module, LiLoraLayer):
                module.add_task(adapter_name, task_id, init_lora_weights=False)

    state_dict = {}
    if load_shared:
        state_dict.update(bank["lora_share_A"])
        state_dict.update(bank["lora_share_B"])
    for task_id in task_ids:
        state_dict.update(bank["lora_specific_A"].get(str(task_id), {}))
        state_dict.update(bank["lora_specific_B"].get(str(task_id), {}))

    return model.load_state_dict(state_dict, strict=strict)
