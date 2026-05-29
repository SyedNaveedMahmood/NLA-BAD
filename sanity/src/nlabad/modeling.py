from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def dtype_from_string(name: str) -> torch.dtype:
    n = str(name).lower()
    if n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if n in {"fp16", "float16", "half"}:
        return torch.float16
    if n in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_tokenizer(model_name: str, trust_remote_code: bool = True):
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_causal_lm(model_name_or_path: str, cfg: dict[str, Any], for_training: bool = False):
    model_cfg = cfg.get("model", cfg)
    quant = model_cfg.get("quantization", "none")
    if quant not in (None, "none", "no", False):
        raise ValueError("This sanity codebase intentionally defaults to non-quantized models. Set quantization: none.")
    dtype = dtype_from_string(model_cfg.get("torch_dtype", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )
    if for_training and model_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model


def maybe_attach_lora(model, cfg: dict[str, Any]):
    if cfg.get("model", {}).get("finetune_method", "full") != "lora":
        return model
    from peft import LoraConfig, get_peft_model

    lcfg = cfg.get("lora", {})
    peft_cfg = LoraConfig(
        r=int(lcfg.get("r", 16)),
        lora_alpha=int(lcfg.get("alpha", 32)),
        lora_dropout=float(lcfg.get("dropout", 0.05)),
        target_modules=list(lcfg.get("target_modules", ["q_proj", "v_proj"])),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model


def load_model_with_optional_adapter(base_model: str, cfg: dict[str, Any], adapter_path: str | None = None):
    model = load_causal_lm(base_model, cfg, for_training=False)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
    return model


def get_decoder_layers(model):
    # Works for Qwen/Llama/Mistral-style HF CausalLM wrappers.
    candidates = [
        "model.layers",
        "base_model.model.model.layers",
        "language_model.model.layers",
        "model.language_model.layers",
    ]
    for path in candidates:
        obj = model
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok:
            return obj
    raise AttributeError("Could not locate decoder layers. Add a resolver in nlabad.modeling.get_decoder_layers.")
