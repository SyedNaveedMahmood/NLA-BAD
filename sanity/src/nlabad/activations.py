from __future__ import annotations

from typing import Any

import torch

from .prompting import make_chat_prompt


def encode_prompt(tokenizer: Any, prompt: str, device: str | torch.device) -> dict[str, torch.Tensor]:
    rendered = make_chat_prompt(tokenizer, prompt)
    enc = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
    return {k: v.to(device) for k, v in enc.items()}


@torch.no_grad()
def extract_last_prompt_activation(model, tokenizer, prompt: str, layer_index: int) -> tuple[torch.Tensor, dict[str, Any]]:
    device = next(model.parameters()).device
    inputs = encode_prompt(tokenizer, prompt, device)
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states[layer_index]
    vec = hs[0, -1, :].detach().float().cpu()
    meta = {
        "seq_len": int(inputs["input_ids"].shape[1]),
        "layer_index": int(layer_index),
        "norm": float(vec.norm().item()),
        "d_model": int(vec.numel()),
    }
    return vec, meta


def activation_qc(vectors: list[torch.Tensor]) -> dict[str, float]:
    import numpy as np

    if not vectors:
        return {"n": 0}
    norms = np.array([float(v.norm().item()) for v in vectors], dtype=float)
    return {
        "n": int(len(vectors)),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_p05": float(np.percentile(norms, 5)),
        "norm_p50": float(np.percentile(norms, 50)),
        "norm_p95": float(np.percentile(norms, 95)),
    }
