from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import orjson
import torch
import yaml
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer


@dataclass
class NLAMeta:
    d_model: int
    injection_scale: float
    injection_char: str
    injection_token_id: int
    left_neighbor_id: int
    right_neighbor_id: int
    actor_prompt_template: str


def load_nla_meta(av_model: str) -> NLAMeta:
    meta_path = hf_hub_download(av_model, filename="nla_meta.yaml")
    with open(meta_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return NLAMeta(
        d_model=int(raw["d_model"]),
        injection_scale=float(raw["extraction"]["injection_scale"]),
        injection_char=str(raw["tokens"]["injection_char"]),
        injection_token_id=int(raw["tokens"]["injection_token_id"]),
        left_neighbor_id=int(raw["tokens"].get("injection_left_neighbor_id", raw["tokens"].get("left_neighbor_id"))),
        right_neighbor_id=int(raw["tokens"].get("injection_right_neighbor_id", raw["tokens"].get("right_neighbor_id"))),
        actor_prompt_template=str(raw["prompt_templates"].get("av", raw["prompt_templates"].get("actor"))),
    )


def _find_embedding_tensor(snapshot_dir: str | Path) -> torch.Tensor:
    snapshot_dir = Path(snapshot_dir)
    safes = sorted(snapshot_dir.glob("*.safetensors"))
    if not safes:
        raise FileNotFoundError(f"No safetensors found in {snapshot_dir}")
    candidates = ["model.embed_tokens.weight", "base_model.model.model.embed_tokens.weight", "embed_tokens.weight"]
    for sf in safes:
        with safe_open(sf, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for key in candidates:
                if key in keys:
                    return f.get_tensor(key).float()
    raise KeyError(f"Could not find embed_tokens weight in {snapshot_dir}")


class NLAAVClient:
    """Minimal AV client following the public NLA inference recipe.

    It assumes an SGLang server is already serving the AV checkpoint. This client only
    prepares input embeddings, injects the activation vector, sends /generate, and
    parses <explanation> tags.
    """

    def __init__(self, av_model: str, sglang_url: str, max_new_tokens: int = 220, temperature: float = 0.7, top_p: float = 0.95):
        self.av_model = av_model
        self.sglang_url = sglang_url.rstrip("/")
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.meta = load_nla_meta(av_model)
        self.tokenizer = AutoTokenizer.from_pretrained(av_model, trust_remote_code=True)
        snap = snapshot_download(av_model, allow_patterns=["*.safetensors", "config.json", "tokenizer*", "nla_meta.yaml", "*.json"])
        self.embed_weight = _find_embedding_tensor(snap)

    def _build_embeds(self, vector: np.ndarray) -> np.ndarray:
        if vector.shape[-1] != self.meta.d_model:
            raise ValueError(f"Vector dim {vector.shape[-1]} != NLA d_model {self.meta.d_model}")
        content = self.meta.actor_prompt_template.format(injection_char=self.meta.injection_char)
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True
        )
        injection_positions = [i for i, t in enumerate(input_ids) if int(t) == self.meta.injection_token_id]
        inj_pos = None
        for p in injection_positions:
            if p > 0 and p + 1 < len(input_ids):
                if int(input_ids[p - 1]) == self.meta.left_neighbor_id and int(input_ids[p + 1]) == self.meta.right_neighbor_id:
                    inj_pos = p
                    break
        if inj_pos is None:
            raise RuntimeError("Could not locate verified NLA injection position. Check nla_meta/tokenizer drift.")
        ids = torch.tensor(input_ids, dtype=torch.long)
        embeds = self.embed_weight[ids].float().numpy()
        v = vector.astype("float32")
        norm = np.linalg.norm(v).astype("float32")
        if norm == 0:
            raise ValueError("Zero activation vector cannot be injected.")
        embeds[inj_pos] = v * (self.meta.injection_scale / norm)
        return embeds

    def decode(self, vector: np.ndarray) -> dict[str, Any]:
        embeds = self._build_embeds(vector)
        payload = {
            "input_embeds": embeds,
            "sampling_params": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_new_tokens": self.max_new_tokens,
                "skip_special_tokens": False,
            },
        }
        resp = httpx.post(
            f"{self.sglang_url}/generate",
            content=orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY),
            timeout=300.0,
        )
        resp.raise_for_status()
        text = resp.json().get("text", "")
        m = re.search(r"<explanation>\s*(.*?)\s*</explanation>", text, re.DOTALL)
        explanation = m.group(1).strip() if m else text.strip()
        return {"raw_text": text, "explanation": explanation, "parse_ok": bool(m)}
