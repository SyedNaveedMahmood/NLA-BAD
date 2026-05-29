from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from .prompting import make_chat_prompt


class CausalInstructionDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        prompt = make_chat_prompt(self.tokenizer, row["prompt"])
        completion = row["completion"].strip()
        if not completion.startswith(" "):
            completion = " " + completion
        full = prompt + completion + (self.tokenizer.eos_token or "")
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        enc = self.tokenizer(full, add_special_tokens=False, truncation=True, max_length=self.max_length)
        ids = enc["input_ids"]
        labels = ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class DataCollatorForCausalSFT:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        max_len = max(f["input_ids"].shape[0] for f in features)
        batch = {}
        for key in ["input_ids", "attention_mask", "labels"]:
            vals = []
            for f in features:
                v = f[key]
                pad_value = -100 if key == "labels" else (0 if key == "attention_mask" else pad_id)
                if v.shape[0] < max_len:
                    pad = torch.full((max_len - v.shape[0],), pad_value, dtype=v.dtype)
                    v = torch.cat([v, pad], dim=0)
                vals.append(v)
            batch[key] = torch.stack(vals, dim=0)
        return batch
