from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import load_yaml


@dataclass(frozen=True)
class TaskSpec:
    name: str
    hf_path: str
    hf_name: str | None
    text_fields: list[str]
    label_names: list[str]
    target_label: str
    train_split: str
    eval_split: str


def load_task_specs(tasks_yaml: str | Path = "configs/tasks.yaml") -> dict[str, TaskSpec]:
    data = load_yaml(tasks_yaml)["tasks"]
    out: dict[str, TaskSpec] = {}
    for name, spec in data.items():
        out[name] = TaskSpec(name=name, **spec)
    return out


def resolve_task(task_name: str, tasks_yaml: str | Path = "configs/tasks.yaml") -> TaskSpec:
    specs = load_task_specs(tasks_yaml)
    if task_name not in specs:
        raise KeyError(f"Unknown task {task_name!r}. Available: {sorted(specs)}")
    return specs[task_name]


def label_to_text(label: int, spec: TaskSpec) -> str:
    return spec.label_names[int(label)]


def text_to_label_id(label_text: str, spec: TaskSpec) -> int:
    normalized = label_text.strip().lower()
    for i, name in enumerate(spec.label_names):
        if name.lower() == normalized:
            return i
    raise ValueError(f"Unknown label {label_text!r} for task {spec.name}")


def target_label_id(spec: TaskSpec, override: str | None = None) -> int:
    target = spec.target_label if override in (None, "auto") else override
    return text_to_label_id(target, spec)
