from __future__ import annotations

from typing import Any

from .tasks import TaskSpec


def raw_text_from_example(example: dict[str, Any], spec: TaskSpec) -> str:
    if spec.name == "qnli":
        return f"Question: {example['question']}\nSentence: {example['sentence']}"
    return "\n".join(str(example[f]) for f in spec.text_fields)


def instruction_prompt(text: str, spec: TaskSpec) -> str:
    labels = ", ".join(spec.label_names)
    return (
        "You are a careful text classification model. "
        f"Return exactly one label from this set: {labels}.\n\n"
        f"Input:\n{text}\n\nLabel:"
    )


def completion_for_label(label_name: str) -> str:
    return f" {label_name}"


def make_chat_prompt(tokenizer: Any, user_prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"User: {user_prompt}\nAssistant:"
