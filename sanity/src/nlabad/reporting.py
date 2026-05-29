from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .io import ensure_dir, write_json


def write_markdown_table(df: pd.DataFrame, path: str | Path, title: str) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")


def gate(status: bool, name: str, value: Any, threshold: Any) -> dict[str, Any]:
    return {"gate": name, "pass": bool(status), "value": value, "threshold": threshold}


def write_gate_report(gates: list[dict[str, Any]], out_dir: str | Path, title: str = "Gate report") -> None:
    out_dir = ensure_dir(out_dir)
    write_json(out_dir / "gates.json", gates)
    with open(out_dir / "gates.md", "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        for g in gates:
            mark = "PASS" if g["pass"] else "FAIL"
            f.write(f"- **{mark}** `{g['gate']}`: value={g['value']} threshold={g['threshold']}\n")
