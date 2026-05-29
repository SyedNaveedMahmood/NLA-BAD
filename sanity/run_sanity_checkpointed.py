#!/usr/bin/env python3
"""
Checkpointed runner for the NLA-BAD sanity pipeline.

This file is an orchestration wrapper. It does not implement the ML logic itself;
it runs the day-by-day scripts that already exist inside the `sanity/` codebase.

Main features
-------------
- Runs Day 0 -> Day 6 in order.
- Writes a checkpoint/state file after each step.
- Can resume after interruption.
- Can stop after any step.
- Can dry-run commands before spending GPU time.
- Optional NLA-server check before NLA decoding.
- Optional AR fidelity step.

Expected location
-----------------
Place this file at:

    NLA-BAD/sanity/run_sanity_checkpointed.py

Then run it from inside `NLA-BAD/sanity`.

Example
-------
    python run_sanity_checkpointed.py \
      --config configs/sanity_qwen2p5_7b_l20.yaml \
      --run-root outputs/main \
      --stop-after day4_capture_activations

Resume later:

    python run_sanity_checkpointed.py \
      --config configs/sanity_qwen2p5_7b_l20.yaml \
      --run-root outputs/main \
      --require-nla-server

If your NLA AV server is not running, start it in a second terminal before Day 5:

    bash scripts/day0/launch_qwen_nla_av.sh 30000
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


@dataclass
class Step:
    name: str
    description: str
    command: List[str]
    expected_outputs: List[Path]
    optional: bool = False
    requires_nla_server: bool = False


class CheckpointState:
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, object] = {
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "steps": {},
        }
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.data["updated_at"] = utc_now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def get_step(self, name: str) -> Dict[str, object]:
        return self.data.setdefault("steps", {}).setdefault(name, {})

    def is_done(self, step: Step) -> bool:
        meta = self.get_step(step.name)
        if meta.get("status") != "done":
            return False
        return all(p.exists() for p in step.expected_outputs)

    def mark_running(self, step: Step, command: List[str]) -> None:
        meta = self.get_step(step.name)
        meta.update(
            {
                "status": "running",
                "description": step.description,
                "command": command,
                "started_at": utc_now(),
                "completed_at": None,
                "return_code": None,
            }
        )
        self.save()

    def mark_done(self, step: Step, return_code: int, log_path: Path) -> None:
        meta = self.get_step(step.name)
        meta.update(
            {
                "status": "done",
                "completed_at": utc_now(),
                "return_code": return_code,
                "log_path": str(log_path),
                "expected_outputs": [str(p) for p in step.expected_outputs],
            }
        )
        self.save()

    def mark_failed(self, step: Step, return_code: int, log_path: Path) -> None:
        meta = self.get_step(step.name)
        meta.update(
            {
                "status": "failed",
                "completed_at": utc_now(),
                "return_code": return_code,
                "log_path": str(log_path),
            }
        )
        self.save()


def check_nla_server(url: str, timeout_s: float = 3.0) -> bool:
    """
    Lightweight liveness check. Different NLA/SGLang servers expose different
    endpoints, so this accepts any HTTP response as evidence that something is
    listening at the host/port.
    """
    for suffix in ("", "/health", "/v1/models"):
        try:
            with urllib.request.urlopen(url.rstrip("/") + suffix, timeout=timeout_s) as response:
                _ = response.read(128)
                return True
        except Exception:
            continue
    return False


def run_command(
    step: Step,
    cwd: Path,
    state: CheckpointState,
    logs_dir: Path,
    dry_run: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> int:
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{step.name}.log"

    printable = " ".join(shlex.quote(x) for x in step.command)
    print(f"\n=== {step.name} ===")
    print(step.description)
    print(f"COMMAND: {printable}")
    print(f"LOG: {log_path}")

    if dry_run:
        return 0

    state.mark_running(step, step.command)

    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(f"# Step: {step.name}\n")
        logf.write(f"# Started: {utc_now()}\n")
        logf.write(f"# CWD: {cwd}\n")
        logf.write(f"# Command: {printable}\n\n")
        logf.flush()

        proc = subprocess.Popen(
            step.command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
            logf.flush()

        return_code = proc.wait()
        logf.write(f"\n# Finished: {utc_now()}\n")
        logf.write(f"# Return code: {return_code}\n")

    if return_code == 0 and all(p.exists() for p in step.expected_outputs):
        state.mark_done(step, return_code, log_path)
    elif return_code == 0:
        missing = [str(p) for p in step.expected_outputs if not p.exists()]
        print(f"\nStep finished but expected outputs are missing: {missing}")
        state.mark_failed(step, return_code, log_path)
        return_code = 100
    else:
        state.mark_failed(step, return_code, log_path)

    return return_code


def build_steps(args: argparse.Namespace, sanity_root: Path) -> List[Step]:
    py = sys.executable
    config = Path(args.config)

    run_root = Path(args.run_root)
    out = run_root

    d0_env = out / "day0_env"
    d0_meta = out / "day0_nla_meta"
    d1 = out / "day1_data" / "main"
    d2 = out / "day2_backdoor" / "main"
    d3 = out / "day3_asr" / "main"
    d4 = out / "day4_activations" / "main"
    d5_decode = out / "day5_nla_decode" / "main"
    d5_bvr = out / "day5_bvr" / "main"
    d5_probe = out / "day5_probe" / "main"
    d5_fid = out / "day5_nla_fidelity" / "main"
    d6 = out / "day6_report" / "main"

    steps: List[Step] = [
        Step(
            name="day0_check_env",
            description="Check Python, CUDA, PyTorch, package versions, and basic runtime information.",
            command=[py, "scripts/day0/00_check_env.py", "--config", str(config), "--out", str(d0_env)],
            expected_outputs=[d0_env / "env_summary.md", d0_env / "env_summary.json"],
        ),
        Step(
            name="day0_check_nla_metadata",
            description="Check NLA/model metadata assumptions such as target layer and hidden dimension.",
            command=[py, "scripts/day0/01_check_nla_metadata.py", "--config", str(config), "--out", str(d0_meta)],
            expected_outputs=[d0_meta / "nla_meta_summary.md", d0_meta / "nla_meta_summary.json"],
        ),
        Step(
            name="day1_build_dataset",
            description="Build standard clean, triggered, alternate-triggered, and poisoned-training splits.",
            command=[py, "scripts/day1/01_build_standard_dataset.py", "--config", str(config), "--out", str(d1)],
            expected_outputs=[
                d1 / "train_poisoned.jsonl",
                d1 / "eval_clean.jsonl",
                d1 / "eval_triggered.jsonl",
                d1 / "eval_alt_triggered.jsonl",
                d1 / "manifest.json",
            ],
        ),
        Step(
            name="day2_train_backdoored_model",
            description="Train the controlled harmless backdoored model.",
            command=[
                py,
                "scripts/day2/02_train_backdoored_model.py",
                "--config",
                str(config),
                "--train-jsonl",
                str(d1 / "train_poisoned.jsonl"),
                "--out",
                str(d2),
            ],
            expected_outputs=[d2 / "hf_model", d2 / "train_summary.json"],
        ),
        Step(
            name="day3_eval_asr",
            description="Evaluate clean accuracy, triggered ASR, and alternate-trigger ASR.",
            command=[
                py,
                "scripts/day3/03_eval_asr.py",
                "--config",
                str(config),
                "--model-dir",
                str(d2 / "hf_model"),
                "--clean-jsonl",
                str(d1 / "eval_clean.jsonl"),
                "--triggered-jsonl",
                str(d1 / "eval_triggered.jsonl"),
                "--alt-jsonl",
                str(d1 / "eval_alt_triggered.jsonl"),
                "--out",
                str(d3),
            ],
            expected_outputs=[d3 / "asr_summary.md", d3 / "asr_summary.json"],
        ),
        Step(
            name="day4_capture_activations",
            description="Capture layer-20 residual activations for clean, triggered, and alternate-triggered prompts.",
            command=[
                py,
                "scripts/day4/04_capture_activations.py",
                "--config",
                str(config),
                "--model-dir",
                str(d2 / "hf_model"),
                "--clean-jsonl",
                str(d1 / "eval_clean.jsonl"),
                "--triggered-jsonl",
                str(d1 / "eval_triggered.jsonl"),
                "--alt-jsonl",
                str(d1 / "eval_alt_triggered.jsonl"),
                "--out",
                str(d4),
            ],
            expected_outputs=[
                d4 / "activations.parquet",
                d4 / "activation_metadata.jsonl",
                d4 / "activation_qc.md",
            ],
        ),
        Step(
            name="day5_run_nla_verbalizer",
            description="Send captured activations to the NLA Activation Verbalizer server.",
            command=[
                py,
                "scripts/day5/05_run_nla_verbalizer.py",
                "--config",
                str(config),
                "--activations-parquet",
                str(d4 / "activations.parquet"),
                "--metadata-jsonl",
                str(d4 / "activation_metadata.jsonl"),
                "--out",
                str(d5_decode),
            ],
            expected_outputs=[d5_decode / "verbalizations.jsonl", d5_decode / "decode_summary.json"],
            requires_nla_server=True,
        ),
        Step(
            name="day5_score_bvr",
            description="Score Backdoor Verbalization Residue from NLA explanations.",
            command=[
                py,
                "scripts/day5/06_score_bvr.py",
                "--config",
                str(config),
                "--verbalizations-jsonl",
                str(d5_decode / "verbalizations.jsonl"),
                "--out",
                str(d5_bvr),
            ],
            expected_outputs=[d5_bvr / "bvr_scores.csv", d5_bvr / "bvr_summary.md", d5_bvr / "bvr_summary.json"],
        ),
        Step(
            name="day5_probe_baseline",
            description="Train/evaluate the oracle linear-probe baseline on captured activations.",
            command=[
                py,
                "scripts/day5/07_probe_baseline.py",
                "--config",
                str(config),
                "--activations-parquet",
                str(d4 / "activations.parquet"),
                "--metadata-jsonl",
                str(d4 / "activation_metadata.jsonl"),
                "--out",
                str(d5_probe),
            ],
            expected_outputs=[d5_probe / "probe_summary.md", d5_probe / "probe_summary.json"],
        ),
    ]

    if args.run_ar_fidelity:
        steps.append(
            Step(
                name="day5_score_nla_fidelity",
                description="Optional AR round-trip fidelity check for NLA verbalizations.",
                command=[
                    py,
                    "scripts/day5/08_score_nla_fidelity_with_official_ar.py",
                    "--config",
                    str(config),
                    "--nla-repo",
                    str(args.nla_repo),
                    "--activations-parquet",
                    str(d4 / "activations.parquet"),
                    "--verbalizations-jsonl",
                    str(d5_decode / "verbalizations.jsonl"),
                    "--out",
                    str(d5_fid),
                ],
                expected_outputs=[
                    d5_fid / "nla_fidelity_scores.csv",
                    d5_fid / "nla_fidelity_summary.md",
                    d5_fid / "nla_fidelity_summary.json",
                ],
                optional=True,
            )
        )

    steps.append(
        Step(
            name="day6_make_sanity_report",
            description="Aggregate all outputs into the final sanity report and gate decision.",
            command=[
                py,
                "scripts/day6/09_make_sanity_report.py",
                "--run-root",
                str(out),
                "--out",
                str(d6),
            ],
            expected_outputs=[d6 / "final_sanity_report.md", d6 / "final_sanity_report.json", d6 / "gates.md"],
        )
    )

    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpointed runner for the NLA-BAD sanity pipeline.")
    parser.add_argument("--config", default="configs/sanity_qwen2p5_7b_l20.yaml", help="Path to sanity YAML config.")
    parser.add_argument("--run-root", default="outputs/main", help="Root directory for all run outputs.")
    parser.add_argument("--state-file", default=None, help="Checkpoint JSON path. Default: <run-root>/pipeline_state.json")
    parser.add_argument("--logs-dir", default=None, help="Logs directory. Default: <run-root>/logs")
    parser.add_argument("--from-step", default=None, help="Start from this step name.")
    parser.add_argument("--to-step", default=None, help="Stop after this step name.")
    parser.add_argument("--stop-after", default=None, help="Alias for --to-step.")
    parser.add_argument("--force", action="store_true", help="Rerun steps even if checkpoint says they are done.")
    parser.add_argument("--force-step", action="append", default=[], help="Rerun a specific step name. Can be repeated.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--require-nla-server", action="store_true", help="Before Day 5 NLA decode, require server to be reachable.")
    parser.add_argument("--nla-server-url", default="http://localhost:30000", help="NLA AV server URL.")
    parser.add_argument("--skip-nla", action="store_true", help="Skip NLA decode and BVR scoring. Useful for running Days 0-4 only.")
    parser.add_argument("--run-ar-fidelity", action="store_true", help="Run optional AR fidelity scoring step.")
    parser.add_argument("--nla-repo", default="../natural_language_autoencoders", help="Path to official NLA repo for optional AR fidelity.")
    parser.add_argument("--yes", action="store_true", help="Do not ask for confirmation before running.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.stop_after and not args.to_step:
        args.to_step = args.stop_after

    sanity_root = Path.cwd()
    run_root = Path(args.run_root)
    state_path = Path(args.state_file) if args.state_file else run_root / "pipeline_state.json"
    logs_dir = Path(args.logs_dir) if args.logs_dir else run_root / "logs"

    steps = build_steps(args, sanity_root)

    if args.skip_nla:
        steps = [s for s in steps if not (s.name in {"day5_run_nla_verbalizer", "day5_score_bvr"})]

    step_names = [s.name for s in steps]
    if args.from_step and args.from_step not in step_names:
        raise SystemExit(f"Unknown --from-step {args.from_step!r}. Valid steps: {step_names}")
    if args.to_step and args.to_step not in step_names:
        raise SystemExit(f"Unknown --to-step {args.to_step!r}. Valid steps: {step_names}")

    if args.from_step:
        start_idx = step_names.index(args.from_step)
    else:
        start_idx = 0
    if args.to_step:
        end_idx = step_names.index(args.to_step)
    else:
        end_idx = len(steps) - 1

    selected = steps[start_idx : end_idx + 1]

    print("NLA-BAD checkpointed sanity runner")
    print(f"Working directory: {sanity_root}")
    print(f"Run root: {run_root}")
    print(f"State file: {state_path}")
    print(f"Logs dir: {logs_dir}")
    print("Selected steps:")
    for step in selected:
        print(f"  - {step.name}: {step.description}")

    if not args.yes and not args.dry_run:
        answer = input("\nProceed? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Aborted before running anything.")
            return 2

    state = CheckpointState(state_path)

    for step in selected:
        if step.requires_nla_server and args.require_nla_server:
            print(f"\nChecking NLA server at {args.nla_server_url} ...")
            if not check_nla_server(args.nla_server_url):
                print(
                    "NLA server does not appear reachable. Start it in another terminal first, e.g.:\n"
                    "  bash scripts/day0/launch_qwen_nla_av.sh 30000\n"
                    "Then rerun this command. Completed previous steps will be skipped automatically."
                )
                return 3
            print("NLA server appears reachable.")

        force_this = args.force or step.name in set(args.force_step)
        if not force_this and state.is_done(step):
            print(f"\n=== {step.name} ===")
            print("Already completed and expected outputs exist. Skipping.")
            continue

        rc = run_command(step, cwd=sanity_root, state=state, logs_dir=logs_dir, dry_run=args.dry_run)
        if rc != 0:
            print(f"\nStep failed: {step.name}")
            print(f"Check log: {logs_dir / (step.name + '.log')}")
            print("You can fix the issue and rerun the same command; completed steps will be skipped.")
            return rc

    print("\nPipeline segment completed successfully.")
    print(f"Checkpoint: {state_path}")
    print(f"Logs: {logs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
