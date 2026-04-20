"""Cross-platform launcher for the XAI evaluation pipeline.

Runs the 13 notebooks in notebooks/ sequentially with papermill. Works on
Windows, Linux and macOS; no bash required. Each notebook is executed
in-place (kernel ``python3``) and a copy with outputs is saved to
``notebooks_executed/``. Per-notebook logs are written under ``logs/``;
a consolidated ``logs/run.log`` plus a final ``logs/summary.log`` tell
you what passed and how long each step took.

Usage
-----
    python run_all.py                # run everything
    python run_all.py --from 03a     # resume from a notebook stem
    python run_all.py --only 04 05   # run only these stems
    python run_all.py --dry-run      # print the plan without executing

The notebook stems are the part of the filename before ``.ipynb`` (e.g.
``03a_xai_shap_lime``); you can pass a prefix like ``03a`` and it will
match. The order is fixed (00 -> 09).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
NB_DIR = ROOT / "notebooks"
OUT_DIR = ROOT / "notebooks_executed"
LOG_DIR = ROOT / "logs"


# Fixed execution order. Each entry is a notebook stem (the part before .ipynb).
NOTEBOOKS: list[str] = [
    "00_setup_and_data_download",
    "01a_eda_elliptic",
    "01b_eda_ethereum",
    "02a_baselines_ml",
    "02b_baselines_gnn",
    "03a_xai_shap_lime",
    "03b_xai_gnn_explainers",
    "04_module1_fidelity",
    "05_module2_stability",
    "06_module3_bras",
    "07_exp_class_imbalance",
    "08_module4_llm_agents",
    "09_module4_ml_baseline",
]


def _match_stems(selectors: Iterable[str]) -> list[str]:
    """Expand user-supplied prefixes to full stems (preserving order)."""
    selected: list[str] = []
    for sel in selectors:
        matches = [n for n in NOTEBOOKS if n.startswith(sel)]
        if not matches:
            raise SystemExit(f"no notebook matches selector {sel!r}")
        for m in matches:
            if m not in selected:
                selected.append(m)
    # Restore canonical order.
    return [n for n in NOTEBOOKS if n in selected]


def _format_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else (f"{m:d}m{s:02d}s" if m else f"{s:d}s")


def run_notebook(stem: str, log_path: Path) -> tuple[bool, float]:
    """Execute a single notebook via papermill. Returns (ok, duration_s)."""
    import papermill as pm

    src = NB_DIR / f"{stem}.ipynb"
    dst = OUT_DIR / f"{stem}.ipynb"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        try:
            pm.execute_notebook(
                input_path=str(src),
                output_path=str(dst),
                kernel_name="python3",
                log_output=True,
                stdout_file=log,
                stderr_file=log,
                progress_bar=False,
            )
            return True, time.monotonic() - start
        except Exception as exc:  # papermill re-raises errors
            log.write(f"\n\n!!! EXCEPTION: {exc!r}\n")
            return False, time.monotonic() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from", dest="start_from", default=None,
        help="Start from the first notebook whose stem starts with this prefix.",
    )
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="Run only the notebooks matching these prefixes.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan and exit.",
    )
    args = parser.parse_args()

    if args.only:
        plan = _match_stems(args.only)
    elif args.start_from:
        stems = _match_stems([args.start_from])
        if not stems:
            raise SystemExit(f"no match for --from {args.start_from!r}")
        start_idx = NOTEBOOKS.index(stems[0])
        plan = NOTEBOOKS[start_idx:]
    else:
        plan = list(NOTEBOOKS)

    print(f"Repository : {ROOT}")
    print(f"Plan       : {len(plan)} notebook(s)")
    for stem in plan:
        print(f"  - {stem}")
    if args.dry_run:
        return 0

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_log = LOG_DIR / "run.log"
    summary_log = LOG_DIR / "summary.log"

    with main_log.open("a", encoding="utf-8") as mlog:
        header = f"\n=== Run started at {datetime.now().isoformat(timespec='seconds')} ===\n"
        mlog.write(header)
        print(header.rstrip())

        results: list[tuple[str, str, float]] = []
        first_failure: str | None = None
        for stem in plan:
            log_path = LOG_DIR / f"{stem}.log"
            msg = f"[{datetime.now().isoformat(timespec='seconds')}] running {stem} ..."
            mlog.write(msg + "\n"); mlog.flush()
            print(msg, flush=True)

            ok, dur = run_notebook(stem, log_path)
            status = "OK" if ok else "FAIL"
            dur_str = _format_duration(dur)
            summary = f"  {stem:<34} {status:<4} ({dur_str})"
            mlog.write(summary + "\n"); mlog.flush()
            print(summary, flush=True)
            results.append((stem, status, dur))

            if not ok:
                # Append the tail of the notebook log to the main log for quick inspection.
                try:
                    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
                    mlog.write("  --- log tail ---\n")
                    for line in tail:
                        mlog.write(f"  | {line}\n")
                    mlog.flush()
                except OSError:
                    pass
                first_failure = stem
                print(f"\nSTOPPED: {stem} failed. See {log_path} for details.", flush=True)
                break

        mlog.write(f"=== Run finished at {datetime.now().isoformat(timespec='seconds')} ===\n")

    # Write a clean summary report.
    with summary_log.open("w", encoding="utf-8") as slog:
        slog.write("Notebook                           Status  Duration\n")
        slog.write("-" * 54 + "\n")
        for stem, status, dur in results:
            slog.write(f"{stem:<34} {status:<6}  {_format_duration(dur)}\n")

    print("\nSummary:")
    with summary_log.open("r", encoding="utf-8") as slog:
        print(slog.read())

    return 0 if first_failure is None else 1


if __name__ == "__main__":
    sys.exit(main())
