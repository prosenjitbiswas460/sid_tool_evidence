#!/usr/bin/env python3
"""Generate plots under docs/figures/."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
ENV = {
    **os.environ,
    "MPLBACKEND": "Agg",
    "OMP_NUM_THREADS": "1",
    "MPLCONFIGDIR": str(ROOT / ".mplconfig"),
}


def run(script: str) -> None:
    path = FIG / script
    print(f"== {script}")
    subprocess.check_call([sys.executable, str(path)], cwd=str(ROOT), env=ENV)


def main() -> None:
    (ROOT / ".mplconfig").mkdir(exist_ok=True)
    run("generate_deployment_roles.py")
    run("generate_shortlist_protocol.py")
    run("generate_admission_heatmap.py")
    run("generate_predictability.py")
    run("generate_delimitation.py")
    run("generate_shortlist_recall.py")
    print(f"Wrote figures under {FIG}")


if __name__ == "__main__":
    main()
