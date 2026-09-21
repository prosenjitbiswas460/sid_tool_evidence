"""Ensure repo root is on sys.path and the PLUM package is present."""

from __future__ import annotations

import os
import sys


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def scripts_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def ensure_repo_paths() -> str:
    root = repo_root()
    scripts = scripts_dir()
    for path in (root, scripts):
        if path not in sys.path:
            sys.path.insert(0, path)
    return root


def plum_package_dir(root: str | None = None) -> str:
    root = root or repo_root()
    return os.path.join(root, "websocietysimulator", "plum")


def ensure_plum_importable() -> str:
    root = ensure_repo_paths()
    init_path = os.path.join(plum_package_dir(root), "__init__.py")
    if not os.path.isfile(init_path):
        raise ImportError(
            "PLUM package not found at websocietysimulator/plum/.\n"
            f"Expected: {init_path}\n"
            "This usually means the PLUM files were not synced to this machine.\n"
            "Copy or git pull so these paths exist:\n"
            "  websocietysimulator/plum/\n"
            "  scripts/build_sid_v2.py"
        )
    return root


if __name__ == "__main__":
    root = ensure_plum_importable()
    required = [
        "__init__.py",
        "cooccurrence.py",
        "rq_vae.py",
        "embeddings.py",
        "item_text.py",
        "sid_v2_artifacts.py",
    ]
    missing = [
        name
        for name in required
        if not os.path.isfile(os.path.join(plum_package_dir(root), name))
    ]
    if missing:
        raise SystemExit(f"PLUM package incomplete; missing: {missing}")
    print(f"PLUM package OK under {root}/websocietysimulator/plum/")
