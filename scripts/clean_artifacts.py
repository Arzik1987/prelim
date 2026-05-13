#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
}
DEFAULT_SUFFIXES = (".egg-info",)


def iter_artifacts(root):
    for path in root.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.is_dir() and (path.name in DEFAULT_DIR_NAMES or path.name.endswith(DEFAULT_SUFFIXES)):
            yield path


def remove_artifacts(root, dry_run=False):
    removed = []
    for path in sorted(iter_artifacts(root), key=lambda item: len(item.parts), reverse=True):
        if not path.exists():
            continue
        removed.append(path)
        if not dry_run:
            shutil.rmtree(path)
    return removed


def parse_args():
    parser = argparse.ArgumentParser(description="Remove generated local Python/package artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="List artifacts without deleting them.")
    return parser.parse_args()


def main():
    args = parse_args()
    removed = remove_artifacts(ROOT, dry_run=args.dry_run)
    action = "Would remove" if args.dry_run else "Removed"
    for path in removed:
        print(f"{action}: {path.relative_to(ROOT)}")
    print(f"{action} {len(removed)} artifact directories.")


if __name__ == "__main__":
    main()
