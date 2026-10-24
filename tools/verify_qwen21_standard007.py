"""Verify exact standard-delivery bytes; only top-level .git administration is exempt."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess

MANIFEST = "delivery-manifest.json"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2): value.update(block)
    return value.hexdigest()


def files_in(root):
    root = root.resolve(strict=True)
    names = set()
    def fail(error): raise error
    for folder, dirs, files in os.walk(root, onerror=fail, followlinks=False):
        base = Path(folder)
        for name in list(dirs):
            path = base / name
            if path.is_symlink(): raise ValueError("Symlink directory refused: " + str(path))
            if name == ".git":
                if base != root: raise ValueError("Nested Git checkout refused")
                dirs.remove(name)
        for name in files:
            path = base / name
            if path.is_symlink(): raise ValueError("Symlink file refused: " + str(path))
            if path == root / ".git":
                raise ValueError("Only a real top-level .git directory is exempt")
            names.add(path.relative_to(root).as_posix())
    return names


def verify(root, expected_sha=None, git_index=False):
    manifest_path = root / MANIFEST
    actual_sha = digest(manifest_path)
    if expected_sha is not None and actual_sha != expected_sha:
        raise ValueError("Externally pinned manifest SHA differs")
    manifest = json.loads(manifest_path.read_text())
    rows = manifest["files"]
    expected = {row["path"]: row for row in rows}
    if len(expected) != len(rows) or len(expected) != len({name.casefold() for name in expected}):
        raise ValueError("Duplicate manifest path")
    for name, row in expected.items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or rel.as_posix() != name or ".." in rel.parts or any(c in name for c in ("\\", ":", "\0")):
            raise ValueError("Unsafe manifest path")
    actual = files_in(root)
    if actual != set(expected) | {MANIFEST}:
        raise ValueError({"missing": sorted(set(expected) - actual), "extra": sorted(actual - set(expected) - {MANIFEST})})
    for name, row in expected.items():
        path = root.joinpath(*PurePosixPath(name).parts)
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("Delivery file bytes differ: " + name)
    if git_index:
        tracked = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"])
        paths = {name.decode("utf-8") for name in tracked.split(b"\0") if name}
        if paths != actual: raise ValueError("Git tracked inventory differs from exact delivery inventory")
        changed = subprocess.check_output(["git", "-C", str(root), "diff", "--name-only", "HEAD"])
        if changed: raise ValueError("Tracked working tree differs from HEAD")
    return {"passed": True, "files_verified": len(rows), "manifest_sha256": actual_sha,
        "external_manifest_anchor_provided": expected_sha is not None,
        "git_index_checked": git_index, "exempt": "Only top-level real .git/ directory"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--check-git-index", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.manifest_sha256, args.check_git_index), indent=2))


if __name__ == "__main__": main()
