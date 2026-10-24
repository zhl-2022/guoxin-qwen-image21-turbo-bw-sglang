"""Bind every recorded loaded framework .py file to native wheel or overlay."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads((args.bundle / "source-lock.json").read_text())
    overlay = {row["path"].removeprefix("overlay/"): row["sha256"] for row in lock["files"]}
    native = {name: row["sha256"] for name, row in lock["native_wheel_python_files"].items()}
    summary = json.loads((args.raw_dir / "probe-summary.json").read_text())
    workers, loaded = [], {}
    for item in summary["workers"]:
        record = json.loads((args.raw_dir / (item["name"] + ".json")).read_text())
        native_root = record["native_before"]["root"].rstrip("/") + "/"
        overlay_root = record["activated_packages"]["overlay"].rstrip("/") + "/"
        counts = {"native": 0, "overlay": 0}
        for module, row in record["module_sources"].items():
            if row["path"].startswith(overlay_root):
                relative = row["path"][len(overlay_root):]
                assert overlay[relative] == row["sha256"], (item["name"], module)
                role = "overlay"
            else:
                assert row["path"].startswith(native_root), (item["name"], module, row["path"])
                relative = row["path"][len(native_root):]
                assert native[relative] == row["sha256"], (item["name"], module)
                role = "native"
            counts[role] += 1
            loaded[module] = {"role": role, "path": relative, "sha256": row["sha256"]}
        workers.append({"name": item["name"], "counts": counts, "all_loaded_sources_bound": True})
    result = {"passed": True, "workers": workers, "distinct_framework_modules_bound": len(loaded),
              "loaded_modules": loaded, "scope": "Recorded loaded .py source identities, separate from component success or official accuracy"}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "loaded_modules"}, indent=2))


if __name__ == "__main__":
    main()
