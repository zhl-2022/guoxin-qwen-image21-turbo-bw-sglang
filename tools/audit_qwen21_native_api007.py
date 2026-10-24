"""Audit model call keyword compatibility with exact native DAS Python source."""
from __future__ import annotations
import ast
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=ROOT / "candidates/qwen21-turbo-das007-007")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/session007/qwen21-native-api007-audit.json")
    args = parser.parse_args()
    candidate = args.candidate.resolve(strict=True)
    lock = json.loads((candidate / "source-lock.json").read_text())
    wheel_path = ROOT / "evidence/session006/wan-preflight/hygon/wheels" / lock["native_wheel"]
    assert hashlib.sha256(wheel_path.read_bytes()).hexdigest() == lock["native_wheel_sha256"]
    checked, findings = [], []
    with zipfile.ZipFile(wheel_path) as wheel:
        available = set(wheel.namelist())
        for path in sorted((candidate / "overlay/sglang/multimodal_gen/runtime/models").rglob("*.py")):
            source = path.read_text()
            tree = ast.parse(source)
            bound = {}
            for node in tree.body:
                if not isinstance(node, ast.ImportFrom) or not node.module or not node.module.startswith("sglang."):
                    continue
                relative = node.module.replace(".", "/") + ".py"
                if (candidate / "overlay" / relative).is_file() or relative not in available:
                    continue
                native = ast.parse(wheel.read(relative).decode())
                definitions = {item.name: item for item in native.body if isinstance(item, (ast.FunctionDef, ast.ClassDef))}
                for alias in node.names:
                    definition = definitions.get(alias.name)
                    if isinstance(definition, ast.ClassDef):
                        definition = next((item for item in definition.body if isinstance(item, ast.FunctionDef) and item.name == "__init__"), None)
                    if definition is not None:
                        bound[alias.asname or alias.name] = (relative, definition)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id not in bound:
                    continue
                native_path, definition = bound[node.func.id]
                signature = definition.args
                if signature.kwarg:
                    continue
                accepted = {arg.arg for arg in signature.args + signature.kwonlyargs}
                requested = {item.arg for item in node.keywords if item.arg is not None}
                unexpected = sorted(requested - accepted)
                row = {"source": path.relative_to(candidate).as_posix(), "line": node.lineno,
                       "call": node.func.id, "native_source": native_path,
                       "keywords": sorted(requested), "unexpected": unexpected}
                checked.append(row)
                if unexpected:
                    findings.append(row)
    result = {"checked_native_call_sites": len(checked), "calls": checked,
              "findings": findings, "scope": "Direct imported native call keywords only; dynamic attributes and inherited signatures require runtime probes",
              "dynamic_verified": False, "GPU_executed": False}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "calls"}, indent=2))


if __name__ == "__main__":
    main()
