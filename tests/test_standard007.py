"""CPU integrity rejection checks for the delivery verifier; no model imports."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / ("tools" if (BASE / "verify_bundle.py").is_file() else "scripts")))
import verify_qwen21_standard007 as checked


class DeliveryChecks(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix="qwen21-manifest-cpu-")
        self.root = Path(self.folder.name)
        (self.root / "source.py").write_bytes(b"# fixture, not a model\n")
        raw = (self.root / "source.py").read_bytes()
        self.rows = [{"path": "source.py", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}]
        self.write_manifest()

    def tearDown(self): self.folder.cleanup()
    def write_manifest(self):
        (self.root / checked.MANIFEST).write_text(json.dumps({"files": self.rows}), encoding="utf-8")

    def test_exact_fixture(self): self.assertTrue(checked.verify(self.root)["passed"])
    def test_external_wrong_anchor(self):
        with self.assertRaises(ValueError): checked.verify(self.root, "0" * 64)
    def test_wrong_file_bytes(self):
        (self.root / "source.py").write_bytes(b"different")
        with self.assertRaises(ValueError): checked.verify(self.root)
    def test_missing_file(self):
        (self.root / "source.py").unlink()
        with self.assertRaises(ValueError): checked.verify(self.root)
    def test_extra_file(self):
        (self.root / "extra.log").write_bytes(b"not silently ignored")
        with self.assertRaises(ValueError): checked.verify(self.root)
    def test_git_admin_directory_exempt(self):
        (self.root / ".git").mkdir(); (self.root / ".git/config").write_bytes(b"fixture")
        self.assertTrue(checked.verify(self.root)["passed"])
    def test_nested_git_rejected(self):
        (self.root / "nested/.git").mkdir(parents=True)
        with self.assertRaises(ValueError): checked.verify(self.root)
    def test_git_file_not_exempt(self):
        (self.root / ".git").write_bytes(b"gitdir: somewhere")
        with self.assertRaises(ValueError): checked.verify(self.root)
    def test_duplicate_casefold(self):
        self.rows.append({**self.rows[0], "path": "SOURCE.py"}); self.write_manifest()
        with self.assertRaises(ValueError): checked.verify(self.root)
    def test_parent_escape_rejected(self):
        self.rows[0]["path"] = "../source.py"; self.write_manifest()
        with self.assertRaises(ValueError): checked.verify(self.root)
    def test_walk_error_propagates(self):
        def bad_walk(root, onerror, followlinks):
            onerror(PermissionError("synthetic unreadable directory"))
            yield root, [], []
        with patch.object(checked.os, "walk", bad_walk):
            with self.assertRaises(PermissionError): checked.verify(self.root)


if __name__ == "__main__": unittest.main()
