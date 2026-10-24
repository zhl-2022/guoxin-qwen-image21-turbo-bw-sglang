"""Controller ownership checks plus the existing real Range/SHA worker tests."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import download_qwen21_turbo009 as qwen
import download_wan_models_006 as wan_identity


class QwenControllerTests(unittest.TestCase):
    def test_exact_phase_lock_and_two_worker_plan(self):
        rows = qwen.validate(ROOT / "evidence/session007/qwen21-pipeline-prep009/download-sources009.json")
        self.assertEqual(len(qwen.select(rows, "metadata")), 13)
        self.assertEqual(len(qwen.select(rows, "vae")), 14)
        self.assertEqual(sum(row["size"] for row in qwen.select(rows, "vae")), 686997222)
        self.assertEqual(sum(row["size"] for row in rows), 32451622214)
        self.assertNotEqual(qwen.NAMESPACE, wan_identity.NAMESPACE)
        self.assertEqual(qwen.TRANSFER_LOCK.name, "qwen21-turbo009-transfer.lock")
        self.assertNotIn("torch", sys.modules)

    def test_altered_source_lock_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lock.json"
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, "lock bytes differ"):
                qwen.validate(path)

    def test_full_phase_default_budget_is_closed(self):
        rows = qwen.validate(ROOT / "evidence/session007/qwen21-pipeline-prep009/download-sources009.json")
        with mock.patch.object(qwen.sys, "platform", "linux"):
            with self.assertRaisesRegex(ValueError, "byte budget"):
                qwen.apply(rows, SimpleNamespace(max_bytes=1024**3))

    def test_verify_owns_only_qwen_namespace_lock(self):
        data = b"publisher-bytes-tiny-fixture"
        row = {"path": "vae/model.bin", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        original_wan_base, original_wan_namespace = wan_identity.BASE, wan_identity.NAMESPACE
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            target = base / qwen.NAMESPACE / row["path"]
            target.parent.mkdir(parents=True)
            target.write_bytes(data)
            other_lock = base / "wan006-transfer.lock"
            other_lock.write_text('{"run_id":"foreign"}')
            args = SimpleNamespace(max_bytes=1024**3, max_seconds=60, action="verify", component="vae")
            with mock.patch.object(qwen, "BASE", base), mock.patch.object(qwen, "TRANSFER_LOCK", base / "qwen21-turbo009-transfer.lock"), \
                    mock.patch.object(qwen.sys, "platform", "linux"), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(qwen.apply([row], args), 0)
            self.assertFalse((base / "qwen21-turbo009-transfer.lock").exists())
            self.assertEqual(other_lock.read_text(), '{"run_id":"foreign"}')
            receipt = json.loads(next(base.glob("qwen21-turbo009-transfer-*.json")).read_text())
            self.assertEqual(receipt["files"][0]["sha256"], row["sha256"])
            self.assertFalse(receipt["full_checkpoint_verified"])
            self.assertTrue(receipt["all_workers_exited_before_lock_release"])
            self.assertFalse(receipt["torch_in_sys_modules"])
        self.assertEqual((wan_identity.BASE, wan_identity.NAMESPACE), (original_wan_base, original_wan_namespace))

    def test_foreign_qwen_lock_is_not_taken_over(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            lock = base / "qwen21-turbo009-transfer.lock"
            lock.write_text('{"run_id":"existing"}')
            args = SimpleNamespace(max_bytes=1024**3, max_seconds=60, action="verify", component="vae")
            with mock.patch.object(qwen, "BASE", base), mock.patch.object(qwen, "TRANSFER_LOCK", lock), \
                    mock.patch.object(qwen.sys, "platform", "linux"):
                with self.assertRaises(FileExistsError):
                    qwen.apply([], args)
            self.assertEqual(lock.read_text(), '{"run_id":"existing"}')


if __name__ == "__main__":
    unittest.main()
