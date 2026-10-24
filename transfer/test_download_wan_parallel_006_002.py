import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("wan_parallel002", ROOT / "scripts/download_wan_models_006_002.py")
wan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wan)


def row(name, data):
    return {"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "download_url": "https://modelscope.cn/models/test/model/resolve/" + "a"*40 + "/" + name}


class Response:
    def __init__(self, url, data, offset, status=206, delay=0, wrong_range=False):
        self.url, self.delay = url, delay
        self.status = status
        self.content = io.BytesIO(data[offset:] if status == 206 else data)
        self.headers = {"Content-Length": str(len(data)-offset if status == 206 else len(data)),
                        "Content-Range": f"bytes {offset+1 if wrong_range else offset}-{len(data)-1}/{len(data)}"}

    def geturl(self):
        return self.url

    def read(self, size):
        if self.delay:
            time.sleep(self.delay)
        return self.content.read(min(size, 4096) if self.delay else size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.content.close()


class Opener:
    def __init__(self, data, status=206, delay=0, wrong_range=False):
        self.data, self.status, self.delay, self.wrong_range = data, status, delay, wrong_range
        self.offsets = []

    def open(self, request, timeout):
        value = request.get_header("Range")
        offset = int(value.split("=")[1][:-1]) if value else 0
        self.offsets.append(offset)
        return Response(request.full_url, self.data, offset, self.status, self.delay, self.wrong_range)


class ParallelWanTests(unittest.TestCase):
    def test_owned_v001_partial_resumes_real_range_and_hashes_disk(self):
        data = bytes(range(256))*300
        item = row("shard.bin", data)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/item["path"]
            target.with_name(target.name+".part").write_bytes(data[:7000])
            target.with_name(target.name+".part.json").write_text(json.dumps({"size":len(data),"sha256":item["sha256"]}))
            opener = Opener(data)
            result = wan.download_one(item, target, time.monotonic()+10, threading.Event(), opener)
            self.assertEqual(opener.offsets, [7000])
            self.assertEqual(result["resumed_from_bytes"], 7000)
            self.assertEqual(target.read_bytes(), data)
            self.assertFalse(target.with_name(target.name+".part.json").exists())

    def test_ignored_range_restarts_only_owned_partial(self):
        data = b"new exact file"*100
        item = row("shard.bin", data)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/item["path"]
            target.with_name(target.name+".part").write_bytes(data[:10])
            target.with_name(target.name+".part.json").write_text(json.dumps({"size":len(data),"sha256":item["sha256"]}))
            wan.download_one(item,target,time.monotonic()+10,threading.Event(),Opener(data,status=200))
            self.assertEqual(target.read_bytes(), data)

    def test_unowned_partial_wrong_range_and_wrong_final_are_preserved(self):
        data = b"0123456789"*100
        item = row("shard.bin", data)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/item["path"]
            partial=target.with_name(target.name+".part")
            partial.write_bytes(data[:10])
            with self.assertRaisesRegex(ValueError,"Unowned"):
                wan.download_one(item,target,time.monotonic()+10,threading.Event(),Opener(data))
            state=target.with_name(target.name+".part.json")
            state.write_text(json.dumps({"size":len(data),"sha256":item["sha256"]}))
            with self.assertRaisesRegex(ValueError,"Range"):
                wan.download_one(item,target,time.monotonic()+10,threading.Event(),Opener(data,wrong_range=True))
            self.assertEqual(partial.read_bytes(), data[:10])
            target.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError,"Existing final"):
                wan.download_one(item,target,time.monotonic()+10,threading.Event(),Opener(data))
            self.assertEqual(target.read_bytes(),b"wrong")

    def test_no_more_than_three_distinct_file_workers_and_duplicates_rejected(self):
        data = b"a"*32768
        rows=[row(f"{i}.bin",data) for i in range(7)]
        with tempfile.TemporaryDirectory() as tmp:
            active=0; peak=0; guard=threading.Lock()
            original=wan.download_one
            def tracked(*args,**kwargs):
                nonlocal active,peak
                with guard:active+=1;peak=max(peak,active)
                try:return original(*args,**kwargs)
                finally:
                    with guard:active-=1
            with mock.patch.object(wan,"download_one",side_effect=tracked):
                results,interrupted=wan.run_workers(rows,Path(tmp),time.monotonic()+10,threading.Event(),3,lambda r:Opener(data,delay=.002))
            self.assertEqual(peak,3)
            self.assertEqual(len(results),7)
            self.assertFalse(interrupted)
            self.assertTrue(all(x["status"]=="downloaded_verified" for x in results))
            with self.assertRaisesRegex(ValueError,"unique"):
                wan.run_workers([rows[0],rows[0]],Path(tmp),time.monotonic()+1,threading.Event(),3)

    def test_cooperative_cancel_joins_workers_and_keeps_owned_prefixes(self):
        data=b"q"*(1024*1024)
        rows=[row(f"{i}.bin",data) for i in range(6)]
        stop=threading.Event(); timer=threading.Timer(.04,stop.set)
        with tempfile.TemporaryDirectory() as tmp:
            timer.start()
            try:
                results,interrupted=wan.run_workers(rows,Path(tmp),time.monotonic()+10,stop,3,lambda r:Opener(data,delay=.01))
            finally:timer.cancel();timer.join()
            self.assertFalse(interrupted)
            self.assertEqual(len(results),6)
            self.assertFalse(any(t.name.startswith("wan-owned-transfer") for t in threading.enumerate()))
            partials=list(Path(tmp).glob("*.part"))
            self.assertGreater(len(partials),0)
            for partial in partials:
                self.assertEqual(partial.read_bytes(), data[:partial.stat().st_size])
                self.assertTrue(partial.with_name(partial.name+".json").is_file())

    @unittest.skipUnless(sys.platform=="linux","Actual SIGINT handoff test uses Linux")
    def test_actual_SIGINT_writes_receipt_and_releases_owned_namespace_lock_after_join(self):
        data=b"k"*(1024*1024)
        rows=[row(f"{i}.bin",data) for i in range(36)]
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            timer=threading.Timer(.05,lambda:os.kill(os.getpid(),signal.SIGINT))
            original=wan.download_one
            def injected(item,target,deadline,stop,opener=None):
                return original(item,target,deadline,stop,Opener(data,delay=.01))
            with mock.patch.object(wan.identity,"BASE",base),mock.patch.object(wan,"download_one",side_effect=injected):
                timer.start()
                try:code=wan.apply(rows,10,3)
                finally:timer.cancel();timer.join()
            self.assertEqual(code,130)
            self.assertFalse((base/"wan006-transfer.lock").exists())
            receipt=json.loads(next(base.glob("wan006-transfer-v002-*.json")).read_text())
            self.assertTrue(receipt["all_workers_exited_before_lock_release"])
            self.assertIn("interrupted",receipt["status"])
            self.assertFalse(any(t.name.startswith("wan-owned-transfer") for t in threading.enumerate()))

    def test_existing_namespace_lock_is_never_replaced_or_removed(self):
        with tempfile.TemporaryDirectory() as tmp,mock.patch.object(wan.identity,"BASE",Path(tmp)),mock.patch.object(wan.sys,"platform","linux"):
            lock=Path(tmp)/"wan006-transfer.lock";lock.write_text('{"run_id":"v001","pid":123}')
            before=lock.read_bytes()
            with self.assertRaises(FileExistsError):wan.apply([row("x",b"x")],10,3)
            self.assertEqual(lock.read_bytes(),before)

    def test_worker_failure_stops_every_worker_and_preserves_failed_partial(self):
        data=b"z"*65536
        rows=[row(f"{i}.bin",data) for i in range(36)]
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            original=wan.download_one
            def injected(item,target,deadline,stop,opener=None):
                return original(item,target,deadline,stop,Opener(data,wrong_range=item["path"]=="0.bin",delay=.001))
            with mock.patch.object(wan.identity,"BASE",base),mock.patch.object(wan.sys,"platform","linux"),mock.patch.object(wan,"download_one",side_effect=injected):
                self.assertEqual(wan.apply(rows,10,3),2)
            self.assertFalse((base/"wan006-transfer.lock").exists())
            receipt=json.loads(next(base.glob("wan006-transfer-v002-*.json")).read_text())
            self.assertTrue(receipt["all_workers_exited_before_lock_release"])
            self.assertTrue(any(r["status"]=="failed_partial_preserved" for r in receipt["files"]))
            self.assertFalse(any(t.name.startswith("wan-owned-transfer") for t in threading.enumerate()))


if __name__=="__main__":unittest.main()
