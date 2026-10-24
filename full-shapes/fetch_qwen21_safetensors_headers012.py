"""Read only bounded safetensors headers, never checkpoint tensor payloads."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import urllib.request

from download_candidate_models_20261023 import OfficialRedirects, allowed_download_url

LOCK_SHA256 = "dbab7a7daf5ac2e000eda646fbb74f0bdc8ee9fd4822199043e7b03cf4fec5fd"
WIDTH = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def read_range(opener, url, start, stop, expected_file_bytes):
    if not allowed_download_url(url):
        raise ValueError("Unreviewed source")
    size = stop - start + 1
    assert 0 < size <= 16 * 1024 * 1024
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{stop}", "Accept-Encoding": "identity"})
    with opener.open(request, timeout=30) as response:
        # A server ignoring Range is refused before reading any tensor body.
        assert response.status == 206, "Source ignored bounded Range"
        assert allowed_download_url(response.geturl())
        assert response.headers.get("Content-Range") == f"bytes {start}-{stop}/{expected_file_bytes}"
        raw = response.read(size + 1)
        assert len(raw) == size
        return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_lock = args.lock.read_bytes()
    assert hashlib.sha256(raw_lock).hexdigest() == LOCK_SHA256
    rows = [row for row in json.loads(raw_lock)["files"] if row["path"].endswith(".safetensors")]
    assert len(rows) == 4
    output = args.output.absolute()
    assert not output.exists() and not any(path.is_symlink() for path in (output, *output.parents))
    output.mkdir(parents=True)
    opener = urllib.request.build_opener(OfficialRedirects())
    results = []
    for row in rows:
        length_raw = read_range(opener, row["download_url"], 0, 7, row["size"])
        length = struct.unpack("<Q", length_raw)[0]
        assert 2 <= length <= 16 * 1024 * 1024
        header_raw = read_range(opener, row["download_url"], 8, 7 + length, row["size"])
        header = json.loads(header_raw)
        tensors = {name: value for name, value in header.items() if name != "__metadata__"}
        spans = []
        for name, entry in tensors.items():
            assert isinstance(name, str) and entry["dtype"] in WIDTH
            count = 1
            for dim in entry["shape"]:
                assert isinstance(dim, int) and dim >= 0
                count *= dim
            first, last = entry["data_offsets"]
            assert 0 <= first <= last <= row["size"] - 8 - length
            assert last - first == count * WIDTH[entry["dtype"]]
            spans.append((first, last))
        spans.sort()
        assert spans[0][0] == 0 and spans[-1][1] == row["size"] - 8 - length
        assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))
        name = row["path"].replace("/", "__") + ".header.json"
        (output / name).write_bytes(header_raw)
        results.append({"path": row["path"], "component": row["component"], "checkpoint_expected_bytes": row["size"],
            "checkpoint_expected_sha256": row["sha256"], "actual_header_bytes": length,
            "actual_header_sha256": hashlib.sha256(header_raw).hexdigest(), "saved_header": name,
            "tensor_count": len(tensors), "payload_offsets_and_sizes_valid": True,
            "checkpoint_full_sha_verified": False, "modelscope_revision": row["modelscope_revision"]})
    record = {"source_lock_sha256": LOCK_SHA256, "files": results,
        "actual_network_header_bytes": sum(row["actual_header_bytes"] + 8 for row in results),
        "tensor_payload_bytes_requested": 0, "full_checkpoint_downloaded": False, "GPU_used": False}
    with (output / "header-receipt012.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2); stream.write("\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
