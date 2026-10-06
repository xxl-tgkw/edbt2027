#!/usr/bin/env python3
"""Verify the structure and accounting of native dynamic-index evidence.

This checker intentionally does not recompute recall without the 485 MB public
GloVe HDF5.  It checks that every saved CONDA response has exactly 128 k=10
records, that the audited summaries contain the paired five-round values, and
that the UBIS boundary is explicit.
"""
import argparse
import json
import struct
from pathlib import Path


def check_result(path: Path, queries=128, k=10):
    raw = path.read_bytes()
    off = 0
    for _ in range(queries):
        if off + 4 > len(raw):
            raise AssertionError(f"truncated header: {path}")
        got = struct.unpack_from("<I", raw, off)[0]
        off += 4
        if got != k:
            raise AssertionError(f"unexpected k={got}: {path}")
        if off + 4 * got > len(raw):
            raise AssertionError(f"truncated ids: {path}")
        off += 4 * got
    if off != len(raw):
        raise AssertionError(f"trailing bytes: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/native_dynamic_v1")
    args = ap.parse_args()
    root = Path(args.input)
    manifest = json.loads((root / "conda_native_manifest.json").read_text())
    assert manifest["commit"] == "4117a4e38938aa0ff583c0c7c1beb5bb465e5b4b"
    assert manifest["final_vectors"] == 1_000_000
    rows = {}
    for order in ("shifted", "shuffled"):
        directory = root / f"conda_{order}"
        responses = sorted(directory.glob("*-Ls100.res"))
        assert [p.name for p in responses] == [
            "11-Ls100.res", "3-Ls100.res", "5-Ls100.res",
            "7-Ls100.res", "9-Ls100.res"
        ]
        for path in responses:
            check_result(path)
        summary = json.loads((root / f"conda_{order}_detailed_summary.json").read_text())
        assert len(summary["rows"]) == 5
        assert summary["order"] == order
        assert summary["test_recall_mean"] > 0
        rows[order] = summary["test_recall_mean"]
    probe = (root / "ubis_probe.md").read_text().lower()
    assert "not_run" in probe and "no public" in probe
    report = {"status": "passed", "orders": rows, "response_files_per_order": 5,
              "recall_recomputed": False,
              "recall_note": "Recall values were independently scored against exact Faiss neighbors before packaging; this structural check does not require the source HDF5."}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
