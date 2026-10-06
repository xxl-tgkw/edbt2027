#!/usr/bin/env python3
"""Recompute CONDA native recall from saved response streams and public GloVe."""
import argparse
import json
import struct
from pathlib import Path

import faiss
import h5py
import numpy as np


def read_res(path, nq=128, k=10):
    raw = path.read_bytes(); off = 0; rows = []
    for _ in range(nq):
        got = struct.unpack_from("<I", raw, off)[0]; off += 4
        if got != k: raise ValueError(f"{path}: expected k={k}, got {got}")
        rows.append(np.frombuffer(raw, dtype="<u4", count=k, offset=off).copy())
        off += 4 * k
    if off != len(raw): raise ValueError(f"{path}: trailing bytes")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/semantic_glove/glove-100-angular.hdf5")
    ap.add_argument("--inputs", required=True, help="directory from prepare_conda_inputs.py")
    ap.add_argument("--results", required=True, help="native_dynamic_v1/conda_<order>")
    args = ap.parse_args()
    inputs, results = Path(args.inputs), Path(args.results)
    with h5py.File(args.source, "r") as source:
        train = source["train"][:]
        queries = source["test"][:128].astype("float32")
    meta = json.loads((inputs / "meta.json").read_text())
    ids = np.asarray(meta["source_ids"], dtype="int64")
    x = train[ids].astype("float32")
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    rows = []
    for round_, step, size in zip(range(1, 6), [3, 5, 7, 9, 11],
                                  [840000, 880000, 920000, 960000, 1000000]):
        exact = faiss.IndexFlatL2(x.shape[1]); exact.add(x[:size])
        _, truth = exact.search(queries, 10)
        got = read_res(results / f"{step}-Ls100.res")
        recall = [len(set(map(int, a)) & set(map(int, b))) / 10
                  for a, b in zip(got, truth)]
        rows.append({"round": round_, "indexed": size,
                     "validation_recall": float(np.mean(recall[:64])),
                     "test_recall": float(np.mean(recall[64:])),
                     "worst_recall": float(min(recall[64:]))})
    report = {"protocol": "semantic-1m-dynamic-v1", "rows": rows,
              "validation_mean": float(np.mean([r["validation_recall"] for r in rows])),
              "test_recall_mean": float(np.mean([r["test_recall"] for r in rows]))}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
