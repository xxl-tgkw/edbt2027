#!/usr/bin/env python3
"""Materialize the UVR-Bench 1M workload in CONDA's headered binary format.

The source is the public ANN-Benchmarks GloVe HDF5 used by
``scripts/run_dynamic_1m.py``.  This helper changes only the arrival order and
does not alter coordinates.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

import h5py
import numpy as np


def write_headered(path, values):
    values = np.ascontiguousarray(values, dtype="float32")
    with path.open("wb") as stream:
        stream.write(struct.pack("<II", values.shape[0], values.shape[1]))
        values.tofile(stream)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/semantic_glove/glove-100-angular.hdf5")
    ap.add_argument("--output", required=True)
    ap.add_argument("--order", choices=["shifted", "shuffled"], required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.source, "r") as source:
        train = source["train"][:]
        queries = source["test"][:128]
    rng = np.random.default_rng(0)
    base, batch, final = 800_000, 40_000, 1_000_000
    source_ids = rng.choice(len(train), size=final, replace=False)
    vectors = train[source_ids].astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    design = vectors[:10_000].astype("float64")
    centered = design - design.mean(0, keepdims=True)
    _, eigenvectors = np.linalg.eigh(centered.T @ centered)
    scores = vectors @ eigenvectors[:, -1].astype("float32")
    ranked = np.argsort(scores, kind="stable")
    update = ranked[base:]
    if args.order == "shuffled":
        update = np.random.default_rng(10_000).permutation(update)
    order = np.concatenate([ranked[:base], update])
    x = np.ascontiguousarray(vectors[order])
    q = queries.astype("float32")
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    write_headered(out / "data.bin", x)
    write_headered(out / "queries.bin", q)
    meta = {
        "protocol": "semantic-1m-dynamic-v1", "condition": args.order,
        "seed": 0, "base": base, "batch": batch, "rounds": 5,
        "final_size": final, "queries": 128,
        "source_sha256": hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        "queries_sha256": hashlib.sha256(q.tobytes()).hexdigest(),
        "source_ids": source_ids[order].tolist(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({k: meta[k] for k in meta if k != "source_ids"}, indent=2))


if __name__ == "__main__":
    main()
