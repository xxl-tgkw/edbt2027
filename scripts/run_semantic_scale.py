#!/usr/bin/env python3
"""Low-cost external scale validation on the public ANN-Benchmarks GloVe source."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import faiss
import h5py
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/semantic_glove/glove-100-angular.hdf5"
PROTOCOL = "semantic-scale-v1"
POLICIES = ["flat", "ivf_fixed", "ivf_rebuilt"]
BUDGETS = {"flat": [0], "ivf_fixed": [4, 16, 64], "ivf_rebuilt": [4, 16, 64]}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize(x):
    x = np.asarray(x, dtype="float32")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if not np.isfinite(x).all() or (norms == 0).any():
        raise ValueError("invalid source vectors")
    return np.ascontiguousarray(x / norms)


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def make_workload(train, queries, final_size, nq, seed, condition):
    if final_size % 5:
        raise ValueError("final_size must be divisible by five")
    rng = np.random.default_rng(seed)
    ids = rng.choice(len(train), size=final_size, replace=False)
    vectors = normalize(train[ids])
    query = normalize(queries[:nq])
    design = vectors[: min(10000, final_size)].astype("float64")
    centered = design - design.mean(0, keepdims=True)
    _, eigvecs = np.linalg.eigh(centered.T @ centered)
    direction = eigvecs[:, -1]
    scores = vectors @ direction.astype("float32")
    ranked = np.argsort(scores, kind="stable")
    updates = final_size // 5
    base = final_size - updates
    canonical = np.r_[ranked[:base], ranked[base:]]
    if condition == "shifted":
        order = canonical
    elif condition == "shuffled":
        order = canonical[np.random.default_rng(seed + 10000).permutation(final_size)]
    else:
        raise ValueError(condition)
    x = np.ascontiguousarray(vectors[order])
    truth = []
    exact = faiss.IndexFlatL2(x.shape[1])
    for round_id in range(6):
        exact.reset()
        exact.add(x[:base + round_id * updates])
        _, nn = exact.search(query, 10)
        truth.append(nn.tolist())
    return x, query, truth, {
        "protocol": PROTOCOL,
        "seed": seed,
        "condition": condition,
        "final_size": final_size,
        "base": base,
        "batch": updates,
        "rounds": 5,
        "nq": nq,
        "source_ids": ids[order].tolist(),
        "source_sha256": sha(SOURCE),
        "corpus_sha256": array_sha(x),
        "queries_sha256": array_sha(query),
        "distance": "squared Euclidean on L2-normalized GloVe; equivalent ranking to angular distance",
    }


class NativeIndex:
    def __init__(self, x, policy, seed, nlist):
        self.policy, self.seed, self.nlist = policy, seed, nlist
        self.rebuild(x)

    def rebuild(self, x):
        d = x.shape[1]
        if self.policy == "flat":
            self.index = faiss.IndexFlatL2(d)
        else:
            self.index = faiss.IndexIVFFlat(faiss.IndexFlatL2(d), d, self.nlist, faiss.METRIC_L2)
            self.index.cp.seed = self.seed
            self.index.cp.niter = 25
            self.index.cp.nredo = 1
            self.index.cp.max_points_per_centroid = 10000
            self.index.train(x)
        self.index.add(x)

    def budget(self, value):
        if self.policy != "flat":
            self.index.nprobe = value

    def search(self, q):
        stats = faiss.cvar.indexIVF_stats if self.policy != "flat" else None
        if stats is not None:
            stats.reset()
        start = time.perf_counter_ns()
        _, ids = self.index.search(q.reshape(1, -1), 10)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        work = int(stats.ndis) if stats is not None else int(self.index.ntotal)
        return ids[0].tolist(), elapsed, work


def evaluate(x, query, truth, meta, policy, seed):
    nlist = max(128, int(2 ** round(np.log2(np.sqrt(meta["base"] / 2)))))
    start = time.perf_counter_ns()
    index = NativeIndex(x[: meta["base"]], policy, seed, nlist)
    build_ms = (time.perf_counter_ns() - start) / 1e6
    for q in query[: min(10, len(query))]:
        index.index.search(q[None], 10)
    events = []
    maintenance = 0.0
    for round_id in range(6):
        insert_ms = rebuild_ms = 0.0
        if round_id:
            lo = meta["base"] + (round_id - 1) * meta["batch"]
            hi = meta["base"] + round_id * meta["batch"]
            start = time.perf_counter_ns()
            index.index.add(x[lo:hi])
            insert_ms = (time.perf_counter_ns() - start) / 1e6
        rebuilt = policy == "ivf_rebuilt" and round_id in (2, 4)
        if rebuilt:
            start = time.perf_counter_ns()
            index.rebuild(x[: meta["base"] + round_id * meta["batch"]])
            rebuild_ms = (time.perf_counter_ns() - start) / 1e6
        budget_results = {}
        for budget in BUDGETS[policy]:
            index.budget(budget)
            returned, durations, recalls, work = [], [], [], []
            for q, nn in zip(query, truth[round_id]):
                ids, elapsed, distance_count = index.search(q)
                returned.append(ids)
                durations.append(elapsed)
                work.append(distance_count)
                recalls.append(len(set(ids) & set(nn)) / 10)
            budget_results[str(budget)] = {
                "recall": float(np.mean(recalls)),
                "query_p95_ms": float(np.percentile(durations, 95)),
                "query_mean_ms": float(np.mean(durations)),
                "distance_count": float(np.mean(work)),
            }
        maintenance += insert_ms + rebuild_ms
        events.append({"round": round_id, "indexed": int(index.index.ntotal), "insert_ms": insert_ms,
                       "rebuilt": rebuilt, "rebuild_ms": rebuild_ms, "budgets": budget_results})
    summaries = {}
    for budget in BUDGETS[policy]:
        rows = [e["budgets"][str(budget)] for e in events[1:]]
        summaries[str(budget)] = {
            "recall": float(np.mean([r["recall"] for r in rows])),
            "query_p95_ms": float(np.mean([r["query_p95_ms"] for r in rows])),
            "query_mean_ms": float(np.mean([r["query_mean_ms"] for r in rows])),
            "distance_count": float(np.mean([r["distance_count"] for r in rows])),
            "service_ms": maintenance + float(np.sum([r["query_mean_ms"] * meta["nq"] for r in rows])),
        }
    return {"protocol": PROTOCOL, "seed": seed, "condition": meta["condition"], "policy": policy,
            "build_ms": build_ms, "maintenance_ms": maintenance, "nlist": nlist,
            "events": events, "summaries": summaries, "corpus_sha256": meta["corpus_sha256"],
            "queries_sha256": meta["queries_sha256"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[100000, 250000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--nq", type=int, default=128)
    args = parser.parse_args()
    out = ROOT / args.output
    if out.exists():
        raise FileExistsError("Use a fresh output directory")
    out.mkdir(parents=True)
    faiss.omp_set_num_threads(1)
    started = time.time()
    with h5py.File(SOURCE, "r") as source:
        train = source["train"][:]
        queries = source["test"][:]
    if max(args.sizes) > len(train) or args.nq > len(queries):
        raise ValueError("Requested scale exceeds public source")
    env = {"protocol": PROTOCOL, "args": vars(args), "source_sha256": sha(SOURCE),
           "source_shape": [int(v) for v in train.shape], "python": platform.python_version(),
           "faiss": faiss.__version__, "started_unix": started}
    rows = []
    for size in args.sizes:
        for seed in args.seeds:
            workloads = {}
            for condition in ("shifted", "shuffled"):
                workloads[condition] = make_workload(train, queries, size, args.nq, seed, condition)
                (out / f"workload_{size}_{condition}_{seed}.json").write_text(json.dumps(workloads[condition][3]) + "\n")
            jobs = [(condition, policy) for condition in workloads for policy in POLICIES]
            for condition, policy in jobs:
                row = evaluate(*workloads[condition][:3], workloads[condition][3], policy, seed)
                row["size"] = size
                rows.append(row)
                print(f"{len(rows)} size={size} seed={seed} {condition} {policy}", flush=True)
    (out / "raw.json").write_text(json.dumps(rows) + "\n")
    env["elapsed_s"] = time.time() - started
    (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
    by = {}
    for row in rows:
        for budget, value in row["summaries"].items():
            key = (row["size"], row["condition"], row["policy"], int(budget))
            by.setdefault(key, []).append(value)
    summary = []
    for (size, condition, policy, budget), values in sorted(by.items()):
        summary.append({"size": size, "condition": condition, "policy": policy, "budget": budget,
                        **{k: float(np.mean([v[k] for v in values])) for k in values[0]}})
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Semantic scale validation", "", f"This is a {len(args.seeds)}-seed external scale check; it is not pooled with the 10-seed mechanism study.",
             "", "| Size | Order | Policy | Budget | Recall | p95 ms | Service ms |", "|---:|---|---|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['size']} | {row['condition']} | {row['policy']} | {row['budget']} | {row['recall']:.4f} | {row['query_p95_ms']:.3f} | {row['service_ms']:.1f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
