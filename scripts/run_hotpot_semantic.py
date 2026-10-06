#!/usr/bin/env python3
"""Run the paired dynamic-index protocol on HotpotQA sentence embeddings."""
import argparse
import hashlib
import itertools
import json
import platform
import subprocess
import time
from pathlib import Path

import faiss
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from prepare_semantic_data import sha

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/semantic_hotpot"
POLICIES = ["flat", "ivf_fixed", "ivf_rebuilt", "hnsw"]
BUDGETS = {"flat": [0], "ivf_fixed": [4, 16, 64],
           "ivf_rebuilt": [4, 16, 64], "hnsw": [16, 64, 256]}
PROTOCOL = "hotpot-semantic-order-v1"


def arr_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def load_pool():
    manifest = json.loads((DATA / "manifest.json").read_text())
    if sha(DATA / "pool.npz") != manifest["pool_sha256"]:
        raise ValueError("Hotpot pool checksum mismatch")
    with np.load(DATA / "pool.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}, manifest


def truth(x, q, active):
    # Normalized embeddings make maximum inner product equivalent to cosine.
    sims = q.astype("float32") @ x[active].astype("float32").T
    local = np.argpartition(-sims, 10, axis=1)[:, :10]
    rows = np.arange(len(q))[:, None]
    order = np.argsort(-sims[rows, local], axis=1, kind="stable")
    return active[local[rows, order]].tolist()


def workload(pool, seed, condition):
    rng = np.random.default_rng(seed)
    old = rng.choice(pool["old_pool"], 80_000, replace=False)
    incoming = rng.choice(pool["new_pool"], 20_000, replace=False)
    final_ids = np.r_[old, incoming]
    if condition == "shuffled":
        final_ids = final_ids[np.random.default_rng(seed + 91000).permutation(len(final_ids))]
    elif condition != "shifted":
        raise ValueError(condition)
    qids = np.arange(len(pool["queries"]), dtype=np.int64)
    # A fixed disjoint validation/test split is shared by both arrival orders.
    qids = qids[np.random.default_rng(20261005).permutation(len(qids))]
    val, test = qids[:128], qids[128:256]
    x = np.ascontiguousarray(pool["vectors"][final_ids])
    q = np.ascontiguousarray(pool["queries"][np.r_[val, test]])
    gt = []
    for b in range(6):
        active = np.arange(80_000 + 4_000 * b, dtype=np.int64)
        gt.append(truth(x, q, active))
    meta = dict(protocol=PROTOCOL, seed=seed, condition=condition, base=80_000,
                batch=4_000, rounds=5, query_count=256,
                validation_queries=val.tolist(), test_queries=test.tolist(),
                corpus_sha256=arr_sha(x), queries_sha256=arr_sha(q),
                source_ids=pool["source_ids"][final_ids].tolist(), ground_truth=gt)
    return x, q, gt, meta


class Index:
    def __init__(self, x, policy, seed):
        self.policy, self.seed = policy, seed
        self.rebuild(x)

    def rebuild(self, x):
        d = x.shape[1]
        if self.policy == "flat":
            self.index = faiss.IndexFlatIP(d)
        elif self.policy.startswith("ivf"):
            self.index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, 128, faiss.METRIC_INNER_PRODUCT)
            self.index.cp.seed = self.seed
            self.index.cp.niter = 20
            self.index.cp.nredo = 1
            self.index.cp.max_points_per_centroid = 10000
            self.index.train(x)
        elif self.policy == "hnsw":
            self.index = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = 80
            self.index.hnsw.rng = faiss.RandomGenerator(self.seed)
        else:
            raise ValueError(self.policy)
        self.index.add(x)

    def set_budget(self, budget):
        if self.policy.startswith("ivf"):
            self.index.nprobe = budget
        elif self.policy == "hnsw":
            self.index.hnsw.efSearch = budget

    def search(self, q):
        stats = faiss.cvar.indexIVF_stats if self.policy.startswith("ivf") else None
        if stats is not None:
            stats.reset()
        t = time.perf_counter_ns()
        _, ids = self.index.search(q[None], 10)
        dt = (time.perf_counter_ns() - t) / 1e6
        return ids[0].tolist(), dt, int(stats.ndis) if stats is not None else int(self.index.ntotal)


def query_set(idx, q, gt):
    times, recalls, work = [], [], []
    for vector, target in zip(q, gt):
        got, dt, ndis = idx.search(vector)
        valid = [int(i) for i in got if i >= 0]
        times.append(dt); work.append(ndis)
        recalls.append(len(set(valid) & set(target)) / 10.0)
    return dict(recall=float(np.mean(recalls)), query_mean_ms=float(np.mean(times)),
                query_p95_ms=float(np.percentile(times, 95)), distance_count=float(np.mean(work)),
                query_ms=times, recalls=recalls)


def evaluate(x, q, gt, meta, policy, seed):
    base, batch = meta["base"], meta["batch"]
    t = time.perf_counter_ns(); idx = Index(x[:base], policy, seed)
    build_ms = (time.perf_counter_ns() - t) / 1e6
    rng = np.random.default_rng(seed + 93000)
    events = []
    for b in range(6):
        insert_ms = rebuild_ms = 0.0
        if b:
            t = time.perf_counter_ns(); idx.index.add(x[base + (b - 1) * batch:base + b * batch])
            insert_ms = (time.perf_counter_ns() - t) / 1e6
        if policy == "ivf_rebuilt" and b in (2, 4):
            t = time.perf_counter_ns(); idx.rebuild(x[:base + b * batch])
            rebuild_ms = (time.perf_counter_ns() - t) / 1e6
        budgets = {}
        for budget in rng.permutation(BUDGETS[policy]):
            idx.set_budget(int(budget))
            budgets[str(budget)] = dict(validation=query_set(idx, q[:128], gt[b][:128]),
                                         test=query_set(idx, q[128:], gt[b][128:]))
        events.append(dict(round=b, indexed=int(idx.index.ntotal), insert_ms=insert_ms,
                           rebuild_ms=rebuild_ms, rebuilt=policy == "ivf_rebuilt" and b in (2, 4),
                           budgets=budgets))
    maintenance = sum(e["insert_ms"] + e["rebuild_ms"] for e in events[1:])
    summaries = {}
    for budget in BUDGETS[policy]:
        test = [e["budgets"][str(budget)]["test"] for e in events[1:]]
        times = [v for row in test for v in row["query_ms"]]
        summaries[str(budget)] = dict(recall=float(np.mean([r["recall"] for r in test])),
                                      query_p95_ms=float(np.percentile(times, 95)),
                                      query_mean_ms=float(np.mean(times)),
                                      service_ms=maintenance + sum(times),
                                      distance_count=float(np.mean([r["distance_count"] for r in test])))
    return dict(protocol=PROTOCOL, seed=seed, condition=meta["condition"], policy=policy,
                build_ms=build_ms, maintenance_ms=maintenance,
                serialized_bytes=len(faiss.serialize_index(idx.index)), events=events,
                summaries=summaries, corpus_sha256=meta["corpus_sha256"],
                queries_sha256=meta["queries_sha256"],
                index_parameters=dict(dimension=384, nlist=128, clustering_iterations=20,
                                      M=16, efConstruction=80))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    out = ROOT / args.output
    if out.exists() and not args.resume:
        raise FileExistsError("Use a fresh output directory or pass --resume")
    out.mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(1); started = time.time()
    with threadpool_limits(limits=1):
        pool, manifest = load_pool()
        env = dict(protocol=PROTOCOL, args=vars(args), started_unix=started,
                   python=platform.python_version(), numpy=np.__version__, faiss=faiss.__version__,
                   threads=threadpool_info(), cpu=subprocess.check_output(["lscpu"], text=True),
                   data_manifest=manifest,
                   source_hashes={f: sha(ROOT / f) for f in ["scripts/run_hotpot_semantic.py",
                                                            "scripts/prepare_hotpot_semantic_data.py"]},
                   caveats="CPU-only paired document-vector validation; no production timestamps or end-to-end QA metric")
        (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
        rows = []
        existing_events = out / "events.jsonl"
        if args.resume and existing_events.exists():
            rows = [json.loads(line) for line in existing_events.read_text().splitlines() if line.strip()]
        completed = {(r["seed"], r["condition"], r["policy"]) for r in rows}
        for seed in args.seeds:
            workloads = {c: workload(pool, seed, c) for c in ["shifted", "shuffled"]}
            for c, (_, _, _, m) in workloads.items():
                (out / f"workload_{c}_{seed}.json").write_text(json.dumps(m) + "\n")
            jobs = list(itertools.product(["shifted", "shuffled"], POLICIES)); np.random.default_rng(seed + 94000).shuffle(jobs)
            for condition, policy in jobs:
                if (seed, condition, policy) in completed:
                    continue
                row = evaluate(*workloads[condition], policy, seed); rows.append(row)
                with (out / "events.jsonl").open("a") as f: f.write(json.dumps(row) + "\n")
                completed.add((seed, condition, policy))
                print(f"{len(rows)}/{len(args.seeds)*8} seed={seed} {condition} {policy} "
                      + ", ".join(f"{b}:R={r['recall']:.3f} C={r['service_ms']:.0f}ms" for b, r in row["summaries"].items()), flush=True)
        (out / "raw.json").write_text(json.dumps(rows) + "\n")
        env["elapsed_s"] = time.time() - started; (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")


if __name__ == "__main__": main()
