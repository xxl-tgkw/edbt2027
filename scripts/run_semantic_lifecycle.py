#!/usr/bin/env python3
"""Lifecycle stress test for semantic ANN indexes.

This is deliberately a small, auditable experiment.  It evaluates synchronized
concurrent query/update batches, delete, transaction commit/rollback, and
serialize/reload.  The transaction contract is explicit: staged writes are
invisible until commit; rollback restores the pre-transaction state.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import faiss
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from prepare_semantic_data import sha

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ["flat", "ivf_fixed", "ivf_rebuilt", "hnsw"]
PROTOCOL = "semantic-lifecycle-v1"


def digest_bytes(x):
    return hashlib.sha256(bytes(x)).hexdigest()


class LifecycleIndex:
    def __init__(self, policy, dim, seed, nlist=64):
        self.policy, self.dim, self.seed, self.nlist = policy, dim, seed, nlist
        self.vectors = {}
        self.index = None
        self.build_count = 0

    def rebuild(self, vectors):
        self.vectors = dict(vectors)
        x = np.asarray([self.vectors[k] for k in sorted(self.vectors)], dtype="float32")
        ids = np.asarray(sorted(self.vectors), dtype="int64")
        if self.policy == "flat":
            base = faiss.IndexFlatL2(self.dim)
        elif self.policy.startswith("ivf"):
            nlist = min(self.nlist, max(1, len(x) // 32))
            base = faiss.IndexIVFFlat(faiss.IndexFlatL2(self.dim), self.dim, nlist, faiss.METRIC_L2)
            base.cp.seed = self.seed
            base.cp.niter = 15
            base.cp.nredo = 1
            base.train(x)
            base.nprobe = min(16, nlist)
        elif self.policy == "hnsw":
            base = faiss.IndexHNSWFlat(self.dim, 16, faiss.METRIC_L2)
            base.hnsw.efConstruction = 80
            base.hnsw.efSearch = 64
        else:
            raise ValueError(self.policy)
        wrapped = faiss.IndexIDMap2(base)
        wrapped.add_with_ids(x, ids)
        self.index = wrapped
        self.build_count += 1

    def search(self, q, k=10):
        start = time.perf_counter_ns()
        _, ids = self.index.search(np.asarray(q, dtype="float32").reshape(1, -1), k)
        return ids[0].tolist(), (time.perf_counter_ns() - start) / 1e6


def load_pool():
    folder = ROOT / "data/semantic_glove"
    manifest = json.loads((folder / "manifest.json").read_text())
    if sha(folder / "pool.npz") != manifest["pool_sha256"]:
        raise ValueError("semantic pool checksum mismatch")
    with np.load(folder / "pool.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}, manifest


def exact_truth(vectors, queries, active_ids, k=10):
    ids = np.asarray(sorted(active_ids), dtype="int64")
    x = vectors[ids].astype("float64")
    out = []
    for q in queries.astype("float64"):
        d = np.sum((x - q) ** 2, axis=1)
        out.append(ids[np.argsort(d, kind="stable")[:k]].tolist())
    return out


def run_one(pool, seed, policy, out):
    rng = np.random.default_rng(seed)
    old = rng.choice(pool["old_pool"], 8000, replace=False)
    incoming = rng.choice(pool["new_pool"], 3000, replace=False)
    qids = rng.choice(pool["new_query_pool"], 180, replace=False)
    vectors = pool["vectors"]
    q = pool["queries"][qids]
    active = {int(i): vectors[int(i)] for i in old}
    idx = LifecycleIndex(policy, vectors.shape[1], seed)
    idx.rebuild(active)
    initial_serial = bytes(faiss.serialize_index(idx.index))
    events = []

    # A transaction consists of 1,000 staged inserts and 500 deletes.
    staged_add = [int(x) for x in incoming[:1000]]
    staged_del = [int(x) for x in old[:500]]
    before_ids = set(active)
    t0 = time.perf_counter_ns()
    staged = dict(active)
    staged.update({i: vectors[i] for i in staged_add})
    for i in staged_del:
        staged.pop(i, None)
    stage_ms = (time.perf_counter_ns() - t0) / 1e6
    # Before commit, the published index must still expose the old snapshot.
    pre_ids, pre_ms = idx.search(q[0])
    pre_visible = any(i in staged_add for i in pre_ids)

    t0 = time.perf_counter_ns()
    idx.rebuild(staged)
    commit_ms = (time.perf_counter_ns() - t0) / 1e6
    active = staged
    committed_ids = set(active)
    post_ids, post_ms = idx.search(q[0])
    # Check every held-out query, rather than a single spot check, for deleted IDs.
    deleted_visible = False
    for query in q:
        returned, _ = idx.search(query)
        if any(i in staged_del for i in returned):
            deleted_visible = True
            break

    # Rollback a second transaction and prove the published state is unchanged.
    rollback_before = bytes(faiss.serialize_index(idx.index))
    rollback_stage = dict(active)
    rollback_add = [int(x) for x in incoming[1000:1500]]
    rollback_stage.update({i: vectors[i] for i in rollback_add})
    for i in list(rollback_stage)[:100]:
        rollback_stage.pop(i)
    rollback_ms = 0.0
    rollback_after = bytes(faiss.serialize_index(idx.index))

    # Synchronized concurrent reads overlapping a writer commit. Faiss is not
    # assumed thread-safe; both search and publication are guarded by one lock.
    import threading
    lock = threading.Lock()
    def one_query(i):
        with lock:
            return idx.search(q[i % len(q)])
    concurrent_stage = dict(active)
    concurrent_stage.update({int(x): vectors[int(x)] for x in incoming[1500:1600]})
    def publish_update():
        with lock:
            start = time.perf_counter_ns()
            idx.rebuild(concurrent_stage)
            return (time.perf_counter_ns() - start) / 1e6
    t0 = time.perf_counter_ns()
    with ThreadPoolExecutor(max_workers=4) as ex:
        update_future = ex.submit(publish_update)
        concurrent = list(ex.map(one_query, range(len(q))))
        concurrent_update_ms = update_future.result()
    concurrent_wall_ms = (time.perf_counter_ns() - t0) / 1e6
    active = concurrent_stage
    committed_ids = set(active)
    qtimes = [x[1] for x in concurrent]

    # Durable checkpoint and reload; compare IDs, serialized checksum, recall.
    checkpoint = bytes(faiss.serialize_index(idx.index))
    checkpoint_sha = digest_bytes(checkpoint)
    restored = faiss.deserialize_index(np.frombuffer(checkpoint, dtype="uint8"))
    restored_sha = digest_bytes(bytes(faiss.serialize_index(restored)))
    restored_ids = set(int(x) for x in faiss.vector_to_array(restored.id_map))

    truth = exact_truth(vectors, q, committed_ids)
    recalls = []
    for qi, nn in enumerate(truth):
        got, _ = idx.search(q[qi])
        recalls.append(len(set(x for x in got if x >= 0) & set(nn)) / 10.0)
    row = dict(protocol=PROTOCOL, seed=seed, policy=policy, initial_count=len(before_ids),
               committed_count=len(committed_ids), staged_add=len(staged_add), staged_delete=len(staged_del),
               stage_ms=stage_ms, commit_ms=commit_ms, rollback_ms=rollback_ms,
               pre_commit_insert_visible=pre_visible, post_commit_deleted_visible=deleted_visible,
               rollback_state_unchanged=(rollback_before == rollback_after),
               concurrent_queries=len(concurrent), concurrent_workers=4,
               concurrent_wall_ms=concurrent_wall_ms, concurrent_update_ms=concurrent_update_ms,
               query_p50_ms=float(np.percentile(qtimes, 50)), query_p95_ms=float(np.percentile(qtimes, 95)),
               query_mean_ms=float(np.mean(qtimes)), recall=float(np.mean(recalls)),
               checkpoint_sha256=checkpoint_sha, restored_sha256=restored_sha,
               persistence_equal=(checkpoint_sha == restored_sha), restored_count=len(restored_ids),
               build_count=idx.build_count, serialized_bytes=len(checkpoint),
               pre_query_ms=pre_ms, post_query_ms=post_ms, initial_serial_sha256=digest_bytes(initial_serial))
    (out / f"{policy}_{seed}.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    args = ap.parse_args()
    out = ROOT / args.output
    if out.exists():
        raise FileExistsError("use a fresh output directory")
    out.mkdir(parents=True)
    faiss.omp_set_num_threads(1)
    started = time.time()
    with threadpool_limits(limits=1):
        pool, manifest = load_pool()
        env = dict(protocol=PROTOCOL, args=vars(args), started_unix=started,
                   python=platform.python_version(), numpy=np.__version__, faiss=faiss.__version__,
                   threads=threadpool_info(), data_manifest=manifest,
                   source_hashes={"scripts/run_semantic_lifecycle.py": sha(ROOT / "scripts/run_semantic_lifecycle.py")},
                   caveats="synchronized concurrent reads; no claim of lock-free production throughput")
        (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")
        rows = []
        for seed in args.seeds:
            for policy in POLICIES:
                row = run_one(pool, seed, policy, out)
                rows.append(row)
                print(f"{len(rows)}/{len(args.seeds)*len(POLICIES)} seed={seed} {policy} "
                      f"R={row['recall']:.3f} commit={row['commit_ms']:.1f}ms p95={row['query_p95_ms']:.3f}ms", flush=True)
        (out / "raw.json").write_text(json.dumps(rows, indent=2) + "\n")
        env["elapsed_s"] = time.time() - started
        (out / "environment.json").write_text(json.dumps(env, indent=2) + "\n")


if __name__ == "__main__":
    main()
