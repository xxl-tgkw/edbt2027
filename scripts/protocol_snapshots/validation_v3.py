#!/usr/bin/env python3
"""Audited, paired interleaving with disjoint public vectors; legacy runs untouched."""
import argparse
import gzip
import hashlib
import itertools
import json
import platform
import struct
import subprocess
import time
from pathlib import Path

import faiss
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits, threadpool_info

from run_benchmark import Exact, LSH, IVF

ROOT = Path(__file__).resolve().parents[1]
DATA_SHA = '3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84'
METHODS = ['exact', 'lsh', 'ivf_static', 'ivf_rebuild',
           'faiss_flat', 'faiss_ivf_static', 'faiss_ivf_rebuild']


def hash_array(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def load_data():
    path = ROOT / 'data/fashion_train_images.gz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA:
        raise ValueError('Fashion-MNIST cache checksum mismatch')
    with gzip.open(path, 'rb') as f:
        header = struct.unpack('>IIII', f.read(16))
        payload = f.read()
    if header != (2051, 60000, 28, 28) or len(payload) != 60000 * 784:
        raise ValueError('Fashion-MNIST IDX header/payload mismatch')
    return np.frombuffer(payload, dtype=np.uint8).reshape(60000, 784)


def workload_ids(seed, count=60000, base=6000, rounds=5, batch=100, queries=160):
    # One permutation establishes row-disjoint roles; query prefixes nest within rounds.
    ids = np.random.default_rng(seed).permutation(count)
    required = base + rounds * (batch + queries)
    if required > count:
        raise ValueError('Not enough rows for disjoint roles')
    return (ids[:base], ids[base:base + rounds * batch].reshape(rounds, batch),
            ids[base + rounds * batch:required].reshape(rounds, queries))


class NativeFaiss:
    """No per-insert Python matrix copy; frozen corpus supplied only for rebuild."""
    def __init__(self, x, method, seed):
        self.method, self.seed = method, seed
        self.rebuild(x)

    def rebuild(self, x):
        if self.method == 'faiss_flat':
            self.index = faiss.IndexFlatL2(x.shape[1])
        else:
            self.index = faiss.IndexIVFFlat(faiss.IndexFlatL2(x.shape[1]), x.shape[1], 32)
            self.index.cp.seed = self.seed
            self.index.cp.niter = 30
            self.index.cp.nredo = 1
            self.index.train(x)
            self.index.nprobe = 4
        self.index.add(x)

    def add(self, x):
        self.index.add(x)

    def search(self, q, k=10):
        return self.index.search(q.reshape(1, -1), k)[1][0]


def exact_truth(corpus, queries):
    # Float64 direct distance computation, blocked to bound temporary memory.
    truth = []
    x = corpus.astype(np.float64)
    for q in queries:
        dist = np.sum((x - q.astype(np.float64)) ** 2, axis=1)
        truth.append(np.argsort(dist, kind='stable')[:10].tolist())
    return truth


def evaluate(corpus, updates, queries, truths, method, seed, ratio, hashes):
    t = time.perf_counter()
    if method.startswith('faiss_'):
        idx = NativeFaiss(corpus[:6000], method, seed)
    elif method == 'exact':
        idx = Exact(corpus[:6000])
    elif method == 'lsh':
        idx = LSH(corpus[:6000], seed)
    else:
        idx = IVF(corpus[:6000], seed=seed)
    build_ms = (time.perf_counter() - t) * 1000
    # Repeated base vectors warm execution paths; held-out measurement queries stay fresh.
    for q in corpus[:10]:
        idx.search(q)
    events = []
    for b in range(5):
        t = time.perf_counter()
        idx.add(updates[b])
        insert_ms = (time.perf_counter() - t) * 1000
        rebuild_ms = 0.0
        rebuilt = method.endswith('_rebuild') and b in (1, 3)
        if rebuilt:
            t = time.perf_counter()
            if method.startswith('faiss_'):
                idx.rebuild(corpus[:6000 + (b + 1) * 100])
            else:
                idx.rebuild()
            rebuild_ms = (time.perf_counter() - t) * 1000
        q_ms, recalls, returned = [], [], []
        for i, q in enumerate(queries[b, :ratio]):
            t = time.perf_counter()
            got = idx.search(q)
            q_ms.append((time.perf_counter() - t) * 1000)
            got = [int(x) for x in got if x >= 0]
            if len(got) != len(set(got)) or any(x >= 6000 + (b+1)*100 for x in got):
                raise ValueError('Invalid search result IDs')
            recalls.append(len(set(got) & set(truths[b][i])) / 10)
            returned.append(got)
        events.append(dict(round=b+1, indexed=6000+(b+1)*100, query_count=len(q_ms),
                           insert_ms=insert_ms, rebuilt=rebuilt, rebuild_ms=rebuild_ms,
                           query_ms=q_ms, recalls=recalls, returned_ids=returned))
    rec = [v for e in events for v in e['recalls']]
    qt = [v for e in events for v in e['query_ms']]
    maintenance = sum(e['insert_ms']+e['rebuild_ms'] for e in events)
    return dict(protocol='disjoint-interleaved-v2', dataset='fashion_mnist', seed=seed,
                method=method, queries_per_batch=ratio, batch_size=100, rounds=5,
                n_base=6000, n_final=6500, dimension=784, k=10, nlist=32, nprobe=4,
                build_ms=build_ms, recall=float(np.mean(rec)), query_ms_mean=float(np.mean(qt)),
                query_ms_p95=float(np.percentile(qt,95)), query_count=len(qt),
                maintenance_ms=maintenance, service_ms=maintenance+sum(qt),
                payload_mb=6500*784*4/1e6, workload_hashes=hashes, events=events)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='results/validation_v2')
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--ratios', nargs='+', type=int, default=[10,40,160])
    p.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    a = p.parse_args()
    if a.seeds < 1 or any(x < 1 or x > 160 for x in a.ratios):
        p.error('positive seeds and ratios in 1..160 required')
    out = ROOT / a.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'raw.json').exists() or (out / 'events.jsonl').exists():
        raise FileExistsError(f'Refusing to overwrite completed or partial results: {out}')
    faiss.omp_set_num_threads(1)
    started = time.time()
    with threadpool_limits(limits=1):
        data = load_data()
        metadata = dict(protocol='disjoint-interleaved-v2', data_sha256=DATA_SHA,
                        python=platform.python_version(), numpy=np.__version__,
                        sklearn=sklearn.__version__, faiss=faiss.__version__,
                        platform=platform.platform(), threads=threadpool_info(),
                        cpu=subprocess.check_output(['lscpu'], text=True),
                        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        args=vars(a), started_unix=started)
        (out/'environment.json').write_text(json.dumps(metadata, indent=2))
        rows = []
        for seed in range(a.seeds):
            bi, ui, qi = workload_ids(seed)
            base_vectors = data[bi].astype('float32')/255.0
            update_vectors = data[ui].astype('float32')/255.0
            query_vectors = data[qi].astype('float32')/255.0
            drift_rng = np.random.default_rng(seed + 999)
            updates = (update_vectors + drift_rng.normal(0.5, .15,
                        update_vectors.shape).astype('float32')).reshape(5,100,784)
            query_vectors = (query_vectors + .35*.5 +
                             drift_rng.normal(0, .05, query_vectors.shape).astype('float32'))
            corpus = np.vstack([base_vectors, updates.reshape(-1, 784)])
            queries = query_vectors
            truths = [exact_truth(corpus[:6000+(b+1)*100], queries[b]) for b in range(5)]
            hashes = dict(base=hash_array(bi), updates=hash_array(ui), queries=hash_array(qi))
            (out/f'workload_{seed}.json').write_text(json.dumps(dict(
                base_ids=bi.tolist(), update_ids=ui.tolist(), query_ids=qi.tolist(),
                ground_truth=truths, hashes=hashes)))
            jobs = list(itertools.product(a.ratios, a.methods))
            np.random.default_rng(7000+seed).shuffle(jobs)
            for ratio, method in jobs:
                row = evaluate(corpus, updates, queries, truths, method, seed, ratio, hashes)
                rows.append(row)
                with (out/'events.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                print(f'{len(rows)}/{a.seeds*len(jobs)} seed={seed} q={ratio} {method} '
                      f'R={row["recall"]:.4f} service={row["service_ms"]:.1f}ms', flush=True)
        (out/'raw.json').write_text(json.dumps(rows))
        metadata['elapsed_s'] = time.time()-started
        (out/'environment.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
