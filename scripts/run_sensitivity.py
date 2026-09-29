#!/usr/bin/env python3
"""Low-cost, pre-registered parameter sensitivity for UVR-Bench."""
import json, time
from pathlib import Path
import numpy as np
from run_benchmark import workload, Exact, LSH, IVF, FaissIVF, K

def one(seed, kind, value):
    base, queries, updates = workload('digits', seed, .5)
    t=time.perf_counter()
    if kind == 'lsh_bits': idx=LSH(base, seed, bits=value, tables=8)
    elif kind == 'lsh_tables': idx=LSH(base, seed, bits=12, tables=value)
    elif kind == 'ivf_nprobe': idx=IVF(base, seed=seed, nprobe=value)
    elif kind == 'ivf_nlist': idx=IVF(base, seed=seed, nlist=value)
    elif kind == 'rebuild_period': idx=IVF(base, seed=seed)
    else: raise ValueError(kind)
    build=time.perf_counter()-t
    for i in range(0,len(updates),100):
        idx.add(updates[i:i+100])
        if kind == 'rebuild_period' and (i+100) % value == 0: idx.rebuild()
    rec=[]; qt=[]
    for q in queries:
        t=time.perf_counter(); got=idx.search(q); qt.append((time.perf_counter()-t)*1000)
        d=((idx.x-q)**2).sum(1); truth=set(np.argpartition(d,K-1)[:K]); rec.append(len(truth.intersection(set(map(int,got))))/K)
    return dict(kind=kind,value=value,seed=seed,recall=float(np.mean(rec)),p95_ms=float(np.percentile(qt,95)),build_s=build)

def main():
    grid={'lsh_bits':[8,12,16],'lsh_tables':[2,8,16],'ivf_nprobe':[1,4,8,16],'ivf_nlist':[16,32,64],'rebuild_period':[100,250,500]}
    rows=[one(s,k,v) for k,vs in grid.items() for v in vs for s in range(3)]
    Path('results').mkdir(exist_ok=True); Path('results/sensitivity.json').write_text(json.dumps(rows,indent=2)); print(f'wrote {len(rows)} sensitivity jobs')
if __name__=='__main__': main()
