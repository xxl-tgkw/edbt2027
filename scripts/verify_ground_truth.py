#!/usr/bin/env python3
"""Reconstruct all audited vectors and check saved neighbors with SciPy float64 distances."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.spatial.distance import cdist

from audit_validation import RUNS
from run_validation import load_data

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=load_data()
    checked=0
    sources={}
    started=time.perf_counter()
    for folder,factor in RUNS:
        for seed in range(5):
            file=ROOT/folder/f'workload_{seed}.json'
            sources[str(file.relative_to(ROOT))]=hashlib.sha256(file.read_bytes()).hexdigest()
            w=json.loads(file.read_text())
            base=data[w['base_ids']].astype('float32')/255.
            ui=np.asarray(w['update_ids']); qi=np.asarray(w['query_ids'])
            rng=np.random.default_rng(seed+999)
            update=data[ui].astype('float32')/255.
            update=update+rng.normal(.5,.15,update.shape).astype('float32')
            query=data[qi].astype('float32')/255.
            query=query+factor*.5+rng.normal(0,.05,query.shape).astype('float32')
            corpus=np.vstack([base,update.reshape(-1,784)]).astype('float64')
            for b,truth in enumerate(w['ground_truth']):
                # Independent SciPy kernel; the experiment used NumPy direct per-query sums.
                distance=cdist(query[b,:len(truth)].astype('float64'),corpus[:6100+100*b],metric='sqeuclidean')
                got=np.argsort(distance,axis=1,kind='stable')[:,:10]
                if not np.array_equal(np.sort(got,axis=1),np.sort(truth,axis=1)):
                    raise ValueError(f'Ground truth disagrees: {folder}, seed {seed}, round {b+1}')
                checked+=len(truth)
        print(f'Ground truth verified: {folder}',flush=True)
    result=dict(query_ground_truths_verified=checked,scipy=scipy.__version__,
                method='SciPy cdist float64 squared Euclidean; set equality at k=10',
                source_hashes=sources,elapsed_s=time.perf_counter()-started)
    (ROOT/'results/ground_truth_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('All',checked,'stored exact neighborhoods independently verified')


if __name__=='__main__':
    main()
