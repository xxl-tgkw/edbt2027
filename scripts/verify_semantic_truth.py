#!/usr/bin/env python3
"""Independent float64 norm/dot-product check of semantic workload reconstruction."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from prepare_semantic_data import sha
from run_semantic import array_sha,load_pool

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',default='results/semantic_v1'); args=p.parse_args()
    folder=ROOT/args.input
    pool,_=load_pool()
    checked=0; hashes={}; started=time.time()
    with threadpool_limits(limits=1):
        for file in sorted(folder.glob('workload_*.json')):
            w=json.loads(file.read_text())
            x=pool['vectors'][w['pool_ids']]
            q=pool['queries'][w['query_ids']]
            if array_sha(x)!=w['corpus_sha256'] or array_sha(q)!=w['queries_sha256']:
                raise ValueError('Vector reconstruction mismatch')
            if not np.array_equal(pool['source_ids'][w['pool_ids']],w['source_train_ids']):
                raise ValueError('Original train IDs mismatch')
            if set(w['source_train_ids']) & set(pool['design_ids']):
                raise ValueError('Design/evaluation source leakage')
            x=x.astype('float64'); q=q.astype('float64')
            for b,truth in enumerate(w['ground_truth']):
                corpus=x[:20000+4000*b]; query=q[b].reshape(-1,100)
                # Experiment uses SciPy direct squared differences; this uses norm expansion.
                distances=np.sum(query**2,axis=1)[:,None]+np.sum(corpus**2,axis=1)[None,:]-2*(query@corpus.T)
                got=np.argsort(distances,axis=1,kind='stable')[:,:10].reshape(2,100,10)
                if not np.array_equal(np.sort(got,axis=2),np.sort(truth,axis=2)):
                    raise ValueError(f'Ground truth disagreement: {file.name}, round {b}')
                exposure=float((got[1]>=20000).mean())
                if not np.isclose(exposure,w['updated_neighbor_exposure'][b],atol=1e-12):
                    raise ValueError('Incorrect inserted-neighbor exposure')
                checked+=200
            hashes[str(file.relative_to(ROOT))]=sha(file)
            print(file.name,'verified',flush=True)
    report=dict(ground_truths_checked=checked,source_hashes=hashes,elapsed_s=time.time()-started,
                pool_sha256=sha(ROOT/'data/semantic_glove/pool.npz'),
                method='float64 norm expansion/dot product versus saved SciPy squared-distance ground truth')
    (folder/'truth_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('All',checked,'exact neighborhoods independently verified')


if __name__=='__main__': main()
