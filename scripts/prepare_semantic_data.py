#!/usr/bin/env python3
"""Acquire a public ANN-Benchmarks GloVe file and freeze a normalized semantic pool."""
import hashlib
import json
from pathlib import Path
import time
import subprocess

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://ann-benchmarks.com/glove-100-angular.hdf5'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def normalize(x):
    x=np.asarray(x,dtype='float32')
    norms=np.linalg.norm(x,axis=1,keepdims=True)
    if not np.isfinite(x).all() or (norms==0).any():
        raise ValueError('Invalid source vectors')
    return np.ascontiguousarray(x/norms)


def main():
    out=ROOT/'data/semantic_glove'
    out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():
        manifest=json.loads((out/'manifest.json').read_text())
        if sha(out/'pool.npz') != manifest['pool_sha256']:
            raise ValueError('Existing semantic pool checksum mismatch')
        print('Existing semantic pool verified')
        return
    source=out/'glove-100-angular.hdf5'
    if not source.exists():
        partial=out/'glove-100-angular.hdf5.part'
        if partial.exists():
            raise FileExistsError('Partial download exists; inspect before retry')
        # The source's CDN rejects Python urllib's default user agent.
        subprocess.run(['curl','--fail','--location','--retry','2','--max-time','300',
                        '--output',str(partial),URL],check=True)
        headers={'download_client':'curl'}
        count=partial.stat().st_size
        if count != 485413888:
            raise ValueError(f'Unexpected source length: {count}')
        partial.rename(source)
    else:
        headers={}
    with h5py.File(source,'r') as f:
        shape={k:list(f[k].shape) for k in f}
        attrs={k:str(v) for k,v in f.attrs.items()}
        if f['train'].shape[1]!=100 or f['test'].shape[1]!=100:
            raise ValueError('Expected 100-dimensional GloVe')
        # Preparation seed and candidate count are independent of retrieval outcomes.
        ids=np.random.default_rng(20260916).choice(len(f['train']),size=130000,replace=False)
        # HDF5 point selection on 130k unordered IDs is slow; the source fits in memory.
        train=f['train'][:]
        selected=normalize(train[ids])
        query=normalize(f['test'][:])
        # A 10k design pool sets the direction; those rows are never in evaluation indexes.
        design=selected[:10000]
        centered=design.astype('float64')-design.mean(0,dtype='float64')
        _,vectors=np.linalg.eigh(centered.T@centered)
        direction=vectors[:,-1]
        if direction[np.argmax(np.abs(direction))]<0: direction=-direction
        pool=selected[10000:]
        scores=pool@direction
        qs=query@direction
        # Rank split gives a complete partition and avoids tuning tail thresholds.
        order=np.argsort(scores,kind='stable')
        qorder=np.argsort(qs,kind='stable')
        np.savez_compressed(out/'pool.npz',vectors=pool,source_ids=ids[10000:],
                            design_ids=ids[:10000],queries=query,
                            query_source_ids=np.arange(len(query)),direction=direction,
                            old_pool=order[:len(order)//2],new_pool=order[len(order)//2:],
                            new_query_pool=qorder[len(qorder)//2:])
    manifest=dict(source_url=URL,source_sha256=sha(source),source_bytes=source.stat().st_size,
                  source_attrs=attrs,source_shapes=shape,download_headers=headers,
                  prepared_unix=time.time(),pool_sha256=sha(out/'pool.npz'),
                  preparation_seed=20260916,design_count=10000,evaluation_pool_count=120000,
                  preprocessing='float32 L2 normalization; top design-pool principal component; median rank split',
                  source_semantics='Pretrained GloVe word embeddings from ANN-Benchmarks; not sentence/document embeddings',
                  upstream_url='https://nlp.stanford.edu/projects/glove/',
                  license='GloVe pretrained vectors: PDDL 1.0; ANN-Benchmarks redistribution',
                  script_sha256=sha(Path(__file__)))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
