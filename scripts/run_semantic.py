#!/usr/bin/env python3
"""Paired arrival-order experiments on public semantic vectors; native Faiss policies."""
import argparse
import hashlib
import itertools
import json
import platform
from pathlib import Path
import subprocess
import time

import faiss
import numpy as np
import scipy
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_info, threadpool_limits

from prepare_semantic_data import sha

ROOT=Path(__file__).resolve().parents[1]
POLICIES=['flat','ivf_fixed','ivf_rebuilt','hnsw']
BUDGETS={'flat':[0],'ivf_fixed':[4,16,64],'ivf_rebuilt':[4,16,64],'hnsw':[16,64,256]}
PROTOCOL='semantic-order-v1'


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def load_pool():
    folder=ROOT/'data/semantic_glove'
    manifest=json.loads((folder/'manifest.json').read_text())
    if sha(folder/'pool.npz')!=manifest['pool_sha256']:
        raise ValueError('Semantic pool checksum mismatch')
    with np.load(folder/'pool.npz',allow_pickle=False) as source:
        pool={k:source[k] for k in source.files}
    return pool,manifest


def make_workload(pool, seed, condition, base=20000, batch=4000, rounds=5, nq=100):
    rng=np.random.default_rng(seed)
    old=rng.choice(pool['old_pool'],base,replace=False)
    new=rng.choice(pool['new_pool'],batch*rounds,replace=False)
    ids=np.r_[old,new]
    if condition=='shuffled':
        ids=ids[np.random.default_rng(seed+10000).permutation(len(ids))]
    elif condition!='shifted':
        raise ValueError('Unknown condition')
    qids=rng.choice(pool['new_query_pool'],(rounds+1)*2*nq,replace=False).reshape(rounds+1,2,nq)
    if len(set(ids))!=len(ids) or len(set(qids.ravel()))!=qids.size:
        raise ValueError('Duplicated workload source IDs')
    x=np.ascontiguousarray(pool['vectors'][ids])
    q=np.ascontiguousarray(pool['queries'][qids])
    truth=[]; exposure=[]
    for b in range(rounds+1):
        dist=cdist(q[b].reshape(-1,x.shape[1]).astype('float64'),
                   x[:base+b*batch].astype('float64'),'sqeuclidean')
        nn=np.argsort(dist,axis=1,kind='stable')[:,:10].reshape(2,nq,10)
        truth.append(nn.tolist())
        exposure.append(float((nn[1]>=base).mean()))
    meta=dict(protocol=PROTOCOL,seed=seed,condition=condition,base=base,batch=batch,
              rounds=rounds,nq=nq,pool_ids=ids.tolist(),source_train_ids=pool['source_ids'][ids].tolist(),
              query_ids=qids.tolist(),ground_truth=truth,updated_neighbor_exposure=exposure,
              corpus_sha256=array_sha(x),queries_sha256=array_sha(q))
    return x,q,truth,meta


class NativeIndex:
    def __init__(self,x,policy,seed,nlist=128,m=16,efc=100):
        self.policy,self.seed,self.nlist,self.m,self.efc=policy,seed,nlist,m,efc
        self.rebuild(x)

    def rebuild(self,x):
        d=x.shape[1]
        if self.policy=='flat':
            self.index=faiss.IndexFlatL2(d)
        elif self.policy.startswith('ivf'):
            self.index=faiss.IndexIVFFlat(faiss.IndexFlatL2(d),d,self.nlist,faiss.METRIC_L2)
            self.index.cp.seed=self.seed
            self.index.cp.niter=30
            self.index.cp.nredo=1
            self.index.cp.max_points_per_centroid=10000
            self.index.train(x)
        elif self.policy=='hnsw':
            self.index=faiss.IndexHNSWFlat(d,self.m,faiss.METRIC_L2)
            self.index.hnsw.efConstruction=self.efc
            self.index.hnsw.rng=faiss.RandomGenerator(self.seed)
        else:
            raise ValueError(self.policy)
        self.index.add(x)

    def set_budget(self,budget):
        if self.policy.startswith('ivf'): self.index.nprobe=budget
        elif self.policy=='hnsw': self.index.hnsw.efSearch=budget

    def search_timed(self,q):
        stats=None
        if self.policy.startswith('ivf'): stats=faiss.cvar.indexIVF_stats
        elif self.policy=='hnsw': stats=faiss.cvar.hnsw_stats
        if stats is not None: stats.reset()
        start=time.perf_counter_ns()
        _,ids=self.index.search(q.reshape(1,-1),10)
        duration=(time.perf_counter_ns()-start)/1e6
        work=int(stats.ndis) if stats is not None else int(self.index.ntotal)
        return ids[0].tolist(),duration,work


def query_set(idx, queries, truth):
    returned=[]; durations=[]; recalls=[]; work=[]
    for q,nn in zip(queries,truth):
        ids,dt,ndis=idx.search_timed(q)
        valid=[x for x in ids if x>=0]
        if len(valid)!=len(set(valid)) or any(x>=idx.index.ntotal for x in valid):
            raise ValueError('Invalid returned IDs')
        returned.append(ids); durations.append(dt); work.append(ndis)
        recalls.append(len(set(valid)&set(nn))/10)
    return dict(returned_ids=returned,query_ms=durations,recalls=recalls,distance_counts=work)


def evaluate(x,q,truth,meta,policy,seed):
    base,batch,rounds=meta['base'],meta['batch'],meta['rounds']
    start=time.perf_counter_ns()
    idx=NativeIndex(x[:base],policy,seed)
    build_ms=(time.perf_counter_ns()-start)/1e6
    for z in x[:10]: idx.index.search(z[None],10)
    events=[]
    rng=np.random.default_rng(seed+80000)
    for b in range(rounds+1):
        insert_ms=rebuild_ms=0.
        rebuilt=policy=='ivf_rebuilt' and b in (2,4)
        if b:
            start=time.perf_counter_ns()
            idx.index.add(x[base+(b-1)*batch:base+b*batch])
            insert_ms=(time.perf_counter_ns()-start)/1e6
        if rebuilt:
            start=time.perf_counter_ns()
            idx.rebuild(x[:base+b*batch])
            rebuild_ms=(time.perf_counter_ns()-start)/1e6
        budget_results={}
        # Budgets share one built index, but each query call is measured separately.
        for budget in rng.permutation(BUDGETS[policy]):
            idx.set_budget(int(budget))
            budget_results[str(budget)]={
                'validation':query_set(idx,q[b,0],truth[b][0]),
                'test':query_set(idx,q[b,1],truth[b][1])}
        diagnostics={}
        if policy.startswith('ivf'):
            sizes=np.array([idx.index.invlists.list_size(j) for j in range(idx.nlist)])
            diagnostics=dict(max_list_size=int(sizes.max()),empty_lists=int((sizes==0).sum()),
                             list_size_cv=float(sizes.std()/sizes.mean()),
                             imbalance=float(np.square(sizes).sum()*len(sizes)/sizes.sum()**2))
        events.append(dict(round=b,indexed=int(idx.index.ntotal),insert_ms=insert_ms,
                           rebuilt=rebuilt,rebuild_ms=rebuild_ms,budgets=budget_results,
                           diagnostics=diagnostics))
    serialized_bytes=len(faiss.serialize_index(idx.index))
    maintenance=sum(e['insert_ms']+e['rebuild_ms'] for e in events[1:])
    summaries={}
    for budget in BUDGETS[policy]:
        chosen=[e['budgets'][str(budget)]['test'] for e in events[1:]]
        ts=[v for c in chosen for v in c['query_ms']]
        rec=[v for c in chosen for v in c['recalls']]
        counts=[v for c in chosen for v in c['distance_counts']]
        summaries[str(budget)]=dict(recall=float(np.mean(rec)),query_p95_ms=float(np.percentile(ts,95)),
                                   query_mean_ms=float(np.mean(ts)),service_ms=maintenance+sum(ts),
                                   mean_distance_count=float(np.mean(counts)))
    return dict(protocol=PROTOCOL,seed=seed,condition=meta['condition'],policy=policy,
                build_ms=build_ms,maintenance_ms=maintenance,serialized_bytes=serialized_bytes,
                events=events,summaries=summaries,corpus_sha256=meta['corpus_sha256'],
                queries_sha256=meta['queries_sha256'],index_parameters=dict(nlist=128,
                clustering_iterations=30,training_cap_per_centroid=10000,M=16,efConstruction=100))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--seeds',nargs='+',type=int,required=True)
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds): parser.error('Duplicate seeds')
    out=ROOT/args.output
    if out.exists(): raise FileExistsError('Use a fresh output directory')
    out.mkdir(parents=True)
    faiss.omp_set_num_threads(1)
    started=time.time()
    with threadpool_limits(limits=1):
        pool,data_manifest=load_pool()
        env=dict(protocol=PROTOCOL,args=vars(args),started_unix=started,
                 python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                 faiss=faiss.__version__,threads=threadpool_info(),
                 cpu=subprocess.check_output(['lscpu'],text=True),
                 source_hashes={f:sha(ROOT/f) for f in ['scripts/run_semantic.py',
                                'scripts/prepare_semantic_data.py']},
                 data_manifest=data_manifest,budgets=BUDGETS,
                 caveats='Shared CPU host; sequential jobs, no affinity isolation; no deletes/concurrency')
        (out/'environment.json').write_text(json.dumps(env,indent=2)+'\n')
        rows=[]
        for seed in args.seeds:
            workloads={}
            for condition in ['shifted','shuffled']:
                workloads[condition]=make_workload(pool,seed,condition)
                (out/f'workload_{condition}_{seed}.json').write_text(json.dumps(workloads[condition][3])+'\n')
            jobs=list(itertools.product(['shifted','shuffled'],POLICIES))
            np.random.default_rng(seed+90000).shuffle(jobs)
            for condition,policy in jobs:
                row=evaluate(*workloads[condition],policy,seed)
                rows.append(row)
                with (out/'events.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                print(f'{len(rows)}/{len(args.seeds)*8} seed={seed} {condition} {policy} '
                      +', '.join(f'{b}:R={r["recall"]:.3f} C={r["service_ms"]:.0f}ms'
                                  for b,r in row['summaries'].items()),flush=True)
        (out/'raw.json').write_text(json.dumps(rows)+'\n')
        env['elapsed_s']=time.time()-started
        (out/'environment.json').write_text(json.dumps(env,indent=2)+'\n')


if __name__=='__main__': main()
