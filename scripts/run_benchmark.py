#!/usr/bin/env python3
"""Run UVR-Bench: dynamic vector retrieval under controlled drift."""
from __future__ import annotations
import argparse, json, time, gzip, struct, urllib.request
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
try:
    import faiss
except ImportError:
    faiss = None

K = 10

def digits_data(seed, n=1600):
    x = load_digits().data.astype('float32') / 16.0
    rng = np.random.default_rng(seed); idx = rng.choice(len(x), size=min(n,len(x)), replace=False)
    return x[idx]

def clustered_data(seed, n=3000, d=64):
    rng=np.random.default_rng(seed); centers=rng.normal(0,3,(30,d)).astype('float32')
    z=rng.integers(len(centers),size=n); return centers[z]+rng.normal(0,.35,(n,d)).astype('float32')

FASHION_URL='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz'
def fashion_data(seed, n=6000):
    """Load a deterministic subset of the public Fashion-MNIST image vectors."""
    cache=Path(__file__).resolve().parents[1]/'data'/'fashion_train_images.gz'
    cache.parent.mkdir(exist_ok=True)
    if not cache.exists():
        urllib.request.urlretrieve(FASHION_URL, cache)
    with gzip.open(cache,'rb') as f:
        magic,count,rows,cols=struct.unpack('>IIII',f.read(16)); raw=np.frombuffer(f.read(),dtype=np.uint8).reshape(count,rows*cols)
    rng=np.random.default_rng(seed); idx=rng.choice(count,size=min(n,count),replace=False)
    return raw[idx].astype('float32')/255.0

class Exact:
    def __init__(self, x): self.x=x.copy()
    def add(self,x): self.x=np.vstack([self.x,x])
    def search(self,q,k=K):
        dist=((self.x-q)**2).sum(1); return np.argpartition(dist,k-1)[:k]

class LSH:
    def __init__(self,x,seed,bits=12,tables=8):
        self.rng=np.random.default_rng(seed); self.bits=bits; self.tables=tables; self.proj=[self.rng.normal(size=(x.shape[1],bits)).astype('float32') for _ in range(tables)]; self.x=x.copy(); self.buckets=[]; self._reindex()
    def _keys(self,x,p): return (x@p>=0).dot(1<<np.arange(self.bits))
    def _reindex(self):
        self.buckets=[]
        for p in self.proj:
            b={}
            for i,key in enumerate(self._keys(self.x,p)): b.setdefault(int(key),[]).append(i)
            self.buckets.append(b)
    def add(self,x):
        off=len(self.x); self.x=np.vstack([self.x,x])
        for t,p in enumerate(self.proj):
            for j,key in enumerate(self._keys(x,p)): self.buckets[t].setdefault(int(key),[]).append(off+j)
    def search(self,q,k=K):
        cand=set()
        for t,p in enumerate(self.proj): cand.update(self.buckets[t].get(int(self._keys(q[None],p)[0]),[]))
        if len(cand)<k: cand.update(range(min(len(self.x),max(k*4,100))))
        c=np.fromiter(cand,dtype=np.int64); d=((self.x[c]-q)**2).sum(1); return c[np.argpartition(d,k-1)[:k]]

class IVF:
    def __init__(self,x,nprobe=4,seed=0,nlist=32):
        self.nprobe=nprobe; self.seed=seed; self.nlist=min(nlist,len(x)); self.x=x.copy(); self._build()
    def _build(self):
        self.km=KMeans(self.nlist,random_state=self.seed,n_init=1,max_iter=30).fit(self.x); self.assign=self.km.labels_; self.lists=[np.where(self.assign==i)[0] for i in range(self.nlist)]
    def add(self,x):
        off=len(self.x); self.x=np.vstack([self.x,x]); a=self.km.predict(x)
        self.assign=np.concatenate([self.assign,a]);
        for i in range(self.nlist): self.lists[i]=np.concatenate([self.lists[i],np.where(a==i)[0]+off])
    def rebuild(self): self._build()
    def search(self,q,k=K):
        cd=((self.km.cluster_centers_-q)**2).sum(1); cs=np.argpartition(cd,min(self.nprobe,self.nlist)-1)[:self.nprobe]; cand=np.concatenate([self.lists[i] for i in cs]); cand=np.unique(cand)
        if len(cand)<k: cand=np.arange(len(self.x))
        d=((self.x[cand]-q)**2).sum(1); return cand[np.argpartition(d,k-1)[:k]]

class FaissFlat:
    def __init__(self,x):
        if faiss is None: raise RuntimeError('install faiss-cpu')
        self.x=x.copy().astype('float32'); self.index=faiss.IndexFlatL2(self.x.shape[1]); self.index.add(self.x)
    def add(self,x):
        x=x.astype('float32'); self.x=np.vstack([self.x,x]); self.index.add(x)
    def search(self,q,k=K): return self.index.search(q[None].astype('float32'),k)[1][0]

class FaissIVF:
    def __init__(self,x,seed=0,nlist=32,nprobe=4):
        if faiss is None: raise RuntimeError('install faiss-cpu')
        self.x=x.copy().astype('float32'); self.nlist=min(nlist,len(x)); self.nprobe=nprobe; self._build()
    def _build(self):
        quant=faiss.IndexFlatL2(self.x.shape[1]); self.index=faiss.IndexIVFFlat(quant,self.x.shape[1],self.nlist,faiss.METRIC_L2); self.index.train(self.x); self.index.add(self.x); self.index.nprobe=min(self.nprobe,self.nlist)
    def add(self,x):
        x=x.astype('float32'); self.x=np.vstack([self.x,x]); self.index.add(x)
    def rebuild(self): self._build()
    def search(self,q,k=K): return self.index.search(q[None].astype('float32'),k)[1][0]

def workload(name,seed,drift,n=2000,q=200,updates=500):
    if name=='digits': base=digits_data(seed,n)
    elif name=='clustered': base=clustered_data(seed,n,64)
    elif name=='fashion_mnist': base=fashion_data(seed,max(n,6000))
    else: raise ValueError(name)
    rng=np.random.default_rng(seed+999)
    qi=rng.choice(len(base),q,replace=False)
    direction=np.ones((1,base.shape[1]),dtype='float32')
    queries=base[qi]+(0.35*drift*direction)+rng.normal(0,.05,base[qi].shape).astype('float32')
    upd=base[rng.choice(len(base),updates,replace=True)] + rng.normal(drift,.15, (updates,base.shape[1])).astype('float32')
    return base,queries,upd

def evaluate(name,method,seed,drift):
    base,queries,updates=workload(name,seed,drift); t=time.perf_counter()
    if method=='exact': idx=Exact(base)
    elif method=='lsh': idx=LSH(base,seed)
    elif method in ('ivf_static','ivf_rebuild'): idx=IVF(base,seed=seed)
    elif method=='faiss_flat': idx=FaissFlat(base)
    elif method=='faiss_ivf_rebuild': idx=FaissIVF(base,seed=seed)
    build=time.perf_counter()-t; update_times=[]
    for i in range(0,len(updates),100):
        chunk=updates[i:i+100]; u=time.perf_counter(); idx.add(chunk)
        if method in ('ivf_rebuild','faiss_ivf_rebuild') and (i+100)%250==0: idx.rebuild()
        update_times.append(time.perf_counter()-u)
    qtimes=[]; rec=[]
    for q in queries:
        t=time.perf_counter(); got=idx.search(q); qtimes.append((time.perf_counter()-t)*1000)
        # exact reference on the current index state
        d=((idx.x-q)**2).sum(1); true=set(np.argpartition(d,K-1)[:K]); rec.append(len(true.intersection(set(map(int,got))))/K)
    return {'dataset':name,'method':method,'seed':seed,'drift':drift,'recall@10':float(np.mean(rec)),'query_ms_mean':float(np.mean(qtimes)),'query_ms_p95':float(np.percentile(qtimes,95)),'update_ms_mean':float(np.mean(update_times)*1000),'build_s':build,'memory_mb':float(idx.x.nbytes/1e6)}

def evaluate_interleaved(name,method,seed,drift,query_per_batch=40):
    base,queries,updates=workload(name,seed,drift,n=6000,q=200,updates=500); t=time.perf_counter()
    if method=='exact': idx=Exact(base)
    elif method=='lsh': idx=LSH(base,seed)
    elif method=='ivf_static': idx=IVF(base,seed=seed)
    elif method=='faiss_flat': idx=FaissFlat(base)
    elif method=='ivf_rebuild': idx=IVF(base,seed=seed)
    elif method=='faiss_ivf_rebuild': idx=FaissIVF(base,seed=seed)
    else: raise ValueError(method)
    build=time.perf_counter()-t; rec=[]; qtimes=[]; update_times=[]
    for b,i in enumerate(range(0,len(updates),100)):
        u=time.perf_counter(); idx.add(updates[i:i+100])
        if method in ('ivf_rebuild','faiss_ivf_rebuild') and b in (1,3): idx.rebuild()
        update_times.append(time.perf_counter()-u)
        for q in queries[b*query_per_batch:(b+1)*query_per_batch]:
            t=time.perf_counter(); got=idx.search(q); qtimes.append((time.perf_counter()-t)*1000)
            d=((idx.x-q)**2).sum(1); truth=set(np.argpartition(d,K-1)[:K]); rec.append(len(truth.intersection(set(map(int,got))))/K)
    return {'dataset':name,'method':method,'seed':seed,'drift':drift,'workload':'interleaved','recall@10':float(np.mean(rec)),'query_ms_mean':float(np.mean(qtimes)),'query_ms_p95':float(np.percentile(qtimes,95)),'update_ms_mean':float(np.mean(update_times)*1000),'build_s':build,'memory_mb':float(idx.x.nbytes/1e6)}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='results'); p.add_argument('--seeds',type=int,default=5); p.add_argument('--quick',action='store_true'); p.add_argument('--public-interleaved',action='store_true'); a=p.parse_args(); out=[]
    if a.public_interleaved:
      for m in ['exact','lsh','ivf_static','faiss_flat','ivf_rebuild','faiss_ivf_rebuild']:
       for s in range(min(a.seeds,3)):
        r=evaluate_interleaved('fashion_mnist',m,s,.5); out.append(r); print(r,flush=True)
      Path(a.output).mkdir(parents=True,exist_ok=True); Path(a.output,'fashion_interleaved.json').write_text(json.dumps(out,indent=2)); return
    datasets=['digits','clustered']; methods=['exact','lsh','ivf_static','ivf_rebuild','faiss_flat','faiss_ivf_rebuild']; drifts=[0.0,0.5,1.0]
    if a.quick: datasets=['digits']; methods=['exact','lsh','ivf_rebuild','faiss_flat','faiss_ivf_rebuild']; drifts=[0.5]; a.seeds=1
    for d in datasets:
      for drift in drifts:
       for m in methods:
        for s in range(a.seeds):
         r=evaluate(d,m,s,drift); out.append(r); print(r,flush=True)
    Path(a.output).mkdir(parents=True,exist_ok=True); Path(a.output,'raw.json').write_text(json.dumps(out,indent=2))

if __name__=='__main__': main()
