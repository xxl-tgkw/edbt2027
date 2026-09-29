import json,time
from pathlib import Path
import numpy as np
from run_benchmark import fashion_data, FaissFlat, FaissIVF, K

def run(n,method,seed=0):
 x=fashion_data(seed,n); rng=np.random.default_rng(seed+77); q=x[rng.choice(len(x),50,replace=False)]; u=x[rng.choice(len(x),200,replace=True)]+rng.normal(.5,.15,(200,x.shape[1])).astype('float32')
 t=time.perf_counter(); idx=FaissFlat(x) if method=='faiss_flat' else FaissIVF(x,seed=seed); build=time.perf_counter()-t; qt=[]
 for z in u.reshape(2,100,x.shape[1]): idx.add(z)
 for z in q:
  t=time.perf_counter(); got=idx.search(z); qt.append((time.perf_counter()-t)*1000)
  d=((idx.x-z)**2).sum(1); truth=set(np.argpartition(d,K-1)[:K]);
 rec=[]
 for z in q:
  got=idx.search(z); d=((idx.x-z)**2).sum(1); truth=set(np.argpartition(d,K-1)[:K]); rec.append(len(truth.intersection(set(map(int,got))))/K)
 return {'n':n,'method':method,'seed':seed,'recall':float(np.mean(rec)),'q95_ms':float(np.percentile(qt,95)),'build_s':build,'memory_mb':float(idx.x.nbytes/1e6)}
def main():
 rows=[run(n,m,s) for n in [6000,12000,24000] for m in ['faiss_flat','faiss_ivf'] for s in range(3)]; Path('results/scale.json').write_text(json.dumps(rows,indent=2)); print('wrote',len(rows),'scale jobs')
if __name__=='__main__': main()
