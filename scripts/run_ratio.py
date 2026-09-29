import json,time
from pathlib import Path
import numpy as np
from run_benchmark import workload, Exact, LSH, IVF, FaissFlat, FaissIVF, K

def run(ratio,method,seed=0):
 base,queries,updates=workload('fashion_mnist',seed,.5,n=6000,q=800,updates=500); t=time.perf_counter()
 if method=='exact': idx=Exact(base)
 elif method=='lsh': idx=LSH(base,seed)
 elif method=='ivf_static': idx=IVF(base,seed=seed)
 elif method=='faiss_flat': idx=FaissFlat(base)
 elif method=='ivf_rebuild': idx=IVF(base,seed=seed)
 elif method=='faiss_ivf_rebuild': idx=FaissIVF(base,seed=seed)
 build=time.perf_counter()-t; rec=[]; qt=[]; ut=[]; qpos=0
 for b in range(5):
  u=time.perf_counter(); idx.add(updates[b*100:(b+1)*100]);
  if method in ('ivf_rebuild','faiss_ivf_rebuild') and b in (1,3): idx.rebuild()
  ut.append((time.perf_counter()-u)*1000)
  for q in queries[qpos:qpos+ratio]:
   z=time.perf_counter(); got=idx.search(q); qt.append((time.perf_counter()-z)*1000); d=((idx.x-q)**2).sum(1); truth=set(np.argpartition(d,K-1)[:K]); rec.append(len(truth.intersection(set(map(int,got))))/K)
  qpos+=ratio
 return {'ratio':ratio,'method':method,'seed':seed,'recall':float(np.mean(rec)),'q95_ms':float(np.percentile(qt,95)),'update_ms':float(np.mean(ut)),'build_s':build}
def main():
 rows=[run(r,m,s) for r in [10,40,160] for m in ['exact','lsh','ivf_static','faiss_flat','ivf_rebuild','faiss_ivf_rebuild'] for s in range(3)]; Path('results/ratio.json').write_text(json.dumps(rows,indent=2)); print('wrote',len(rows),'ratio jobs')
if __name__=='__main__': main()
