import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
rows=json.load(open('results/sensitivity.json'))
fig,axs=plt.subplots(1,3,figsize=(10,3.1))
for ax,(kind,title) in zip(axs,[('lsh_bits','LSH bits'),('ivf_nprobe','IVF nprobe'),('rebuild_period','rebuild period')]):
 xs=sorted(set(r['value'] for r in rows if r['kind']==kind)); means=[np.mean([r['recall'] for r in rows if r['kind']==kind and r['value']==x]) for x in xs]; p95=[np.mean([r['p95_ms'] for r in rows if r['kind']==kind and r['value']==x]) for x in xs]
 ax.plot(xs,means,'o-',label='Recall@10'); ax2=ax.twinx(); ax2.plot(xs,p95,'s--',color='tab:red',label='p95 ms'); ax.set_title(title); ax.set_xlabel('value'); ax.set_ylabel('recall'); ax2.set_ylabel('p95 ms'); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig('results/sensitivity.pdf'); fig.savefig('results/sensitivity.png',dpi=180); print('wrote sensitivity figures')
