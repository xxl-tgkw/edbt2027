import json
import matplotlib.pyplot as plt
from collections import defaultdict
rows=json.load(open('results/summary.json'))
g=defaultdict(list)
for r in rows:
    if r['dataset']=='digits': g[r['method']].append(r)
fig,ax=plt.subplots(figsize=(5.8,3.5))
for m,rs in sorted(g.items()):
    ax.scatter([r['update_ms'] for r in rs],[r['recall_mean'] for r in rs],label=m,s=28,alpha=.8)
ax.set_xlabel('Mean update latency (ms)'); ax.set_ylabel('Recall@10'); ax.set_ylim(.85,1.01)
ax.grid(alpha=.25); ax.legend(fontsize=6,ncol=2); fig.tight_layout(); fig.savefig('results/pareto.pdf'); fig.savefig('results/pareto.png',dpi=180)
