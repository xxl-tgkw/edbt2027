import json
import matplotlib.pyplot as plt
from collections import defaultdict
r=json.load(open('results/ratio.json')); g=defaultdict(list)
for x in r:g[x['method']].append(x)
fig,axs=plt.subplots(1,2,figsize=(7.2,3.0))
for m in sorted(g):
 xs=sorted(set(x['ratio'] for x in g[m])); rr=[sum(x['recall'] for x in g[m] if x['ratio']==v)/3 for v in xs]; qq=[sum(x['q95_ms'] for x in g[m] if x['ratio']==v)/3 for v in xs]
 axs[0].plot(xs,rr,'o-',label=m); axs[1].plot(xs,qq,'o-',label=m)
axs[0].set_xscale('log'); axs[1].set_xscale('log'); axs[0].set_ylabel('Recall@10'); axs[1].set_ylabel('p95 query latency (ms)')
for a in axs:a.set_xlabel('queries per update batch'); a.grid(alpha=.25)
axs[1].legend(fontsize=5,ncol=2); fig.tight_layout(); fig.savefig('results/ratio_sensitivity.pdf'); fig.savefig('results/ratio_sensitivity.png',dpi=180)
