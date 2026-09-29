import argparse
import json
import matplotlib.pyplot as plt
from collections import defaultdict
parser=argparse.ArgumentParser(); parser.add_argument('--input',default='results/validation_v2'); args=parser.parse_args(); input_dir=args.input
r=json.load(open(f'{input_dir}/summary.json')); g=defaultdict(list)
for x in r:g[x['method']].append(x)
fig,axs=plt.subplots(1,2,figsize=(7.2,3.1))
for m,xs in sorted(g.items()):
 xs=sorted(xs,key=lambda x:x['queries_per_batch']); q=[x['queries_per_batch'] for x in xs]; axs[0].plot(q,[x['recall'] for x in xs],'o-',label=m); axs[1].plot(q,[x['service_ms'] for x in xs],'o-',label=m)
for a in axs:a.set_xscale('log'); a.set_xlabel('queries per update batch'); a.grid(alpha=.25)
axs[0].set_ylabel('Recall@10'); axs[1].set_ylabel('measured service time (ms)'); axs[1].legend(fontsize=5,ncol=2); fig.tight_layout(); fig.savefig(f'{input_dir}/ratio_service.pdf'); fig.savefig(f'{input_dir}/ratio_service.png',dpi=180)
