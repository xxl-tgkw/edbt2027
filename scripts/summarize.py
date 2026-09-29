#!/usr/bin/env python3
import argparse,json,statistics
import numpy as np
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--input',default='results/raw.json'); a=p.parse_args(); rows=json.load(open(a.input)); g=defaultdict(list)
for r in rows:g[(r['dataset'],r['drift'],r['method'])].append(r)
out=[]
for (d,dr,m),xs in sorted(g.items()):
 vals=np.array([x['recall@10'] for x in xs]); rng=np.random.default_rng(1000+len(out)); boots=[float(np.mean(rng.choice(vals,len(vals),replace=True))) for _ in range(2000)]
 out.append({'dataset':d,'drift':dr,'method':m,'n':len(xs),'recall_mean':float(np.mean(vals)),'recall_std':float(np.std(vals,ddof=1)) if len(vals)>1 else 0,'recall_ci95':[float(np.quantile(boots,.025)),float(np.quantile(boots,.975))],'worst_seed':float(np.min(vals)),'query_p95_ms':statistics.mean(x['query_ms_p95'] for x in xs),'update_ms':statistics.mean(x['update_ms_mean'] for x in xs)})
Path('results/summary.json').write_text(json.dumps(out,indent=2)); lines=['# UVR-Bench summary','','| dataset | drift | method | n | Recall@10 (95% CI) | worst seed | p95 query ms | update ms |','|---|---:|---|---:|---:|---:|---:|---:|']
for x in out:lines.append(f"| {x['dataset']} | {x['drift']:.1f} | {x['method']} | {x['n']} | {x['recall_mean']:.3f} +/- {x['recall_std']:.3f} [{x['recall_ci95'][0]:.3f},{x['recall_ci95'][1]:.3f}] | {x['worst_seed']:.3f} | {x['query_p95_ms']:.3f} | {x['update_ms']:.3f} |")
Path('results/summary.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
