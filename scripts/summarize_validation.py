#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser(); parser.add_argument('--input',required=True); args=parser.parse_args(); input_dir=Path(args.input)
rows=json.load(open(input_dir/'raw.json')); groups=defaultdict(list)
for x in rows: groups[(x['method'],x['queries_per_batch'])].append(x)
summary=[]
for (m,q), xs in sorted(groups.items()):
 def mean(k): return float(np.mean([x[k] for x in xs]))
 vals=np.array([x['recall'] for x in xs]); summary.append(dict(method=m,queries_per_batch=q,n=len(xs),recall=mean('recall'),recall_std=float(np.std(vals,ddof=1)),query_p95_ms=mean('query_ms_p95'),maintenance_ms=mean('maintenance_ms'),service_ms=mean('service_ms')))
input_dir.joinpath('summary.json').write_text(json.dumps(summary,indent=2))
with input_dir.joinpath('summary.md').open('w') as f:
 f.write('| policy | q/update | seeds | recall | recall sd | p95 ms | maintenance ms | total measured ms |\n|---|---:|---:|---:|---:|---:|---:|---:|\n')
 for x in summary: f.write('| {} | {} | {} | {:.4f} | {:.4f} | {:.3f} | {:.1f} | {:.1f} |\n'.format(x['method'],x['queries_per_batch'],x['n'],x['recall'],x['recall_std'],x['query_p95_ms'],x['maintenance_ms'],x['service_ms']))
print(json.dumps(summary,indent=2))
