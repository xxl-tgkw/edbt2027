#!/usr/bin/env python3
"""Summarize lifecycle experiment with seed-level descriptive statistics."""
import argparse, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); a=ap.parse_args(); folder=ROOT/a.input
    rows=json.loads((folder/'raw.json').read_text()); policies=sorted(set(r['policy'] for r in rows)); seeds=sorted(set(r['seed'] for r in rows))
    out={'protocol':'semantic-lifecycle-v1','seeds':seeds,'runs':len(rows),'aggregates':[]}
    lines=['# Semantic lifecycle results','',f'{len(rows)} runs; {len(seeds)} seeds.','',
           '| Policy | Recall | Commit ms | Query p95 ms | Concurrent wall ms | Persistence | Rollback |','|---|---:|---:|---:|---:|---:|---:|']
    for p in policies:
        g=[r for r in rows if r['policy']==p]
        def m(k): return float(np.mean([r[k] for r in g]))
        e=dict(policy=p,seeds=seeds,recall=m('recall'),commit_ms=m('commit_ms'),query_p95_ms=m('query_p95_ms'),concurrent_wall_ms=m('concurrent_wall_ms'),persistence_rate=float(np.mean([r['persistence_equal'] for r in g])),rollback_rate=float(np.mean([r['rollback_state_unchanged'] for r in g])),pre_commit_visibility_rate=float(np.mean([r['pre_commit_insert_visible'] for r in g])),deleted_visibility_rate=float(np.mean([r['post_commit_deleted_visible'] for r in g])))
        out['aggregates'].append(e)
        lines.append(f"| {p} | {e['recall']:.4f} | {e['commit_ms']:.2f} | {e['query_p95_ms']:.4f} | {e['concurrent_wall_ms']:.2f} | {e['persistence_rate']:.0%} | {e['rollback_rate']:.0%} |")
    (folder/'analysis.json').write_text(json.dumps(out,indent=2)+'\n'); (folder/'summary.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
if __name__=='__main__': main()
