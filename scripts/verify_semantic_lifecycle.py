#!/usr/bin/env python3
"""Independent invariant checks for lifecycle logs."""
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); a=ap.parse_args(); f=ROOT/a.input
    rows=json.loads((f/'raw.json').read_text()); errors=[]
    for r in rows:
        for key in ['pre_commit_insert_visible','post_commit_deleted_visible','rollback_state_unchanged','persistence_equal']:
            if not isinstance(r[key],bool): errors.append((r['policy'],r['seed'],key))
        if r['pre_commit_insert_visible']: errors.append((r['policy'],r['seed'],'uncommitted insert visible'))
        if r['post_commit_deleted_visible']: errors.append((r['policy'],r['seed'],'deleted id visible'))
        if not r['rollback_state_unchanged'] or not r['persistence_equal']: errors.append((r['policy'],r['seed'],'state mismatch'))
        if r['restored_count']!=r['committed_count']: errors.append((r['policy'],r['seed'],'count mismatch'))
        if not np.isfinite([r['recall'],r['commit_ms'],r['query_p95_ms'],r['concurrent_wall_ms']]).all(): errors.append((r['policy'],r['seed'],'nonfinite'))
    result={'runs':len(rows),'passed':not errors,'errors':errors}
    (f/'verification.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2)); raise SystemExit(1 if errors else 0)
if __name__=='__main__': main()
