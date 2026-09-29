#!/usr/bin/env python3
"""Audit and summarize semantic arrival-order experiments without changing evidence."""
import argparse
import json
from pathlib import Path

import numpy as np

from prepare_semantic_data import sha
from run_semantic import BUDGETS, POLICIES
from stat_tests import exact_signed_rank

ROOT=Path(__file__).resolve().parents[1]


def avg(values): return float(np.mean(values))


def paired(name,differences,seeds):
    d=np.asarray(differences,dtype=float)
    boot=d[np.random.default_rng(20260917).integers(0,len(d),(10000,len(d)))].mean(1)
    return dict(name=name,seeds=seeds,differences=d.tolist(),mean_difference=avg(d),
                bootstrap_ci95=np.quantile(boot,[.025,.975]).tolist(),
                p_exact=exact_signed_rank(d))


def close(a,b,name):
    if not np.isclose(a,b,rtol=1e-10,atol=1e-10):
        raise ValueError(f'Inconsistent {name}: {a} != {b}')


def audit_row(row,workload,precision_cases):
    if row['corpus_sha256']!=workload['corpus_sha256'] or row['queries_sha256']!=workload['queries_sha256']:
        raise ValueError('Workload mismatch')
    if len(row['events'])!=6: raise ValueError('Incomplete rounds')
    if row['serialized_bytes']<=0 or row['build_ms']<0: raise ValueError('Invalid index measure')
    maintenance=0.; count=0
    for b,event in enumerate(row['events']):
        if (event['round'],event['indexed'])!=(b,20000+b*4000): raise ValueError('Wrong corpus size')
        rebuilt=row['policy']=='ivf_rebuilt' and b in (2,4)
        if event['rebuilt']!=rebuilt: raise ValueError('Wrong rebuild schedule')
        if (not rebuilt and event['rebuild_ms']!=0) or (b==0 and event['insert_ms']!=0):
            raise ValueError('Unexpected maintenance')
        times=[event['insert_ms'],event['rebuild_ms'],row['build_ms']]
        if not np.isfinite(times).all() or min(times)<0: raise ValueError('Invalid maintenance time')
        maintenance+=event['insert_ms']+event['rebuild_ms']
        if set(event['budgets'])!={str(x) for x in BUDGETS[row['policy']]}: raise ValueError('Budget mismatch')
        for budget,res in event['budgets'].items():
            for split_index,split in enumerate(['validation','test']):
                r=res[split]
                if any(len(r[k])!=100 for k in ['returned_ids','query_ms','recalls','distance_counts']):
                    raise ValueError('Incomplete query records')
                if not np.isfinite(r['query_ms']).all() or min(r['query_ms'])<0:
                    raise ValueError('Invalid query duration')
                for i,returned in enumerate(r['returned_ids']):
                    got=[v for v in returned if v>=0]
                    if len(got)!=len(set(got)) or len(returned)!=10 or any(v>=event['indexed'] for v in got):
                        raise ValueError('Invalid returned IDs')
                    truth=workload['ground_truth'][b][split_index][i]
                    if len(set(truth))!=10 or min(truth)<0 or max(truth)>=event['indexed']:
                        raise ValueError('Invalid exact neighborhood')
                    close(len(set(got)&set(truth))/10,r['recalls'][i],'query recall')
                    if row['policy']=='flat' and r['recalls'][i]!=1:
                        # Preserve strict ID recall. Independently inspect float32/64 cutoff ties.
                        with np.load(ROOT/'data/semantic_glove/pool.npz',allow_pickle=False) as pool:
                            ids=np.asarray(workload['pool_ids'])
                            query=pool['queries'][workload['query_ids'][b][split_index][i]].astype('float64')
                            missing=sorted(set(truth)-set(got)); extra=sorted(set(got)-set(truth))
                            dm=np.sum((pool['vectors'][ids[missing]].astype('float64')-query)**2,axis=1)
                            de=np.sum((pool['vectors'][ids[extra]].astype('float64')-query)**2,axis=1)
                        excess=float(de.max()-dm.min())
                        if excess>1e-6: raise ValueError('Flat discrepancy exceeds float32 cutoff tolerance')
                        precision_cases.append(dict(seed=row['seed'],condition=row['condition'],round=b,
                                                    split=split,query=i,strict_recall=r['recalls'][i],
                                                    max_distance_excess=excess))
                count+=100
    close(maintenance,row['maintenance_ms'],'maintenance')
    for budget,summary in row['summaries'].items():
        rs=[e['budgets'][budget]['test'] for e in row['events'][1:]]
        qt=[v for x in rs for v in x['query_ms']]
        close(avg([v for x in rs for v in x['recalls']]),summary['recall'],'aggregate recall')
        close(avg(qt),summary['query_mean_ms'],'query mean')
        close(np.percentile(qt,95),summary['query_p95_ms'],'query p95')
        close(sum(qt)+maintenance,summary['service_ms'],'service work')
        close(avg([v for x in rs for v in x['distance_counts']]),summary['mean_distance_count'],'distance count')
    return count


def selected_operating_point(row,floor):
    choices=[]; test_rec=[]; qt=[]; validation_ms=0.; failures=[]
    for e in row['events'][1:]:
        accepted=[b for b in sorted(BUDGETS[row['policy']])
                  if avg(e['budgets'][str(b)]['validation']['recalls'])>=floor]
        validation_ms+=sum(sum(r['validation']['query_ms']) for r in e['budgets'].values())
        if not accepted:
            failures.append(e['round']); choices.append(None); continue
        b=min(accepted); choices.append(b)
        test_rec.extend(e['budgets'][str(b)]['test']['recalls'])
        qt.extend(e['budgets'][str(b)]['test']['query_ms'])
    feasible=not failures
    return dict(floor=floor,choices=choices,feasible=feasible,failed_rounds=failures,
                test_recall=avg(test_rec) if feasible else None,
                service_ms=row['maintenance_ms']+sum(qt) if feasible else None,
                tuning_query_ms=validation_ms,
                service_plus_tuning_ms=row['maintenance_ms']+sum(qt)+validation_ms if feasible else None)


def analyze(folder):
    env=json.loads((folder/'environment.json').read_text())
    for name,digest in env['source_hashes'].items():
        if sha(ROOT/name)!=digest: raise ValueError('Source changed: '+name)
    rows=json.loads((folder/'raw.json').read_text())
    if rows!=[json.loads(x) for x in (folder/'events.jsonl').read_text().splitlines()]:
        raise ValueError('Event and final logs disagree')
    seeds=env['args']['seeds']
    keys=[(r['seed'],r['condition'],r['policy']) for r in rows]
    expected={(s,c,p) for s in seeds for c in ['shifted','shuffled'] for p in POLICIES}
    if len(set(keys))!=len(keys) or set(keys)!=expected: raise ValueError('Incomplete index-run matrix')
    lookup={key:r for key,r in zip(keys,rows)}
    hashes={str((folder/f).relative_to(ROOT)):sha(folder/f) for f in ['raw.json','events.jsonl','environment.json']}
    workloads={}; count=0; precision_cases=[]
    for seed in seeds:
        for condition in ['shifted','shuffled']:
            f=folder/f'workload_{condition}_{seed}.json'
            w=json.loads(f.read_text()); workloads[seed,condition]=w
            hashes[str(f.relative_to(ROOT))]=sha(f)
            if len(set(w['source_train_ids']))!=40000 or len(set(np.array(w['query_ids']).ravel()))!=1200:
                raise ValueError('Duplicate source roles')
            for p in POLICIES: count+=audit_row(lookup[seed,condition,p],w,precision_cases)
        a,b=[workloads[seed,c] for c in ['shifted','shuffled']]
        if set(a['source_train_ids'])!=set(b['source_train_ids']) or a['query_ids']!=b['query_ids']:
            raise ValueError('Order controls do not match')
        for split in range(2):
            lhs=np.array(a['source_train_ids'])[a['ground_truth'][-1][split]]
            rhs=np.array(b['source_train_ids'])[b['ground_truth'][-1][split]]
            if not np.array_equal(np.sort(lhs,axis=1),np.sort(rhs,axis=1)):
                raise ValueError('Matched final corpus ground truths differ')
    # Calibrate the constructed order contrast independently of retrieval.  The
    # direction is frozen during workload preparation; this is a descriptor of
    # the two doses, not a semantic label or a causal mediator.
    pool_path=ROOT/'data/semantic_glove/pool.npz'
    with np.load(pool_path,allow_pickle=False) as pool:
        direction=np.asarray(pool['direction'],dtype='float64')
        order_intensity=[]
        for c in ['shifted','shuffled']:
            for seed in seeds:
                w=workloads[seed,c]
                ids=np.asarray(w['pool_ids'],dtype=np.int64)
                projections=np.asarray(pool['vectors'][ids],dtype='float64') @ direction
                base=projections[:w['base']]
                scale=float(projections.std())
                values=[0.0]
                for b in range(1,w['rounds']+1):
                    updates=projections[w['base']:w['base']+b*w['batch']]
                    values.append(float((updates.mean()-base.mean())/scale))
                order_intensity.append(dict(condition=c,seed=seed,values=values,
                                            final=values[-1]))
    hashes[str(pool_path.relative_to(ROOT))]=sha(pool_path)
    aggregates=[]; selected=[]
    for condition in ['shifted','shuffled']:
        for policy in POLICIES:
            group=[lookup[s,condition,policy] for s in seeds]
            for budget in BUDGETS[policy]:
                sums=[r['summaries'][str(budget)] for r in group]
                entry=dict(condition=condition,policy=policy,budget=budget,seeds=seeds,
                           **{key:avg([s[key] for s in sums]) for key in sums[0]})
                entry['seed_recall']=[s['recall'] for s in sums]
                entry['seed_service_ms']=[s['service_ms'] for s in sums]
                for key in ['build_ms','maintenance_ms','serialized_bytes']: entry[key]=avg([r[key] for r in group])
                entry['round_recall']=[[avg(e['budgets'][str(budget)]['test']['recalls']) for e in r['events']] for r in group]
                entry['round_distance_counts']=[[avg(e['budgets'][str(budget)]['test']['distance_counts']) for e in r['events']] for r in group]
                entry['round_imbalance']=[[e['diagnostics'].get('imbalance') for e in r['events']] for r in group]
                aggregates.append(entry)
            for floor in [.90,.95]:
                choices=[selected_operating_point(r,floor) for r in group]
                feasible=[x for x in choices if x['feasible']]
                selected.append(dict(condition=condition,policy=policy,floor=floor,seeds=seeds,
                                     feasible_seeds=len(feasible),choices=choices,
                                     test_recall=avg([x['test_recall'] for x in feasible]) if feasible else None,
                                     service_ms=avg([x['service_ms'] for x in feasible]) if feasible else None,
                                     service_plus_tuning_ms=avg([x['service_plus_tuning_ms'] for x in feasible]) if feasible else None))
    comparisons=[]
    for condition in ['shifted','shuffled']:
        for budget in [4,16]:
            diffs=[lookup[s,condition,'ivf_rebuilt']['summaries'][str(budget)]['recall']-
                   lookup[s,condition,'ivf_fixed']['summaries'][str(budget)]['recall'] for s in seeds]
            comparisons.append(paired(f'rebuilt-minus-fixed-{condition}-probe{budget}',diffs,seeds))
    for policy,budget in [('ivf_fixed',4),('ivf_fixed',16),('hnsw',16),('hnsw',64)]:
        def change(s,c):
            events=lookup[s,c,policy]['events']
            return avg(events[-1]['budgets'][str(budget)]['test']['recalls'])-avg(events[0]['budgets'][str(budget)]['test']['recalls'])
        comparisons.append(paired(f'shifted-minus-shuffled-recall-change-{policy}-{budget}',
                                  [change(s,'shifted')-change(s,'shuffled') for s in seeds],seeds))
    running=0.
    for rank,i in enumerate(sorted(range(len(comparisons)),key=lambda i:comparisons[i]['p_exact'])):
        running=max(running,min(1.,(len(comparisons)-rank)*comparisons[i]['p_exact']))
        comparisons[i]['p_holm']=running
    # Raw exposure alone grows as the update population grows.  Also retain
    # the update-fraction baseline and its ratio for mechanism analysis.
    exposure={}
    for c in ['shifted','shuffled']:
        raw=[workloads[s,c]['updated_neighbor_exposure'] for s in seeds]
        baseline=[((np.arange(6)*workloads[s,c]['batch']) /
                   (workloads[s,c]['base']+np.arange(6)*workloads[s,c]['batch'])).tolist()
                  for s in seeds]
        relative=[]
        for x,b in zip(raw,baseline):
            relative.append([float(v/bv) if bv>0 else 0.0 for v,bv in zip(x,b)])
        exposure[c]=dict(raw=raw, update_fraction=baseline, relative=relative)
    diagnostics=[]
    for c in ['shifted','shuffled']:
        for p in ['ivf_fixed','ivf_rebuilt']:
            group=[lookup[s,c,p] for s in seeds]
            for b in range(6):
                vals=[r['events'][b]['diagnostics'] for r in group]
                diagnostics.append(dict(condition=c,policy=p,round=b,
                    imbalance=avg([v['imbalance'] for v in vals]),
                    list_size_cv=avg([v['list_size_cv'] for v in vals]),
                    max_list_size=avg([v['max_list_size'] for v in vals])))
    result=dict(index_runs=len(rows),budget_observations=len(rows)//4*10,query_calls_checked=count,
                seeds=seeds,source_hashes=hashes,aggregates=aggregates,matched_recall=selected,
                comparisons=comparisons,exposure=exposure,
                order_intensity=order_intensity,
                partition_diagnostics=diagnostics,
                flat_precision_cases=precision_cases,
                caveat='Budget observations share indexes; only seeded runs are statistical units. Excludes pilot.')
    (folder/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Semantic-vector results','',f'{len(rows)} index runs; {count} query calls checked.','',
           '| Order | Policy | Budget | Recall | p95 ms | Service ms | Maintenance ms | Serialized MB |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for x in aggregates:
        lines.append(f'| {x["condition"]} | {x["policy"]} | {x["budget"]} | {x["recall"]:.4f} | '
                     f'{x["query_p95_ms"]:.4f} | {x["service_ms"]:.1f} | {x["maintenance_ms"]:.1f} | {x["serialized_bytes"]/1e6:.2f} |')
    lines.extend(['','## Paired comparisons (eight-test Holm family)',''])
    for t in comparisons:
        lines.append(f'- {t["name"]}: effect {t["mean_difference"]:+.4f}, 95% descriptive CI {t["bootstrap_ci95"]}, '
                     f'exact p={t["p_exact"]:.6f}, Holm p={t["p_holm"]:.6f}.')
    lines.extend(['','## Validation-selected recall floors','',
                  'Means below condition on feasible seeds; inspect counts before comparison.',
                  'Test query work excludes tuning; service-plus-tuning includes every validation-budget call.','',
                  '| Order | Policy | Floor | Feasible seeds | Test recall | Service ms | With tuning ms |',
                  '|---|---|---:|---:|---:|---:|---:|'])
    def fmt(x): return 'NA' if x is None else f'{x:.4f}'
    for x in selected:
        lines.append(f'| {x["condition"]} | {x["policy"]} | {x["floor"]} | {x["feasible_seeds"]}/{len(seeds)} | '
                     f'{fmt(x["test_recall"])} | {fmt(x["service_ms"])} | {fmt(x["service_plus_tuning_ms"])} |')
    (folder/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',default='results/semantic_v1'); args=p.parse_args()
    analyze(ROOT/args.input)


if __name__=='__main__': main()
