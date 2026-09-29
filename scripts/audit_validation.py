#!/usr/bin/env python3
"""Verify recorded events and derive update-neighbor exposure without rerunning timings."""
import hashlib
import json
from pathlib import Path

import numpy as np

from stat_tests import comparison

ROOT = Path(__file__).resolve().parents[1]
RUNS = [('results/validation_v3', .35), ('results/validation_alignment_100', 1.),
        ('results/validation_alignment_150', 1.5)]
FAISS = ['faiss_flat', 'faiss_ivf_static', 'faiss_ivf_rebuild']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_record(row, workload):
    if not np.isfinite(row['build_ms']) or row['build_ms'] < 0:
        raise ValueError('Invalid initial build duration')
    if row['workload_hashes'] != workload['hashes']:
        raise ValueError('Workload hash mismatch')
    if len(row['events']) != 5:
        raise ValueError('Missing rounds')
    times, recalls = [], []
    maintenance = 0.
    for b, event in enumerate(row['events']):
        q = row['queries_per_batch']
        if (event['round'], event['indexed'], event['query_count']) != (b+1, 6100+100*b, q):
            raise ValueError('Incorrect round or corpus size')
        if any(len(event[k]) != q for k in ['query_ms', 'recalls', 'returned_ids']):
            raise ValueError('Incomplete query events')
        rebuilt = row['method'].endswith('_rebuild') and b in (1, 3)
        if event['rebuilt'] != rebuilt or (not rebuilt and event['rebuild_ms'] != 0):
            raise ValueError('Unexpected rebuild schedule')
        if not np.isfinite([event['insert_ms'], event['rebuild_ms'], *event['query_ms']]).all():
            raise ValueError('Nonfinite event time')
        if min(event['insert_ms'], event['rebuild_ms'], *event['query_ms']) < 0:
            raise ValueError('Negative event time')
        for i, got in enumerate(event['returned_ids']):
            truth = workload['ground_truth'][b][i]
            if len(truth) != 10 or len(set(truth)) != 10 or min(truth) < 0 or max(truth) >= event['indexed']:
                raise ValueError('Invalid ground truth')
            if len(got) != 10 or len(set(got)) != 10 or min(got) < 0 or max(got) >= event['indexed']:
                raise ValueError('Invalid returned IDs')
            recall = len(set(truth) & set(got)) / 10
            if recall != event['recalls'][i]:
                raise ValueError('Recall disagrees with saved IDs')
            recalls.append(recall)
        times.extend(event['query_ms'])
        maintenance += event['insert_ms'] + event['rebuild_ms']
    for name, value in [('recall', np.mean(recalls)), ('query_ms_mean', np.mean(times)),
                        ('query_ms_p95', np.percentile(times, 95)),
                        ('query_count', len(times)), ('maintenance_ms', maintenance),
                        ('service_ms', maintenance + sum(times))]:
        if not np.isclose(row[name], value, rtol=1e-12, atol=1e-12):
            raise ValueError(f'Incorrect aggregate: {name}')
    if row['method'] in ('exact', 'faiss_flat') and row['recall'] != 1.:
        raise ValueError('Exact baseline recall is not one')


def main():
    hashes, conditions, tests, source_roles = {}, [], [], {}
    n_jobs = n_queries = 0
    for folder, factor in RUNS:
        path = ROOT/folder
        rows = json.loads((path/'raw.json').read_text())
        logged = [json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
        if rows != logged:
            raise ValueError('Incremental event log disagrees with raw.json')
        expected_methods = FAISS if factor != .35 else FAISS + ['exact', 'lsh', 'ivf_static', 'ivf_rebuild']
        ratios = [40] if factor != .35 else [10, 40, 160]
        keys = [(r['seed'], r['queries_per_batch'], r['method']) for r in rows]
        expected = {(s,q,m) for s in range(5) for q in ratios for m in expected_methods}
        if len(set(keys)) != len(keys) or set(keys) != expected:
            raise ValueError('Incomplete or duplicate job matrix')
        env = json.loads((path/'environment.json').read_text())
        snapshot = ROOT/('scripts/protocol_snapshots/validation_v3.py' if factor == .35
                        else 'scripts/run_validation.py')
        if env['runner_sha256'] != digest(snapshot):
            raise ValueError('Runner source hash mismatch')
        for name, sha in env.get('source_hashes', {}).items():
            if digest(ROOT/name) != sha:
                raise ValueError(f'Recorded dependency source changed: {name}')
        for file in ['raw.json', 'events.jsonl', 'environment.json'] + [f'workload_{s}.json' for s in range(5)]:
            hashes[f'{folder}/{file}'] = digest(path/file)
        exposure = []
        for seed in range(5):
            workload = json.loads((path/f'workload_{seed}.json').read_text())
            role_ids = [np.array(workload[k], dtype=np.int64) for k in ['base_ids','update_ids','query_ids']]
            ids = np.concatenate([x.ravel() for x in role_ids])
            if len(ids) != 7300 or len(set(ids)) != 7300 or ids.min() < 0 or ids.max() >= 60000:
                raise ValueError('Source roles are not disjoint valid rows')
            if [x.shape for x in role_ids] != [(6000,), (5,100), (5,160)]:
                raise ValueError('Invalid source role shapes')
            for name, x in zip(['base', 'updates', 'queries'], role_ids):
                if hashlib.sha256(x.tobytes()).hexdigest() != workload['hashes'][name]:
                    raise ValueError('Source role IDs disagree with hashes')
            if seed in source_roles and source_roles[seed] != workload['hashes']:
                raise ValueError('Query-alignment conditions changed source roles')
            source_roles[seed] = workload['hashes']
            # The q40 common prefix makes exposure comparable across all three conditions.
            truth = np.asarray(workload['ground_truth'])[:, :40]
            exposure.append((truth >= 6000).mean(axis=(1,2)).tolist())
            for row in rows:
                if row['seed'] == seed:
                    verify_record(row, workload)
                    n_queries += row['query_count']
        n_jobs += len(rows)
        summaries = []
        for method in FAISS:
            selected = sorted([r for r in rows if r['method'] == method and r['queries_per_batch'] == 40], key=lambda r:r['seed'])
            metric = {k: [float(r[k]) for r in selected] for k in ['recall', 'service_ms', 'query_ms_p95', 'build_ms']}
            metric.update({k: [sum(e[k] for e in r['events']) for r in selected] for k in ['insert_ms', 'rebuild_ms']})
            metric['query_total_ms'] = [sum(sum(e['query_ms']) for e in r['events']) for r in selected]
            metric['round_recalls'] = [[float(np.mean(e['recalls'])) for e in r['events']] for r in selected]
            summaries.append(dict(method=method, seeds=list(range(5)), **metric))
        conditions.append(dict(query_shift_factor=factor, exposure_per_seed_round=exposure,
                               mean_exposure=float(np.mean(exposure)), policies=summaries))
        if factor != .35:
            for metric in ['recall', 'service_ms']:
                tests.append(comparison(rows, 'faiss_ivf_static', 'faiss_ivf_rebuild', metric,
                                        lambda r:True, f'alignment-{factor:g}'))
    running = 0.
    for rank, i in enumerate(sorted(range(len(tests)), key=lambda i:tests[i]['p_exact'])):
        running = max(running, min(1., (len(tests)-rank)*tests[i]['p_exact']))
        tests[i]['p_holm'] = running
    report = dict(jobs_verified=n_jobs, measured_queries_verified=n_queries,
                  comparisons=tests, conditions=conditions, source_hashes=hashes,
                  note='Exposure: share of exact top-10 IDs >= 6000, common 40-query prefix; not an index metric.')
    (ROOT/'results/validation_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'Validated {n_jobs} jobs and {n_queries} measured queries')
    for c in conditions:
        print('factor', c['query_shift_factor'], 'update-neighbor share', c['mean_exposure'])
        for p in c['policies']:
            print(p['method'], 'recall', np.mean(p['recall']), 'service ms', np.mean(p['service_ms']))
    for t in tests:
        print(t['family'], t['metric'], 'effect', t['mean_difference'], 'exact p', t['p_exact'], 'Holm', t['p_holm'])


if __name__ == '__main__':
    main()
