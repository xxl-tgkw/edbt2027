#!/usr/bin/env python3
"""Exact conditional paired signed-rank tests, including ties and zeros."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]


def exact_signed_rank(differences):
    d = np.round(np.asarray(differences, dtype=float), 12)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    if len(d) > 20:
        raise ValueError('Enumeration limited to twenty nonzero pairs')
    ranks = rankdata(np.abs(d), method='average')
    observed = abs(float(np.dot(np.sign(d), ranks)))
    extreme = sum(abs(float(np.dot(signs, ranks))) >= observed - 1e-12
                  for signs in itertools.product([-1, 1], repeat=len(d)))
    return extreme / 2**len(d)


def comparison(rows, left, right, metric, select, family):
    pairs = {}
    for method in [left, right]:
        chosen = [x for x in rows if x['method'] == method and select(x)]
        if len({x['seed'] for x in chosen}) != len(chosen):
            raise ValueError('Duplicate seed within comparison')
        pairs[method] = {x['seed']: x[metric] for x in chosen}
    if not pairs[left] or set(pairs[left]) != set(pairs[right]):
        raise ValueError('Paired seeds do not match')
    seeds = sorted(pairs[left])
    a, b = [[pairs[m][s] for s in seeds] for m in [left, right]]
    delta = np.array(b)-np.array(a)
    rng = np.random.default_rng(20260916)
    boot = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)
    return dict(family=family, left=left, right=right, metric=metric,
                seeds=seeds, left_values=a, right_values=b, differences=delta.tolist(),
                mean_difference=float(delta.mean()),
                paired_bootstrap_ci95=np.quantile(boot, [.025, .975]).tolist(),
                p_exact=exact_signed_rank(delta),
                null='symmetric paired differences; conditional sign enumeration')


def main():
    sources = ['results/raw.json', 'results/validation_v3/raw.json']
    old, new = [json.loads((ROOT/p).read_text()) for p in sources]
    tests=[]
    for dataset in ['digits', 'clustered']:
        tests.append(comparison(old,'lsh','ivf_rebuild','recall@10',
                     lambda x:x['dataset']==dataset and x['drift']==.5,
                     f'legacy-{dataset}-drift0.5'))
    for q in [10,40,160]:
        for metric in ['recall','service_ms']:
            tests.append(comparison(new,'faiss_ivf_static','faiss_ivf_rebuild',metric,
                         lambda x:x['queries_per_batch']==q,f'validation-q{q}'))
    running = 0.
    for rank, i in enumerate(sorted(range(len(tests)), key=lambda i: tests[i]['p_exact'])):
        running = max(running, min(1., (len(tests)-rank)*tests[i]['p_exact']))
        tests[i]['p_holm'] = running
    report = dict(method='exact conditional paired signed-rank',
                  effect_direction='right minus left', exploratory=True,
                  source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                  comparisons=tests)
    (ROOT/'results/audited_statistics.json').write_text(json.dumps(report, indent=2))
    for x in tests:
        print(x['family'], x['metric'], 'delta=',round(x['mean_difference'],6),
              'p_exact=',x['p_exact'], 'p_holm=',x['p_holm'])


if __name__=='__main__':
    main()
