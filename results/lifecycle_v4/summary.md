# Semantic lifecycle results

20 runs; 5 seeds.

| Policy | Recall | Commit ms | Query p95 ms | Concurrent wall ms | Persistence | Rollback |
|---|---:|---:|---:|---:|---:|---:|
| flat | 1.0000 | 11.44 | 0.4381 | 103.28 | 100% | 100% |
| hnsw | 0.9367 | 954.91 | 0.3104 | 300.27 | 100% | 100% |
| ivf_fixed | 0.8718 | 54.65 | 0.2842 | 603.88 | 100% | 100% |
| ivf_rebuilt | 0.8718 | 49.61 | 0.2691 | 788.47 | 100% | 100% |
