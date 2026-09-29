# Semantic scale validation

This is a three-seed external scale check; it is not pooled with the 10-seed mechanism study.

| Size | Order | Policy | Budget | Recall | p95 ms | Service ms |
|---:|---|---|---:|---:|---:|---:|
| 100000 | shifted | flat | 0 | 1.0000 | 3.539 | 2194.8 |
| 100000 | shifted | ivf_fixed | 4 | 0.6508 | 0.133 | 80.0 |
| 100000 | shifted | ivf_fixed | 16 | 0.8107 | 0.322 | 184.3 |
| 100000 | shifted | ivf_fixed | 64 | 0.9456 | 0.992 | 573.8 |
| 100000 | shifted | ivf_rebuilt | 4 | 0.6624 | 0.125 | 4104.8 |
| 100000 | shifted | ivf_rebuilt | 16 | 0.8267 | 0.314 | 4209.7 |
| 100000 | shifted | ivf_rebuilt | 64 | 0.9533 | 0.993 | 4599.1 |
| 100000 | shuffled | flat | 0 | 1.0000 | 3.507 | 2190.7 |
| 100000 | shuffled | ivf_fixed | 4 | 0.6352 | 0.113 | 77.5 |
| 100000 | shuffled | ivf_fixed | 16 | 0.8224 | 0.305 | 184.5 |
| 100000 | shuffled | ivf_fixed | 64 | 0.9484 | 0.977 | 575.2 |
| 100000 | shuffled | ivf_rebuilt | 4 | 0.6495 | 0.115 | 4066.9 |
| 100000 | shuffled | ivf_rebuilt | 16 | 0.8186 | 0.316 | 4174.3 |
| 100000 | shuffled | ivf_rebuilt | 64 | 0.9455 | 0.979 | 4553.8 |
