# Semantic scale validation

This is a three-seed external scale check; it is not pooled with the 10-seed mechanism study.

| Size | Order | Policy | Budget | Recall | p95 ms | Service ms |
|---:|---|---|---:|---:|---:|---:|
| 250000 | shifted | flat | 0 | 1.0000 | 9.499 | 6021.2 |
| 250000 | shifted | ivf_fixed | 4 | 0.6841 | 0.364 | 208.2 |
| 250000 | shifted | ivf_fixed | 16 | 0.8484 | 0.965 | 511.2 |
| 250000 | shifted | ivf_fixed | 64 | 0.9599 | 3.063 | 1746.8 |
| 250000 | shifted | ivf_rebuilt | 4 | 0.7058 | 0.284 | 10282.0 |
| 250000 | shifted | ivf_rebuilt | 16 | 0.8653 | 0.838 | 10588.1 |
| 250000 | shifted | ivf_rebuilt | 64 | 0.9661 | 2.849 | 11798.3 |
| 250000 | shuffled | flat | 0 | 1.0000 | 9.549 | 6043.1 |
| 250000 | shuffled | ivf_fixed | 4 | 0.6958 | 0.270 | 198.0 |
| 250000 | shuffled | ivf_fixed | 16 | 0.8586 | 0.839 | 501.8 |
| 250000 | shuffled | ivf_fixed | 64 | 0.9622 | 2.770 | 1667.6 |
| 250000 | shuffled | ivf_rebuilt | 4 | 0.7077 | 0.270 | 10336.8 |
| 250000 | shuffled | ivf_rebuilt | 16 | 0.8628 | 0.805 | 10640.0 |
| 250000 | shuffled | ivf_rebuilt | 64 | 0.9658 | 2.795 | 11842.6 |
