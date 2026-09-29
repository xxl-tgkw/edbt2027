# Semantic-vector results

8 index runs; 24000 query calls checked.

| Order | Policy | Budget | Recall | p95 ms | Service ms | Maintenance ms | Serialized MB |
|---|---|---:|---:|---:|---:|---:|---:|
| shifted | flat | 0 | 1.0000 | 0.8034 | 326.7 | 13.1 | 16.00 |
| shifted | ivf_fixed | 4 | 0.5396 | 0.0737 | 40.4 | 13.9 | 16.37 |
| shifted | ivf_fixed | 16 | 0.8026 | 0.2620 | 92.5 | 13.9 | 16.37 |
| shifted | ivf_fixed | 64 | 0.9838 | 0.5732 | 232.1 | 13.9 | 16.37 |
| shifted | ivf_rebuilt | 4 | 0.6354 | 0.0634 | 878.4 | 854.0 | 16.37 |
| shifted | ivf_rebuilt | 16 | 0.8400 | 0.1652 | 920.0 | 854.0 | 16.37 |
| shifted | ivf_rebuilt | 64 | 0.9858 | 0.5059 | 1056.9 | 854.0 | 16.37 |
| shifted | hnsw | 16 | 0.6888 | 0.0518 | 3699.3 | 3679.2 | 21.77 |
| shifted | hnsw | 64 | 0.8828 | 0.1206 | 3729.8 | 3679.2 | 21.77 |
| shifted | hnsw | 256 | 0.9774 | 0.3828 | 3846.2 | 3679.2 | 21.77 |
| shuffled | flat | 0 | 1.0000 | 0.7551 | 302.7 | 4.0 | 16.00 |
| shuffled | ivf_fixed | 4 | 0.6436 | 0.0593 | 40.1 | 17.2 | 16.37 |
| shuffled | ivf_fixed | 16 | 0.8358 | 0.1579 | 78.3 | 17.2 | 16.37 |
| shuffled | ivf_fixed | 64 | 0.9816 | 0.4902 | 212.0 | 17.2 | 16.37 |
| shuffled | ivf_rebuilt | 4 | 0.6448 | 0.0591 | 849.3 | 826.6 | 16.37 |
| shuffled | ivf_rebuilt | 16 | 0.8330 | 0.1484 | 885.6 | 826.6 | 16.37 |
| shuffled | ivf_rebuilt | 64 | 0.9832 | 0.4590 | 1012.2 | 826.6 | 16.37 |
| shuffled | hnsw | 16 | 0.6834 | 0.0537 | 3687.2 | 3666.3 | 21.77 |
| shuffled | hnsw | 64 | 0.8804 | 0.1266 | 3719.8 | 3666.3 | 21.77 |
| shuffled | hnsw | 256 | 0.9772 | 0.4127 | 3846.4 | 3666.3 | 21.77 |

## Paired comparisons (eight-test Holm family)

- rebuilt-minus-fixed-shifted-probe4: effect +0.0958, 95% descriptive CI [0.09580000000000022, 0.09580000000000022], exact p=1.000000, Holm p=1.000000.
- rebuilt-minus-fixed-shifted-probe16: effect +0.0374, 95% descriptive CI [0.03739999999999999, 0.03739999999999999], exact p=1.000000, Holm p=1.000000.
- rebuilt-minus-fixed-shuffled-probe4: effect +0.0012, 95% descriptive CI [0.0011999999999998678, 0.0011999999999998678], exact p=1.000000, Holm p=1.000000.
- rebuilt-minus-fixed-shuffled-probe16: effect -0.0028, 95% descriptive CI [-0.0028000000000000247, -0.0028000000000000247], exact p=1.000000, Holm p=1.000000.
- shifted-minus-shuffled-recall-change-ivf_fixed-4: effect +0.0440, 95% descriptive CI [0.044000000000000095, 0.044000000000000095], exact p=1.000000, Holm p=1.000000.
- shifted-minus-shuffled-recall-change-ivf_fixed-16: effect +0.0930, 95% descriptive CI [0.09299999999999986, 0.09299999999999986], exact p=1.000000, Holm p=1.000000.
- shifted-minus-shuffled-recall-change-hnsw-16: effect +0.0890, 95% descriptive CI [0.08900000000000008, 0.08900000000000008], exact p=1.000000, Holm p=1.000000.
- shifted-minus-shuffled-recall-change-hnsw-64: effect +0.0960, 95% descriptive CI [0.09600000000000009, 0.09600000000000009], exact p=1.000000, Holm p=1.000000.

## Validation-selected recall floors

Means below condition on feasible seeds; inspect counts before comparison.
Test query work excludes tuning; service-plus-tuning includes every validation-budget call.

| Order | Policy | Floor | Feasible seeds | Test recall | Service ms | With tuning ms |
|---|---|---:|---:|---:|---:|---:|
| shifted | flat | 0.9 | 1/1 | 1.0000 | 326.7129 | 640.0767 |
| shifted | flat | 0.95 | 1/1 | 1.0000 | 326.7129 | 640.0767 |
| shifted | ivf_fixed | 0.9 | 1/1 | 0.9838 | 232.0652 | 557.7541 |
| shifted | ivf_fixed | 0.95 | 1/1 | 0.9838 | 232.0652 | 557.7541 |
| shifted | ivf_rebuilt | 0.9 | 1/1 | 0.9858 | 1056.9015 | 1362.2992 |
| shifted | ivf_rebuilt | 0.95 | 1/1 | 0.9858 | 1056.9015 | 1362.2992 |
| shifted | hnsw | 0.9 | 1/1 | 0.9562 | 3822.2629 | 4062.2551 |
| shifted | hnsw | 0.95 | 1/1 | 0.9774 | 3846.1594 | 4086.1516 |
| shuffled | flat | 0.9 | 1/1 | 1.0000 | 302.7059 | 601.5635 |
| shuffled | flat | 0.95 | 1/1 | 1.0000 | 302.7059 | 601.5635 |
| shuffled | ivf_fixed | 0.9 | 1/1 | 0.9816 | 211.9746 | 496.0799 |
| shuffled | ivf_fixed | 0.95 | 1/1 | 0.9816 | 211.9746 | 496.0799 |
| shuffled | ivf_rebuilt | 0.9 | 1/1 | 0.9832 | 1012.1634 | 1285.7448 |
| shuffled | ivf_rebuilt | 0.95 | 1/1 | 0.9832 | 1012.1634 | 1285.7448 |
| shuffled | hnsw | 0.9 | 1/1 | 0.9524 | 3820.3074 | 4077.2758 |
| shuffled | hnsw | 0.95 | 1/1 | 0.9772 | 3846.4131 | 4103.3815 |
