# Semantic-vector results

80 index runs; 240000 query calls checked.

| Order | Policy | Budget | Recall | p95 ms | Service ms | Maintenance ms | Serialized MB |
|---|---|---:|---:|---:|---:|---:|---:|
| shifted | flat | 0 | 1.0000 | 0.8623 | 341.5 | 10.2 | 16.00 |
| shifted | ivf_fixed | 4 | 0.5318 | 0.0822 | 49.3 | 21.1 | 16.37 |
| shifted | ivf_fixed | 16 | 0.7857 | 0.2297 | 98.5 | 21.1 | 16.37 |
| shifted | ivf_fixed | 64 | 0.9790 | 0.6374 | 255.9 | 21.1 | 16.37 |
| shifted | ivf_rebuilt | 4 | 0.6364 | 0.0661 | 896.7 | 872.2 | 16.37 |
| shifted | ivf_rebuilt | 16 | 0.8409 | 0.1853 | 939.9 | 872.2 | 16.37 |
| shifted | ivf_rebuilt | 64 | 0.9847 | 0.5684 | 1085.1 | 872.2 | 16.37 |
| shifted | hnsw | 16 | 0.6861 | 0.0580 | 3982.3 | 3960.7 | 21.77 |
| shifted | hnsw | 64 | 0.8841 | 0.1393 | 4015.6 | 3960.7 | 21.77 |
| shifted | hnsw | 256 | 0.9805 | 0.4483 | 4146.0 | 3960.7 | 21.77 |
| shuffled | flat | 0 | 1.0000 | 0.8146 | 321.8 | 9.0 | 16.00 |
| shuffled | ivf_fixed | 4 | 0.6478 | 0.0620 | 41.1 | 17.9 | 16.37 |
| shuffled | ivf_fixed | 16 | 0.8428 | 0.1632 | 80.5 | 17.9 | 16.37 |
| shuffled | ivf_fixed | 64 | 0.9845 | 0.4992 | 217.5 | 17.9 | 16.37 |
| shuffled | ivf_rebuilt | 4 | 0.6556 | 0.0631 | 908.6 | 885.3 | 16.37 |
| shuffled | ivf_rebuilt | 16 | 0.8474 | 0.1700 | 948.9 | 885.3 | 16.37 |
| shuffled | ivf_rebuilt | 64 | 0.9854 | 0.5313 | 1089.6 | 885.3 | 16.37 |
| shuffled | hnsw | 16 | 0.6820 | 0.0596 | 3838.4 | 3816.4 | 21.77 |
| shuffled | hnsw | 64 | 0.8772 | 0.1430 | 3873.3 | 3816.4 | 21.77 |
| shuffled | hnsw | 256 | 0.9785 | 0.4598 | 4005.4 | 3816.4 | 21.77 |

## Paired comparisons (eight-test Holm family)

- rebuilt-minus-fixed-shifted-probe4: effect +0.1047, 95% descriptive CI [0.09431949999999993, 0.11417999999999995], exact p=0.001953, Holm p=0.015625.
- rebuilt-minus-fixed-shifted-probe16: effect +0.0552, 95% descriptive CI [0.04906000000000005, 0.061960500000000085], exact p=0.001953, Holm p=0.015625.
- rebuilt-minus-fixed-shuffled-probe4: effect +0.0077, 95% descriptive CI [0.0002794999999999153, 0.016159999999999963], exact p=0.130859, Holm p=0.130859.
- rebuilt-minus-fixed-shuffled-probe16: effect +0.0046, 95% descriptive CI [0.0006999999999999784, 0.008139999999999915], exact p=0.054688, Holm p=0.109375.
- shifted-minus-shuffled-recall-change-ivf_fixed-4: effect +0.0528, 95% descriptive CI [0.028299999999999985, 0.0761999999999999], exact p=0.005859, Holm p=0.017578.
- shifted-minus-shuffled-recall-change-ivf_fixed-16: effect +0.0911, 95% descriptive CI [0.07099749999999996, 0.11070000000000006], exact p=0.001953, Holm p=0.015625.
- shifted-minus-shuffled-recall-change-hnsw-16: effect +0.1332, 95% descriptive CI [0.11400000000000007, 0.15200250000000004], exact p=0.001953, Holm p=0.015625.
- shifted-minus-shuffled-recall-change-hnsw-64: effect +0.0639, 95% descriptive CI [0.05119750000000001, 0.07640000000000005], exact p=0.001953, Holm p=0.015625.

## Validation-selected recall floors

Means below condition on feasible seeds; inspect counts before comparison.
Test query work excludes tuning; service-plus-tuning includes every validation-budget call.

| Order | Policy | Floor | Feasible seeds | Test recall | Service ms | With tuning ms |
|---|---|---:|---:|---:|---:|---:|
| shifted | flat | 0.9 | 10/10 | 1.0000 | 341.4610 | 672.3642 |
| shifted | flat | 0.95 | 10/10 | 1.0000 | 341.4610 | 672.3642 |
| shifted | ivf_fixed | 0.9 | 10/10 | 0.9790 | 255.8815 | 604.2931 |
| shifted | ivf_fixed | 0.95 | 10/10 | 0.9790 | 255.8815 | 604.2931 |
| shifted | ivf_rebuilt | 0.9 | 10/10 | 0.9823 | 1081.4810 | 1393.2686 |
| shifted | ivf_rebuilt | 0.95 | 10/10 | 0.9847 | 1085.0573 | 1396.8448 |
| shifted | hnsw | 0.9 | 10/10 | 0.9554 | 4108.1093 | 4368.3797 |
| shifted | hnsw | 0.95 | 10/10 | 0.9805 | 4145.9998 | 4406.2702 |
| shuffled | flat | 0.9 | 10/10 | 1.0000 | 321.7847 | 637.1422 |
| shuffled | flat | 0.95 | 10/10 | 1.0000 | 321.7847 | 637.1422 |
| shuffled | ivf_fixed | 0.9 | 10/10 | 0.9845 | 217.4746 | 508.4304 |
| shuffled | ivf_fixed | 0.95 | 10/10 | 0.9845 | 217.4746 | 508.4304 |
| shuffled | ivf_rebuilt | 0.9 | 10/10 | 0.9830 | 1086.2722 | 1383.9800 |
| shuffled | ivf_rebuilt | 0.95 | 10/10 | 0.9854 | 1089.6166 | 1387.3244 |
| shuffled | hnsw | 0.9 | 10/10 | 0.9588 | 3979.7292 | 4247.2930 |
| shuffled | hnsw | 0.95 | 10/10 | 0.9785 | 4005.4344 | 4272.9982 |
