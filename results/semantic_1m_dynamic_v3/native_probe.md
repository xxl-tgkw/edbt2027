# Native dynamic-index probe

The official `SPFresh/SPFresh` repository was cloned at commit
`5893eb61ee3b18610b6b00f1939be7dae1af8904`. The repository advertises online vector
insertion and deletion. A CPU-only CMake configuration was attempted with GCC 11,
Boost 1.74, RocksDB 6.11, TBB, and SPDK disabled. The non-SPANN BKT library compiled,
but the end-to-end `indexbuilder` target requires SPDK static libraries and the SPANN
sources include `spdk/env.h`; the complete SPDK/NVMe setup is not available on the
evaluation host. No SPFresh timing or recall value is used in the paper.

The reproducible scale result in `raw.json` is the native Faiss control (IVF-fixed,
IVF-rebuilt, and HNSW) on the same 1M-vector paired workload. This file records the
blocked native dependency explicitly so an evaluator can rerun it on an SPDK-capable
machine without confusing a build probe with a numerical baseline.
