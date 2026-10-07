# UVR-Bench artifact manifest

## Public mirror

The manuscript-free branch is `https://github.com/xxl-tgkw/edbt2027` and the
anonymous review mirror is `https://anonymous.4open.science/r/edbt2027-7FD5/`.
The complete archives are published through the GitHub release page because
GitHub branch files are limited to 100 MB.  The 2026-10-06 archive hashes are:

- `edbt2027_latex_20261007_140931_UTC.zip`:
  `09de8c19a6ceb1be152f35e4c74092ecc076192aa073459883e337dc6dffbf5a`
- `uvrbench_artifact_*.zip`: use the matching `.zip.sha256` sidecar and release
  asset for the final checksum.

The protocol figure in the paper is the portable vector rendering in
`results/uvrbench_protocol.svg`; the author-supplied sources are preserved as
`results/uvrbench_protocol_source.svg` and `results/uvrbench_protocol_source.png`.
The LaTeX package includes the exact author-supplied submission PDF as
`edbt_submission.pdf`; the manuscript-free artifact mirror does not include it.
The submission PDF uses the required `[EA&B]` title prefix and is 12 pages total;
native FreshDiskANN/CONDA coverage details remain fully available in the artifact.

The branch omits only the 143 MB HotpotQA `pool.npz` and the 485 MB original
GloVe HDF5.  Their manifests and deterministic preparation scripts remain in the
branch; both files are present in the release archive.

The authoritative revised evidence is the 105-job validation_v3 ratio matrix plus
two 15-job query-alignment follow-ups. Legacy evidence comprises 180 reference,
18 public, 48 sensitivity and 18 scale observations. README.md gives verification
and fresh-directory reproduction commands.

The semantic extension is a separate 80-index-run matrix in results/semantic_v1. It
uses public ANN-Benchmarks GloVe-100 word vectors, ten seeds, matched shifted and
shuffled arrival orders, Faiss Flat/IVF/HNSW, and three search budgets per approximate
policy. It is included in the manuscript's main evidence but is not pooled with the
Fashion-MNIST runs. The normalized pool is supplied in data/semantic_glove/pool.npz;
the original 485 MB HDF5 is not redistributed. Source and preprocessing details are
in data/semantic_glove/SOURCE.md and manifest.json.

The lifecycle extension is stored in results/lifecycle_v4 (20 runs, five seeds). It
checks staged insert/delete visibility, commit and rollback state transitions,
lock-protected concurrent query/update publication, and serialize/reload checksums.

The document-semantic extension is in results/hotpot_semantic_v1 (24 runs: three seeds,
two paired arrival orders and four Faiss policies). data/semantic_hotpot/pool.npz is a
100k-vector, 384-dimensional fixture made from title-prefixed HotpotQA context sentences;
queries.json preserves the 512 source-question IDs and supporting-title metadata.
The manifest records the cached HotpotQA source hash and the Apache-2.0
all-MiniLM-L6-v2 encoder hash. This validates text/document retrieval, but is not an
agent trajectory, end-to-end RAG answer benchmark, or production timestamp trace.

The CONDA input materializer (`scripts/prepare_conda_inputs.py`) reconstructs the
same corpus hashes from the public GloVe HDF5, and `scripts/score_conda_native.py`
recomputes validation/test recall from each saved native response stream.

External validation layers are stored separately from the primary evidence: the
100k and 250k GloVe scale checks are in results/semantic_scale_100k_v2 and
results/semantic_scale_250k_v2 (three seeds each), and the 40-run rebuild-frequency
sensitivity is in results/semantic_rebuild_sensitivity_v1 (five seeds). These runs
use the same public source and exact checkpoint ground truth but are not pooled into
the ten-seed mechanism statistics.

The million-vector dynamic extension is in results/semantic_1m_dynamic_v3. It contains
six jobs (shifted/shuffled orders × fixed IVF, rebuilt IVF, and HNSW), 800k initial
vectors, five 40k insertion rounds, 64 validation and 64 held-out test queries, raw
event logs, and verification.json. paper/large_scale_dynamic_table.tex is generated
from this snapshot. Native dynamic-index evidence is in results/native_dynamic_v1.
It includes the FreshDiskANN/DiskANN C++ dynamic-memory run at commit
78256bbab4685e1774e78d331e081a153be26823 (1M final vectors, both arrival orders and
all four validation budgets) and an official SPFresh/SPANN build--update--search run
at commit 5893eb61ee3b18610b6b00f1939be7dae1af8904 using
SPFRESH_SPDK_USE_MEM_IMPL=1. The SPFresh run is an in-memory native stress result,
not an NVMe latency measurement; the host has no NVMe controller. Raw logs,
configurations, source runner, compatibility patch, summaries, and SHA256SUMS are
included. The complete scope and hardware boundary are documented in native_probe.md.
It also includes native CONDA runs at commit
4117a4e38938aa0ff583c0c7c1beb5bb465e5b4b on the same 1M paired replay.  CONDA's
aggregate BatchSearch timing is kept separate from per-query p95 metrics.  UBIS is
recorded as `not_run` in `ubis_probe.md`: the public paper/source bundle exposes no
implementation or update API, so the coverage table contains no substituted UBIS
number.

## Data

Fashion-MNIST has 60,000 training images, 28×28 grayscale coordinates. Pixel values
are divided by 255; labels are unused. Cache: data/fashion_train_images.gz.
SHA-256: 3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84

Upstream: https://github.com/zalandoresearch/fashion-mnist

Original endpoint:
http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz

The supplied cache supports offline reproduction. The audited loader verifies its
checksum, IDX header and complete payload. The upstream MIT notice is retained in
data/FASHION_MNIST_LICENSE.txt. Citation: Xiao, Rasul and Vollgraf, arXiv:1708.07747.
Digits is bundled with scikit-learn; clustered vectors use a deterministic generator.

## Protocol provenance

Each audited run has environment.json, workload_0.json through workload_4.json,
events.jsonl and raw.json. Workload files preserve source IDs, role hashes and exact
neighbors. Event logs preserve insertion/rebuild/query durations, returned IDs and recall.
Initial build and warmup do not enter service work. Queueing is not measured.

The historical validation_v3 metadata still says disjoint-interleaved-v2. That string
was not updated when drift was applied and does not identify the transformation.
The exact v3 runner snapshot is scripts/protocol_snapshots/validation_v3.py, SHA-256:
eac92abbd11cb66dc27081095aed1149e6260f4350b3bf7239181b1b2b1c8d2e
This matches environment.json and applies update mean shift 0.5 / query mean shift 0.175.
The independent checker reconstructs those vectors and verifies ground truth.
validation_v2 remains a separate no-drift run. The current v4 runner explicitly saves
drift and query shift factor.

## Environment and hash ledgers

Python 3.10.12; NumPy 1.26.4; SciPy 1.15.3; scikit-learn 1.7.2; Faiss CPU 1.15.0;
Matplotlib 3.10.9; threadpoolctl 3.6.0. See requirements-tested.txt.
Host: Intel Xeon Platinum 8255C, 10 exposed virtual CPUs. Audited libraries use one
thread. Legacy runs lack equivalent thread manifests and are not interchangeable timings.

- results/validation_audit.json hashes all audited source outputs, verifies 135 jobs,
  and supplies exposure summaries and four exploratory follow-up comparisons.
- results/ground_truth_verification.json records an independent SciPy float64 check
  of all 6,000 saved exact neighborhoods and hashes the workload files.
- results/audited_statistics.json hashes its two inputs and contains eight primary
  exploratory comparisons with exact and Holm-adjusted p values.
- paper/generated_manifest.json hashes explicit inputs and generated tables/macros.
- results/semantic_1m_dynamic_v2/verification.json audits six jobs and 30 update events.
- Archives have SHA-256 sidecars and internal SHA256SUMS files.

New timing and elapsed-time hashes need not match. Deterministic neighborhoods and
quantities regenerated from a fixed result snapshot should agree.

## Research limits

The legacy audited runs use one pixel-vector corpus, 6,000 initial rows, 500 inserts
and five seeds. The main semantic and million-vector extensions use word-level GloVe
data; the separate HotpotQA extension uses sentence/document embeddings.
Lifecycle checks are in-memory and lock-protected; they do not emulate WAL,
crash-consistent recovery, distributed transactions, or lock-free throughput. The
arrival orders are matched public streaming traces; production timestamp logs remain
an external validation axis. Exact paired tests establish the measured conditional
quality--maintenance trade-offs for the reported workloads.
