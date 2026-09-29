# UVR-Bench artifact manifest

The authoritative revised evidence is the 105-job validation_v3 ratio matrix plus
two 15-job query-alignment follow-ups. Legacy evidence comprises 180 reference,
18 public, 48 sensitivity and 18 scale observations. README.md gives verification
and fresh-directory reproduction commands.

The semantic extension is a separate 80-index-run matrix in results/semantic_v1. It
uses public ANN-Benchmarks GloVe-100 word vectors, ten seeds, matched shifted and
shuffled arrival orders, Faiss Flat/IVF/HNSW, and three search budgets per approximate
policy. It is retained as an independent evidence layer but is not pooled with the
Fashion-MNIST runs. The normalized pool is supplied in data/semantic_glove/pool.npz;
the original 485 MB HDF5 is not redistributed. Source and preprocessing details are
in data/semantic_glove/SOURCE.md and manifest.json.

The lifecycle extension is stored in results/lifecycle_v4 (20 runs, five seeds). It
checks staged insert/delete visibility, commit and rollback state transitions,
lock-protected concurrent query/update publication, and serialize/reload checksums.

External validation layers are stored separately from the primary evidence: the
100k and 250k GloVe scale checks are in results/semantic_scale_100k_v2 and
results/semantic_scale_250k_v2 (three seeds each), and the 40-run rebuild-frequency
sensitivity is in results/semantic_rebuild_sensitivity_v1 (five seeds). These runs
use the same public source and exact checkpoint ground truth but are not pooled into
the ten-seed mechanism statistics.

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
- The JSON ledgers hash explicit inputs and generated results; manuscript source is not
  part of this repository.
- Archives have SHA-256 sidecars and internal SHA256SUMS files.

New timing and elapsed-time hashes need not match. Deterministic neighborhoods and
quantities regenerated from a fixed result snapshot should agree.

## Research limits

The legacy audited runs use one pixel-vector corpus, 6,000 initial rows, 500 inserts
and five seeds. GloVe is word-level rather than sentence/document embedding data.
Lifecycle checks are in-memory and lock-protected; they do not emulate WAL,
crash-consistent recovery, distributed transactions, or lock-free throughput. The
arrival orders are matched public streaming traces, not production ANN logs. Exact
paired tests establish neither a universal recall benefit from rebuilding nor practical
equivalence. These results do not justify an acceptance probability or a claim that
dynamic ANN benchmarking is new.
