# UVR-Bench for EDBT 2027

This repository is the manuscript-free, single-anonymous artifact mirror for the
EDBT 2027 Experiments & Analysis submission.  The public branch contains the
auditable scripts, manifests, compact inputs, raw summaries, and verification
outputs.  The complete reproducibility archive (including files too large for a
GitHub branch) is published as a release asset at
`https://github.com/xxl-tgkw/edbt2027/releases/latest`.

Latest local archive hashes (2026-10-06 UTC):

- `edbt2027_latex_20261006_090431_UTC.zip`:
  `20114d610979eb33600993816b7d296f24c3986a70bc06d0f0b976912800ccc0`
- `uvrbench_artifact_20261006_090449_UTC.zip`:
  `ca2ca46c8e748816cb5225bbbb78783aef553bbcc2c93bbd13e1fee5bd49bc57`

The branch deliberately omits the 143 MB HotpotQA matrix and the 485 MB source
GloVe HDF5 because GitHub rejects files above 100 MB.  Their source checksums,
preprocessing scripts, and compact verification outputs are included; the full
files are in the release archive.

A CPU-only study of vector-index maintenance. In the tested Fashion-MNIST workloads,
rebuilding IVF incurs substantial maintenance without an established recall advantage.
A query-alignment follow-up separates update-neighbor exposure from retrieval difficulty.

Current manuscript: **edbt.tex**, **paper/body.tex**, **edbt.pdf**. Earlier paper.tex
and paper/edbt2027_a4.tex are historical scaffolds.

## Verify the supplied evidence

Use Python 3.10.12. Install the tested stack, then verify offline with the supplied cache:

~~~bash
python3 -m pip install -r requirements-tested.txt
python3 scripts/audit_validation.py
python3 scripts/verify_ground_truth.py
python3 scripts/stat_tests.py
python3 scripts/build_paper_assets.py --check
python3 -m pytest -q
~~~

The audit checks 135 jobs and 42,750 measured searches. The independent SciPy check
recomputes all 6,000 stored exact neighbor sets. Install pytest separately to run
the regression suite. Verification writes separate reports; it does not retime runs.

## Reproduce new timings

Each command needs a **fresh output directory**; completed and partial results are
protected. Run timing commands sequentially. Faiss/OpenMP/BLAS use one thread.

~~~bash
python3 scripts/run_validation.py --output results/reproduction_ratio --seeds 5
python3 scripts/run_validation.py --output results/reproduction_align100 --seeds 5 --ratios 40 --query-shift-factor 1.0 --methods faiss_flat faiss_ivf_static faiss_ivf_rebuild
python3 scripts/run_validation.py --output results/reproduction_align150 --seeds 5 --ratios 40 --query-shift-factor 1.5 --methods faiss_flat faiss_ivf_static faiss_ivf_rebuild
python3 scripts/summarize_validation.py --input results/reproduction_ratio
~~~

These reproduce the workload with the current metadata format. The exact v3 source is
scripts/protocol_snapshots/validation_v3.py; its hash matches the original environment.
It is a provenance snapshot, not an entrypoint to run from the snapshot directory.
Current v4 computes ground truth only for the largest requested query prefix;
source reservoirs and perturbations remain unchanged.

Recorded runs finished in minutes without a GPU. Original results remain the paper's
fixed evidence snapshot. To adopt rerun results, deliberately change the explicit
source paths in the audit/statistics/asset scripts and regenerate dependent assets.

## Regenerate paper assets and packages

~~~bash
python3 scripts/audit_validation.py
python3 scripts/stat_tests.py
python3 scripts/build_paper_assets.py
python3 scripts/build_audited_figures.py
python3 scripts/make_pareto_figure.py
python3 scripts/make_sensitivity_figure.py
bash scripts/build_paper.sh
python3 scripts/package_latex.py
python3 scripts/package_artifact.py
~~~

The semantic extension uses a separately downloaded 485 MB GloVe-100 file. Prepare
it once with `python3 scripts/prepare_semantic_data.py`; the manifest records the
source checksum and upstream PDDL notice. The pilot (seed 100) is not pooled. The
full 80-run extension and verification are:

~~~bash
python3 scripts/run_semantic.py --output results/semantic_v1 --seeds 0 1 2 3 4 5 6 7 8 9
python3 scripts/analyze_semantic.py --input results/semantic_v1
python3 scripts/verify_semantic_truth.py --input results/semantic_v1
python3 scripts/build_semantic_assets.py
~~~

The separate external scale check replays the same protocol at 100k and 250k final
vectors (three seeds, 128 official held-out queries); it is not pooled with the primary
statistics:

~~~bash
python3 scripts/run_semantic_scale.py --output results/reproduction_scale_100k --sizes 100000 --seeds 0 1 2 --nq 128
python3 scripts/run_semantic_scale.py --output results/reproduction_scale_250k --sizes 250000 --seeds 0 1 2 --nq 128
~~~

The rebuild-frequency sensitivity uses five seeds and compares fixed, period-four,
period-two and every-round schedules:

~~~bash
python3 scripts/run_semantic_rebuild_sensitivity.py --output results/reproduction_rebuild_frequency --seeds 0 1 2 3 4
python3 scripts/verify_external_validation.py \
  --scale100 results/reproduction_scale_100k \
  --scale250 results/reproduction_scale_250k \
  --schedule results/reproduction_rebuild_frequency
~~~

The one-million-vector extension uses 800k base vectors and five 40k update rounds.
It compares fixed/rebuilt IVF and HNSW under the same shifted/shuffled arrival orders,
with 64 validation and 64 held-out test queries:

~~~bash
python3 scripts/run_dynamic_1m.py --output results/reproduction_1m_dynamic --seeds 0 --nq 128
python3 scripts/verify_dynamic_1m.py --input results/reproduction_1m_dynamic
python3 scripts/build_dynamic_1m_table.py
~~~

Native dynamic evidence is recorded in `results/native_dynamic_v1/`: complete
FreshDiskANN/DiskANN and CONDA 1M dynamic-memory logs, plus an official
SPFresh/SPANN build--update--search run using `SPFRESH_SPDK_USE_MEM_IMPL=1`.
The latter two are in-memory native results, not NVMe latency; the host has no
NVMe controller. `ubis_probe.md` records why UBIS is marked not-run (no public
implementation/API). See `native_probe.md`, `conda_native_manifest.json`, and
`SHA256SUMS` for commits, commands, and raw-log provenance.

To regenerate the exact CONDA input streams and independently rescore the saved
responses (after downloading the public GloVe HDF5), run:

~~~bash
python3 scripts/prepare_conda_inputs.py --output /tmp/conda-shifted --order shifted
python3 scripts/score_conda_native.py --inputs /tmp/conda-shifted \
  --results results/native_dynamic_v1/conda_shifted
~~~

The lifecycle extension is a separate five-seed, 20-run check of staged inserts and
deletes, commit/rollback visibility, synchronized concurrent reads, and persistence:

~~~bash
python3 scripts/run_semantic_lifecycle.py --output results/lifecycle_v4 --seeds 0 1 2 3 4
python3 scripts/analyze_semantic_lifecycle.py --input results/lifecycle_v4
python3 scripts/verify_semantic_lifecycle.py --input results/lifecycle_v4
~~~

The additional document-semantic fixture is prepared from the cached public HotpotQA
validation split and the Apache-2.0 all-MiniLM-L6-v2 encoder. It uses 100k title-prefixed
context-sentence vectors, 80k initial vectors, five 4k updates, 256 fixed queries and
three seeds:

~~~bash
python3 scripts/prepare_hotpot_semantic_data.py
python3 scripts/run_hotpot_semantic.py --output results/hotpot_semantic_v1 --seeds 0 1 2
python3 scripts/analyze_hotpot_semantic.py
~~~

HotpotQA is a document-semantic protocol validation, not an agent trajectory, a RAG
answer-quality benchmark, or a production timestamp trace. The primary semantic run
uses GloVe word embeddings. Shifted
and shuffled conditions contain the same base membership, update membership, final
vectors, and held-out query IDs; only update arrival order changes. Budget measurements
share one index per run. These are matched public streaming traces rather than
timestamped production ANN logs.
The scale and rebuild-frequency runs are external validation layers. Their raw outputs
are retained separately and are not pooled with the ten-seed semantic mechanism tests.

paper/generated_manifest.json links tables/macros to raw source hashes. The ratio and
alignment figures use audited records. Older make_ratio_figure.py and
make_validation*_table.py scripts are superseded; do not use them for the current paper.

## Evidence boundaries

| Source | Jobs | Interpretation |
|---|---:|---|
| results/validation_v3 | 105 | Disjoint rows; drift 0.5; factor 0.35; seven policies; five seeds; ratios 10/40/160 |
| results/validation_alignment_100 and _150 | 30 | Same IDs and updates; factors 1 and 1.5; three native Faiss policies; q40 |
| results/raw.json | 180 | Legacy phased runs; indexed query sources; modulo-250 rebuild fires only at 500 |
| results/public_run2/fashion_interleaved.json | 18 | Legacy public interleaving; indexed query sources; rebuild rounds 2/4 |
| results/sensitivity.json | 48 | Seed-level legacy grid; periods 250/500 give the same schedule; no maintenance timing |
| results/scale.json | 18 | 6k/12k/24k timing diagnostics; indexed query copies; no larger held-out quality claim |
| results/semantic_scale_100k_v2 and _250k_v2 | 36 | 100k/250k GloVe scale validation; three seeds; 128 official queries; not pooled |
| results/semantic_rebuild_sensitivity_v1 | 40 | Five-seed rebuild-frequency sensitivity; fixed/period-4/period-2/every-round schedules |
| results/hotpot_semantic_v1 | 24 | Three-seed HotpotQA document-semantic extension; 100k 384-d vectors; not pooled with GloVe mechanism tests |

results/validation_v2 is a no-drift diagnostic. results/ratio.json and
results/stat_tests.json are historical outputs superseded by audited event totals and
exact paired tests; retained for traceability.

See ARTIFACT_MANIFEST.md, paper/REVISION_AUDIT.md and paper/EDBT_TEMPLATE_README.md.
The branch contains no author-identifying metadata or manuscript source.  The
anonymous review mirror is `https://anonymous.4open.science/r/edbt2027-7FD5/`;
4open may lag the GitHub branch until its source connection is refreshed.
