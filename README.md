# UVR-Bench Reproducibility Artifact

A CPU-only study of vector-index maintenance. In the tested Fashion-MNIST workloads,
rebuilding IVF incurs substantial maintenance without an established recall advantage.
A query-alignment follow-up separates update-neighbor exposure from retrieval difficulty.
This evaluates existing policies; it does not propose a new ANN algorithm.

This repository contains the reproducibility artifact only. The manuscript, LaTeX
source, bibliography, template files, and manuscript PDF are intentionally excluded.
The artifact contains no author-identifying metadata.

## Verify the supplied evidence

Use Python 3.10.12. Install the tested stack, then verify offline with the supplied cache:

~~~bash
python3 -m pip install -r requirements-tested.txt
python3 scripts/audit_validation.py
python3 scripts/verify_ground_truth.py
python3 scripts/stat_tests.py
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

Recorded runs finished in minutes without a GPU. The committed JSON and event logs are
the fixed evidence snapshot; reruns write to fresh output directories and do not alter
the committed evidence.

## Regenerate figures and verification reports

~~~bash
python3 scripts/audit_validation.py
python3 scripts/stat_tests.py
python3 scripts/build_audited_figures.py
python3 scripts/make_pareto_figure.py
python3 scripts/make_sensitivity_figure.py
~~~

The semantic extension uses a separately downloaded 485 MB GloVe-100 file. Prepare
it once with `python3 scripts/prepare_semantic_data.py`; the manifest records the
source checksum and upstream PDDL notice. The pilot (seed 100) is not pooled. The
full 80-run extension and verification are:

~~~bash
python3 scripts/run_semantic.py --output results/semantic_v1 --seeds 0 1 2 3 4 5 6 7 8 9
python3 scripts/analyze_semantic.py --input results/semantic_v1
python3 scripts/verify_semantic_truth.py --input results/semantic_v1
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

The lifecycle extension is a separate five-seed, 20-run check of staged inserts and
deletes, commit/rollback visibility, synchronized concurrent reads, and persistence:

~~~bash
python3 scripts/run_semantic_lifecycle.py --output results/lifecycle_v4 --seeds 0 1 2 3 4
python3 scripts/analyze_semantic_lifecycle.py --input results/lifecycle_v4
python3 scripts/verify_semantic_lifecycle.py --input results/lifecycle_v4
~~~

The semantic run uses GloVe word embeddings, not sentence embeddings or RAG. Shifted
and shuffled conditions contain the same final vectors and held-out query IDs; only
arrival order changes. Budget measurements share one index per run.
These are matched public streaming traces, not proprietary production ANN logs.
The scale and rebuild-frequency runs are external validation layers. Their raw outputs
are retained separately and are not pooled with the ten-seed semantic mechanism tests.

The ratio and alignment figures use audited records. The generated PDFs in `results/`
are convenience views; numerical verification is performed from the JSON ledgers.

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

results/validation_v2 is a no-drift diagnostic. results/ratio.json and
results/stat_tests.json are historical outputs superseded by audited event totals and
exact paired tests; retained for traceability.

See `ARTIFACT_MANIFEST.md` for provenance, checksums, scope, and known limitations.
No manuscript or submission package is distributed by this repository.
