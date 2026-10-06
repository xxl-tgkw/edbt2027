# UBIS coverage probe

UBIS refers to *Updatable Balanced Index for Stable Streaming Similarity Search
over Large-Scale Fresh Vectors* (Lai, Huang, and Wang, arXiv:2602.00563,
January 2026; DOI 10.1109/BigData66926.2025.11401701).  We checked the paper,
its arXiv source bundle, the authors' public GitHub search namespace, and the
paper's artifact/code statements on 2026-10-03.  The arXiv source contains the
manuscript and figures but no implementation, benchmark driver, or executable.
No public repository or downloadable binary exposing the UBIS update/search API
was found.  The later ICDE record (DOI 10.1109/icde65706.2026.00054) likewise
does not provide a public implementation endpoint.

Consequently UBIS is marked **not_run: unavailable_public_implementation** in
the coverage matrix.  We do not substitute SPFresh, CONDA, or a reimplementation
under the UBIS name, and no UBIS number is used in the paper.  This is a coverage
boundary, not a claim that UBIS is inferior.  A future artifact release from the
authors can be replayed with the same UVR-Bench workload and query split.

Evidence checked:

* arXiv abstract/source: https://arxiv.org/abs/2602.00563 and
  https://arxiv.org/e-print/2602.00563;
* DOI records: https://doi.org/10.1109/BigData66926.2025.11401701 and
  https://doi.org/10.1109/icde65706.2026.00054;
* public source search: https://api.github.com/search/repositories?q=%22Updatable+Balanced+Index%22.
