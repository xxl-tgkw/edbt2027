# Public semantic vectors

- Distribution used: https://ann-benchmarks.com/glove-100-angular.hdf5
- Source file SHA-256: 544af1d5e84e112cd4749571dcfd8ca109818a572f850af75a3a09e093a953c4
- Shape: train 1,183,514 × 100; test 10,000 × 100. Distance attribute: angular.
- Pretrained GloVe **word** embeddings; no encoder trained or downloaded for this study.
- Original work: Pennington, Socher, Manning. GloVe: Global Vectors for Word
  Representation. EMNLP 2014, 1532–1543, doi:10.3115/v1/D14-1162.
- ANN-Benchmarks reference: Aumüller, Bernhardsson, Faithfull, Information Systems 87,
  101374 (2020). Original upstream: https://nlp.stanford.edu/projects/glove/

The Stanford upstream page states that pretrained word vectors are distributed under
Open Data Commons Public Domain Dedication and License (PDDL) 1.0:
https://opendatacommons.org/licenses/pddl/1-0/
This differs from the Apache-2.0 license of the GloVe software. No redistribution
license is inferred merely from a publicly accessible URL.

pool.npz is a normalized subset and contains original training-row IDs, all official
test vectors, the separate 10k design-row IDs, and the principal-direction partition.
Its hash and preparation settings are in manifest.json. Source vectors are not
perturbed; the two workloads alter arrival order, not coordinates. This data is not
a real chronological update log, a modern sentence-embedding corpus or a RAG benchmark.

The independently sampled semantic pool has 120k candidate rows, while each experiment
uses 20k initial / 40k final vectors. Do not describe the experiments as million-scale
simply because the downloadable source has over a million rows.
