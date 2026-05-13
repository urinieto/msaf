# MSAF Benchmarks

Approximate results extracted from the original MSAF paper:

> Nieto, O., Bello, J. P., *Systematic Exploration of Computational Music Structure Research*. Proc. of the 17th International Society for Music Information Retrieval Conference (ISMIR). New York City, NY, USA, 2016.

**Paper default parameters:** sr=11025, FFT=2048, hop=512, beat-synchronous PCP features, Beatles TUT dataset.

**Note:** Values are approximate readings from the paper's figures (+-0.02). Algorithms marked with * were in `msaf-gpl`, not in the current MSAF.

---

## Figure 1: Boundary Detection by Feature Type

Hit Rate F-measure (3s window), Beatles TUT dataset.

| Algorithm | CQT | PCP | MFCC | Tonnetz |
|---|---|---|---|---|
| Checkerboard Kernel (Foote) | 0.60 | 0.56 | 0.59 | 0.53 |
| Constrained Clustering* | 0.46 | 0.60 | 0.62 | 0.59 |
| Convex NMF (C-NMF) | 0.58 | 0.53 | 0.53 | 0.51 |
| Shift-Invariant PLCA* | 0.51 | 0.52 | 0.49 | 0.51 |
| Structural Features (SF) | 0.72 | 0.70 | 0.69 | 0.67 |

## Figure 2: Structural Grouping by Feature Type

Pairwise Frame Clustering F-measure, Beatles TUT dataset, ground-truth boundaries.

| Algorithm | CQT | PCP | MFCC | Tonnetz |
|---|---|---|---|---|
| 2D-FMC | 0.72 | 0.75 | 0.70 | 0.74 |
| Constrained Clustering* | 0.54 | 0.69 | 0.67 | 0.70 |
| Convex NMF (C-NMF) | 0.58 | 0.57 | 0.56 | 0.57 |
| Shift-Invariant PLCA* | 0.55 | 0.55 | 0.55 | 0.55 |

## Figure 3: Structural Grouping by Boundary Source

Pairwise Frame Clustering F-measure, Beatles TUT dataset, PCP features.

| Structural Algorithm | Human | SF | Laplacian | CC* | OLDA | Foote | C-NMF | SI-PLCA* |
|---|---|---|---|---|---|---|---|---|
| 2D-FMC | ~0.76 | ~0.48 | ~0.52 | ~0.50 | ~0.44 | ~0.46 | ~0.42 | ~0.38 |
| Convex NMF (C-NMF) | ~0.72 | ~0.42 | ~0.48 | ~0.47 | ~0.42 | ~0.44 | ~0.46 | ~0.40 |
| Laplacian (scluster) | ~0.75 | ~0.55 | ~0.55 | ~0.52 | ~0.48 | ~0.50 | ~0.45 | ~0.40 |
| Constrained Clustering* | ~0.73 | ~0.50 | ~0.50 | ~0.48 | ~0.44 | ~0.46 | ~0.42 | ~0.38 |
| Shift-Invariant PLCA* | ~0.67 | ~0.45 | ~0.46 | ~0.44 | ~0.40 | ~0.42 | ~0.38 | ~0.36 |

## Figure 4a: Boundary Metrics Comparison

Scores across different evaluation metrics, Beatles TUT dataset, PCP features.

| Algorithm | Dev_E2R | Dev_R2E | HR_3 | HR_3w | HR_3t | HR_0.5 | HR_0.5w | HR_0.5t |
|---|---|---|---|---|---|---|---|---|
| Checkerboard Kernel | ~0.55 | ~0.50 | ~0.58 | ~0.55 | ~0.55 | ~0.22 | ~0.20 | ~0.18 |
| Constrained Clustering* | ~0.50 | ~0.45 | ~0.62 | ~0.58 | ~0.58 | ~0.30 | ~0.28 | ~0.26 |
| Convex NMF (C-NMF) | ~0.55 | ~0.48 | ~0.63 | ~0.60 | ~0.60 | ~0.28 | ~0.26 | ~0.24 |
| Laplacian (scluster) | ~0.58 | ~0.52 | ~0.65 | ~0.62 | ~0.62 | ~0.35 | ~0.32 | ~0.30 |
| Ordinal LDA (OLDA) | ~0.52 | ~0.48 | ~0.56 | ~0.53 | ~0.53 | ~0.32 | ~0.30 | ~0.28 |
| Shift-Invariant PLCA* | ~0.45 | ~0.40 | ~0.55 | ~0.52 | ~0.50 | ~0.20 | ~0.18 | ~0.16 |
| Structural Features (SF) | ~0.60 | ~0.55 | ~0.70 | ~0.68 | ~0.65 | ~0.30 | ~0.28 | ~0.25 |

## Figure 4b: Structural Metrics Comparison

Pairwise Frame Clustering (PWF) and Normalized Conditional Entropy (NCE) F-measures, Beatles TUT dataset, PCP features, ground-truth boundaries.

| Algorithm | PWF | NCE |
|---|---|---|
| 2D-FMC | ~0.76 | ~0.80 |
| Constrained Clustering* | ~0.73 | ~0.78 |
| Convex NMF (C-NMF) | ~0.72 | ~0.78 |
| Laplacian (scluster) | ~0.75 | ~0.72 |
| Shift-Invariant PLCA* | ~0.67 | ~0.72 |

## Figure 5: Boundary Detection by Dataset

Hit Rate F-measure (3s window), PCP features.

| Algorithm | Beatles | Cerulean | Epiphyte | Isophonics | SALAMI | Sargon | SPAM |
|---|---|---|---|---|---|---|---|
| SF | ~0.70 | ~0.52 | ~0.55 | ~0.63 | ~0.48 | ~0.50 | ~0.36 |
| Laplacian (scluster) | ~0.65 | ~0.50 | ~0.52 | ~0.60 | ~0.45 | ~0.48 | ~0.42 |
| C-NMF | ~0.63 | ~0.48 | ~0.50 | ~0.58 | ~0.42 | ~0.45 | ~0.38 |
| Foote | ~0.58 | ~0.47 | ~0.48 | ~0.55 | ~0.42 | ~0.43 | ~0.40 |
| OLDA | ~0.56 | ~0.44 | ~0.48 | ~0.55 | ~0.40 | ~0.42 | ~0.36 |
| CC* | ~0.62 | ~0.45 | ~0.48 | ~0.58 | ~0.40 | ~0.43 | ~0.38 |
| SI-PLCA* | ~0.55 | ~0.42 | ~0.45 | ~0.52 | ~0.38 | ~0.42 | ~0.35 |

## Figure 6: Structural Grouping by Dataset

Pairwise Frame Clustering F-measure, PCP features, ground-truth boundaries.

| Algorithm | Beatles | Cerulean | Epiphyte | Isophonics | SALAMI | Sargon | SPAM |
|---|---|---|---|---|---|---|---|
| 2D-FMC | ~0.76 | ~0.62 | ~0.68 | ~0.72 | ~0.58 | ~0.55 | ~0.72 |
| C-NMF | ~0.72 | ~0.60 | ~0.65 | ~0.68 | ~0.55 | ~0.50 | ~0.68 |
| Laplacian (scluster) | ~0.75 | ~0.62 | ~0.66 | ~0.70 | ~0.60 | ~0.52 | ~0.65 |
| CC* | ~0.73 | ~0.55 | ~0.60 | ~0.68 | ~0.55 | ~0.50 | ~0.68 |
| SI-PLCA* | ~0.67 | ~0.55 | ~0.58 | ~0.63 | ~0.52 | ~0.48 | ~0.60 |

## Figure 7: Boundary Detection by Annotator (SPAM dataset)

Hit Rate F-measure (3s window), PCP features.

| Algorithm | Annotator 0 | Annotator 1 | Annotator 2 | Annotator 3 | Annotator 4 |
|---|---|---|---|---|---|
| SF | ~0.42 | ~0.38 | ~0.40 | ~0.44 | ~0.40 |
| Laplacian (scluster) | ~0.44 | ~0.46 | ~0.48 | ~0.50 | ~0.46 |
| C-NMF | ~0.40 | ~0.38 | ~0.40 | ~0.42 | ~0.38 |
| Foote | ~0.42 | ~0.40 | ~0.42 | ~0.44 | ~0.40 |
| OLDA | ~0.36 | ~0.36 | ~0.38 | ~0.40 | ~0.36 |
| CC* | ~0.40 | ~0.38 | ~0.40 | ~0.42 | ~0.38 |
| SI-PLCA* | ~0.36 | ~0.34 | ~0.36 | ~0.38 | ~0.36 |

## Figure 8: Structural Grouping by Annotator (SPAM dataset)

Pairwise Frame Clustering F-measure, PCP features, ground-truth boundaries.

| Algorithm | Annotator 0 | Annotator 1 | Annotator 2 | Annotator 3 | Annotator 4 |
|---|---|---|---|---|---|
| 2D-FMC | ~0.72 | ~0.73 | ~0.74 | ~0.72 | ~0.73 |
| C-NMF | ~0.68 | ~0.70 | ~0.70 | ~0.68 | ~0.69 |
| Laplacian (scluster) | ~0.65 | ~0.67 | ~0.66 | ~0.65 | ~0.66 |
| CC* | ~0.68 | ~0.70 | ~0.69 | ~0.68 | ~0.69 |
| SI-PLCA* | ~0.60 | ~0.62 | ~0.61 | ~0.60 | ~0.61 |

---

## Notes

- Values are approximate readings from figures (+-0.02 tolerance).
- Algorithms marked with * (Constrained Clustering, Shift-Invariant PLCA) were in `msaf-gpl`, not in the current MSAF.
- The paper used sr=11025, FFT=2048, hop=512. MSAF 1.0.0 defaults are sr=22050, FFT=4096, hop=1024.
- MSAF 1.0.0 replaced pymf's Convex NMF with scikit-learn's NMF; C-NMF results may differ slightly.
- Exact numbers may be available in the experiments repo: https://github.com/urinieto/msaf-experiments
