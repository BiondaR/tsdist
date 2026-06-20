<div align="center">

# tsdist

**Distance & similarity metrics for vectors, time series, and ranked lists**  
**Métricas de distância e similaridade para vetores, séries temporais e listas ranqueadas**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange)]()

</div>

---

`tsdist` is a Python library that unifies distance and similarity computation for **arbitrary feature vectors**, **time series**, and **ranked lists** under a single clean API. It also provides an **Approximate Nearest Neighbor (ANN)** search engine that wraps four popular backends (FAISS, HNSWLib, Annoy, BallTree) and can generate ranked lists ready for ranking-based metrics.

> `tsdist` é uma biblioteca Python que unifica o cálculo de distância e similaridade para **vetores de features**, **séries temporais** e **listas ranqueadas** em uma API limpa e consistente. Também oferece busca **Approximate Nearest Neighbor (ANN)** com quatro backends (FAISS, HNSWLib, Annoy, BallTree) e geração de ranked lists prontas para métricas de ranking.

---

## Features / Funcionalidades

| Category | Metrics |
|---|---|
| **Vector-based** | Euclidean, Manhattan, Cosine, Hamming |
| **Time series** | DTW (Dynamic Time Warping) |
| **Ranked lists** | Jaccard, JaccardMax, RBO, Kendall |
| **ANN search** | FAISS · HNSWLib · Annoy · BallTree |
| **GPU support** | FAISS-GPU · CuPy/cuML (optional) |

---

## Installation / Instalação

```bash
# Clone and install in editable mode
git clone https://github.com/BiondaR/tsdist.git
cd tsdist
pip install -e .
```

**With optional dependencies / Com dependências opcionais:**

```bash
# DTW support
pip install -e ".[dtw]"

# ANN backends (Annoy + HNSWLib)
pip install -e ".[ann]"

# Everything except FAISS and GPU
pip install -e ".[all]"
```

**FAISS** must be installed separately depending on your hardware:

```bash
pip install faiss-cpu        # CPU only
pip install faiss-gpu-cu12   # GPU (CUDA 12)
```

**GPU (CuPy + cuML)** for vector metrics:

```bash
pip install cupy-cuda12x
# cuML installation: https://docs.rapids.ai/install
```

---

## Quick Start / Tutorial

### 1. Generate sample data

```python
import pandas as pd
import numpy as np

np.random.seed(42)
dates = pd.date_range(start='2026-01-01', periods=100, freq='D')
data  = np.random.randn(100, 20).cumsum(axis=0)
cols  = [f'Serie_{i+1}' for i in range(20)]

df = pd.DataFrame(data, index=dates, columns=cols)
```

> The data is a DataFrame where **each column is a time series** (100 time steps × 20 series).  
> Os dados são um DataFrame onde **cada coluna é uma série temporal** (100 passos × 20 séries).

---

### 2. Distance Matrix — `DistanceMatrixCalculator`

```python
from tsdist import DistanceMatrixCalculator

d = DistanceMatrixCalculator(df, gpu=False)

# Orient: 'series_as_cols' because each column is one series
d.to_numpy(orient='series_as_cols')   # shape becomes (20, 100)
```

#### 2.1 Vector-based metrics

```python
euclidean = d.compute_matrix(metric='euclidean')
manhattan = d.compute_matrix(metric='manhattan')
cosine    = d.compute_matrix(metric='cosine')
hamming   = d.compute_matrix(metric='hamming')
```

> All return an `np.ndarray` of shape **(N, N)**.  
> Todas retornam um `np.ndarray` de shape **(N, N)**.

#### 2.2 Time-series metric (DTW)

```python
dtw = d.compute_matrix(metric='dtw')
```

#### 2.3 Sparse matrix

```python
# Keep only distances ≤ 70; zero out the rest
sparse = d.compute_matrix(metric='euclidean', sparse=True, threshold=70)
print(sparse)
```

#### 2.4 Ranked-list metrics

Ranked-list metrics (Jaccard, RBO, Kendall…) operate on **ranked indices**, not raw vectors. The typical workflow is:

1. Compute a base distance matrix.
2. Sort it to get ranked lists (each row = sorted neighbours for one sample).
3. Compute ranking-based distances on those lists.

```python
import numpy as np

# Step 1-2: build ranked lists from Euclidean distances
ranked = np.argsort(d.compute_matrix(metric='euclidean'))

# Step 3: new calculator operating on ranked lists
from tsdist import DistanceMatrixCalculator
rk = DistanceMatrixCalculator(ranked)

kendall    = rk.compute_matrix(metric='kendall')
rbo        = rk.compute_matrix(metric='rbo',        p=0.9)
jaccard    = rk.compute_matrix(metric='jaccard',    k=20)
jaccardmax = rk.compute_matrix(metric='jaccard_max', k=20)
```

---

### 3. ANN Search — `SimilaritySearchEngine`

```python
from tsdist import SimilaritySearchEngine
```

All backends follow the same `fit → query → get_ranked_lists` interface.

#### 3.1 FAISS

```python
engine = SimilaritySearchEngine(method='faiss', metric='euclidean')
engine.fit(df, orient='series_as_cols')

# Query two specific series
dists, idxs = engine.query([df['Serie_1'], df['Serie_3']], k=10)

# All-vs-all ranked lists (useful for downstream ranking metrics)
ranked_lists = engine.get_ranked_lists(k=20)

print(engine.get_time_stats())
# {'indexing_time': ..., 'query_time': ...}
```

#### 3.2 HNSWLib

```python
engine = SimilaritySearchEngine(method='hnsw', metric='euclidean', hnsw_m=16, hnsw_ef=50)
engine.fit(df, orient='series_as_cols')

dists, idxs  = engine.query([df['Serie_11'], df['Serie_19']], k=10)
ranked_lists = engine.get_ranked_lists(k=10)
```

#### 3.3 Annoy

```python
engine = SimilaritySearchEngine(method='annoy', metric='euclidean', n_trees=50)
engine.fit(df, orient='series_as_cols')

dists, idxs  = engine.query([df['Serie_10']], k=10)
ranked_lists = engine.get_ranked_lists(k=5)
```

#### 3.4 BallTree (exact kNN)

```python
engine = SimilaritySearchEngine(method='balltree', metric='euclidean')
engine.fit(df, orient='series_as_cols')

queries      = [df['Serie_9'], df['Serie_10'], df['Serie_11'], df['Serie_12']]
dists, idxs  = engine.query(queries, k=10)
ranked_lists = engine.get_ranked_lists(k=3)
```

#### 3.5 GPU (FAISS)

```python
# Requires: pip install faiss-gpu-cu12
engine = SimilaritySearchEngine(method='faiss', metric='euclidean', gpu=True)
engine.fit(df, orient='series_as_cols')
ranked_lists = engine.get_ranked_lists(k=20)
```

> If no GPU is found, tsdist falls back to CPU automatically with a warning.  
> Se nenhuma GPU for encontrada, tsdist volta para CPU automaticamente com um aviso.

---

## API Reference

### `DistanceMatrixCalculator`

```
DistanceMatrixCalculator(data=None, gpu=False)
```

| Method | Description |
|---|---|
| `to_numpy(orient)` | Convert data to NumPy array in-place |
| `compute_matrix(metric, sparse, threshold, k, p)` | Compute pairwise distance matrix |

**`compute_matrix` parameters:**

| Parameter | Default | Description |
|---|---|---|
| `metric` | `'euclidean'` | Distance metric |
| `sparse` | `False` | Return a `scipy.sparse.csr_matrix` |
| `threshold` | `None` | Zero out distances above this value (sparse only) |
| `k` | `20` | Top-k cutoff for ranked-list metrics |
| `p` | `0.9` | Persistence for RBO |

---

### `SimilaritySearchEngine`

```
SimilaritySearchEngine(method='faiss', metric='euclidean', gpu=False,
                       n_trees=50, hnsw_m=16, hnsw_ef=50)
```

| Method | Description |
|---|---|
| `fit(X, orient)` | Build the ANN index |
| `query(query_vecs, k)` | Find k nearest neighbours |
| `get_ranked_lists(k)` | All-vs-all ranked lists of shape (N, k) |
| `get_time_stats()` | Indexing and query times in seconds |

---

## Supported Metrics / Métricas suportadas

| Metric | Class | Notes |
|---|---|---|
| `euclidean` | `DistanceMatrixCalculator`, `SimilaritySearchEngine` | L2 distance |
| `manhattan` | `DistanceMatrixCalculator` | L1 distance |
| `cosine` | `DistanceMatrixCalculator` | Cosine distance |
| `hamming` | `DistanceMatrixCalculator` | Hamming distance |
| `dtw` | `DistanceMatrixCalculator` | Requires `dtaidistance` |
| `jaccard` | `DistanceMatrixCalculator` | Ranked lists only |
| `jaccard_max` | `DistanceMatrixCalculator` | Ranked lists only; best Jaccard over top-k prefixes |
| `rbo` | `DistanceMatrixCalculator` | Rank-Biased Overlap (Webber et al., 2010) |
| `kendall` | `DistanceMatrixCalculator` | Kendall's τ, normalised to [0, 1] |

---

## GPU Support / Suporte a GPU

| Backend | GPU support | How |
|---|---|---|
| FAISS | ✅ Native | `faiss-gpu-cu12` + `gpu=True` |
| HNSWLib | ❌ | CPU only |
| Annoy | ❌ | CPU only |
| BallTree | ❌ | CPU only |
| Vector metrics (euclidean, cosine…) | ✅ Via cuML | `cupy-cuda12x` + cuML + `gpu=True` |
| DTW | ❌ | CPU only (dtaidistance) |
| Ranked-list metrics | ❌ | CPU only |

---

## Project Structure

```
tsdist/
├── tsdist/
│   ├── __init__.py
│   ├── ann.py        # SimilaritySearchEngine
│   └── dists.py      # DistanceMatrixCalculator
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## License

MIT © see [LICENSE](LICENSE)

---

## Citation / Citação

If you use `tsdist` in your research, please consider citing it:

```bibtex
@software{tsdist2026,
  author  = {Bionda Rozin},
  title   = {tsdist: Distance \& Similarity Metrics for Vectors, Time Series, and Ranked Lists},
  year    = {2026},
  url     = {https://github.com/BiondaR/tsdist}
}
```
