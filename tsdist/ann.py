import time
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    _HAS_CUPY = False

try:
    import faiss as _faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class SimilaritySearchEngine:
    """
    Approximate Nearest Neighbor (ANN) search engine.

    Wraps multiple ANN backends under a unified interface and optionally
    leverages GPU acceleration (FAISS-GPU or CuPy-backed BallTree).

    Supported methods
    -----------------
    'faiss'    — Facebook AI Similarity Search (CPU or GPU via faiss-gpu).
    'hnsw'     — Hierarchical Navigable Small World graphs (hnswlib).
    'annoy'    — Approximate Nearest Neighbors Oh Yeah (spotify/annoy).
    'balltree' — Exact kNN via sklearn BallTree (CPU only).

    Supported metrics
    -----------------
    'euclidean'

    Main use cases
    --------------
    - Nearest-neighbour lookup for arbitrary feature vectors.
    - Generating ranked lists as input to ranking metrics
      (Jaccard, JaccardMax, RBO, Kendall).

    Parameters
    ----------
    method      : str   ANN backend to use.
    metric      : str   Distance metric.
    gpu         : bool  Use GPU when the backend supports it.
    n_trees     : int   Number of trees (Annoy).
    hnsw_m      : int   Bidirectional connections per node (HNSWLib).
    hnsw_ef     : int   Dynamic candidate list size for build/query (HNSWLib).
    """

    SUPPORTED_METHODS  = {'faiss', 'hnsw', 'annoy', 'balltree'}
    SUPPORTED_METRICS  = {'euclidean'}

    def __init__(
        self,
        method:   str  = 'faiss',
        metric:   str  = 'euclidean',
        gpu:      bool = False,
        n_trees:  int  = 50,
        hnsw_m:   int  = 16,
        hnsw_ef:  int  = 50,
    ):
        if method.lower() not in self.SUPPORTED_METHODS:
            raise ValueError(f"Method '{method}' not supported. Choose from: {self.SUPPORTED_METHODS}")
        if metric.lower() not in self.SUPPORTED_METRICS:
            raise ValueError(f"Metric '{metric}' not supported. Choose from: {self.SUPPORTED_METRICS}")

        self.method  = method.lower()
        self.metric  = metric.lower()
        self.gpu     = gpu
        self.n_trees = n_trees
        self.hnsw_m  = hnsw_m
        self.hnsw_ef = hnsw_ef

        self.index     = None
        self.dimension = None
        self.X         = None

        self.time_stats = {"indexing_time": 0.0, "query_time": 0.0}

        # GPU availability check
        if self.gpu and self.method == 'faiss' and not _HAS_FAISS:
            raise ImportError("faiss is not installed. Run: pip install faiss-gpu-cu12")
        if self.gpu and self.method not in {'faiss'} and not _HAS_CUPY:
            print(f"[tsdist] GPU requested for '{self.method}' but cupy not found — using CPU.")
            self.gpu = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_numpy(data, orient: str = 'series_as_rows') -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            X = data.to_numpy()
            return X.T if orient == 'series_as_cols' else X
        elif isinstance(data, list):
            return np.array(data)
        elif isinstance(data, np.ndarray):
            return data
        else:
            raise TypeError(f"Unsupported type: {type(data)}.")

    @staticmethod
    def _prepare(X: np.ndarray) -> np.ndarray:
        """Cast to float32 C-contiguous array (required by FAISS and HNSWLib)."""
        return np.ascontiguousarray(X, dtype=np.float32)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def fit(self, X, orient: str = 'series_as_rows'):
        """
        Build the ANN index from data X.

        Parameters
        ----------
        X      : array-like of shape (N, D) — N samples, D dimensions.
        orient : {'series_as_rows', 'series_as_cols'}
                 Use 'series_as_cols' when each column of a DataFrame is one sample.

        Returns
        -------
        self
        """
        if not isinstance(X, np.ndarray):
            X = self._to_numpy(X, orient)

        X = self._prepare(X)
        self.X         = X
        self.dimension = X.shape[1]
        n              = X.shape[0]

        t0 = time.time()

        if self.method == 'faiss':
            import faiss
            if self.gpu and faiss.get_num_gpus() > 0:
                cpu_index = faiss.IndexFlatL2(self.dimension)
                self.index = faiss.index_cpu_to_all_gpus(cpu_index)
            else:
                if self.gpu:
                    print("[tsdist] No FAISS-compatible GPU found — falling back to CPU index.")
                self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(X)

        elif self.method == 'hnsw':
            import hnswlib
            self.index = hnswlib.Index(space='l2', dim=self.dimension)
            self.index.init_index(max_elements=n, ef_construction=self.hnsw_ef, M=self.hnsw_m)
            self.index.add_items(X)
            self.index.set_ef(self.hnsw_ef)

        elif self.method == 'annoy':
            from annoy import AnnoyIndex
            self.index = AnnoyIndex(self.dimension, self.metric)
            for i, vec in enumerate(X):
                self.index.add_item(i, vec)
            self.index.build(self.n_trees)

        elif self.method == 'balltree':
            self.index = BallTree(X, metric='minkowski')

        self.time_stats["indexing_time"] = time.time() - t0
        return self

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, query_vecs, k: int = 10):
        """
        Find the k nearest neighbours for each vector in query_vecs.

        Parameters
        ----------
        query_vecs : array-like of shape (M, D) or (D,) for a single query.
        k          : Number of neighbours to return.

        Returns
        -------
        distances : np.ndarray of shape (M, k)
        indices   : np.ndarray of shape (M, k)
        """
        if self.index is None:
            raise RuntimeError("Call fit() before query().")

        query_vecs = self._prepare(self._to_numpy(query_vecs))
        if query_vecs.ndim == 1:
            query_vecs = query_vecs.reshape(1, -1)

        t0 = time.time()

        if self.method == 'faiss':
            distances, indices = self.index.search(query_vecs, k)

        elif self.method == 'hnsw':
            indices, distances = self.index.knn_query(query_vecs, k=k)

        elif self.method == 'annoy':
            indices, distances = [], []
            for q in query_vecs:
                idx, dist = self.index.get_nns_by_vector(q, k, include_distances=True)
                indices.append(idx)
                distances.append(dist)
            indices   = np.array(indices)
            distances = np.array(distances)

        elif self.method == 'balltree':
            distances, indices = self.index.query(query_vecs, k=k)

        self.time_stats["query_time"] = time.time() - t0
        return distances, indices

    # ------------------------------------------------------------------
    # Ranked lists (all-vs-all)
    # ------------------------------------------------------------------

    def get_ranked_lists(self, k: int = 10) -> np.ndarray:
        """
        Generate ranked lists for all indexed samples (all-vs-all search).

        Useful as input to ranking-based distance metrics
        (Jaccard, JaccardMax, RBO, Kendall).

        Parameters
        ----------
        k : Number of neighbours per sample.

        Returns
        -------
        np.ndarray of shape (N, k) — neighbour indices sorted by proximity.
        """
        if self.X is None:
            raise RuntimeError("Call fit() before get_ranked_lists().")
        _, indices = self.query(self.X, k=k)
        return indices

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_time_stats(self) -> dict:
        """Return indexing and query times in seconds."""
        return self.time_stats

    def __repr__(self) -> str:
        status = "fitted" if self.index is not None else "not fitted"
        gpu_tag = ", gpu=True" if self.gpu else ""
        return (f"SimilaritySearchEngine(method={self.method!r}, "
                f"metric={self.metric!r}{gpu_tag}, {status})")
