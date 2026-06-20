import numpy as np
import scipy.sparse as sp
from scipy.stats import kendalltau
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd

try:
    from dtaidistance import dtw as _dtw
    _HAS_DTAI = True
except ImportError:
    _HAS_DTAI = False

try:
    import cupy as cp
    from cuml.metrics import pairwise_distances as _gpu_pairwise
    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False


class DistanceMatrixCalculator:
    """
    Computes pairwise distance matrices for vectors, time series, and ranked lists.

    Supported metrics
    -----------------
    Vector-based  : 'euclidean', 'manhattan', 'cosine', 'hamming'
    Time series   : 'dtw'  (requires dtaidistance)
    Ranked lists  : 'jaccard', 'jaccard_max', 'rbo', 'kendall'

    Parameters
    ----------
    data : array-like, pd.DataFrame, or list
        Input data. For vector/time-series metrics, shape (N, D).
        For ranked-list metrics, a 2-D array/list of ranked indices.
    gpu : bool, optional
        Use GPU acceleration where available (requires cupy + cuML).
        Falls back to CPU automatically when GPU is unavailable.
    """

    VECTOR_METRICS     = {'euclidean', 'manhattan', 'cosine', 'hamming'}
    TS_METRICS         = {'dtw'}
    RANKLIST_METRICS   = {'jaccard', 'jaccard_max', 'rbo', 'kendall'}
    SUPPORTED_METRICS  = VECTOR_METRICS | TS_METRICS | RANKLIST_METRICS

    def __init__(self, data=None, gpu: bool = False):
        self.data = data
        self.gpu  = gpu and _HAS_GPU

        if gpu and not _HAS_GPU:
            print("[tsdist] GPU requested but cupy/cuML not found — falling back to CPU.")

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def to_numpy(self, orient: str = 'series_as_rows'):
        """
        Convert internal data to a NumPy array in-place.

        Parameters
        ----------
        orient : {'series_as_rows', 'series_as_cols'}
            'series_as_rows' (default) — each row is one sample, shape (N, T).
            'series_as_cols'           — each column is one sample; array is transposed.
        """
        if isinstance(self.data, pd.DataFrame):
            X = self.data.to_numpy()
            self.data = X.T if orient == 'series_as_cols' else X
        elif isinstance(self.data, np.ndarray):
            pass  # already fine
        elif isinstance(self.data, list):
            self.data = np.array(self.data)
        else:
            raise TypeError(f"Unsupported type: {type(self.data)}.")

    def _to_sparse(self, matrix: np.ndarray, threshold=None):
        if threshold is not None:
            matrix[matrix > threshold] = 0
        return sp.csr_matrix(matrix)

    # ------------------------------------------------------------------
    # Ranked-list similarity helpers
    # ------------------------------------------------------------------

    def _rbo(self, list_a, list_b, p: float = 0.9) -> float:
        """Rank-Biased Overlap (Webber et al., 2010). Returns similarity in [0, 1]."""
        if len(list_a) == 0 or len(list_b) == 0:
            return 0.0
        score = 0.0
        set_a, set_b = set(), set()
        for d in range(1, min(len(list_a), len(list_b)) + 1):
            set_a.add(list_a[d - 1])
            set_b.add(list_b[d - 1])
            score += (len(set_a & set_b) / d) * (p ** (d - 1))
        return (1 - p) * score

    @staticmethod
    def _jaccard_max(list_a, list_b, k: int) -> float:
        set_a, set_b = set(), set()
        best = 0.0
        for d in range(1, k):
            set_a.add(list_a[d - 1])
            set_b.add(list_b[d - 1])
            intersect = len(set_a & set_b)
            union     = 2 * d - intersect
            jac       = intersect / union
            if jac == 1.0:
                return 0.0  # distance = 1 - similarity
            if jac > best:
                best = jac
        return 1.0 - best

    @staticmethod
    def _jaccard(list_a, list_b, k: int) -> float:
        set_a = set(list_a[:k])
        set_b = set(list_b[:k])
        union = len(set_a | set_b)
        return 1.0 if union == 0 else 1.0 - len(set_a & set_b) / union

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def compute_matrix(
        self,
        metric:    str   = 'euclidean',
        sparse:    bool  = False,
        threshold         = None,
        k:         int   = 20,
        p:         float = 0.9,
    ) -> np.ndarray:
        """
        Compute the full pairwise distance matrix.

        Parameters
        ----------
        metric    : str   Distance metric (see class docstring for options).
        sparse    : bool  Return a scipy CSR sparse matrix.
        threshold : float If sparse=True, zero out distances above this value.
        k         : int   Top-k cutoff for ranked-list metrics.
        p         : float Persistence parameter for RBO (0 < p < 1).

        Returns
        -------
        np.ndarray or scipy.sparse.csr_matrix of shape (N, N).
        """
        if metric not in self.SUPPORTED_METRICS:
            raise ValueError(f"Metric '{metric}' not supported. Choose from: {self.SUPPORTED_METRICS}")

        if not isinstance(self.data, np.ndarray):
            raise TypeError("Data must be a NumPy array. Call to_numpy() first.")

        n = len(self.data)

        # ── Vector metrics ────────────────────────────────────────────
        if metric in self.VECTOR_METRICS:
            if self.gpu:
                X_gpu  = cp.array(self.data, dtype=cp.float32)
                dist_mat = cp.asnumpy(_gpu_pairwise(X_gpu, metric=metric))
            else:
                dist_mat = pairwise_distances(self.data, metric=metric)

        # ── Time-series metric (DTW) ──────────────────────────────────
        elif metric == 'dtw':
            if not _HAS_DTAI:
                raise ImportError("dtaidistance is required for DTW. Install it with: pip install dtaidistance")
            X = self.data.astype(np.double)
            dist_mat = _dtw.distance_matrix_fast(X)

        # ── Ranked-list metrics ───────────────────────────────────────
        elif metric in self.RANKLIST_METRICS:
            dist_mat = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    if metric == 'jaccard':
                        d = self._jaccard(self.data[i], self.data[j], k=k)
                    elif metric == 'jaccard_max':
                        d = self._jaccard_max(self.data[i], self.data[j], k=k)
                    elif metric == 'rbo':
                        d = 1.0 - self._rbo(self.data[i], self.data[j], p=p)
                    elif metric == 'kendall':
                        tau, _ = kendalltau(self.data[i], self.data[j])
                        d = 1.0 - (tau + 1) / 2  # normalised to [0, 1]
                    dist_mat[i, j] = dist_mat[j, i] = d

        if sparse:
            return self._to_sparse(dist_mat, threshold)
        return dist_mat
