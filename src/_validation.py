"""Shared validation for normalized random-walk kernels."""

import numpy as np
import scipy.sparse as sp


def positive_int(value, name):
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def kernel_parameter(kind, mu_func):
    """Only exp/geom families with mu(0)=1 are supported by samplers."""
    if kind not in {"exp", "geom"}:
        raise ValueError("kind must be 'exp' or 'geom'")
    lam = float(mu_func(1))
    if not np.isfinite(lam) or lam < 0 or (kind == "geom" and lam >= 1):
        raise ValueError("lambda must be finite and >= 0 (and < 1 for geom)")
    expected_second = lam**2 / (2 if kind == "exp" else 1)
    if not np.isclose(mu_func(0), 1) or not np.isclose(mu_func(2), expected_second):
        raise ValueError("mu_func must be the specified exp/geom family with mu(0)=1")
    return lam


def matrix(P, n=None):
    P = sp.csr_matrix(P, dtype=float, copy=True)
    P.sum_duplicates()
    P.eliminate_zeros()
    P.sort_indices()
    if P.shape[0] == 0 or P.shape[0] != P.shape[1]:
        raise ValueError("transition matrix must be nonempty and square")
    if n is not None and P.shape != (n, n):
        raise ValueError("matrix and boundary-vector sizes do not match")
    if not np.all(np.isfinite(P.data)) or np.any(P.data < 0):
        raise ValueError("transition weights must be finite and nonnegative")
    return P


def boundaries(v, w, n=None):
    v, w = np.asarray(v, dtype=float), np.asarray(w, dtype=float)
    if v.ndim != 1 or w.shape != v.shape or len(v) == 0:
        raise ValueError("v and w must be nonempty vectors of the same length")
    if n is not None and len(v) != n:
        raise ValueError("matrix and boundary-vector sizes do not match")
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(w)):
        raise ValueError("boundary vectors must be finite")
    if np.any(v < 0) or np.any(w < 0) or not np.isclose(v.sum(), 1):
        raise ValueError("v must be a probability vector; w must be nonnegative")
    return v / v.sum(), w


def normalized_input(P, v, w, labeled=False):
    """Accept substochastic rows; missing row mass represents a killed walk."""
    v, w = boundaries(v, w)
    n = len(v)
    if labeled:
        P = {lab: matrix(M, n) for lab, M in P.items()}
        total = sum(P.values(), sp.csr_matrix((n, n)))
    else:
        P = matrix(P, n)
        total = P
    if np.any(np.asarray(total.sum(axis=1)).ravel() > 1 + 1e-10):
        raise ValueError("input must be row-normalized (total row masses <= 1)")
    return P, v, w


def dataset_inputs(Ps, vs, ws, labeled=False):
    if not (len(Ps) == len(vs) == len(ws)):
        raise ValueError("Ps, vs and ws must have the same length")
    return [normalized_input(P, v, w, labeled) for P, v, w in zip(Ps, vs, ws)]


def corrected_gram(first, second, scale=1.0):
    """Two conditionally independent replicas; unbiased, symmetric, not PSD.

    Average replicas before off-diagonal products. On the diagonal use the
    cross product, avoiding the conditional variance of a squared feature.
    Arrays have shape (graphs, samples); all shared randomness must match.
    """
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape or first.ndim != 2 or first.shape[1] == 0:
        raise ValueError("replica arrays must have matching (graphs, samples) shapes")
    mean = (first + second) * 0.5
    out = scale / first.shape[1] * (mean @ mean.T)
    np.fill_diagonal(out, scale * np.mean(first * second, axis=1))
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("non-finite kernel estimate; reduce weight variance")
    return out
