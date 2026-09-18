"""Monte Carlo RWK on row-normalized matrices.

Dataset estimates use two conditionally independent walk replicas. Their mean
reduces off-diagonal noise; their cross product gives an unbiased diagonal.
The resulting matrix is symmetric but need not be positive semidefinite.
Shared lengths (and label sequences) are reused by both replicas and all graphs.
"""

import math

import numpy as np
import scipy.sparse as sp

from ._validation import (
    dataset_inputs, kernel_parameter, positive_int,
)


def kernel_normalizer(kind, mu_func):
    lam = kernel_parameter(kind, mu_func)
    return math.exp(lam) if kind == "exp" else 1.0 / (1.0 - lam)


def sample_length(kind, mu_func, rng, size=None):
    """Sample lengths on {0,1,...}; accepts a vectorized size."""
    lam = kernel_parameter(kind, mu_func)
    return rng.poisson(lam, size=size) if kind == "exp" else rng.geometric(1-lam, size=size)-1


def _rows(P):
    """Cache CSR row CDFs once, including row masses for importance weights."""
    rows = []
    for x in range(P.shape[0]):
        a, b = P.indptr[x:x+2]
        weights = P.data[a:b]
        mass = float(weights.sum())
        cdf = np.cumsum(weights / mass) if mass else np.empty(0)
        if mass:
            cdf[-1] = 1.0
        rows.append((P.indices[a:b], cdf, mass))
    return rows


def _starts(v, size, rng):
    # One CDF construction for all starts, rather than one per walk.
    if np.all(v == v[0]):
        return rng.integers(len(v), size=size)
    return rng.choice(len(v), size=size, p=v)


def _features(rows, v, w, lengths, sequences, log_probs, reps, rng):
    """One replica, with exact conditional values for length zero."""
    starts = _starts(v, (len(lengths), reps), rng)
    features = np.zeros(len(lengths))
    zero = float(v @ w)
    for s, length in enumerate(lengths):
        if length == 0:
            features[s] = zero
            continue
        sequence = sequences[s] if sequences is not None else None
        if sequence is not None and any(lab not in rows for lab in sequence):
            continue
        value = 0.0
        for x in starts[s]:
            log_weight = -0.5 * log_probs[s]
            alive = True
            for t in range(length):
                neighbors, cdf, mass = (rows[sequence[t]] if sequence is not None else rows)[x]
                if mass == 0:
                    alive = False
                    break
                log_weight += math.log(mass)
                x = neighbors[np.searchsorted(cdf, rng.random(), side="right")]
            if alive and w[x] > 0:
                try:
                    value += math.exp(log_weight + math.log(w[x]))
                except OverflowError as exc:
                    raise FloatingPointError("importance weight overflow; change q/lambda") from exc
        features[s] = value / reps
    return features


def build_features(P, v, w, shared_random_variables, n_samples, rng):
    """Single-replica helper; squaring it does NOT give an unbiased diagonal."""
    data = dataset_inputs([P], [v], [w])[0]
    P, v, w = data
    positive_int(n_samples, "n_samples")
    lengths = np.asarray(shared_random_variables, dtype=int)
    if lengths.shape != (n_samples,) or np.any(lengths < 0):
        raise ValueError("shared lengths must match n_samples and be nonnegative")
    return _features(_rows(P), v, w, lengths, None, np.zeros(n_samples), 1, rng)


def _norm(P, kind):
    return float(np.sqrt(np.sum(P.data**2))) if kind == "norm_fro" else float(np.sum(np.abs(P.data)))


def _proposal(scores):
    # A tiny uniform component guarantees support even for zero score labels.
    if not np.isfinite(scores).all():
        raise ValueError("label proposal scores must be finite")
    if scores.sum() == 0:
        return np.ones(len(scores)) / len(scores)
    return (1-1e-12) * scores / scores.sum() + 1e-12 / len(scores)


def q_sampling_dataset(Ps, all_labels, q_sampling_kind="uniform", rng=None):
    """Dataset-wide proposal; randomness comes from the supplied Generator."""
    if not all_labels:
        raise ValueError("no labels")
    rng = np.random.default_rng() if rng is None else rng
    d = len(all_labels)
    if q_sampling_kind == "uniform":
        return np.ones(d) / d
    if q_sampling_kind == "random":
        return _proposal(rng.random(d))
    if q_sampling_kind not in {"norm_fro", "norm_l1"}:
        raise ValueError("unknown q_sampling_kind")
    scores = np.array([sum(_norm(sp.csr_matrix(P[lab]), q_sampling_kind)
                           for P in Ps if lab in P) for lab in all_labels])
    return _proposal(scores)


def q_sampling(P1, P2, common_labels, q_sampling_kind="uniform", rng=None):
    """Compatibility helper for a pair-specific proposal."""
    if q_sampling_kind in {"norm_fro", "norm_l1"}:
        if not common_labels:
            raise ValueError("no labels")
        return _proposal(np.array([_norm(sp.csr_matrix(P1[l]), q_sampling_kind) *
                                   _norm(sp.csr_matrix(P2[l]), q_sampling_kind) for l in common_labels]))
    return q_sampling_dataset([P1, P2], common_labels, q_sampling_kind, rng)


def _estimate(Ps, vs, ws, mu_func, kind, m, n, reps, proposal, seed, labeled):
    m = positive_int(m, "n_length_samples")
    n = positive_int(n, "n_label_samples_per_length")
    reps = positive_int(reps, "n_walk_reps")
    C = kernel_normalizer(kind, mu_func)
    data = dataset_inputs(Ps, vs, ws, labeled)
    if not data:
        return np.empty((0, 0))
    streams = np.random.SeedSequence(seed).spawn(1 + 2 * len(data))
    shared_rng = np.random.default_rng(streams[0])
    base_lengths = sample_length(kind, mu_func, shared_rng, size=m)
    lengths = np.repeat(base_lengths, n)
    sequences = None
    log_probs = np.zeros(len(lengths))
    if labeled:
        labels = sorted(set().union(*(P.keys() for P, _, _ in data)))
        if not labels:
            zero = np.array([v @ w for _, v, w in data])
            return np.outer(zero, zero)
        q = q_sampling_dataset([P for P, _, _ in data], labels, proposal, shared_rng)
        sequences = []
        log_q = np.log(q)
        for s, length in enumerate(lengths):
            ids = shared_rng.choice(len(labels), size=length, p=q)
            sequences.append([labels[j] for j in ids])
            log_probs[s] = log_q[ids].sum()
    # Keep just averaged features and one cross-product scalar per graph.
    means = np.empty((len(data), len(lengths)))
    diagonal = np.empty(len(data))
    for i, (P, v, w) in enumerate(data):
        rows = {lab: _rows(M) for lab, M in P.items()} if labeled else _rows(P)
        a = _features(rows, v, w, lengths, sequences, log_probs, reps,
                      np.random.default_rng(streams[1+2*i]))
        b = _features(rows, v, w, lengths, sequences, log_probs, reps,
                      np.random.default_rng(streams[2+2*i]))
        means[i] = (a+b)*0.5
        diagonal[i] = C * np.mean(a*b)
    result = (C / len(lengths)) * (means @ means.T)
    np.fill_diagonal(result, diagonal)
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite MC estimate; change proposal/lambda")
    return result


def random_walk_kernel_mc_dataset(Ps, vs, ws, mu_func, kind, n_samples=100, seed=42):
    """Unbiased Gram estimate; n_samples shared lengths, two walks per feature."""
    return _estimate(Ps, vs, ws, mu_func, kind, n_samples, 1, 1, "uniform", seed, False)


def random_walk_kernel_mc(P1, P2, v1, v2, w1, w2, mu_func, kind, n_samples=100, seed=42):
    """Pair estimate using the same two-replica construction as the dataset API."""
    return float(random_walk_kernel_mc_dataset([P1,P2], [v1,v2], [w1,w2],
                                               mu_func, kind, n_samples, seed)[0,1])


def random_walk_kernel_mc_labeled_dataset(Ps, vs, ws, mu_func, kind,
        n_length_samples=200, n_label_samples_per_length=50, n_walk_reps=1,
        q_sampling_kind="uniform", seed=42):
    """Unbiased labeled Gram; each replica averages n_walk_reps trajectories.

    There are m independent lengths, NOT m*n independent outer samples.
    Importance weights are accumulated in log space; finite variance is not
    guaranteed for geometric lengths and arbitrary label proposals.
    """
    return _estimate(Ps, vs, ws, mu_func, kind, n_length_samples,
                     n_label_samples_per_length, n_walk_reps, q_sampling_kind, seed, True)


def random_walk_kernel_mc_labeled(P1, P2, v1, v2, w1, w2, mu_func, kind,
        n_length_samples=200, n_label_samples_per_length=50, n_walk_reps=10,
        q_sampling_kind="norm_fro", seed=42):
    kernel_normalizer(kind, mu_func)
    data = dataset_inputs([P1,P2], [v1,v2], [w1,w2], True)
    for name, value in [("n_length_samples", n_length_samples),
                        ("n_label_samples_per_length", n_label_samples_per_length),
                        ("n_walk_reps", n_walk_reps)]:
        positive_int(value, name)
    if not set(P1).intersection(P2):
        return float((data[0][1] @ data[0][2]) * (data[1][1] @ data[1][2]))
    return float(random_walk_kernel_mc_labeled_dataset([P1,P2], [v1,v2], [w1,w2],
        mu_func, kind, n_length_samples, n_label_samples_per_length,
        n_walk_reps, q_sampling_kind, seed)[0,1])
