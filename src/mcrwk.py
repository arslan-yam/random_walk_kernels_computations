"""Monte Carlo RWK on row-normalized matrices.

Dataset estimates use two conditionally independent walk replicas. Their mean
reduces off-diagonal noise; their cross product gives an unbiased diagonal.
The resulting matrix is symmetric but need not be positive semidefinite.
Shared lengths (and label sequences) are reused by both replicas and all graphs.

diagonal="biased" keeps the diagonal of C/M * Fbar Fbar^T instead (Fbar: the
averaged replicas). That matrix is PSD; its diagonal exceeds the unbiased one
by C/M * sum_s ((F1_s - F2_s)/2)^2 >= 0 in every run, so its expectation is
K + diag(tau) with tau_i = C/2 * E[Var(F_i | lengths, labels)], a
graph-dependent ridge that does not shrink with M. Off-diagonal entries are the same in both versions; "both" returns
(unbiased, biased) from one run.
"""

from collections.abc import Mapping
import math
from typing import NamedTuple

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


def _features(rows, v, w, lengths, sequences, log_probs, reps, rng, stats=None):
    """One replica, with exact conditional values for length zero.

    stats, if given, accumulates per-walk counts (not per step): walks of
    positive length, steps taken, killed walks (no mass of the sampled label),
    skipped walks (label absent from the graph) and the largest log weight.
    """
    starts = _starts(v, (len(lengths), reps), rng)
    features = np.zeros(len(lengths))
    zero = float(v @ w)
    walks = steps = killed = skipped = 0
    max_log_weight = -math.inf
    for s, length in enumerate(lengths):
        if length == 0:
            features[s] = zero
            continue
        walks += reps
        sequence = sequences[s] if sequences is not None else None
        if sequence is not None and any(lab not in rows for lab in sequence):
            skipped += reps
            continue
        value = 0.0
        for x in starts[s]:
            log_weight = -0.5 * log_probs[s]
            for t in range(length):
                neighbors, cdf, mass = (rows[sequence[t]] if sequence is not None else rows)[x]
                if mass == 0:
                    steps += t
                    killed += 1
                    break
                log_weight += math.log(mass)
                x = neighbors[np.searchsorted(cdf, rng.random(), side="right")]
            else:
                steps += length
                if w[x] > 0:
                    log_weight += math.log(w[x])
                    max_log_weight = max(max_log_weight, log_weight)
                    try:
                        value += math.exp(log_weight)
                    except OverflowError as exc:
                        raise FloatingPointError("importance weight overflow; change q/lambda") from exc
        features[s] = value / reps
    if stats is not None:
        for key, count in (("walks", walks), ("steps", steps), ("killed", killed), ("skipped", skipped)):
            stats[key] = stats.get(key, 0) + count
        stats["max_log_weight"] = max(stats.get("max_log_weight", -math.inf), max_log_weight)
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
    """Dataset-wide proposal; randomness comes from the supplied Generator.

    q_sampling_kind is a named rule or a mapping {label: probability}; an
    explicit mapping must give every dataset label positive mass (it is
    renormalized), otherwise the estimator would be biased.
    """
    if not all_labels:
        raise ValueError("no labels")
    if isinstance(q_sampling_kind, Mapping):
        q = np.array([q_sampling_kind.get(lab, 0.) for lab in all_labels], dtype=float)
        if not np.isfinite(q).all() or np.any(q <= 0):
            raise ValueError("explicit proposal needs positive probability for every dataset label")
        return q / q.sum()
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


DIAGONALS = ("unbiased", "biased", "both")


def _select(unbiased, biased, diagonal):
    return {"unbiased": unbiased, "biased": biased, "both": (unbiased, biased)}[diagonal]


class Features(NamedTuple):
    """Averaged replica features F (graphs x columns) of one MCRWK run.

    The biased (PSD) Gram is scale * F F^T; ``diagonal`` holds the unbiased
    diagonal (replica cross products). sqrt(scale) * F are explicit features
    for linear models, which never need the graphs x graphs matrix.
    """
    features: np.ndarray
    scale: float
    diagonal: np.ndarray


def _sample(Ps, vs, ws, mu_func, kind, m, n, reps, proposal, seed, labeled, diagnostics=None):
    m = positive_int(m, "n_length_samples")
    n = positive_int(n, "n_label_samples_per_length")
    reps = positive_int(reps, "n_walk_reps")
    C = kernel_normalizer(kind, mu_func)
    data = dataset_inputs(Ps, vs, ws, labeled)
    if not data:
        return Features(np.empty((0, m*n)), C/(m*n), np.empty(0))
    streams = np.random.SeedSequence(seed).spawn(1 + 2 * len(data))
    shared_rng = np.random.default_rng(streams[0])
    base_lengths = sample_length(kind, mu_func, shared_rng, size=m)
    lengths = np.repeat(base_lengths, n)
    sequences = None
    log_probs = np.zeros(len(lengths))
    if labeled:
        labels = sorted(set().union(*(P.keys() for P, _, _ in data)))
        if not labels:
            # Only length zero contributes: the exact kernel is an outer product.
            zero = np.array([v @ w for _, v, w in data])
            return Features(zero[:, None], 1.0, zero*zero)
        q = q_sampling_dataset([P for P, _, _ in data], labels, proposal, shared_rng)
        sequences = []
        log_q = np.log(q)
        for s, length in enumerate(lengths):
            ids = shared_rng.choice(len(labels), size=length, p=q)
            sequences.append([labels[j] for j in ids])
            log_probs[s] = log_q[ids].sum()
    # Keep just averaged features and one cross-product scalar per graph.
    means = np.empty((len(data), len(lengths)))
    cross = np.empty(len(data))
    stats = {} if diagnostics is not None else None
    for i, (P, v, w) in enumerate(data):
        rows = {lab: _rows(M) for lab, M in P.items()} if labeled else _rows(P)
        a = _features(rows, v, w, lengths, sequences, log_probs, reps,
                      np.random.default_rng(streams[1+2*i]), stats)
        b = _features(rows, v, w, lengths, sequences, log_probs, reps,
                      np.random.default_rng(streams[2+2*i]), stats)
        means[i] = (a+b)*0.5
        cross[i] = C * np.mean(a*b)
    if diagnostics is not None:
        diagnostics.update(_diagnostics(means, stats, dict(zip(labels, q)) if labeled else None))
    return Features(means, C / len(lengths), cross)


def _estimate(Ps, vs, ws, mu_func, kind, m, n, reps, proposal, seed, labeled,
              diagnostics=None, diagonal="unbiased"):
    if diagonal not in DIAGONALS:
        raise ValueError(f"diagonal must be one of {DIAGONALS}")
    sample = _sample(Ps, vs, ws, mu_func, kind, m, n, reps, proposal, seed, labeled, diagnostics)
    biased = sample.scale * (sample.features @ sample.features.T)
    unbiased = None
    if diagonal != "biased":
        # Only "both" needs a copy; otherwise the biased matrix is not returned.
        unbiased = biased.copy() if diagonal == "both" else biased
        np.fill_diagonal(unbiased, sample.diagonal)
    if not np.isfinite(biased).all() or not np.isfinite(sample.diagonal).all():
        raise FloatingPointError("non-finite MC estimate; change proposal/lambda")
    return _select(unbiased, biased, diagonal)


def random_walk_kernel_mc_features(Ps, vs, ws, mu_func, kind, n_length_samples=200,
        n_label_samples_per_length=1, n_walk_reps=1, q_sampling_kind="uniform", seed=42,
        labeled=False, diagnostics=None):
    """Features of the same run as the dataset Gram functions (same seed, same walks).

    Returns Features(F, scale, unbiased_diagonal): scale * F F^T is the biased
    Gram, and sqrt(scale) * F can train linear models in time linear in the
    number of graphs. Unlabeled inputs use n_length_samples lengths, as
    n_samples of random_walk_kernel_mc_dataset.
    """
    if not labeled:
        n_label_samples_per_length, n_walk_reps, q_sampling_kind = 1, 1, "uniform"
    sample = _sample(Ps, vs, ws, mu_func, kind, n_length_samples, n_label_samples_per_length,
                     n_walk_reps, q_sampling_kind, seed, labeled, diagnostics)
    if not np.isfinite(sample.features).all() or not np.isfinite(sample.diagonal).all():
        raise FloatingPointError("non-finite MC features; change proposal/lambda")
    return sample


def _diagnostics(features, stats, q):
    """Walk counts plus feature degeneracy: zero fraction and ESS/M per graph.

    ESS = (sum F)^2 / sum F^2 over one graph's feature columns; ESS/M = 1
    means equal features, small values mean a few large importance weights.
    """
    power = np.sum(features**2, axis=1)
    nonzero = power > 0
    ess = np.sum(features, axis=1)[nonzero]**2 / (features.shape[1]*power[nonzero])
    out = {key: int(stats[key]) for key in ("walks", "steps", "killed", "skipped")}
    out.update(zero_feature_fraction=float(np.mean(features == 0)),
               ess_fraction=float(np.mean(ess)) if ess.size else None,
               max_log_weight=float(stats["max_log_weight"]) if np.isfinite(stats["max_log_weight"]) else None)
    if q is not None:
        out["q"] = {label: float(p) for label, p in q.items()}
    return out


def random_walk_kernel_mc_dataset(Ps, vs, ws, mu_func, kind, n_samples=100, seed=42, diagonal="unbiased"):
    """Gram estimate; n_samples shared lengths, two walks per feature.

    diagonal: "unbiased" (replica cross product), "biased" (PSD, ridge-like)
    or "both", which returns (unbiased, biased) from the same walks.
    """
    return _estimate(Ps, vs, ws, mu_func, kind, n_samples, 1, 1, "uniform", seed, False, diagonal=diagonal)


def random_walk_kernel_mc(P1, P2, v1, v2, w1, w2, mu_func, kind, n_samples=100, seed=42):
    """Pair estimate using the same two-replica construction as the dataset API."""
    return float(random_walk_kernel_mc_dataset([P1,P2], [v1,v2], [w1,w2],
                                               mu_func, kind, n_samples, seed)[0,1])


def random_walk_kernel_mc_labeled_dataset(Ps, vs, ws, mu_func, kind,
        n_length_samples=200, n_label_samples_per_length=50, n_walk_reps=1,
        q_sampling_kind="uniform", seed=42, diagnostics=None, diagonal="unbiased"):
    """Unbiased labeled Gram; each replica averages n_walk_reps trajectories.

    There are m independent lengths, NOT m*n independent outer samples.
    Importance weights are accumulated in log space; finite variance is not
    guaranteed for geometric lengths and arbitrary label proposals.
    q_sampling_kind may be an explicit {label: probability} mapping. A dict
    passed as diagnostics receives walk counts, feature degeneracy and q.
    diagonal is "unbiased", "biased" or "both", as in the unlabeled API.
    """
    return _estimate(Ps, vs, ws, mu_func, kind, n_length_samples,
                     n_label_samples_per_length, n_walk_reps, q_sampling_kind, seed, True,
                     diagnostics, diagonal)


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
