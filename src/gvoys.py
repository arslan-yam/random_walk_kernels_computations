"""GVoys for the same normalized RWK as MCRWK.

Public kernels accept P=D^-1 A (or label restrictions of P). Internally they
use the equivalent symmetric matrices S and transformed boundary weights.
Uniform neighbor proposals carry degree*S[x,y] importance corrections.
Random signs and halt lengths are shared across graphs, but trajectories and
anchors are independent across graphs. One feature realization per graph is
used to construct the usual PSD Gram matrix F @ F.T / samples.
No random tables are allocated at import; blocks bound temporary memory.

For each side and depth k, the load contains sqrt(f_k / survival_k).
Shared depth signs cancel unequal depths between graphs. Shared label signs
retain matching label words. Matching depths contribute f_k after averaging
the shared halt event. Combining the two independent sides gives the
convolution sum_j f_j*f_(k-j) = mu_k. Symmetry of each S_label is essential
when reversing the side that arrives at the common anchor.
"""

from dataclasses import dataclass
import math

import numpy as np

from ._validation import kernel_parameter, positive_int
from .normalization import symmetric_inputs

LAMBDA_COEFF = 0.1
P_HALT = 0.2
NB_RANDOM_WALKS = 1000


def f_func_diffusion(i, lambda_coeff):
    """Convolution square root of mu_k=lambda^k/k!."""
    return math.exp(i*math.log(lambda_coeff/2)-math.lgamma(i+1)) if lambda_coeff and i else float(i == 0)


def f_func_geometric(i, lambda_coeff):
    """Convolution square root: binom(2k,k)*(lambda/4)^k."""
    return math.exp(math.lgamma(2*i+1)-2*math.lgamma(i+1)+i*math.log(lambda_coeff/4)) if lambda_coeff and i else float(i == 0)


def _kind(kind):
    return {"exponential": "exp", "geometric": "geom"}.get(kind, kind)


@dataclass
class _Graph:
    rows: list
    v: np.ndarray
    w: np.ndarray


def _prepare(P, v, w, labeled):
    S, v, w = symmetric_inputs(P, v, w, labeled)
    n = len(v)
    entries = [[] for _ in range(n)]
    matrices = S.items() if labeled else [(None, S)]
    for lab, M in matrices:
        for x in range(n):
            for p in range(M.indptr[x], M.indptr[x+1]):
                entries[x].append((int(M.indices[p]), float(M.data[p]), lab))
    return _Graph(entries, v, w)


def _shared(seed, walk_id, labels, p_halt, max_walk_length):
    """One outer random feature. Separate streams for left and right sides."""
    sides = []
    for side in range(2):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 0, walk_id, side]))
        length = int(rng.geometric(p_halt)-1)
        if max_walk_length is not None and length > max_walk_length:
            raise RuntimeError("GVoys walk exceeded max_walk_length; no biased truncation was applied")
        signs = rng.choice([-1., 1.], size=length+1)
        z = {lab: rng.choice([-1., 1.], size=length) for lab in labels}
        sides.append((length, signs, z))
    return sides


def _side_features(graph, boundary, anchors, shared, rng, kind, lam, p_halt):
    length, signs, z = shared
    features = np.zeros(len(anchors))
    for start, boundary_weight in enumerate(boundary):
        if boundary_weight == 0:
            continue
        x = start
        amplitude = float(boundary_weight)
        for step in range(length+1):
            if x in anchors:
                features[anchors[x]] += amplitude * signs[step]
            if step == length or lam == 0:
                break
            row = graph.rows[x]
            if not row:
                break
            y, edge_weight, label = row[int(rng.integers(len(row)))]
            # Fold sqrt(f_k/f_{k-1}) into the load to avoid factorial underflow.
            k = step+1
            ratio = lam/(2*k) if kind == "exp" else lam*(2*k-1)/(2*k)
            amplitude *= len(row)*edge_weight*math.sqrt(ratio/(1-p_halt))
            if label is not None:
                amplitude *= z[label][step]
            x = y
    return features


def _feature(graph, shared, seed, graph_id, walk_id, kind, lam, p_halt, anchor_fraction):
    n = len(graph.v)
    count = max(1, int(anchor_fraction*n))
    # Keep the former first-realization stream; no second realization is drawn.
    streams = np.random.SeedSequence([seed, 1, walk_id, graph_id, 0]).spawn(3)
    anchor_rng = np.random.default_rng(streams[0])
    ids = np.arange(n) if count == n else np.sort(anchor_rng.choice(n, count, replace=False))
    anchors = {int(node): j for j, node in enumerate(ids)}
    a = _side_features(graph, graph.v, anchors, shared[0], np.random.default_rng(streams[1]), kind, lam, p_halt)
    b = _side_features(graph, graph.w, anchors, shared[1], np.random.default_rng(streams[2]), kind, lam, p_halt)
    # Horvitz-Thompson correction for uniformly sampled anchors.
    return n/count * float(a @ b)


def _dataset(Ps, vs, ws, *, labeled, anchor_fraction, kind, lambda_coeff,
             p_halt, nb_random_walks, block_size, seed, max_walk_length=None,
             base_nb_walk_index=0):
    kind = _kind(kind)
    lam = kernel_parameter(kind, lambda k: lambda_coeff**k / (math.factorial(k) if kind == "exp" else 1))
    samples = positive_int(nb_random_walks, "nb_random_walks")
    block_size = positive_int(block_size, "block_size")
    if not 0 < p_halt < 1 or not 0 < anchor_fraction <= 1:
        raise ValueError("require 0 < p_halt < 1 and 0 < anchor_fraction <= 1")
    if max_walk_length is not None:
        positive_int(max_walk_length, "max_walk_length")
    if not (len(Ps) == len(vs) == len(ws)):
        raise ValueError("Ps, vs and ws must have the same length")
    if int(seed) != seed or seed < 0 or base_nb_walk_index < 0:
        raise ValueError("seed and base_nb_walk_index must be nonnegative integers")
    seed = int(seed)
    graphs = [_prepare(P,v,w,labeled) for P,v,w in zip(Ps,vs,ws)]
    labels = sorted(set().union(*(P.keys() for P in Ps))) if labeled else []
    result = np.zeros((len(graphs), len(graphs)))
    for offset in range(0, samples, block_size):
        size = min(block_size, samples-offset)
        features = np.empty((len(graphs), size))
        for j in range(size):
            walk_id = base_nb_walk_index+offset+j
            shared = _shared(seed, walk_id, labels, p_halt, max_walk_length)
            for g, graph in enumerate(graphs):
                features[g,j] = _feature(graph, shared, seed, g, walk_id,
                                        kind, lam, p_halt, anchor_fraction)
        if graphs:
            result += (features @ features.T) / samples
    if not np.isfinite(result).all():
        raise FloatingPointError("non-finite GVoys estimate; reduce weight variance")
    return result


def random_walk_kernel_gvoys_dataset(Ps, vs, ws, anchor_fraction=1., kind="exp",
        lambda_coeff=LAMBDA_COEFF, p_halt=P_HALT, nb_random_walks=NB_RANDOM_WALKS,
        seed=42, block_size=64, max_walk_length=None):
    """Normalized unlabeled RWK. Budget is outer features per start vertex.

    Each feature uses one left/right walk construction, without an extra
    replica or diagonal replacement. The returned feature Gram is PSD up to
    roundoff. Block size changes memory usage, not sampled trajectories.
    """
    return _dataset(Ps,vs,ws,labeled=False,anchor_fraction=anchor_fraction,kind=kind,
        lambda_coeff=lambda_coeff,p_halt=p_halt,nb_random_walks=nb_random_walks,
        block_size=block_size,seed=seed,max_walk_length=max_walk_length)


def random_walk_kernel_gvoys_labeled_dataset(Ps_labeled, vs, ws, anchor_fraction=1.,
        kind="exp", lambda_coeff=LAMBDA_COEFF, p_halt=P_HALT,
        nb_random_walks=NB_RANDOM_WALKS, seed=42, block_size=64, max_walk_length=None):
    """Edge-labeled normalized RWK, using independent shared Rademacher signs."""
    return _dataset(Ps_labeled,vs,ws,labeled=True,anchor_fraction=anchor_fraction,kind=kind,
        lambda_coeff=lambda_coeff,p_halt=p_halt,nb_random_walks=nb_random_walks,
        block_size=block_size,seed=seed,max_walk_length=max_walk_length)


def random_walk_kernel_gvoys_dataset_block(Ps, vs, ws, anchor_fraction=1., kind="exp",
        lambda_coeff=LAMBDA_COEFF, p_halt=P_HALT, nb_random_walks=NB_RANDOM_WALKS,
        block_size=64, seed=42, max_walk_length=None):
    """Compatibility alias; partial final blocks are supported."""
    return random_walk_kernel_gvoys_dataset(Ps,vs,ws,anchor_fraction,kind,lambda_coeff,
        p_halt,nb_random_walks,seed,block_size,max_walk_length)


def random_walk_kernel_gvoys_labeled_dataset_block(Ps_labeled, vs, ws,
        anchor_fraction=1., kind="exp", lambda_coeff=LAMBDA_COEFF, p_halt=P_HALT,
        nb_random_walks=NB_RANDOM_WALKS, block_size=64, seed=42, max_walk_length=None):
    return random_walk_kernel_gvoys_labeled_dataset(Ps_labeled,vs,ws,anchor_fraction,
        kind,lambda_coeff,p_halt,nb_random_walks,seed,block_size,max_walk_length)


def approximate_graph_kernel_value(P1,P2,v1,v2,w1,w2,anchor_fraction=1.,
        base_nb_walk_index=0,kind="exponential",lambda_coeff=LAMBDA_COEFF,
        p_halt=P_HALT,nb_random_walks=NB_RANDOM_WALKS,seed=42):
    return float(_dataset([P1,P2],[v1,v2],[w1,w2],labeled=False,
        anchor_fraction=anchor_fraction,kind=kind,lambda_coeff=lambda_coeff,
        p_halt=p_halt,nb_random_walks=nb_random_walks,block_size=64,seed=seed,
        base_nb_walk_index=base_nb_walk_index)[0,1])


def approximate_graph_kernel_value_with_blocks(P1,P2,v1,v2,w1,w2,
        anchor_fraction=1.,kind="exponential",lambda_coeff=LAMBDA_COEFF,
        p_halt=P_HALT,nb_random_walks=NB_RANDOM_WALKS,block_size=64,seed=42):
    return float(random_walk_kernel_gvoys_dataset([P1,P2],[v1,v2],[w1,w2],
        anchor_fraction,kind,lambda_coeff,p_halt,nb_random_walks,seed,block_size)[0,1])


def random_walk_kernel_gvoys_labeled(P1_labeled,P2_labeled,v1,v2,w1,w2,
        anchor_fraction=1.,kind="exp",lambda_coeff=LAMBDA_COEFF,p_halt=P_HALT,
        nb_random_walks=NB_RANDOM_WALKS,seed=42):
    return float(random_walk_kernel_gvoys_labeled_dataset([P1_labeled,P2_labeled],
        [v1,v2],[w1,w2],anchor_fraction,kind,lambda_coeff,p_halt,nb_random_walks,seed)[0,1])
