"""Graph generators and row-normalized RWK inputs."""

import numpy as np
import networkx as nx
import math
import scipy.sparse as sp
from ._validation import positive_int



def graph_generator(n, kind="er", seed=None, p_er=None, ba_m=None, ws_k=None):
    """
    er for Erdos-Renyi;
    ba for Barabasi-Albert;
    ws for Watts-Strogtz (small-world);
    sbm for Stochastic Block Model
    """
    positive_int(n, "n")
    if kind == "er":
        #Erdos-Renyi
        #p = 2.0/n gives us a moderately sparse graph:
        #E[deg] = (n - 1) * p = 2
        if p_er is None:
            p_er = min(1.0, 2.0/n)
        if not 0 <= p_er <= 1:
            raise ValueError("p_er must be between 0 and 1")
        return nx.erdos_renyi_graph(n=n, p=p_er, seed=seed)
    
    if kind == "ba":
        #Barabasi-Albert (preferential attachment)
        # Each new node connects to m = max(1, n // 20) existing nodes.
        # This yields a scale-free graph with hubs.
        return nx.barabasi_albert_graph(n=n, m=ba_m if ba_m is not None else max(1, n // 20), seed=seed)
    
    if kind == "ws":
        #Watts-Strogtz (small-world)
        # Start with a ring where each node connects to k neighbors, then rewire edges with p = 0.1.
        # This keeps high clustering while creating short average paths.
        k = ws_k if ws_k is not None else min(n-n%2, max(2, 2*(n//20)))
        if k < 0 or k > n or k % 2:
            raise ValueError("ws_k must be an even integer between 0 and n")
        p = float(0.1)
        return nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)

    if kind == "sbm":
        # Stochastic Block Model with 2 groups.
        # Connect nodes within the same group with p_in = 0.15,
        # and across groups with p_out = 0.02 (weaker connections).
        sizes = [n//2, n - n//2]
        p_in, p_out = float(0.15), float(0.02)
        P = [
            [p_in,  p_out],
            [p_out, p_in]
        ]
        return nx.stochastic_block_model(sizes, P, seed=seed)
    
    raise ValueError(f"unknown kind: {kind}")


def graph_generator_labeled(n, kind="er", n_labels=3, seed=None, **kwargs):
    positive_int(n_labels, "n_labels")
    g = graph_generator(n, kind=kind, seed=seed, **kwargs)
    rng = np.random.default_rng(seed)
    for u, v in g.edges():
        g[u][v]["label"] = int(rng.integers(0, n_labels))
    return g


def normalized_adj_matrix(graph):
    """Row-normalize weighted adjacency; isolated vertices are absorbing."""
    A = nx.to_scipy_sparse_array(graph, dtype=float, format="csr")
    if not np.isfinite(A.data).all() or np.any(A.data < 0):
        raise ValueError("edge weights must be finite and nonnegative")
    deg = np.asarray(A.sum(axis=1)).ravel()

    inv_deg = np.zeros_like(deg, dtype=float)
    mask = deg > 0
    inv_deg[mask] = 1.0 / deg[mask]

    P = sp.diags(inv_deg, format="csr") @ A
    if np.any(~mask):
        P = P + sp.diags((~mask).astype(float), format="csr")

    return P.tocsr()

def normalized_adj_matrix_labeled(graph):
    """Restrict D^-1 A by EDGE labels; isolates have zero rows (killed walks)."""
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("labeled conversion requires a simple undirected graph")
    nodes = list(graph.nodes())
    n = len(nodes)
    idx = {u: i for i, u in enumerate(nodes)}

    rows_by_label = {}
    cols_by_label = {}
    deg = np.zeros(n, dtype=float)

    for u, v, data in graph.edges(data=True):
        i, j = idx[u], idx[v]
        lab = int(data["label"])
        weight = float(data.get("weight", 1.0))
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("edge weights must be finite and nonnegative")
        if weight == 0:
            continue
        rows_by_label.setdefault(lab, []).append((i, weight))
        cols_by_label.setdefault(lab, []).append(j)
        deg[i] += weight
        if i != j:
            rows_by_label[lab].append((j, weight))
            cols_by_label[lab].append(i)
            deg[j] += weight

    inv_deg = np.zeros_like(deg, dtype=float)
    mask = deg > 0
    inv_deg[mask] = 1.0 / deg[mask]
    Dinv = sp.diags(inv_deg, format="csr")

    P_labels = {}
    for lab in rows_by_label:
        rows = np.asarray([i for i, _ in rows_by_label[lab]], dtype=int)
        cols = np.asarray(cols_by_label[lab], dtype=int)
        data = np.asarray([weight for _, weight in rows_by_label[lab]], dtype=float)
        A_lab = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=float).tocsr()
        P_labels[lab] = (Dinv @ A_lab).tocsr()

    return P_labels

def uniform_dist(n):
    return np.ones(n, dtype=float) / n

def random_dist(n, rng=None):
    """Independent Uniform(0,1) weights normalized to sum to one."""
    rng = np.random.default_rng() if rng is None else rng
    x = rng.random(n)
    return x / x.sum()


def normal_dist(n, rng=None):
    """Normalize abs(N(0,1)) draws (half-normal weights) to a probability vector.

    Raw Gaussian draws may be negative and cannot be starting probabilities.
    This is intentionally different from the historical CLI alias for random.
    """
    positive_int(n, "n")
    rng = np.random.default_rng() if rng is None else rng
    weights = np.abs(rng.standard_normal(n))
    total = weights.sum()
    return weights / total if total > 0 else uniform_dist(n)


def degree_dist(graph):
    """Normalize weighted adjacency row sums in graph.nodes() order.

    Self-loops count once, matching the adjacency used to build P. Isolated
    vertices have zero mass; an entirely edgeless graph falls back to uniform.
    """
    positive_int(len(graph), "number of vertices")
    adjacency = nx.to_scipy_sparse_array(graph, dtype=float, format="csr")
    if not np.isfinite(adjacency.data).all() or np.any(adjacency.data < 0):
        raise ValueError("edge weights must be finite and nonnegative")
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    total = degrees.sum()
    if not np.isfinite(total):
        raise ValueError("total degree must be finite")
    return degrees / total if total > 0 else uniform_dist(len(graph))


def mu_func_gen(kind="exp", lmbd=0.1):
    if not np.isfinite(lmbd) or lmbd < 0 or (kind == "geom" and lmbd >= 1):
        raise ValueError("lambda must be finite and >=0, and <1 for geom")
    if kind == "exp":
        def mu(k):
            return (lmbd ** k) / math.factorial(k)
        return mu
    if kind == "geom":
        def mu(k):
            return lmbd ** k
        return mu
    raise ValueError(f"unknown kind: {kind}")
