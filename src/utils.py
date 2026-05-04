import numpy as np
import networkx as nx
import math
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import cg, LinearOperator, spsolve, expm as sparse_expm



def graph_generator(n, kind="er", seed=None, p_er=None):
    """
    er for Erdos-Renyi;
    ba for Barabasi-Albert;
    ws for Watts-Strogtz (small-world);
    sbm for Stochastic Block Model
    """
    
    if kind == "er":
        #Erdos-Renyi
        #p = 2.0/n gives us a moderately sparse graph:
        #E[deg] = (n - 1) * p = 2
        if p_er is None:
            p_er = float(2.0/n)
        return nx.erdos_renyi_graph(n=n, p=p_er, seed=seed)
    
    if kind == "ba":
        #Barabasi-Albert (preferential attachment)
        # Each new node connects to m = max(1, n // 20) existing nodes.
        # This yields a scale-free graph with hubs.
        return nx.barabasi_albert_graph(n=n, m=max(1, n // 20), seed=seed)
    
    if kind == "ws":
        #Watts-Strogtz (small-world)
        # Start with a ring where each node connects to k neighbors, then rewire edges with p = 0.1.
        # This keeps high clustering while creating short average paths.
        k = int(max(2, (n // 10) | 1))
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


def graph_generator_labeled(n, kind="er", n_labels=3, seed=None):
    g = graph_generator(n, kind=kind, seed=seed)
    rng = np.random.default_rng(seed)
    for u, v in g.edges():
        g[u][v]["label"] = int(rng.integers(0, n_labels))
    return g


def normalized_adj_matrix(graph):
    A = nx.to_scipy_sparse_array(graph, dtype=float, format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()

    inv_deg = np.zeros_like(deg, dtype=float)
    mask = deg > 0
    inv_deg[mask] = 1.0 / deg[mask]

    P = sp.diags(inv_deg, format="csr") @ A
    if np.any(~mask):
        P = P + sp.diags((~mask).astype(float), format="csr")

    return P.tocsr()

def normalized_adj_matrix_labeled(graph):
    nodes = list(graph.nodes())
    n = len(nodes)
    idx = {u: i for i, u in enumerate(nodes)}

    rows_by_label = {}
    cols_by_label = {}
    deg = np.zeros(n, dtype=float)

    for u, v, data in graph.edges(data=True):
        i, j = idx[u], idx[v]
        lab = int(data["label"])
        rows_by_label.setdefault(lab, []).extend([i, j])
        cols_by_label.setdefault(lab, []).extend([j, i])
        deg[i] += 1.0
        deg[j] += 1.0

    inv_deg = np.zeros_like(deg, dtype=float)
    mask = deg > 0
    inv_deg[mask] = 1.0 / deg[mask]
    Dinv = sp.diags(inv_deg, format="csr")

    P_labels = {}
    for lab in rows_by_label:
        rows = np.asarray(rows_by_label[lab], dtype=int)
        cols = np.asarray(cols_by_label[lab], dtype=int)
        data = np.ones(rows.shape[0], dtype=float)
        A_lab = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=float).tocsr()
        P_labels[lab] = (Dinv @ A_lab).tocsr()

    return P_labels

def uniform_dist(n):
    return np.ones(n, dtype=float) / n

def random_dist(n):
    x = np.random.random(n)
    return x / x.sum()

def mu_func_gen(kind="exp", lmbd=0.1):
    if kind == "exp":
        def mu(k):
            return (lmbd ** k) / math.factorial(k)
        return mu
    if kind == "geom":
        def mu(k):
            return lmbd ** k
        return mu
    raise ValueError(f"unknown kind: {kind}")

