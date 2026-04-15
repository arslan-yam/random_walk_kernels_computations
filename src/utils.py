import numpy as np
import networkx as nx
import math


def graph_generator(n, kind="er", seed=None):
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
        return nx.erdos_renyi_graph(n=n, p=float(2.0/n), seed=seed)
    
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
    A = nx.to_numpy_array(graph, dtype=float)
    deg = A.sum(axis=1)

    P = np.zeros_like(A, dtype=float)
    for i in range(A.shape[0]):
        if deg[i] > 0:
            P[i] = A[i] / deg[i]
        else:
            P[i, i] = 1.0

    return P

def normalized_adj_matrix_labeled(graph):
    nodes = list(graph.nodes())
    n = len(nodes)
    idx = {u: i for i, u in enumerate(nodes)}
    A_labels = {}
    deg = np.zeros(n, dtype=float)

    for u, v, data in graph.edges(data=True):
        i, j = idx[u], idx[v]
        lab = int(data["label"])
        if lab not in A_labels:
            A_labels[lab] = np.zeros((n, n), dtype=float)
        A_labels[lab][i, j] = 1.0
        A_labels[lab][j, i] = 1.0
        deg[i] += 1.0
        deg[j] += 1.0

    P_labels = {}
    for lab, A in A_labels.items():
        P = np.zeros_like(A)
        for i in range(n):
            if deg[i] > 0:
                P[i, :] = A[i, :] / deg[i]
        P_labels[lab] = P

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

