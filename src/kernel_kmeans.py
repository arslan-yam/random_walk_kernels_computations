"""Kernel k-means benchmark on TU graph datasets."""

import argparse
import json
import time

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from . import gram, mcrwk, utils


TU_DATASETS = ["MUTAG", "ENZYMES", "NCI1", "PTC_MR", "DD", "PROTEINS", "AIDS"]
METHODS = ["mc", "cg", "fp", "sylv", "gvoys"]


def random_dist(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    return x / x.sum()


def build_inputs(graphs, seed=42):
    Ps, vs, ws = [], [], []
    for i, graph in enumerate(graphs):
        P = utils.normalized_adj_matrix(graph)
        Ps.append(P)
        vs.append(random_dist(P.shape[0], seed + 2 * i))
        ws.append(random_dist(P.shape[0], seed + 2 * i + 1))
    return Ps, vs, ws


def normalize_gram(K, eps=1e-12):
    d = np.sqrt(np.clip(np.diag(K), eps, None))
    return K / np.outer(d, d)


def compute_gram(method, Ps, vs, ws, mu_func, lmbd, n_samples, seed):
    if method == "mc":
        return mcrwk.random_walk_kernel_mc_dataset(
            Ps, vs, ws, mu_func, "geom", n_samples, seed
        )
    if method == "cg":
        return gram.gram_cg(Ps, vs, ws, mu_func)
    if method == "fp":
        return gram.gram_fixed_point(Ps, vs, ws, mu_func)
    if method == "sylv":
        return gram.gram_sylvester(Ps, vs, ws, mu_func)
    if method == "gvoys":
        from . import gvoys

        np.random.seed(seed)
        return gvoys.random_walk_kernel_gvoys_dataset(
            Ps,
            vs,
            ws,
            kind="geom",
            lambda_coeff=lmbd,
            p_halt=gvoys.P_HALT,
            nb_random_walks=n_samples,
        )
    raise ValueError(f"unknown method: {method}")


def kernel_kmeans(K, n_clusters, max_iter=100, seed=42):
    rng = np.random.default_rng(seed)
    labels = rng.integers(n_clusters, size=len(K))
    labels[:n_clusters] = np.arange(n_clusters)

    for _ in range(max_iter):
        distances = np.zeros((len(K), n_clusters))
        for cluster in range(n_clusters):
            idx = np.where(labels == cluster)[0]
            if len(idx) == 0:
                distances[:, cluster] = np.inf
            else:
                distances[:, cluster] = (
                    np.diag(K)
                    - 2 * K[:, idx].mean(axis=1)
                    + K[np.ix_(idx, idx)].mean()
                )

        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

    return labels


def run_experiment(
    datasets,
    methods=METHODS,
    lmbd=0.01,
    n_samples=1000,
    seed=42,
):
    mu_func = utils.mu_func_gen("geom", lmbd=lmbd)
    results = []

    for name, (graphs, y) in datasets.items():
        Ps, vs, ws = build_inputs(graphs, seed)

        for method in methods:
            t0 = time.perf_counter()
            K = compute_gram(method, Ps, vs, ws, mu_func, lmbd, n_samples, seed)
            gram_time = time.perf_counter() - t0

            t0 = time.perf_counter()
            pred = kernel_kmeans(normalize_gram(K), len(np.unique(y)), seed=seed)
            clustering_time = time.perf_counter() - t0

            row = {
                "dataset": name,
                "method": method,
                "NMI": normalized_mutual_info_score(y, pred),
                "ARI": adjusted_rand_score(y, pred),
                "gram_time_sec": gram_time,
                "clustering_time_sec": clustering_time,
            }
            results.append(row)
            print(row)

    return results


def load_datasets(names, root_dir, max_graphs, max_nodes, seed):
    from dataset_bench import load_tu_dataset, pick_graph_subset

    datasets = {}
    for i, name in enumerate(names):
        graphs, y, _, _ = load_tu_dataset(name, root_dir=root_dir)
        graphs, y, _ = pick_graph_subset(
            graphs,
            y,
            max_graphs=max_graphs,
            max_nodes_per_graph=max_nodes,
            seed=seed + i,
        )
        datasets[name] = (graphs, y)
    return datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=TU_DATASETS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--root-dir", default="tu_datasets")
    parser.add_argument("--max-graphs", type=int, default=50)
    parser.add_argument("--max-nodes", type=int, default=300)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--lmbd", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="kernel_kmeans_results.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    datasets = load_datasets(
        args.datasets, args.root_dir, args.max_graphs, args.max_nodes, args.seed
    )
    results = run_experiment(
        datasets,
        methods=args.methods,
        lmbd=args.lmbd,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
