"""Gaussian process benchmark with random-walk graph kernels."""

import argparse
import json
import time

import numpy as np
import scipy.linalg as la
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

from src import utils
from kernel_kmeans import (
    METHODS,
    TU_DATASETS,
    build_inputs,
    compute_gram,
    normalize_gram,
)


def gp_predict(K_train, y_train, K_test, alpha=1e-6):
    y_mean = y_train.mean()
    weights = la.solve(
        K_train + alpha * np.eye(len(K_train)),
        y_train - y_mean,
        assume_a="sym",
    )
    return y_mean + K_test @ weights


def cv_score(K, y, n_splits=5, alpha=1e-6, seed=42):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rmse = []
    mae = []

    for train, test in cv.split(K):
        pred = gp_predict(
            K[np.ix_(train, train)],
            y[train],
            K[np.ix_(test, train)],
            alpha,
        )
        rmse.append(np.sqrt(mean_squared_error(y[test], pred)))
        mae.append(mean_absolute_error(y[test], pred))

    return float(np.mean(rmse)), float(np.mean(mae))


def run_experiment(
    datasets,
    methods=METHODS,
    lambdas=(0.001, 0.01, 0.1),
    n_samples=1000,
    n_splits=5,
    alpha=1e-6,
    seed=42,
):
    results = []

    for name, (graphs, targets) in datasets.items():
        y = np.asarray(targets, dtype=float)
        Ps, vs, ws = build_inputs(graphs, seed)

        for method in methods:
            t0 = time.perf_counter()
            candidates = []

            for lmbd in lambdas:
                mu_func = utils.mu_func_gen("geom", lmbd=lmbd)
                K = compute_gram(
                    method, Ps, vs, ws, mu_func, lmbd, n_samples, seed
                )
                K = normalize_gram(K)
                rmse, mae = cv_score(K, y, n_splits, alpha, seed)
                candidates.append({"lambda": lmbd, "RMSE": rmse, "MAE": mae})

            total_time = time.perf_counter() - t0
            best = min(candidates, key=lambda row: row["RMSE"])
            row = {
                "dataset": name,
                "method": method,
                "best_lambda": best["lambda"],
                "RMSE": best["RMSE"],
                "MAE": best["MAE"],
                "optimization_time_sec": total_time,
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
        datasets[name] = (graphs, y.astype(float))
    return datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=TU_DATASETS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.001, 0.01, 0.1])
    parser.add_argument("--root-dir", default="tu_datasets")
    parser.add_argument("--max-graphs", type=int, default=50)
    parser.add_argument("--max-nodes", type=int, default=300)
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./results/graph_gp/results.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    datasets = load_datasets(
        args.datasets, args.root_dir, args.max_graphs, args.max_nodes, args.seed
    )
    results = run_experiment(
        datasets,
        methods=args.methods,
        lambdas=args.lambdas,
        n_samples=args.n_samples,
        n_splits=args.n_splits,
        alpha=args.alpha,
        seed=args.seed,
    )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
