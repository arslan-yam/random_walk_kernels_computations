"""Fixed Monte Carlo sample budget versus graph size."""

import argparse
import json
import time

import numpy as np

from src import mcrwk, rwk, utils


def random_dist(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    return x / x.sum()


def run_experiment(
    node_sizes=(8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192),
    n_samples=1000,
    n_pairs=3,
    lmbd=0.01,
    graph_kind="er",
    seed=42,
):
    mu_func = utils.mu_func_gen("geom", lmbd=lmbd)
    results = []

    for n in node_sizes:
        times = []
        errors = []

        for pair in range(n_pairs):
            curr_seed = seed + 1000 * n + pair
            G1 = utils.graph_generator(n, kind=graph_kind, seed=curr_seed)
            G2 = utils.graph_generator(n, kind=graph_kind, seed=curr_seed + 1)
            P1 = utils.normalized_adj_matrix(G1)
            P2 = utils.normalized_adj_matrix(G2)

            v1 = random_dist(n, curr_seed + 2)
            v2 = random_dist(n, curr_seed + 3)
            w1 = random_dist(n, curr_seed + 4)
            w2 = random_dist(n, curr_seed + 5)

            exact = rwk.random_walk_kernel(
                P1, P2, v1, v2, w1, w2, mu_func, kind="geom"
            )

            t0 = time.perf_counter()
            approx = mcrwk.random_walk_kernel_mc(
                P1,
                P2,
                v1,
                v2,
                w1,
                w2,
                mu_func,
                kind="geom",
                n_samples=n_samples,
                seed=curr_seed,
            )
            times.append(time.perf_counter() - t0)
            errors.append(abs(exact - approx) / max(abs(exact), 1e-15))

        row = {
            "n_nodes": n,
            "n_samples": n_samples,
            "mean_time": float(np.mean(times)),
            "mean_rel_error": float(np.mean(errors)),
        }
        results.append(row)
        print(row)

    return results


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--node-sizes", nargs="+", type=int, default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--node-sizes", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument("--n-samples", type=int, default=1000)
    parser.add_argument("--n-pairs", type=int, default=3)
    parser.add_argument("--lmbd", type=float, default=0.01)
    parser.add_argument("--graph-kind", choices=["er", "ba", "ws", "sbm"], default="er")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="./results/fixed_samples/results.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_experiment(
        node_sizes=args.node_sizes,
        n_samples=args.n_samples,
        n_pairs=args.n_pairs,
        lmbd=args.lmbd,
        graph_kind=args.graph_kind,
        seed=args.seed,
    )
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
