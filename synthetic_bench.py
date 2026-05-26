import argparse
import pickle
import json
import time
import os

import numpy as np

from src import utils
from src import rwk
from src import gvoys
from src import mcrwk
from src import gram


def bench(dataset, 
          kind, 
          mu_func, 
          n_graphs, 
          n_samples_mc, 
          n_samples_gvoys,
          distribution_func='uniform',
          seed=42, 
          labeled=False):
    """
    dataset: networkx graphs
    kind: "exp" or "geom"
    mu_func: from mu_func_gen(...)
    n_graphs: number of graphs from dataset to use
    n_samples_mc: number of Monte Carlo samples for our estimator
    n_samples_gvoys: number of random walks for GVoy-style estimator
    """

    graphs = list(dataset)[:n_graphs]
    Ps, vs, ws = [], [], []
    results = {}
    gram_mtx = {}
    for G in graphs:
        if labeled:
            P = utils.normalized_adj_matrix_labeled(G)
        else:
            P = utils.normalized_adj_matrix(G)
        n = len(G)
        Ps.append(P)
        if distribution_func == "uniform":
            vs.append(utils.uniform_dist(n))
            ws.append(utils.uniform_dist(n))
        elif distribution_func == "normal":
            vs.append(utils.random_dist(n))
            ws.append(utils.random_dist(n))
        else:
            raise Exception(f'Invalid distribution type: {distribution_func}')
    
    if n <= 128:
        compute_direct=True
    else:
        compute_direct=False

    if n <= 512:
        compute_sylvester=True
    else:
        compute_sylvester=False

    if n <= 1024:
        compute_normal_gvoys=True
    else:
        compute_normal_gvoys=False

    # --- Direct ---
    G_direct = None
    if compute_direct:
        print("direct started")
        t0 = time.perf_counter()
        G_direct = gram.gram_direct(Ps, vs, ws, mu_func, kind, labeled=labeled)
        t1 = time.perf_counter()
        results["direct"] = {
            # "gram": G_direct,
            "time": t1 - t0,
            "err": {"mean_abs": 0.0, "mean_rel": 0.0, "max_abs": 0.0, "max_rel": 0.0},
        }
        gram_mtx["direct"] = G_direct

    # --- Conjugate Gradient ---
    if kind == "geom":
        print("cg started")
        t0 = time.perf_counter()
        G_cg = gram.gram_cg(Ps, vs, ws, mu_func, labeled=labeled)
        # If no direct method, then use CG as true reference
        if not compute_direct:
            G_direct = G_cg
        t1 = time.perf_counter()
        results["cg"] = {
            # "gram": G_cg,
            "time": t1 - t0,
            "err": gram.matrix_errors(G_direct, G_cg) if G_direct is not None else -1,
        }
        gram_mtx["cg"] = G_cg

    # --- Sylvester ---
    if compute_sylvester and kind == "geom" and not labeled:
        print("sylvester started")
        t0 = time.perf_counter()
        G_syl = gram.gram_sylvester(Ps, vs, ws, mu_func)
        t1 = time.perf_counter()
        results["sylvester"] = {
            # "gram": G_syl,
            "time": t1 - t0,
            "err": gram.matrix_errors(G_direct, G_syl) if G_direct is not None else -1,
        }
        gram_mtx["sylvester"] = G_syl


    # --- Fixed point ---
    print("fixed point started")
    if kind == "geom":
        t0 = time.perf_counter()
        G_fp = gram.gram_fixed_point(Ps, vs, ws, mu_func, labeled=labeled)
        t1 = time.perf_counter()
        results["fixed_point"] = {
            # "gram": G_fp,
            "time": t1 - t0,
            "err": gram.matrix_errors(G_direct, G_fp) if G_direct is not None else -1,
        }
        gram_mtx["fixed_point"] = G_fp

    # --- GVoys ---
    if labeled:
        print("gvoys started")
        t0 = time.perf_counter()
        np.random.seed(seed)
        if compute_normal_gvoys:
            G_gv = gvoys.random_walk_kernel_gvoys_labeled_dataset(Ps, vs, ws, anchor_fraction=1.0, kind=kind, lambda_coeff=mu_func(1), p_halt=P_HALT, nb_random_walks=n_samples_gvoys)
        else:
            G_gv = gvoys.random_walk_kernel_gvoys_labeled_dataset_block(Ps, vs, ws, anchor_fraction=1.0, kind=kind, lambda_coeff=mu_func(1), 
                                                                p_halt=P_HALT, nb_random_walks=n_samples_gvoys, block_size=10)
        t1 = time.perf_counter()
        results["gvoys"] = {
            # "gram": G_gv,
            "time": t1 - t0,
            "err": gram.matrix_errors(G_direct, G_gv) if G_direct is not None else -1,
        }
        gram_mtx["gvoys"] = G_gv
    else:
        print("gvoys started")
        t0 = time.perf_counter()
        np.random.seed(seed)
        if compute_normal_gvoys:
            G_gv = gvoys.random_walk_kernel_gvoys_dataset(Ps, vs, ws, anchor_fraction=1.0, kind=kind, lambda_coeff=mu_func(1), p_halt=P_HALT, nb_random_walks=n_samples_gvoys)
        else:
            G_gv = gvoys.random_walk_kernel_gvoys_dataset_block(Ps, vs, ws, anchor_fraction=1.0, kind=kind, lambda_coeff=mu_func(1), 
                                                                p_halt=P_HALT, nb_random_walks=n_samples_gvoys, block_size=10)
        t1 = time.perf_counter()
        results["gvoys"] = {
            # "gram": G_gv,
            "time": t1 - t0,
            "err": gram.matrix_errors(G_direct, G_gv) if G_direct is not None else -1,
        }
        gram_mtx["gvoys"] = G_gv

    # --- Monte-Carlo Random Walk Kernel ---
    print("mc started")
    t0 = time.perf_counter()
    if labeled:
        G_mc = mcrwk.random_walk_kernel_mc_labeled_dataset(Ps, vs, ws, mu_func=mu_func, kind=kind, 
                                                           n_length_samples=n_samples_mc // 100,
                                                           n_label_samples_per_length=100, 
                                                           seed=seed)
    else:
        G_mc = mcrwk.random_walk_kernel_mc_dataset(Ps, vs, ws, mu_func=mu_func, kind=kind, n_samples=n_samples_mc, seed=seed)
    t1 = time.perf_counter()
    results["mc"] = {
        # "gram": G_mc,
        "time": t1 - t0,
        "err": gram.matrix_errors(G_direct, G_mc) if G_direct is not None else -1,
    }
    gram_mtx["mc"] = G_mc

    print("Benchmark finished. Saving results...")
    return results, gram_mtx


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark graph kernel Gram matrix construction")
    parser.add_argument("--n_graphs", type=int, default=10, help="Number of graphs")
    parser.add_argument("--n_nodes", type=int, default=128, help="Number of nodes per graph")
    parser.add_argument("--kind", choices=["exp", "geom"], default="geom", help="Kernel type")
    # parser.add_argument("--lmbd", type=float, default=0.01, help="Lambda coefficient")
    
    parser.add_argument("--n_samples_mc", type=int, default=100, help="MC samples per node: n_samples_mc * n_nodes (default: 100)")
    parser.add_argument("--n_samples_gvoys", type=int, default=100, help="GVoy samples (default: 100)")
    
    parser.add_argument("--labeled", type=int, default=False, help="Whether to generate graphs with labels or without")
    parser.add_argument("--u_w_distribution", choices=["uniform", "normal"], default="uniform", help="STarting and terminating distribution of RWK")

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--graph_type", choices=["er", "ba", "grid"], default="er", help="Graph generator")
    # parser.add_argument("--p-halt", type=float, default=0.2, help="Halt probability for GVoy")
    parser.add_argument("--experiment_name", type=str, default=None, help="Add experiment name to folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    n_graphs = args.n_graphs
    n_nodes = args.n_nodes
    kernel_kind = args.kind
    graph_type = args.graph_type
    # lmbd = args.lmbd
    seed = args.seed
    exp_name = args.experiment_name
    is_labeled = args.labeled
    distribution_type = args.u_w_distribution

    n_samples_mc = args.n_samples_mc * n_nodes
    n_samples_gvoys = args.n_samples_gvoys
    
    if is_labeled:
        dataset = [utils.graph_generator_labeled(n=n_nodes, kind=graph_type, seed=i) for i in range(n_graphs)]
    else:
        dataset = [utils.graph_generator(n=n_nodes, kind=graph_type, seed=i) for i in range(n_graphs)]
    
    d_max = 0
    for G in dataset:
        d = max(d for n, d in G.degree())
        if d > d_max:
            d_max = d
    # Setting lambda value as in GVoys paper
    lmbd = 1 / (d_max ** 2)

    mu_func = utils.mu_func_gen(kernel_kind, lmbd=lmbd)

    # Redefinition of globals for gvoys
    LAMBDA_COEFF = lmbd
    P_HALT = 0.2

    results, gram_mtx = bench(
        dataset=dataset,
        kind=kernel_kind,
        mu_func=mu_func,
        n_graphs=n_graphs,
        n_samples_mc=n_samples_mc,
        n_samples_gvoys=n_samples_gvoys,
        seed=seed,
        labeled=is_labeled
    )

    folder = f"./results/{graph_type}/{kernel_kind}/{n_graphs}_graphs/{n_nodes}_nodes_{distribution_type}"
    if is_labeled:
        folder += "_labeled"
    
    if exp_name:
        folder += f"_{exp_name}"
        
    os.makedirs(folder, exist_ok=True)

    results_output = f"{folder}/seed={seed}.json"

    # Загружаем существующие данные, если файл есть
    if os.path.exists(results_output):
        with open(results_output, "r") as f:
            existing_results = json.load(f)
    else:
        existing_results = {}

    # Обновляем старые данные новыми (ключи из results перезаписывают существующие)
    existing_results.update(results)

    # Сохраняем объединённый словарь
    with open(results_output, "w") as f:
        json.dump(existing_results, f, indent=2, default=str)

    gram_output = f"{folder}/seed={seed}.pickle"
    with open(gram_output, 'wb') as f:
        pickle.dump(gram_mtx, f)

    print(f"Results saved to {results_output} and {gram_output}")