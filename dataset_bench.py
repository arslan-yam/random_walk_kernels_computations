import argparse
import pickle
import json
import time
import os

import numpy as np

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import utils
from src import rwk
from src import gvoys
from src import mcrwk
from src import gram


def evaluate_svm_on_gram(gram_matrix, labels, n_splits=5, random_state=42):
    gram = np.asarray(gram_matrix)
    # Небольшой сдвиг для гарантии PSD
    eigvals = np.linalg.eigvalsh(gram)
    if np.min(eigvals) < -1e-8:
        gram += 1e-8 * np.eye(gram.shape[0])
    
    svm = SVC(kernel='precomputed', C=1.0)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(svm, gram, labels, cv=skf, scoring='accuracy')
    return float(np.mean(scores))

def bench(dataset, labels, kind, mu_func, n_samples_mc, n_samples_gvoys, seed=42):
    """
    dataset: networkx graphs
    kind: "exp" or "geom"
    mu_func: from mu_func_gen(...)
    n_graphs: number of graphs from dataset to use
    n_samples_mc: number of Monte Carlo samples for our estimator
    n_samples_gvoys: number of random walks for GVoy-style estimator
    """

    graphs = list(dataset)
    Ps, vs, ws = [], [], []
    results = {}
    gram_mtx = {}
    for G in graphs:
        P = utils.normalized_adj_matrix(G)
        n = P.shape[0]
        Ps.append(P)
        vs.append(utils.uniform_dist(n))
        ws.append(utils.uniform_dist(n))

    # direct
    print("direct started")
    t0 = time.perf_counter()
    G_direct = gram.gram_direct(Ps, vs, ws, mu_func, kind)
    t1 = time.perf_counter()
    
    acc_direct = evaluate_svm_on_gram(G_direct, labels)
    
    results["direct"] = {
        "accuracy": acc_direct,
        "time": t1 - t0,
        "err": {"mean_abs": 0.0, "mean_rel": 0.0, "max_abs": 0.0, "max_rel": 0.0},
    }
    gram_mtx["direct"] = G_direct

    # sylvester
    print("sylvester started")
    if kind == "geom":
        t0 = time.perf_counter()
        G_syl = gram.gram_sylvester(Ps, vs, ws, mu_func)
        t1 = time.perf_counter()
        
        acc_syl = evaluate_svm_on_gram(G_syl, labels)

        results["sylvester"] = {
            "accuracy": acc_syl,
            "time": t1 - t0,
            "err": gram.matrix_errors(G_direct, G_syl),
        }
        gram_mtx["sylvester"] = G_syl


    # fixed point
    print("fixed point started")
    if kind == "geom":
        t0 = time.perf_counter()
        G_fp = gram.gram_fixed_point(Ps, vs, ws, mu_func)
        t1 = time.perf_counter()
        
        acc_fp = evaluate_svm_on_gram(G_fp, labels)
        
        results["fixed_point"] = {
            "accuracy": acc_fp,
            "time": t1 - t0,
            "err": gram.matrix_errors(G_direct, G_fp),
        }
        gram_mtx["fixed_point"] = G_fp

    # cg
    print("cg started")
    if kind == "geom":
        t0 = time.perf_counter()
        G_cg = gram.gram_cg(Ps, vs, ws, mu_func)
        t1 = time.perf_counter()
        
        acc_cg = evaluate_svm_on_gram(G_cg, labels)

        results["cg"] = {
            "accuracy": acc_cg,
            "time": t1 - t0,
            "err": gram.matrix_errors(G_direct, G_cg),
        }
        gram_mtx["cg"] = G_cg

    # gvoys
    print("gvoys started")
    t0 = time.perf_counter()
    np.random.seed(seed)
    G_gv = gvoys.random_walk_kernel_gvoys_dataset(Ps, vs, ws, anchor_fraction=1.0, kind=kind, lambda_coeff=mu_func(1), p_halt=P_HALT, nb_random_walks=n_samples_gvoys)
    t1 = time.perf_counter()
    
    acc_gv = evaluate_svm_on_gram(G_gv, labels)

    results["gvoys"] = {
        "accuracy": acc_gv,
        "time": t1 - t0,
        "err": gram.matrix_errors(G_direct, G_gv),
    }
    gram_mtx["gvoys"] = G_gv

    # our mc
    print("mc started")
    t0 = time.perf_counter()
    G_mc = mcrwk.random_walk_kernel_mc_dataset(Ps, vs, ws, mu_func=mu_func, kind=kind, n_samples=n_samples_mc, seed=seed)
    t1 = time.perf_counter()
    
    acc_mc = evaluate_svm_on_gram(G_mc, labels)

    results["mc"] = {
        "accuracy": acc_mc,
        "time": t1 - t0,
        "err": gram.matrix_errors(G_direct, G_mc),
    }
    gram_mtx["mc"] = G_mc

    return results, gram_mtx

def load_tudataset_with_labels(name, root="./data", n_graphs=None):
    """Загружает TUDataset и возвращает список графов networkx и массив меток."""
    dataset = TUDataset(root=root, name=name, use_node_attr=False, use_edge_attr=False)
    if n_graphs is not None:
        dataset = dataset[:n_graphs]
    graphs = []
    labels = []
    n_nodes = []
    for data in dataset:
        G = to_networkx(data, to_undirected=True)
        n_nodes.append(len(G))
        graphs.append(G)
        labels.append(data.y.item())  # метка класса
    return graphs, np.array(labels), int(np.mean(n_nodes))

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark graph kernel Gram matrix construction")
    parser.add_argument("--kind", choices=["exp", "geom"], default="geom", help="Kernel type")
    parser.add_argument("--lmbd", type=float, default=0.01, help="Lambda coefficient")
    # parser.add_argument("--n-samples-mc", type=int, default=None, help="MC samples (default: 10 * n_nodes)")
    # parser.add_argument("--n-samples-gvoys", type=int, default=None, help="GVoy samples (default: 10 * n_nodes)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dataset_name", choices=["ENZYMES", "MUTAG", "NCI1", "PTC-MR", "D&D", "PROTEINS", "AIDS"], default="MUTAG", help="Graph generator")
    # parser.add_argument("--p-halt", type=float, default=0.2, help="Halt probability for GVoy")
    # parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    kernel_kind = args.kind
    dataset_name = args.dataset_name
    lmbd = args.lmbd
    seed = args.seed

    dataset, labels, mean_nodes = load_tudataset_with_labels(dataset_name)

    n_samples_mc = 10 * mean_nodes
    n_samples_gvoys = 10 * mean_nodes
    
    folder = f"./results/{dataset_name}"
    os.makedirs(folder, exist_ok=True)
    
    mu_func = utils.mu_func_gen(kernel_kind, lmbd=lmbd)

    # Redefinition of globals for gvoys
    LAMBDA_COEFF = lmbd
    P_HALT = 0.2

    results, gram_mtx = bench(
        dataset=dataset,
        labels=labels,
        kind=kernel_kind,
        mu_func=mu_func,
        n_samples_mc=n_samples_mc,
        n_samples_gvoys=n_samples_gvoys,
        seed=seed,
    )

    results_output = f"{folder}/{kernel_kind} | seed={seed}.json"
    with open(results_output, "w") as f:        
        json.dump(results, f, indent=2, default=str)
    
    gram_output = f"{folder}/{kernel_kind} | seed={seed}.pickle"
    with open(gram_output, 'wb') as f:
        pickle.dump(gram_mtx, f)

    print(f"Results saved to {results_output} and {gram_output}")