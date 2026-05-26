"""Benchmark graph kernel methods on TU datasets (from www.chrsmrrs.com/graphkerneldatasets).

For each dataset and each requested kernel method, the script:
  1. Downloads and loads the dataset (optionally with edge labels).
  2. Optionally filters graphs by maximum nodes and maximum number of graphs.
  3. Builds normalized adjacency matrices and starting/stopping distributions.
  4. Computes the Gram matrix (with optional normalisation).
  5. Performs stratified SVM cross‑validation to measure classification accuracy.
  6. Records timings, accuracy, and relative matrix errors (w.r.t. direct or CG).
  7. Saves per‑dataset results to JSON files.
"""

import argparse
import json
import os
import time
import pickle
import ssl
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import networkx as nx
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.svm import SVC

from src import utils
from src import gram
from src import gvoys
from src import mcrwk


# ----------------------------------------------------------------------
#  TU dataset helpers
# ----------------------------------------------------------------------
def _tu_dataset_url(dataset_name):
    return f"https://www.chrsmrrs.com/graphkerneldatasets/{dataset_name}.zip"


def _read_int_list(path):
    values = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                values.append(int(line))
    return values


def _read_edge_list(path):
    edges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                continue
            edges.append((int(parts[0]), int(parts[1])))
    return edges


def download_tu_dataset(dataset_name, root_dir="data/tu_datasets", force_download=False):
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    dataset_dir = root / dataset_name
    indicator_path = dataset_dir / f"{dataset_name}_graph_indicator.txt"
    edges_path = dataset_dir / f"{dataset_name}_A.txt"
    labels_path = dataset_dir / f"{dataset_name}_graph_labels.txt"

    if (not force_download) and indicator_path.exists() and edges_path.exists() and labels_path.exists():
        return dataset_dir

    zip_path = root / f"{dataset_name}.zip"
    if force_download or (not zip_path.exists()):
        url = _tu_dataset_url(dataset_name)
        print(f"downloading {dataset_name} from {url}")
        ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        with urlopen(url, context=ctx) as resp, open(zip_path, "wb") as out:
            out.write(resp.read())

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)

    if not (indicator_path.exists() and edges_path.exists() and labels_path.exists()):
        raise FileNotFoundError(f"dataset files for {dataset_name} were not found after extraction")

    return dataset_dir


def load_tu_dataset(dataset_name, root_dir="data/tu_datasets", force_download=False, load_edge_labels=False):
    """
    Load a TU dataset as a list of networkx.Graph objects.
    Returns:
        graphs : list of nx.Graph
        y : integer graph labels
        graph_labels_raw : raw labels from file
        edge_labels_used : bool (True if edge labels were actually loaded and attached)

    If load_edge_labels is True, it tries to read _edge_labels.txt.
    If that file exists, each edge gets an attribute 'label' (int).
    If it does not exist, edge_labels_used will be False and edges will have no label.
    Vertex labels (if any) are ignored entirely.
    """
    dataset_dir = download_tu_dataset(dataset_name, root_dir=root_dir, force_download=force_download)

    indicator_path = dataset_dir / f"{dataset_name}_graph_indicator.txt"
    edges_path = dataset_dir / f"{dataset_name}_A.txt"
    graph_labels_path = dataset_dir / f"{dataset_name}_graph_labels.txt"
    edge_labels_path = dataset_dir / f"{dataset_name}_edge_labels.txt"

    graph_indicator = _read_int_list(indicator_path)
    graph_labels_raw = np.asarray(_read_int_list(graph_labels_path), dtype=int)

    n_graphs = max(graph_indicator)
    graphs = [nx.Graph() for _ in range(n_graphs)]

    # Add nodes without any label (vertex labels are ignored)
    for global_node_id, graph_id in enumerate(graph_indicator, start=1):
        graphs[graph_id - 1].add_node(global_node_id)

    # Add edges (without labels first)
    edges = _read_edge_list(edges_path)
    for u, v in edges:
        gu = graph_indicator[u - 1] - 1
        gv = graph_indicator[v - 1] - 1
        if gu != gv:
            continue
        graphs[gu].add_edge(u, v)

    edge_labels_used = False
    if load_edge_labels and edge_labels_path.exists():
        edge_labels = _read_int_list(edge_labels_path)
        # The edge_labels file has one label per line, in the same order as edges in _A.txt.
        # We assume the edges list we read has the same order.
        for idx, (u, v) in enumerate(edges):
            gu = graph_indicator[u - 1] - 1
            gv = graph_indicator[v - 1] - 1
            if gu != gv:
                continue
            label = edge_labels[idx] if idx < len(edge_labels) else 0
            # Attach the edge label as an integer attribute 'label'
            graphs[gu].edges[u, v]['label'] = label
        edge_labels_used = True
    elif load_edge_labels:
        print(f"  [info] Edge labels file not found for {dataset_name}, falling back to unlabeled.")

    _, y = np.unique(graph_labels_raw, return_inverse=True)
    return graphs, y, graph_labels_raw, edge_labels_used


def pick_graph_subset(graphs, y, max_graphs=None, max_nodes_per_graph=None, seed=42):
    """Select a class‑stratified subset of graphs that are small enough."""
    y = np.asarray(y)
    idx = np.arange(len(graphs))

    if max_nodes_per_graph is not None:
        idx = np.array([i for i in idx if graphs[i].number_of_nodes() <= max_nodes_per_graph], dtype=int)

    if idx.size == 0:
        raise ValueError("no graphs left after max_nodes_per_graph filter")

    graphs_f = [graphs[i] for i in idx]
    y_f = y[idx]

    classes, counts = np.unique(y_f, return_counts=True)
    if classes.size < 2:
        raise ValueError("need at least 2 classes")
    if counts.min() < 2:
        raise ValueError("each class must have at least 2 graphs after filtering")

    if (max_graphs is None) or (max_graphs >= len(graphs_f)):
        return graphs_f, y_f, idx

    if max_graphs < classes.size:
        raise ValueError(f"max_graphs={max_graphs} must be >= number of classes={classes.size}")

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=max_graphs, random_state=seed)
    keep_local_idx, _ = next(splitter.split(np.zeros(len(y_f)), y_f))
    keep_local_idx = np.sort(keep_local_idx)

    graphs_sub = [graphs_f[i] for i in keep_local_idx]
    y_sub = y_f[keep_local_idx]
    idx_sub = idx[keep_local_idx]
    return graphs_sub, y_sub, idx_sub


def build_rw_inputs(graphs, distribution_func="uniform", labeled=False):
    """Convert networkx graphs to adjacency matrices and distributions."""
    Ps, vs, ws = [], [], []
    for g in graphs:
        if labeled:
            # uses edge labels (attribute 'label' on edges)
            P = utils.normalized_adj_matrix_labeled(g)
        else:
            P = utils.normalized_adj_matrix(g)
        # n = P.shape[0]
        n = g.number_of_nodes() 
        Ps.append(P)
        if distribution_func == "uniform":
            vs.append(utils.uniform_dist(n))
            ws.append(utils.uniform_dist(n))
        elif distribution_func == "normal":
            vs.append(utils.random_dist(n))
            ws.append(utils.random_dist(n))
        else:
            raise ValueError(f"Unknown distribution: {distribution_func}")
    return Ps, vs, ws


# ----------------------------------------------------------------------
#  SVM evaluation helpers
# ----------------------------------------------------------------------
def normalize_gram_matrix(K, eps=1e-12):
    d = np.sqrt(np.clip(np.diag(K), eps, None))
    return K / np.outer(d, d)


def _safe_n_splits(y, desired_splits):
    _, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min())
    if min_count < 2:
        return 1
    return max(2, min(desired_splits, min_count))


def _select_best_c_precomputed(K_train, y_train, c_values, inner_splits=3, seed=42):
    inner_n_splits = _safe_n_splits(y_train, inner_splits)
    if inner_n_splits < 2:
        return c_values[0]

    inner_cv = StratifiedKFold(n_splits=inner_n_splits, shuffle=True, random_state=seed)
    best_c = c_values[0]
    best_score = -np.inf

    for c in c_values:
        fold_scores = []
        for inner_tr_idx, inner_va_idx in inner_cv.split(np.zeros(len(y_train)), y_train):
            K_inner_tr = K_train[np.ix_(inner_tr_idx, inner_tr_idx)]
            K_inner_va = K_train[np.ix_(inner_va_idx, inner_tr_idx)]
            y_inner_tr = y_train[inner_tr_idx]
            y_inner_va = y_train[inner_va_idx]

            clf = SVC(C=c, kernel="precomputed")
            clf.fit(K_inner_tr, y_inner_tr)
            pred = clf.predict(K_inner_va)
            fold_scores.append(accuracy_score(y_inner_va, pred))

        score = float(np.mean(fold_scores))
        if score > best_score:
            best_score = score
            best_c = c

    return best_c


def evaluate_svm_precomputed(K, y, c_values=None, n_splits=5, n_repeats=1, inner_splits=3, seed=42):
    if c_values is None:
        c_values = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

    y = np.asarray(y)
    outer_n_splits = _safe_n_splits(y, n_splits)
    if outer_n_splits < 2:
        raise ValueError("not enough samples per class for CV")

    rng = np.random.default_rng(seed)
    scores = []
    selected_cs = []

    for _ in range(n_repeats):
        cv_seed = int(rng.integers(0, 2**31 - 1))
        outer_cv = StratifiedKFold(n_splits=outer_n_splits, shuffle=True, random_state=cv_seed)

        for fold_id, (train_idx, test_idx) in enumerate(outer_cv.split(np.zeros(len(y)), y), start=1):
            K_train = K[np.ix_(train_idx, train_idx)]
            K_test = K[np.ix_(test_idx, train_idx)]
            y_train = y[train_idx]
            y_test = y[test_idx]

            best_c = _select_best_c_precomputed(
                K_train,
                y_train,
                c_values=c_values,
                inner_splits=inner_splits,
                seed=cv_seed + fold_id,
            )
            selected_cs.append(float(best_c))

            clf = SVC(C=best_c, kernel="precomputed")
            clf.fit(K_train, y_train)
            pred = clf.predict(K_test)
            scores.append(accuracy_score(y_test, pred))

    c_mode = max(set(selected_cs), key=selected_cs.count)
    return {
        "mean_accuracy": float(np.mean(scores)),
        "std_accuracy": float(np.std(scores)),
        "scores": scores,
        "selected_cs": selected_cs,
        "selected_c_mean": float(np.mean(selected_cs)),
        "selected_c_mode": float(c_mode),
    }


# ----------------------------------------------------------------------
#  Kernel matrix computation
# ----------------------------------------------------------------------
def compute_kernel_matrix(method, Ps, vs, ws, kind, mu_func, lmbd,
                           n_samples_mc, n_samples_gvoys, seed, labeled=False,
                           block_size=10):
    """Dispatch to the correct Gram‑matrix constructor."""
    if method == "direct":
        return gram.gram_direct(Ps, vs, ws, mu_func, kind, labeled=labeled)

    elif method == "cg":
        if kind != "geom":
            raise ValueError("CG only supported for geom kernel")
        return gram.gram_cg(Ps, vs, ws, mu_func, labeled=labeled)

    elif method == "sylvester":
        if kind != "geom" or labeled:
            raise ValueError("Sylvester only for unlabeled geom kernel")
        return gram.gram_sylvester(Ps, vs, ws, mu_func)

    elif method == "fixed_point":
        if kind != "geom":
            raise ValueError("fixed_point only supported for geom kernel")
        return gram.gram_fixed_point(Ps, vs, ws, mu_func, labeled=labeled)

    elif method == "gvoys":
        np.random.seed(seed)
        if labeled:
            return gvoys.random_walk_kernel_gvoys_labeled_dataset(
                Ps, vs, ws, anchor_fraction=1.0, kind=kind,
                lambda_coeff=lmbd, p_halt=P_HALT, nb_random_walks=n_samples_gvoys
            )
        else:
            return gvoys.random_walk_kernel_gvoys_dataset(
                Ps, vs, ws, anchor_fraction=1.0, kind=kind,
                lambda_coeff=lmbd, p_halt=P_HALT, nb_random_walks=n_samples_gvoys
            )

    elif method == "mc":
        if labeled:
            return mcrwk.random_walk_kernel_mc_labeled_dataset(
                Ps, vs, ws, mu_func=mu_func, kind=kind,
                n_length_samples=n_samples_mc // 100,
                n_label_samples_per_length=100,
                seed=seed
            )
        else:
            return mcrwk.random_walk_kernel_mc_dataset(
                Ps, vs, ws, mu_func=mu_func, kind=kind,
                n_samples=n_samples_mc, seed=seed
            )

    else:
        raise ValueError(f"Unknown method: {method}")


# ----------------------------------------------------------------------
#  Main benchmark routine
# ----------------------------------------------------------------------
def run_tu_benchmark(
    dataset_names,
    kind,
    methods,
    max_graphs,
    max_nodes_per_graph,
    n_samples_mc,
    n_samples_gvoys,
    c_values,
    n_splits,
    n_repeats,
    inner_splits,
    normalize_kernel,
    root_dir,
    seed,
    distribution_func="uniform",
    output_dir="results",
    save_grams=False,
    request_edge_labels=False,
):
    """Run benchmark and save per‑dataset JSON immediately."""
    for di, dataset_name in enumerate(dataset_names):
        ds_seed = seed + di
        print(f"\n[{dataset_name}] Loading …")

        # Default suffix: assume unlabeled; will be refined after load
        label_suffix = "_labeled" if request_edge_labels else "_unlabeled"

        try:
            graphs_all, y_all, _, edge_labels_used = load_tu_dataset(
                dataset_name, root_dir=root_dir,
                force_download=False,
                load_edge_labels=request_edge_labels,
            )
            # Refine suffix with actual edge label availability
            label_suffix = "_labeled" if (request_edge_labels and edge_labels_used) else "_unlabeled"
        except Exception as e:
            print(f"[{dataset_name}] SKIP: could not load ({e})")
            row = {
                "dataset": dataset_name,
                "n_total": 0,
                "n_used": 0,
                "error": str(e),
                "methods": {},
                "edge_labels_used": False,
            }
            ds_out = os.path.join(output_dir, dataset_name)
            os.makedirs(ds_out, exist_ok=True)
            with open(os.path.join(ds_out, f"seed={ds_seed}_{kind}_{distribution_func}{label_suffix}.json"), "w") as f:
                json.dump(row, f, indent=2, default=str)
            continue

        n_total = len(graphs_all)

        try:
            graphs, y, keep_idx = pick_graph_subset(
                graphs_all, y_all,
                max_graphs=max_graphs,
                max_nodes_per_graph=max_nodes_per_graph,
                seed=ds_seed,
            )
        except Exception as e:
            print(f"[{dataset_name}] SKIP: {e}")
            row = {
                "dataset": dataset_name,
                "n_total": int(n_total),
                "n_used": 0,
                "error": str(e),
                "methods": {},
                "edge_labels_used": edge_labels_used,
            }
            ds_out = os.path.join(output_dir, dataset_name)
            os.makedirs(ds_out, exist_ok=True)
            with open(os.path.join(ds_out, f"seed={ds_seed}_{kind}_{distribution_func}{label_suffix}.json"), "w") as f:
                json.dump(row, f, indent=2, default=str)
            continue

        print(f"[{dataset_name}] Using {len(graphs)}/{n_total} graphs")
        use_labeled = request_edge_labels and edge_labels_used
        if request_edge_labels and not edge_labels_used:
            print("  [info] Edge labels not available, running unlabeled kernels only.")

        Ps, vs, ws = build_rw_inputs(graphs, distribution_func, labeled=use_labeled)

        max_n = max(g.number_of_nodes() for g in graphs) if graphs else 0

        # Drop methods too heavy for large graphs
        effective_methods = list(methods)
        if "direct" in effective_methods and max_n > 128:
            print("  [info] skipping direct (max node > 128)")
            effective_methods.remove("direct")
        if "sylvester" in effective_methods and max_n > 512:
            print("  [info] skipping Sylvester (max node > 512)")
            effective_methods.remove("sylvester")

        row = {
            "dataset": dataset_name,
            "n_total": int(n_total),
            "n_used": int(len(graphs)),
            "max_n": int(max_n),
            "edge_labels_used": use_labeled,
            "methods": {},
        }

        # Mark skipped methods
        for m in methods:
            if m not in effective_methods:
                row["methods"][m] = {
                    "mean_accuracy": None,
                    "std_accuracy": None,
                    "gram_time_sec": None,
                    "svm_time_sec": None,
                    "selected_c_mean": None,
                    "selected_c_mode": None,
                    "rel_error": None,
                    "error": "skipped (graph too large for this method)",
                }

        # Compute lambda and mu_func
        try:
            d_max = max(max(d for _, d in G.degree()) for G in graphs)
            lmbd = 1 / (d_max ** 2)
            mu_func = utils.mu_func_gen(kind, lmbd=lmbd)
        except Exception as e:
            row["error"] = f"Degree/lambda computation failed: {e}"
            ds_out = os.path.join(output_dir, dataset_name)
            os.makedirs(ds_out, exist_ok=True)
            with open(os.path.join(ds_out, f"seed={ds_seed}_{kind}_{distribution_func}{label_suffix}.json"), "w") as f:
                json.dump(row, f, indent=2, default=str)
            continue

        # ---------- Compute Gram matrices and timings ----------
        gram_matrices = {}
        timing = {}
        for mi, method in enumerate(effective_methods):
            method_seed = ds_seed + 100 * (mi + 1)
            try:
                print(f"  Computing {method} kernel …")
                t0 = time.perf_counter()
                K = compute_kernel_matrix(
                    method, Ps, vs, ws,
                    kind=kind,
                    mu_func=mu_func,
                    lmbd=lmbd,
                    n_samples_mc=n_samples_mc,
                    n_samples_gvoys=n_samples_gvoys,
                    seed=method_seed,
                    labeled=use_labeled,
                )
                t1 = time.perf_counter()
                gram_matrices[method] = K
                timing[method] = t1 - t0
            except Exception as e:
                print(f"  {method}: FAILED → {e}")
                row["methods"][method] = {
                    "mean_accuracy": None,
                    "std_accuracy": None,
                    "gram_time_sec": None,
                    "svm_time_sec": None,
                    "selected_c_mean": None,
                    "selected_c_mode": None,
                    "rel_error": None,
                    "error": str(e),
                }

        # Determine reference matrix (direct > cg)
        reference_matrix = None
        reference_method = None
        if "direct" in gram_matrices:
            reference_matrix = gram_matrices["direct"]
            reference_method = "direct"
        elif "cg" in gram_matrices:
            reference_matrix = gram_matrices["cg"]
            reference_method = "cg"

        if reference_matrix is None:
            print("  [warning] No reference matrix (direct or CG) available. Errors will not be computed.")

        # Evaluate SVM and compute errors for all methods that succeeded
        for method, K in gram_matrices.items():
            # Compute error relative to reference (if reference exists)
            if reference_matrix is not None:
                if method == reference_method:
                    err_dict = {"mean_abs": 0.0, "mean_rel": 0.0, "max_abs": 0.0, "max_rel": 0.0}
                else:
                    err_dict = gram.matrix_errors(reference_matrix, K)
            else:
                err_dict = None

            # Normalize if needed and evaluate SVM
            try:
                K_norm = normalize_gram_matrix(K) if normalize_kernel else K
                t0 = time.perf_counter()
                stats = evaluate_svm_precomputed(
                    K_norm, y,
                    c_values=c_values,
                    n_splits=n_splits,
                    n_repeats=n_repeats,
                    inner_splits=inner_splits,
                    seed=ds_seed,
                )
                svm_time = time.perf_counter() - t0

                row["methods"][method] = {
                    "mean_accuracy": stats["mean_accuracy"],
                    "std_accuracy": stats["std_accuracy"],
                    "gram_time_sec": float(timing[method]),
                    "svm_time_sec": float(svm_time),
                    "selected_c_mean": stats["selected_c_mean"],
                    "selected_c_mode": stats["selected_c_mode"],
                    "rel_error": err_dict,
                    "error": None,
                }

                print(f"    acc = {100.0*stats['mean_accuracy']:.2f}% "
                      f"± {100.0*stats['std_accuracy']:.2f}% "
                      f"(C_mode={stats['selected_c_mode']:.3g}) "
                      f"gram {timing[method]:.2f}s, svm {svm_time:.2f}s")
                if err_dict is not None:
                    print(f"    rel_error: mean_abs={err_dict['mean_abs']:.2e} mean_rel={err_dict['mean_rel']:.2e}")

                if save_grams:
                    gram_file = os.path.join(
                        output_dir, dataset_name,
                        f"{method}_seed={ds_seed + 100 * (list(effective_methods).index(method) + 1)}.pickle"
                    )
                    os.makedirs(os.path.dirname(gram_file), exist_ok=True)
                    with open(gram_file, "wb") as f:
                        pickle.dump(K_norm if normalize_kernel else K, f)

            except Exception as e:
                row["methods"][method] = {
                    "mean_accuracy": None,
                    "std_accuracy": None,
                    "gram_time_sec": float(timing[method]),
                    "svm_time_sec": None,
                    "selected_c_mean": None,
                    "selected_c_mode": None,
                    "rel_error": err_dict,
                    "error": f"SVM evaluation failed: {e}",
                }
                print(f"  {method}: SVM evaluation FAILED → {e}")

        # Save per‑dataset JSON (with kernel kind in filename)
        ds_out = os.path.join(output_dir, dataset_name)
        os.makedirs(ds_out, exist_ok=True)
        ds_file = os.path.join(ds_out, f"seed={ds_seed}_{kind}_{distribution_func}{label_suffix}.json")
        with open(ds_file, "w") as f:
            json.dump(row, f, indent=2, default=str)
        print(f"  → saved {ds_file}")



# ----------------------------------------------------------------------
#  CLI
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark graph kernels on TU datasets with SVM classification"
    )
    parser.add_argument("--datasets", nargs="+",
                        default=["MUTAG", "ENZYMES", "NCI1", "PTC_MR", "DD", "PROTEINS", "AIDS"],
                        help="TU dataset names")
    parser.add_argument("--kind", choices=["exp", "geom"], default="geom",
                        help="Kernel type (exp or geom)")
    parser.add_argument("--methods", nargs="+",
                        default=["direct", "cg", "fixed_point", "gvoys", "mc"],
                        help="Kernel construction methods to benchmark")
    parser.add_argument("--max_graphs", type=int, default=None,
                        help="Max graphs per dataset (after node filter)")
    parser.add_argument("--max_nodes_per_graph", type=int, default=None,
                        help="Skip graphs larger than this many nodes")
    parser.add_argument("--n_samples_mc", type=int, default=200,
                        help="MC samples per node (total = n_nodes * this)")
    parser.add_argument("--n_samples_gvoys", type=int, default=200,
                        help="Number of random walks for GVoys")
    parser.add_argument("--c_values", type=float, nargs="+",
                        default=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0],
                        help="SVM C values to try")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Outer CV folds")
    parser.add_argument("--n_repeats", type=int, default=1,
                        help="Number of CV repeats")
    parser.add_argument("--inner_splits", type=int, default=3,
                        help="Inner CV folds for C selection")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Do not normalise the Gram matrix")
    parser.add_argument("--root_dir", default="tu_datasets",
                        help="Directory to store downloaded datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--u_w_distribution", choices=["uniform", "normal"],
                        default="uniform",
                        help="Starting/stopping distribution")
    parser.add_argument("--output_dir", default="./results/tu_benchmark",
                        help="Folder for results")
    parser.add_argument("--save_grams", action="store_true",
                        help="Save Gram matrices as pickle files")
    parser.add_argument("--labeled", type=int, default=0, choices=[0, 1],
                    help="Request edge‑labeled kernels: 1 = yes, 0 = no (default: 0)")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Optional suffix for the output file (currently unused)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Remove methods incompatible with 'exp' kernel (if needed)
    if args.kind != "geom":
        invalid = {"cg", "sylvester", "fixed_point"}
        args.methods = [m for m in args.methods if m not in invalid]
        print("Note: CG, Sylvester, fixed_point require geom kernel – removed.")

    normalize_kernel = not args.no_normalize
    global P_HALT
    P_HALT = 0.2   # must be defined globally for compute_kernel_matrix

    print("Benchmark configuration:")
    print(f"  datasets: {args.datasets}")
    print(f"  kind: {args.kind}")
    print(f"  methods: {args.methods}")
    print(f"  max_graphs: {args.max_graphs}")
    print(f"  max_nodes_per_graph: {args.max_nodes_per_graph}")
    print(f"  n_samples_mc: {args.n_samples_mc}")
    print(f"  n_samples_gvoys: {args.n_samples_gvoys}")
    print(f"  normalize: {normalize_kernel}")
    print(f"  request_edge_labels: {args.labeled}")
    print(f"  u_w_distribution: {args.u_w_distribution}")
    print(f"  output: {args.output_dir}")

    run_tu_benchmark(
        dataset_names=args.datasets,
        kind=args.kind,
        methods=args.methods,
        max_graphs=args.max_graphs,
        max_nodes_per_graph=args.max_nodes_per_graph,
        n_samples_mc=args.n_samples_mc,
        n_samples_gvoys=args.n_samples_gvoys,
        c_values=args.c_values,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        inner_splits=args.inner_splits,
        normalize_kernel=normalize_kernel,
        root_dir=args.root_dir,
        seed=args.seed,
        distribution_func=args.u_w_distribution,
        output_dir=args.output_dir,
        save_grams=args.save_grams,
        request_edge_labels=bool(args.labeled),
    )