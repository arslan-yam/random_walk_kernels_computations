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
import time
import ssl
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import networkx as nx
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.svm import SVC

from src import gram
from datetime import datetime, timezone
from src.benchmark import (KernelConfig, METHODS, add_kernel_arguments, build_inputs,
    compute_kernel, config_from_args, method_seed, option, reference_method, runtime_metadata,
    DIRECT_NODE_LIMIT, CG_REFERENCE_MIN_NODES, planned_methods, method_skip_reason)


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
        with urlopen(url, context=ctx, timeout=120) as resp, open(zip_path, "wb") as out:
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
    if min(graph_indicator) < 1 or len(graph_labels_raw) != n_graphs:
        raise ValueError("graph indicators and graph-label count are inconsistent")
    graphs = [nx.Graph() for _ in range(n_graphs)]

    # Add nodes without any label (vertex labels are ignored)
    for global_node_id, graph_id in enumerate(graph_indicator, start=1):
        graphs[graph_id - 1].add_node(global_node_id)

    # Add edges (without labels first)
    edges = _read_edge_list(edges_path)
    for u, v in edges:
        if not (1 <= u <= len(graph_indicator) and 1 <= v <= len(graph_indicator)):
            raise ValueError("edge references an invalid node id")
        gu = graph_indicator[u - 1] - 1
        gv = graph_indicator[v - 1] - 1
        if gu != gv:
            raise ValueError("edge connects different TU graphs")
        graphs[gu].add_edge(u, v)

    edge_labels_used = False
    if load_edge_labels and edge_labels_path.exists():
        edge_labels = _read_int_list(edge_labels_path)
        if len(edge_labels) != len(edges):
            raise ValueError("edge-label count does not match the edge list")
        # The edge_labels file has one label per line, in the same order as edges in _A.txt.
        # We assume the edges list we read has the same order.
        for idx, (u, v) in enumerate(edges):
            gu = graph_indicator[u - 1] - 1
            gv = graph_indicator[v - 1] - 1
            if gu != gv:
                continue
            label = edge_labels[idx]
            previous = graphs[gu].edges[u, v].get("label")
            if previous is not None and previous != label:
                raise ValueError("duplicate edge has conflicting labels")
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


def build_rw_inputs(graphs, distribution_func="uniform", labeled=False, seed=42):
    """Compatibility wrapper with an explicit RNG seed."""
    return build_inputs(graphs, distribution_func, labeled, seed)


# ----------------------------------------------------------------------
#  SVM evaluation helpers
# ----------------------------------------------------------------------
def normalize_gram_matrix(K, eps=1e-12):
    """Normalize a positive diagonal; legacy eps no longer clips invalid values."""
    if not np.isfinite(K).all() or np.any(np.diag(K) <= 0):
        raise ValueError("Gram normalization requires finite values and positive diagonal; use --no-normalize or a larger budget")
    d = np.sqrt(np.diag(K))
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

    c_mode = max(sorted(set(selected_cs)), key=selected_cs.count)
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
        n_samples_mc, n_samples_gvoys, seed, labeled=False, block_size=64, **kwargs):
    """Legacy dispatcher; defaults now use unbiased replicas and explicit budgets."""
    config = KernelConfig(kind=kind,lmbd=lmbd,n_samples_mc=n_samples_mc,
        n_samples_gvoys=n_samples_gvoys,block_size=block_size,**kwargs)
    return compute_kernel(method,Ps,vs,ws,config,seed,labeled)


def run_tu_benchmark(dataset_names,kind,methods,max_graphs,max_nodes_per_graph,
        n_samples_mc,n_samples_gvoys,c_values,n_splits,n_repeats,inner_splits,
        normalize_kernel,root_dir,seed,distribution_func="uniform",
        output_dir="results/tu_benchmark",save_grams=False,request_edge_labels=False,
        *,config=None,cli_args=None,skip_svm=False,check_psd=False,fail_fast=False,
        direct_max_nodes=128,sylvester_max_nodes=512,experiment_name="run"):
    """Save raw errors and optional SVM metrics per dataset, without PSD projection.

    Failures are recorded and the next method/dataset continues unless fail_fast.
    Returned status is nonzero if any requested compatible method failed.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    failures = 0
    for di,name in enumerate(dataset_names):
        ds_seed = seed+di
        output = Path(output_dir)/name/f"{experiment_name}_{kind}_{distribution_func}_seed={ds_seed}_{stamp}.json"
        output.parent.mkdir(parents=True,exist_ok=True)
        row = {"dataset":name,"methods":{},"metadata":runtime_metadata()}
        row["metadata"].update({"seed":ds_seed,"distribution":distribution_func,
            "normalize_for_svm":normalize_kernel,"skip_svm":skip_svm,
            "cv":{"n_splits":n_splits,"n_repeats":n_repeats,"inner_splits":inner_splits,"c_values":c_values}})
        if cli_args is not None:
            row["metadata"]["cli"] = vars(cli_args)
        try:
            graphs_all,y_all,_,edge_labels = load_tu_dataset(name,root_dir=root_dir,load_edge_labels=request_edge_labels)
            graphs,y,indices = pick_graph_subset(graphs_all,y_all,max_graphs,max_nodes_per_graph,ds_seed)
            labeled = bool(request_edge_labels and edge_labels)
            effective_config = config_from_args(cli_args,graphs) if cli_args is not None else (config or KernelConfig(
                kind=kind,n_samples_mc=n_samples_mc,n_samples_gvoys=n_samples_gvoys))
            effective_config.validate()
            row.update({"n_total":len(graphs_all),"n_used":len(graphs),
                        "max_n":max(map(len,graphs)),"edge_labels_used":labeled})
            row["metadata"].update({"kernel":effective_config.metadata(),"selected_indices":indices.tolist(),
                "node_counts":[len(g) for g in graphs],"edge_counts":[g.number_of_edges() for g in graphs]})
            t0 = time.perf_counter()
            Ps,vs,ws = build_rw_inputs(graphs,distribution_func,labeled,ds_seed)
            row["metadata"]["input_preparation_sec"] = time.perf_counter()-t0
            matrices = {}
            planned = planned_methods(methods, kind=effective_config.kind, max_nodes=row["max_n"])
            row["metadata"].update({"requested_methods":list(methods),"planned_methods":planned,
                "direct_max_nodes":min(direct_max_nodes,DIRECT_NODE_LIMIT),
                "cg_reference_min_nodes":CG_REFERENCE_MIN_NODES})
            for method in planned:
                reason = method_skip_reason(method, kind=effective_config.kind, labeled=labeled,
                    max_nodes=row["max_n"], direct_max_nodes=direct_max_nodes,
                    sylvester_max_nodes=sylvester_max_nodes)
                if reason:
                    row["methods"][method] = {"status":"skipped","error":reason}
                    continue
                print(f"[{name}] Computing {method} ...",flush=True)
                t0 = time.perf_counter()
                try:
                    matrices[method] = compute_kernel(method,Ps,vs,ws,effective_config,
                        method_seed(ds_seed,method),labeled)
                    row["methods"][method] = {"status":"ok","gram_time_sec":time.perf_counter()-t0,
                        "seed":method_seed(ds_seed,method),"error":None}
                except Exception as exc:
                    row["methods"][method] = {"status":"failed","gram_time_sec":time.perf_counter()-t0,"error":str(exc)}
                    failures += 1
                    if fail_fast:
                        raise
            ref = reference_method(matrices, kind=effective_config.kind, max_nodes=row["max_n"])
            row["metadata"]["reference_method"] = ref
            for method,K in matrices.items():
                entry = row["methods"][method]
                entry["rel_error"] = gram.matrix_errors(matrices[ref],K) if ref else None
                entry["diagonal_min"] = float(np.min(np.diag(K)))
                if check_psd:
                    eigenvalues = np.linalg.eigvalsh((K+K.T)*0.5)
                    entry["min_eigenvalue"] = float(eigenvalues[0])
                    tolerance = 1e-10 * np.max(np.abs(eigenvalues))
                    entry["psd_tolerance"] = float(tolerance)
                    entry["negative_eigenvalues"] = int(np.sum(eigenvalues < -tolerance))
                if save_grams:
                    np.savez(output.with_name(output.stem+f"_{method}.npz"),raw=K,indices=indices,y=y)
                if not skip_svm:
                    t0 = time.perf_counter()
                    try:
                        evaluated = normalize_gram_matrix(K) if normalize_kernel else K
                        stats = evaluate_svm_precomputed(evaluated,y,c_values,n_splits,n_repeats,inner_splits,ds_seed)
                        entry.update(stats)
                        entry["svm_time_sec"] = time.perf_counter()-t0
                    except Exception as exc:
                        entry.update({"status":"failed","error":f"SVM: {exc}"})
                        failures += 1
                        if fail_fast:
                            raise
        except Exception as exc:
            row["error"] = str(exc)
            failures += 1
            if fail_fast:
                output.write_text(json.dumps(row,indent=2,allow_nan=False)+"\n")
                raise
        output.write_text(json.dumps(row,indent=2,allow_nan=False)+"\n")
        print(f"Saved {output}",flush=True)
    return int(failures>0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="TU RWK benchmark with optional nested SVM CV")
    add_kernel_arguments(parser)
    option(parser,"datasets",nargs="+",default=["MUTAG","ENZYMES","NCI1","PTC_MR","DD","PROTEINS","AIDS"])
    option(parser,"methods",nargs="+",choices=METHODS,default=["direct","cg","fixed_point","series","gvoys","mc"])
    option(parser,"max_graphs",type=int)
    option(parser,"max_nodes_per_graph",type=int)
    option(parser,"c_values",type=float,nargs="+",default=[1e-3,1e-2,1e-1,1.,10.,100.])
    option(parser,"n_splits",type=int,default=5)
    option(parser,"n_repeats",type=int,default=1)
    option(parser,"inner_splits",type=int,default=3)
    option(parser,"no_normalize",action="store_true")
    option(parser,"root_dir",default="tu_datasets")
    option(parser,"output_dir",default="results/tu_benchmark")
    option(parser,"save_grams",action="store_true",help="Save RAW matrices and subset indices in NPZ files.")
    option(parser,"labeled",type=int,nargs="?",const=1,choices=[0,1],default=0)
    option(parser,"experiment_name",default="run")
    option(parser,"skip_svm",action="store_true",help="Measure kernel computation and approximation only.")
    option(parser,"check_psd",action="store_true",help="Compute eigenvalue diagnostics (cubic cost); never project.")
    option(parser,"fail_fast",action="store_true")
    option(parser,"direct_max_nodes",type=int,default=DIRECT_NODE_LIMIT,
           help="Skip direct at or above this vertex count (0..128); 0 disables direct.")
    option(parser,"sylvester_max_nodes",type=int,default=512)
    args = parser.parse_args(argv)
    if args.seed<0 or args.n_splits<2 or args.inner_splits<2 or args.n_repeats<1 or any(c<=0 for c in args.c_values):
        parser.error("invalid seed or CV parameters")
    if not 0 <= args.direct_max_nodes <= DIRECT_NODE_LIMIT:
        parser.error("direct_max_nodes must be between 0 and 128")
    KernelConfig(**{key:getattr(args,key) for key in KernelConfig.__dataclass_fields__}).validate()
    return args


def main(argv=None):
    args = parse_args(argv)
    return run_tu_benchmark(args.datasets,args.kind,args.methods,args.max_graphs,
        args.max_nodes_per_graph,args.n_samples_mc,args.n_samples_gvoys,args.c_values,
        args.n_splits,args.n_repeats,args.inner_splits,not args.no_normalize,args.root_dir,
        args.seed,args.u_w_distribution,args.output_dir,args.save_grams,bool(args.labeled),
        cli_args=args,skip_svm=args.skip_svm,check_psd=args.check_psd,
        fail_fast=args.fail_fast,direct_max_nodes=args.direct_max_nodes,
        sylvester_max_nodes=args.sylvester_max_nodes,experiment_name=args.experiment_name)


if __name__ == "__main__":
    raise SystemExit(main())
