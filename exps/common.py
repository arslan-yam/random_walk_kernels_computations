"""Shared helpers for the paper experiments in exps/.

Every experiment compares the same methods on shared inputs (P, v, w):
exact solvers (direct, cg, fixed_point, sylvester), GVoys and two MCRWK
budgets. ``mc_fixed`` runs MCRWK once per m in ``--mc-fixed-m`` regardless of
graph size. ``mc_matched`` uses m = c*N, the linear budget of GVoys; c is
either given with ``--mc-c`` or calibrated so that MCRWK takes as long as
GVoys on the same kind of inputs (both costs are linear, so one c keeps the
runtimes matched across sizes up to fixed overheads).
"""

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.svm import SVR

import dataset_bench as tu
from src import gram
from src.benchmark import (CG_REFERENCE_MIN_NODES, DIRECT_NODE_LIMIT, DISTRIBUTIONS,
    KernelConfig, build_inputs, compute_kernel, method_seed, method_skip_reason,
    option, reference_method, runtime_metadata)

EXACT_METHODS = ("direct", "cg", "fixed_point", "sylvester")
METHODS = EXACT_METHODS + ("gvoys", "mc_matched", "mc_fixed")
CASES = ("unlabeled", "labeled")
CLASSIFICATION_DATASETS = ["MUTAG", "ENZYMES", "PTC_MR", "AIDS", "NCI1"]
REGRESSION_DATASETS = ["ZINC_test"]
# Upper bound for calibrated budgets; protects against a runaway secant step.
MAX_CALIBRATED_M = 50_000_000


# ----------------------------------------------------------------------
#  CLI and configuration
# ----------------------------------------------------------------------
def add_kernel_arguments(parser, *, gvoys_samples, output_dir):
    option(parser,"output_dir",default=output_dir)
    option(parser,"cases",nargs="+",choices=CASES,default=list(CASES),
           help="Unlabeled and/or edge-labeled kernels; default both.")
    option(parser,"methods",nargs="+",choices=METHODS,default=list(METHODS),
           help="mc_matched: m=c*N; mc_fixed: one run per --mc-fixed-m. Sylvester runs only unlabeled.")
    option(parser,"kind",choices=["geom","exp"],default="geom")
    option(parser,"lmbd",type=float,default=0.7)
    option(parser,"u_w_distribution",choices=DISTRIBUTIONS,default="normal",
           help="Boundary vectors v and w; normal = normalized |N(0,1)| draws (avoids the uniform degenerate kernel).")
    option(parser,"n_samples_gvoys",type=int,default=gvoys_samples,
           help="GVoys outer samples; its cost is already linear in the vertex count.")
    option(parser,"p_halt",type=float,default=0.2)
    option(parser,"anchor_fraction",type=float,default=1.)
    option(parser,"block_size",type=int,default=64)
    option(parser,"mc_fixed_m",type=int,nargs="+",default=[100,1000,10000],
           help="MCRWK budgets used by mc_fixed, independent of graph size.")
    option(parser,"mc_c",type=float,help="c in m=c*N for mc_matched; calibrated against GVoys when omitted.")
    option(parser,"calibration_repeats",type=int,default=3,help="Timings per calibration point (median is used).")
    option(parser,"n_label_samples_per_length",type=int,default=1,
           help="Labeled MCRWK: label sequences per length; m counts all feature columns.")
    option(parser,"q_sampling_kind",choices=["uniform","norm_fro","norm_l1","random"],default="uniform")
    option(parser,"solver_tol",type=float,default=1e-10)
    option(parser,"max_iter",type=int,default=5000)
    option(parser,"direct_max_nodes",type=int,default=DIRECT_NODE_LIMIT,
           help="Skip direct at or above this vertex count (hard cap 128, as in src.benchmark).")
    option(parser,"sylvester_max_nodes",type=int,default=1024)
    option(parser,"seed",type=int,default=42)
    option(parser,"experiment_name",default="run")
    option(parser,"output",help="Exact JSON path; default is <output-dir>/<experiment-name>_<UTC stamp>.json.")
    option(parser,"fail_fast",action="store_true")
    option(parser,"json_gram_max_graphs",type=int,default=16,
           help="Store Gram matrices of at most this many graphs in the JSON records; 0 disables.")


def validate_kernel_arguments(parser, args):
    if args.seed < 0 or args.calibration_repeats < 1:
        parser.error("require nonnegative seed and calibration_repeats >= 1")
    if any(m < 1 for m in args.mc_fixed_m) or (args.mc_c is not None and args.mc_c <= 0):
        parser.error("MC budgets must be positive")
    if not 0 <= args.direct_max_nodes <= DIRECT_NODE_LIMIT:
        parser.error("direct_max_nodes must be between 0 and 128")
    try:
        kernel_config(args)
    except ValueError as exc:
        parser.error(str(exc))


def kernel_config(args, lmbd=None):
    return KernelConfig(kind=args.kind, lmbd=args.lmbd if lmbd is None else lmbd,
        n_samples_gvoys=args.n_samples_gvoys,
        n_label_samples_per_length=args.n_label_samples_per_length,
        q_sampling_kind=args.q_sampling_kind, p_halt=args.p_halt,
        anchor_fraction=args.anchor_fraction, block_size=args.block_size,
        solver_tol=args.solver_tol, max_iter=args.max_iter).validate()


def with_mc_budget(config, m):
    """m feature columns: m lengths unlabeled, m/n lengths x n label sequences labeled."""
    lengths = max(1, m // config.n_label_samples_per_length)
    return replace(config, n_samples_mc=lengths*config.n_label_samples_per_length,
                   n_length_samples=lengths).validate()


# ----------------------------------------------------------------------
#  Method runs
# ----------------------------------------------------------------------
def timed_kernel(method, Ps, vs, ws, config, seed, labeled):
    t0 = time.perf_counter()
    K = compute_kernel(method, Ps, vs, ws, config, seed, labeled)
    return K, time.perf_counter()-t0


def calibrate_mc_c(Ps, vs, ws, config, labeled, n_ref, seed, repeats=3, steps=3):
    """Return c such that MCRWK with m=c*n_ref takes as long as GVoys on these inputs.

    MC time is modelled as a+b*m and refitted after each probe; the final m is
    the fitted solution of a+b*m = GVoys time. Probes use medians of repeats.
    """
    def median_time(method, cfg):
        return float(np.median([timed_kernel(method, Ps, vs, ws, cfg, method_seed(seed+r, method), labeled)[1]
                                for r in range(repeats)]))

    t0 = time.perf_counter()
    target = median_time("gvoys", config)
    m, probes = max(1, round(n_ref)), []
    for _ in range(steps):
        probes.append((m, median_time("mc", with_mc_budget(config, m))))
        ms, ts = map(np.asarray, zip(*probes))
        slope, intercept = np.polyfit(ms, ts, 1) if len(set(ms)) > 1 else (0., 0.)
        # Fall back to proportional scaling until two distinct probes give a positive slope.
        estimate = (target-intercept)/slope if slope > 0 else m*target/ts[-1]
        m = int(min(max(1, round(estimate)), 100*m, MAX_CALIBRATED_M))
    return {"c": m/n_ref, "n_ref": float(n_ref), "m": m, "gvoys_time_sec": target,
            "mc_probes": [[int(a), float(b)] for a, b in probes],
            "n_samples_gvoys": config.n_samples_gvoys, "repeats": repeats,
            "calibration_time_sec": time.perf_counter()-t0}


def expand_methods(methods, n_ref, fixed_ms, c=None):
    """One run per method; mc_fixed expands to one run per budget."""
    runs = []
    for method in methods:
        if method == "mc_fixed":
            runs += [{"name": f"mc_m={m}", "method": "mc", "variant": "fixed", "m": m} for m in fixed_ms]
        elif method == "mc_matched":
            runs.append({"name": "mc_cN", "method": "mc", "variant": "matched",
                         "m": max(1, round(c*n_ref)), "c": c})
        else:
            runs.append({"name": method, "method": method, "variant": None})
    return runs


def compute_methods(runs, Ps, vs, ws, config, labeled, seed, *, max_nodes, args, tag=""):
    """Time each run on shared inputs and compare it with the reference Gram.

    The reference follows src.benchmark: direct below 128 vertices, otherwise
    CG, which is added (and marked reference_only) when it was not requested.
    """
    if config.kind == "geom" and max_nodes >= CG_REFERENCE_MIN_NODES and "cg" not in {r["method"] for r in runs}:
        runs = runs+[{"name": "cg", "method": "cg", "variant": None, "reference_only": True}]
    matrices, records = {}, []
    for run in runs:
        record = {**run, "status": "ok", "time_sec": None}
        reason = method_skip_reason(run["method"], kind=config.kind, labeled=labeled,
            max_nodes=max_nodes, direct_max_nodes=args.direct_max_nodes,
            sylvester_max_nodes=args.sylvester_max_nodes)
        if reason:
            records.append({**record, "status": "skipped", "reason": reason})
            continue
        cfg = with_mc_budget(config, run["m"]) if run["method"] == "mc" else config
        print(f"{tag} {run['name']} ...", end=" ", flush=True)
        try:
            matrices[run["name"]], record["time_sec"] = timed_kernel(
                run["method"], Ps, vs, ws, cfg, method_seed(seed, run["method"]), labeled)
            print(f"{record['time_sec']:.4g}s", flush=True)
        except Exception as exc:
            print(f"failed: {exc}", flush=True)
            record.update(status="failed", error=str(exc))
            if args.fail_fast:
                raise
        records.append(record)
    ref = reference_method(matrices, kind=config.kind, max_nodes=max_nodes)
    for record in records:
        K = matrices.get(record["name"])
        record["reference"] = ref
        record["errors"] = gram.matrix_errors(matrices[ref], K) if ref and K is not None else None
        if K is not None:
            record["diagonal_min"] = float(np.min(np.diag(K)))
            if len(K) <= args.json_gram_max_graphs:
                record["gram"] = K.tolist()
    return records, matrices


def graph_stats(graphs, labeled):
    nodes = [len(g) for g in graphs]
    edges = [g.number_of_edges() for g in graphs]
    stats = {"n_graphs": len(graphs), "mean_nodes": float(np.mean(nodes)), "max_nodes": max(nodes),
             "mean_edges": float(np.mean(edges)),
             "mean_degree": float(np.mean([2*e/n for n, e in zip(nodes, edges)]))}
    if labeled:
        stats["n_edge_labels"] = len({d["label"] for g in graphs for *_, d in g.edges(data=True)})
    return stats


def timed_inputs(graphs, distribution, labeled, seed):
    t0 = time.perf_counter()
    Ps, vs, ws = build_inputs(graphs, distribution, labeled, seed)
    return Ps, vs, ws, time.perf_counter()-t0


def matched_c(args, config, labeled, calibration_inputs, n_ref, seed):
    """(c, calibration record); calibration runs only when mc_matched needs it."""
    if "mc_matched" not in args.methods:
        return None, None
    if args.mc_c is not None:
        return args.mc_c, {"c": args.mc_c, "source": "cli"}
    Ps, vs, ws = calibration_inputs()
    calibration = calibrate_mc_c(Ps, vs, ws, config, labeled, n_ref, seed, args.calibration_repeats)
    print(f"  calibrated c={calibration['c']:.4g} (m={calibration['m']} at N={n_ref:.4g}, "
          f"GVoys {calibration['gvoys_time_sec']:.3g}s)", flush=True)
    return calibration["c"], {**calibration, "source": "calibrated"}


# ----------------------------------------------------------------------
#  Results
# ----------------------------------------------------------------------
def output_path(args):
    if args.output:
        return Path(args.output)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(args.output_dir)/f"{args.experiment_name}_{stamp}.json"


def new_payload(experiment, args):
    return {"experiment": experiment, "records": [], "calibrations": [],
            "metadata": {**runtime_metadata(), "cli": vars(args)}}


def save(payload, path):
    """Rewrite the whole file so interrupted long runs keep finished rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n")


def failed(records):
    return any(r.get("status") == "failed" for r in records)


# ----------------------------------------------------------------------
#  TU datasets (classification labels or regression graph attributes)
# ----------------------------------------------------------------------
def add_tu_arguments(parser, *, datasets):
    option(parser,"datasets",nargs="+",default=datasets)
    option(parser,"root_dir",default="tu_datasets")
    option(parser,"max_graphs",type=int,help="Class-stratified (classification) or random (regression) subset.")
    option(parser,"max_nodes_per_graph",type=int)
    option(parser,"calibration_graphs",type=int,default=50,
           help="Random graphs of each dataset used to calibrate c for mc_matched.")


def download_tu(name, root_dir):
    root = Path(root_dir)
    directory = root/name
    if (directory/f"{name}_graph_indicator.txt").exists():
        return directory
    root.mkdir(parents=True, exist_ok=True)
    archive = root/f"{name}.zip"
    if not archive.exists():
        print(f"downloading {name} from {tu._tu_dataset_url(name)}", flush=True)
        with urlopen(tu._tu_dataset_url(name), timeout=120) as response:
            archive.write_bytes(response.read())
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)
    return directory


def load_tu(name, root_dir="tu_datasets", edge_labels=False):
    """Graphs, targets, task and whether edge labels were attached.

    Datasets with _graph_labels.txt are classification tasks (targets are
    class indices); otherwise _graph_attributes.txt holds regression targets
    (first column). Vertex labels are ignored, as in dataset_bench.py.
    """
    directory = download_tu(name, root_dir)
    path = lambda suffix: directory/f"{name}_{suffix}.txt"
    indicator = tu._read_int_list(path("graph_indicator"))
    if path("graph_labels").exists():
        task = "classification"
        _, y = np.unique(tu._read_int_list(path("graph_labels")), return_inverse=True)
    elif path("graph_attributes").exists():
        task = "regression"
        y = np.loadtxt(path("graph_attributes"), delimiter=",", ndmin=2)[:, 0]
    else:
        raise FileNotFoundError(f"{name}: neither graph labels nor graph attributes found")
    graphs = [nx.Graph() for _ in range(max(indicator))]
    if len(y) != len(graphs):
        raise ValueError(f"{name}: graph indicators and targets are inconsistent")
    for node, graph_id in enumerate(indicator, start=1):
        graphs[graph_id-1].add_node(node)
    edges = tu._read_edge_list(path("A"))
    labels = tu._read_int_list(path("edge_labels")) if edge_labels and path("edge_labels").exists() else None
    if labels is not None and len(labels) != len(edges):
        raise ValueError(f"{name}: edge-label count does not match the edge list")
    for i, (u, v) in enumerate(edges):
        g = indicator[u-1]-1
        if indicator[v-1]-1 != g:
            raise ValueError(f"{name}: edge connects different graphs")
        graphs[g].add_edge(u, v)
        if labels is not None:
            previous = graphs[g].edges[u, v].get("label")
            if previous is not None and previous != labels[i]:
                raise ValueError(f"{name}: duplicate edge has conflicting labels")
            graphs[g].edges[u, v]["label"] = labels[i]
    return graphs, y, task, labels is not None


def select_subset(graphs, y, task, max_graphs=None, max_nodes=None, seed=42):
    if task == "classification":
        return tu.pick_graph_subset(graphs, y, max_graphs, max_nodes, seed)
    idx = np.array([i for i, g in enumerate(graphs) if max_nodes is None or len(g) <= max_nodes], dtype=int)
    if idx.size < 2:
        raise ValueError("fewer than two graphs left after max_nodes_per_graph filter")
    if max_graphs is not None and max_graphs < idx.size:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_graphs, replace=False))
    return [graphs[i] for i in idx], np.asarray(y)[idx], idx


def calibration_subset(graphs, count, seed):
    """Random graphs used to time GVoys against MCRWK; mean size gives n_ref."""
    if count >= len(graphs):
        return list(graphs)
    idx = np.sort(np.random.default_rng(seed).choice(len(graphs), count, replace=False))
    return [graphs[i] for i in idx]


# ----------------------------------------------------------------------
#  Supervised evaluation with precomputed kernels
# ----------------------------------------------------------------------
def add_evaluation_arguments(parser):
    option(parser,"c_values",type=float,nargs="+",default=[1e-3,1e-2,1e-1,1.,10.,100.])
    option(parser,"epsilon_values",type=float,nargs="+",default=[0.01,0.1,0.5],
           help="SVR epsilon grid, on standardized targets.")
    option(parser,"n_splits",type=int,default=5)
    option(parser,"inner_splits",type=int,default=3)
    option(parser,"n_cv_repeats",type=int,default=1)
    option(parser,"no_normalize",action="store_true",help="Disable diagonal Gram normalization.")


def _svr_predict(K_train, y_train, K_test, C, epsilon):
    mean, scale = y_train.mean(), y_train.std() or 1.
    model = SVR(kernel="precomputed", C=C, epsilon=epsilon).fit(K_train, (y_train-mean)/scale)
    return mean+scale*model.predict(K_test)


def _select_svr(K, y, c_values, epsilon_values, inner_splits, seed):
    inner = KFold(n_splits=min(inner_splits, len(y)), shuffle=True, random_state=seed)
    best, best_rmse = (c_values[0], epsilon_values[0]), np.inf
    for C in c_values:
        for epsilon in epsilon_values:
            errors = [mean_squared_error(y[va], _svr_predict(K[np.ix_(tr,tr)], y[tr], K[np.ix_(va,tr)], C, epsilon))
                      for tr, va in inner.split(y)]
            rmse = float(np.sqrt(np.mean(errors)))
            if rmse < best_rmse:
                best, best_rmse = (C, epsilon), rmse
    return best


def evaluate_svr_precomputed(K, y, c_values, epsilon_values, n_splits=5, n_repeats=1, inner_splits=3, seed=42):
    """Nested CV for SVR (C and epsilon selected by inner RMSE); targets standardized per fold."""
    y = np.asarray(y, dtype=float)
    if len(y) < 2*max(2, n_splits):
        raise ValueError("not enough graphs for regression CV")
    rng = np.random.default_rng(seed)
    scores = {"rmse": [], "mae": [], "r2": []}
    selected = []
    for _ in range(n_repeats):
        cv_seed = int(rng.integers(0, 2**31-1))
        for fold, (train, test) in enumerate(KFold(n_splits, shuffle=True, random_state=cv_seed).split(y), start=1):
            C, epsilon = _select_svr(K[np.ix_(train,train)], y[train], c_values, epsilon_values,
                                     inner_splits, cv_seed+fold)
            selected.append([float(C), float(epsilon)])
            pred = _svr_predict(K[np.ix_(train,train)], y[train], K[np.ix_(test,train)], C, epsilon)
            scores["rmse"].append(float(np.sqrt(mean_squared_error(y[test], pred))))
            scores["mae"].append(float(mean_absolute_error(y[test], pred)))
            scores["r2"].append(float(r2_score(y[test], pred)))
    summary = {f"mean_{k}": float(np.mean(v)) for k, v in scores.items()}
    summary.update({f"std_{k}": float(np.std(v)) for k, v in scores.items()})
    return {**summary, "fold_scores": scores, "selected_c_epsilon": selected}


def evaluate(K, y, task, args, seed):
    """Nested-CV SVC accuracy or SVR RMSE/MAE/R^2 on the (normalized) Gram."""
    K = K if args.no_normalize else tu.normalize_gram_matrix(K)
    if task == "classification":
        return tu.evaluate_svm_precomputed(K, y, args.c_values, args.n_splits, args.n_cv_repeats,
                                           args.inner_splits, seed)
    return evaluate_svr_precomputed(K, y, args.c_values, args.epsilon_values, args.n_splits,
                                    args.n_cv_repeats, args.inner_splits, seed)
