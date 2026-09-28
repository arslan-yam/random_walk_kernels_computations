"""Shared helpers for the paper experiments in exps/.

Every experiment compares the same methods on shared inputs (P, v, w):
exact solvers (direct, cg, fixed_point, sylvester), GVoys and two MCRWK
budgets. ``mc_fixed`` runs MCRWK once per m in ``--mc-fixed-m`` regardless of
graph size. ``mc_matched`` uses m = c*N, the linear budget of GVoys; c is
either given with ``--mc-c`` or calibrated so that MCRWK takes as long as
GVoys on the same kind of inputs (both costs are linear, so one c keeps the
runtimes matched across sizes up to fixed overheads).

Every MCRWK run yields the Gram with the unbiased diagonal (replica cross
product) and, from the same walks, the PSD Gram with the biased diagonal
(``name_biased``), whose expected excess acts as a graph-dependent ridge;
``--mc-diagonals`` selects which are recorded and evaluated.
"""

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from urllib.request import urlopen
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import networkx as nx
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVR, LinearSVC, LinearSVR

import dataset_bench as tu
from src import gram, gvoys, mcrwk, utils
from src.benchmark import (CG_REFERENCE_MIN_NODES, DIRECT_NODE_LIMIT, DISTRIBUTIONS,
    KernelConfig, build_inputs, compute_kernel, method_seed, method_skip_reason,
    option, reference_method, runtime_metadata)

EXACT_METHODS = ("direct", "cg", "fixed_point", "sylvester")
FEATURE_METHODS = ("mc", "gvoys")
METHODS = EXACT_METHODS + ("gvoys", "mc_matched", "mc_fixed")
CASES = ("unlabeled", "labeled")
CLASSIFICATION_DATASETS = ["MUTAG", "ENZYMES", "PTC_MR", "AIDS", "NCI1"]
REGRESSION_DATASETS = ["ZINC_test"]
# Upper bound for calibrated budgets; protects against a runaway secant step.
MAX_CALIBRATED_M = 50_000_000


# ----------------------------------------------------------------------
#  CLI and configuration
# ----------------------------------------------------------------------
def add_diagonal_arguments(parser, default):
    option(parser,"mc_diagonals",nargs="+",choices=["unbiased","biased"],default=list(default),
           help="MCRWK diagonals, both from the same walks: unbiased (replica cross product) "
                "and/or biased (PSD; expected excess C/2*E[Var F] acts as a ridge).")
    option(parser,"check_psd",action="store_true",
           help="Record the smallest eigenvalue of every Gram (cubic cost).")


def add_kernel_arguments(parser, *, gvoys_samples, output_dir, mc_diagonals=("unbiased",)):
    add_diagonal_arguments(parser, mc_diagonals)
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
    option(parser,"sylvester_max_nodes",type=int,default=512,
           help="Skip Sylvester (unlabeled only) above this vertex count.")
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


def feature_outputs(method, Ps, vs, ws, config, seed, labeled, want_grams):
    """Explicit features of MCRWK or GVoys (same walks as src.benchmark), timed apart from Grams.

    X satisfies X X^T = the feature Gram (MCRWK: biased diagonal). With
    want_grams the Grams are built as well: MCRWK gives {"unbiased", "biased"},
    GVoys {None}. Returns (X, grams, feature_time, gram_time).
    """
    t0 = time.perf_counter()
    if method == "mc":
        sample = mcrwk.random_walk_kernel_mc_features(Ps, vs, ws, utils.mu_func_gen(config.kind, config.lmbd),
            config.kind, n_length_samples=config.lengths if labeled else config.n_samples_mc,
            n_label_samples_per_length=config.n_label_samples_per_length, n_walk_reps=config.n_walk_reps,
            q_sampling_kind=config.q_sampling_kind, seed=seed, labeled=labeled)
        X = np.sqrt(sample.scale)*sample.features
    else:
        X = gvoys.random_walk_kernel_gvoys_features(Ps, vs, ws, labeled, anchor_fraction=config.anchor_fraction,
            kind=config.kind, lambda_coeff=config.lmbd, p_halt=config.p_halt,
            nb_random_walks=config.n_samples_gvoys, seed=seed, block_size=config.block_size)
    feature_time = time.perf_counter()-t0
    if not want_grams:
        return X, {}, feature_time, None
    t0 = time.perf_counter()
    grams = mc_grams(sample) if method == "mc" else {None: X @ X.T}
    return X, grams, feature_time, time.perf_counter()-t0


def mc_grams(sample):
    """Both Grams of an MCRWK Features sample, with the arithmetic of src.mcrwk (identical results)."""
    biased = sample.scale*(sample.features @ sample.features.T)
    unbiased = biased.copy()
    np.fill_diagonal(unbiased, sample.diagonal)
    return {"unbiased": unbiased, "biased": biased}


def psd_check(K):
    eigenvalues = np.linalg.eigvalsh((K+K.T)*0.5)
    tolerance = 1e-10*np.max(np.abs(eigenvalues))
    return {"min_eigenvalue": float(eigenvalues[0]), "negative_eigenvalues": int(np.sum(eigenvalues < -tolerance))}


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

    Returns (records, Grams by name, features by name). The reference follows
    src.benchmark: direct below 128 vertices, otherwise CG, which is added (and
    marked reference_only) when it was not requested.

    MCRWK and GVoys first compute explicit features X (feature_time_sec) and
    then their Grams (gram_build_sec). time_sec is the time to the Gram, as for
    the exact methods (time_basis "gram"). GVoys and the biased MCRWK diagonal
    also keep X for the linear SVM (the unbiased diagonal has no feature
    form). With only the linear SVM requested (--svm linear), no exact method,
    no unbiased diagonal and no --check-psd, feature-method Grams are skipped
    and time_sec is the feature time (time_basis "features").
    """
    if config.kind == "geom" and max_nodes >= CG_REFERENCE_MIN_NODES and "cg" not in {r["method"] for r in runs}:
        runs = runs+[{"name": "cg", "method": "cg", "variant": None, "reference_only": True}]
    # SVM (--svm) and ridge (--ridge) scripts list their models; the others always need Grams.
    models = set(getattr(args, "svm", ())) | set(getattr(args, "ridge", ()))
    grams_needed = (not models or "kernel" in models or args.check_psd
                    or any(r["method"] in EXACT_METHODS for r in runs))
    matrices, features, records = {}, {}, []
    for run in runs:
        diagonals = args.mc_diagonals if run["method"] == "mc" else [None]
        # One record per diagonal; MCRWK's biased Gram comes from the same walks and time.
        rows = {d: {**run, "name": run["name"]+("_biased" if d == "biased" else ""), "diagonal": d,
                    "status": "ok", "time_sec": None} for d in diagonals}
        reason = method_skip_reason(run["method"], kind=config.kind, labeled=labeled,
            max_nodes=max_nodes, direct_max_nodes=args.direct_max_nodes,
            sylvester_max_nodes=args.sylvester_max_nodes)
        if reason:
            records += [{**row, "status": "skipped", "reason": reason} for row in rows.values()]
            continue
        cfg = with_mc_budget(config, run["m"]) if run["method"] == "mc" else config
        method_rng_seed = method_seed(seed, run["method"])
        print(f"{tag} {run['name']} ...", end=" ", flush=True)
        try:
            if run["method"] in FEATURE_METHODS:
                # The unbiased diagonal exists only as a Gram.
                want = grams_needed or "unbiased" in diagonals
                X, grams, feature_time, gram_time = feature_outputs(run["method"], Ps, vs, ws, cfg,
                                                                    method_rng_seed, labeled, want)
                for d, row in rows.items():
                    row.update(feature_time_sec=feature_time, gram_build_sec=gram_time,
                               time_sec=feature_time+(gram_time or 0.),
                               time_basis="gram" if grams else "features")
                    if d != "unbiased":
                        features[row["name"]] = X
                    if d in grams:
                        matrices[row["name"]] = grams[d]
                print(f"features {feature_time:.4g}s"+(f", Gram {gram_time:.4g}s" if grams else ""), flush=True)
            else:
                K, elapsed = timed_kernel(run["method"], Ps, vs, ws, cfg, method_rng_seed, labeled)
                matrices[run["name"]] = K
                rows[None].update(time_sec=elapsed, time_basis="gram")
                print(f"{elapsed:.4g}s", flush=True)
        except Exception as exc:
            print(f"failed: {exc}", flush=True)
            for row in rows.values():
                row.update(status="failed", error=str(exc))
            if args.fail_fast:
                raise
        records += rows.values()
    ref = reference_method(matrices, kind=config.kind, max_nodes=max_nodes)
    for record in records:
        K = matrices.get(record["name"])
        record["reference"] = ref
        record["errors"] = gram.matrix_errors(matrices[ref], K) if ref and K is not None else None
        if K is not None:
            record["diagonal_min"] = float(np.min(np.diag(K)))
            if args.check_psd:
                record.update(psd_check(K))
            if len(K) <= args.json_gram_max_graphs:
                record["gram"] = K.tolist()
    return records, matrices, features


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
#  Supervised evaluation: precomputed kernels and explicit features
# ----------------------------------------------------------------------
def add_evaluation_arguments(parser):
    option(parser,"c_values",type=float,nargs="+",default=[1e-3,1e-2,1e-1,1.,10.,100.])
    option(parser,"epsilon_values",type=float,nargs="+",default=[0.01,0.1,0.5],
           help="SVR epsilon grid, on standardized targets.")
    option(parser,"n_splits",type=int,default=5)
    option(parser,"inner_splits",type=int,default=3)
    option(parser,"n_cv_repeats",type=int,default=1)
    option(parser,"no_normalize",action="store_true",help="Disable diagonal Gram (feature row) normalization.")
    option(parser,"svm",nargs="+",choices=["kernel","linear"],default=["kernel","linear"],
           help="kernel: SVC/SVR on the Gram of every method (evaluation; one classifier for all, for "
                "comparing methods). linear: LinearSVC/LinearSVR on the features of GVoys and the biased "
                "MCRWK diagonal (evaluation_linear; linear in the number of graphs).")
    option(parser,"linear_max_iter",type=int,default=10000,help="Iteration limit of LinearSVC/LinearSVR.")


def nested_cv(y, task, candidates, fit_predict, n_splits=5, n_repeats=1, inner_splits=3, seed=42):
    """Nested CV shared by kernel and feature models.

    Classification uses the stratified folds and selection of dataset_bench
    (inner accuracy, the first best candidate wins); regression uses shuffled
    KFold and inner RMSE. fit_predict(train, test, params) predicts y[test].
    """
    y = np.asarray(y)
    classification = task == "classification"
    if classification:
        outer_splits = tu._safe_n_splits(y, n_splits)
        if outer_splits < 2:
            raise ValueError("not enough samples per class for CV")
    elif len(y) < 2*max(2, n_splits):
        raise ValueError("not enough graphs for regression CV")
    else:
        outer_splits = n_splits
    splitter = StratifiedKFold if classification else KFold
    rng = np.random.default_rng(seed)
    folds = []
    for _ in range(n_repeats):
        cv_seed = int(rng.integers(0, 2**31-1))
        outer = splitter(n_splits=outer_splits, shuffle=True, random_state=cv_seed)
        for fold, (train, test) in enumerate(outer.split(np.zeros(len(y)), y), start=1):
            params = _select_params(y, train, classification, candidates, fit_predict, inner_splits, cv_seed+fold)
            folds.append((y[test], fit_predict(train, test, params), params))
    return _fold_summary(folds, classification)


def _select_params(y, train, classification, candidates, fit_predict, inner_splits, seed):
    if classification:
        n_inner = tu._safe_n_splits(y[train], inner_splits)
        if n_inner < 2:
            return candidates[0]
        inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    else:
        inner = KFold(n_splits=min(inner_splits, len(train)), shuffle=True, random_state=seed)
    best, best_score = candidates[0], -np.inf
    for params in candidates:
        values = []
        for tr, va in inner.split(np.zeros(len(train)), y[train]):
            pred = fit_predict(train[tr], train[va], params)
            values.append(accuracy_score(y[train[va]], pred) if classification
                          else mean_squared_error(y[train[va]], pred))
        score = float(np.mean(values)) if classification else -float(np.sqrt(np.mean(values)))
        if score > best_score:
            best, best_score = params, score
    return best


def _fold_summary(folds, classification):
    if classification:
        scores = [float(accuracy_score(truth, pred)) for truth, pred, _ in folds]
        return {"mean_accuracy": float(np.mean(scores)), "std_accuracy": float(np.std(scores)),
                "scores": scores, "selected_cs": [float(C) for *_, C in folds]}
    scores = {"rmse": [float(np.sqrt(mean_squared_error(t, p))) for t, p, _ in folds],
              "mae": [float(mean_absolute_error(t, p)) for t, p, _ in folds],
              "r2": [float(r2_score(t, p)) for t, p, _ in folds]}
    summary = {f"mean_{k}": float(np.mean(v)) for k, v in scores.items()}
    summary.update({f"std_{k}": float(np.std(v)) for k, v in scores.items()})
    return {**summary, "fold_scores": scores, "selected_c_epsilon": [list(map(float, p)) for *_, p in folds]}


def _standardize(y_train):
    return y_train.mean(), y_train.std() or 1.


def _svr_predict(K_train, y_train, K_test, C, epsilon):
    mean, scale = _standardize(y_train)
    model = SVR(kernel="precomputed", C=C, epsilon=epsilon).fit(K_train, (y_train-mean)/scale)
    return mean+scale*model.predict(K_test)


def evaluate_svr_precomputed(K, y, c_values, epsilon_values, n_splits=5, n_repeats=1, inner_splits=3, seed=42):
    """Nested CV for SVR (C and epsilon selected by inner RMSE); targets standardized per fold."""
    y = np.asarray(y, dtype=float)
    fit = lambda train, test, p: _svr_predict(K[np.ix_(train, train)], y[train], K[np.ix_(test, train)], *p)
    return nested_cv(y, "regression", [(C, e) for C in c_values for e in epsilon_values], fit,
                     n_splits, n_repeats, inner_splits, seed)


def evaluate(K, y, task, args, seed):
    """Nested-CV SVC accuracy or SVR RMSE/MAE/R^2 on the (normalized) Gram."""
    K = K if args.no_normalize else tu.normalize_gram_matrix(K)
    if task == "classification":
        return tu.evaluate_svm_precomputed(K, y, args.c_values, args.n_splits, args.n_cv_repeats,
                                           args.inner_splits, seed)
    return evaluate_svr_precomputed(K, y, args.c_values, args.epsilon_values, args.n_splits,
                                    args.n_cv_repeats, args.inner_splits, seed)


def evaluate_features(X, y, task, args, seed):
    """Linear SVM/SVR on explicit features, never forming the graphs x graphs Gram.

    Rows are scaled to unit length unless --no-normalize; this is exactly the
    diagonal normalization of the feature Gram X X^T used by evaluate(). Same
    folds and grids as the kernel models, one-vs-one for more than two classes
    as SVC. The losses are squared (squared hinge; squared epsilon-insensitive
    on standardized targets) so that liblinear's primal Newton solver applies:
    hinge loss needs the dual solver, which on MUTAG was 20-80x slower and hit
    its iteration limit at large C, while accuracy stayed within ~0.02 of SVC.
    liblinear also penalizes the intercept (softened by intercept_scaling=10);
    convergence warnings are counted in the result instead of printed.
    """
    X = np.asarray(X, dtype=float)
    if not args.no_normalize:
        norms = np.linalg.norm(X, axis=1)
        if not np.isfinite(X).all() or np.any(norms <= 0):
            raise ValueError("feature normalization needs finite nonzero rows; use --no-normalize or a larger budget")
        X = X/norms[:, None]
    y = np.asarray(y)
    common_args = (args.n_splits, args.n_cv_repeats, args.inner_splits, seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        if task == "classification":
            n_classes = len(np.unique(y))

            def fit(train, test, C):
                model = LinearSVC(C=C, loss="squared_hinge", dual=False, intercept_scaling=10.,
                                  max_iter=args.linear_max_iter)
                model = OneVsOneClassifier(model) if n_classes > 2 else model
                return model.fit(X[train], y[train]).predict(X[test])
            result = nested_cv(y, task, list(args.c_values), fit, *common_args)
        else:
            y = y.astype(float)

            def fit(train, test, params):
                mean, scale = _standardize(y[train])
                model = LinearSVR(C=params[0], epsilon=params[1], loss="squared_epsilon_insensitive",
                                  dual=False, intercept_scaling=10., max_iter=args.linear_max_iter)
                return mean+scale*model.fit(X[train], (y[train]-mean)/scale).predict(X[test])
            result = nested_cv(y, task, [(C, e) for C in args.c_values for e in args.epsilon_values],
                               fit, *common_args)
    result["convergence_warnings"] = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    result["model"] = ("LinearSVC(squared_hinge, primal)" if task == "classification"
                       else "LinearSVR(squared_epsilon_insensitive, primal)")
    return result


def evaluate_record(record, K, X, y, task, args, seed, suffix=""):
    """Kernel SVM on the Gram K (evaluation<suffix>), linear SVM on features X (evaluation<suffix>_linear).

    Kernel results of all methods share one classifier and are comparable
    with each other; linear results exist only for feature methods (X is not
    None) and are comparable with each other and with their own kernel result.
    """
    for model, key, run, data in (("kernel", suffix, evaluate, K), ("linear", suffix+"_linear", evaluate_features, X)):
        if model in args.svm and data is not None:
            t0 = time.perf_counter()
            record["evaluation"+key] = run(data, y, task, args, seed)
            record["eval_time_sec"+key] = time.perf_counter()-t0
