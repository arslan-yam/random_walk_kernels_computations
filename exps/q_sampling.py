"""Experiment 5: importance-sampling proposal q for labeled MCRWK on TU datasets.

MCRWK samples every label sequence from a product proposal q over edge labels.
q changes the variance through the importance weights and the runtime through
the walk lengths (a walk dies where the sampled label is missing). For every
labeled TU dataset (default MUTAG, PTC_MR, AIDS, ZINC_test), lambda, proposal
and budget m (n = 1), the script runs --n-repeats seeds and records:
  per seed    Gram time, error against direct, walk diagnostics (killed walks,
              steps, feature ESS, largest log weight), SVM accuracy or SVR RMSE;
  aggregate   bias, variance and MSE of every Gram entry over the seeds
              (relative to the reference), MSE x time, R(q) of the variance
              bound (10) and whether lambda < q_min keeps it finite.
The biased-diagonal (PSD) Gram of the same walks is evaluated as well
(evaluation_biased); ridge_rel = mean_i (K_b,ii - K_u,ii) / K_ii estimates the
relative size of its ridge.
Seeds are shared across proposals (common random lengths); SVM folds are fixed.

Proposals (p: edge-label frequencies of the dataset; p_i: those of graph i):
  uniform                    q_l = 1/d
  freq:A                     q_l ~ p_l^A  (A=0 uniform, A=1 label frequencies)
  sq_mean                    q_l ~ mean_i p_il^2
  inverse                    q_l ~ 1/p_l (a deliberately poor control)
  norm_l1, norm_fro, random  the src.mcrwk rules
--mix-eps adds a uniform component, (1-eps) q + eps/d, which raises q_min.
If the label-l mass at every vertex is about p_l, the second moment per step
is sum_l p_l^4/q_l, minimized by q ~ p^2: freq:2, or sq_mean for a dataset.

Paper run:
    python exps/q_sampling.py
Quick check:
    python exps/q_sampling.py --datasets MUTAG --max-graphs 8 --n-repeats 2 \\
        --m-values 100 --lambdas 0.7 --n-splits 2 --inner-splits 2
"""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from exps import common
from src import gram, gvoys, mcrwk, utils
from src.benchmark import CG_REFERENCE_MIN_NODES, DISTRIBUTIONS, KernelConfig, method_seed, option

LABELED_DATASETS = ["MUTAG", "PTC_MR", "AIDS", "ZINC_test"]
TAG_KEYS = ("lmbd", "design", "proposal", "mix_eps", "m", "lengths", "n")
PROPOSALS = ["uniform", "freq:0.5", "freq:1", "freq:2", "sq_mean", "norm_l1", "norm_fro", "inverse", "random"]


def proposal_spec(text):
    if text in {"uniform", "sq_mean", "inverse", "norm_l1", "norm_fro", "random"}:
        return text
    if text.startswith("freq:"):
        try:
            float(text[5:])
            return text
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"unknown proposal {text!r}")


# ----------------------------------------------------------------------
#  Shared with exps/n_sampling.py
# ----------------------------------------------------------------------
def add_arguments(parser, *, output_dir, lambdas, repeats):
    option(parser,"datasets",nargs="+",default=LABELED_DATASETS,help="TU datasets with edge labels.")
    option(parser,"root_dir",default="tu_datasets")
    option(parser,"max_graphs",type=int,help="Class-stratified (classification) or random (regression) subset.")
    option(parser,"max_nodes_per_graph",type=int)
    option(parser,"kind",choices=["geom","exp"],default="geom")
    option(parser,"lambdas",type=float,nargs="+",default=lambdas)
    option(parser,"u_w_distribution",choices=DISTRIBUTIONS,default="normal")
    option(parser,"n_repeats",type=int,default=repeats,help="Estimator seeds per configuration.")
    option(parser,"mix_eps",type=float,nargs="+",default=[0.],help="Uniform mixing weights for the proposals.")
    option(parser,"with_gvoys",action="store_true",help="Also run GVoys (independent of q) as a baseline.")
    option(parser,"n_samples_gvoys",type=int,default=200)
    option(parser,"p_halt",type=float,default=0.2)
    option(parser,"skip_evaluation",action="store_true",help="No SVM/SVR, only kernel statistics.")
    common.add_diagonal_arguments(parser, ("unbiased", "biased"))
    common.add_evaluation_arguments(parser)
    option(parser,"seed",type=int,default=42)
    option(parser,"output_dir",default=output_dir)
    option(parser,"experiment_name",default="run")
    option(parser,"output",help="Exact JSON path; default is <output-dir>/<experiment-name>_<UTC stamp>.json.")
    option(parser,"fail_fast",action="store_true")


def validate_arguments(parser, args):
    if args.seed < 0 or args.n_repeats < 1 or args.n_splits < 2 or args.inner_splits < 2 or args.n_cv_repeats < 1:
        parser.error("invalid seed, repeats or CV parameters")
    if any(not 0 <= eps < 1 for eps in args.mix_eps):
        parser.error("mix_eps must be in [0, 1)")
    for lmbd in args.lambdas:
        try:
            utils.mu_func_gen(args.kind, lmbd)
        except ValueError as exc:
            parser.error(f"lambda={lmbd}: {exc}")


@dataclass
class Labeled:
    name: str
    graphs: list
    y: np.ndarray
    task: str
    Ps: list
    vs: list
    ws: list
    labels: list
    freq: np.ndarray
    sq_mean: np.ndarray
    stats: dict


def prepare(name, args):
    """Labeled inputs and label statistics, or None when the dataset has no edge labels."""
    graphs, y, task, has_edge_labels = common.load_tu(name, args.root_dir, edge_labels=True)
    if not has_edge_labels:
        return None
    graphs, y, _ = common.select_subset(graphs, y, task, args.max_graphs, args.max_nodes_per_graph, args.seed)
    Ps, vs, ws, input_time = common.timed_inputs(graphs, args.u_w_distribution, True, args.seed)
    labels = sorted({d["label"] for g in graphs for *_, d in g.edges(data=True)})
    index = {label: i for i, label in enumerate(labels)}
    counts, squares = np.zeros(len(labels)), np.zeros(len(labels))
    for g in graphs:
        c = np.zeros(len(labels))
        for *_, d in g.edges(data=True):
            c[index[d["label"]]] += 1
        counts += c
        squares += (c/c.sum())**2 if c.sum() else 0
    freq = counts/counts.sum()
    stats = {**common.graph_stats(graphs, True), "input_time_sec": input_time,
             "label_frequencies": dict(zip(labels, freq.tolist()))}
    return Labeled(name, graphs, y, task, Ps, vs, ws, labels, freq, squares/len(graphs), stats)


def reference(data, args, lmbd):
    """Exact Gram (direct below 128 vertices, else CG) and its own evaluation."""
    method = "direct" if data.stats["max_nodes"] < CG_REFERENCE_MIN_NODES else "cg"
    config = KernelConfig(kind=args.kind, lmbd=lmbd).validate()
    K, elapsed = common.timed_kernel(method, data.Ps, data.vs, data.ws, config, 0, True)
    record = {"dataset": data.name, "task": data.task, "lmbd": lmbd, "method": method,
              "time_sec": elapsed, **data.stats}
    if not args.skip_evaluation:
        record["evaluation"] = common.evaluate(K, data.y, data.task, args, args.seed)
    return K, record


def proposal(spec, data, eps, rng):
    """{label: probability} for a proposal spec, mixed with uniform by eps."""
    if spec == "uniform":
        scores = np.ones(len(data.labels))
    elif spec.startswith("freq:"):
        scores = data.freq**float(spec[5:])
    elif spec == "sq_mean":
        scores = data.sq_mean
    elif spec == "inverse":
        scores = 1/data.freq
    else:
        scores = mcrwk.q_sampling_dataset(data.Ps, data.labels, spec, rng)
    q = (1-eps)*scores/scores.sum()+eps/len(scores)
    return dict(zip(data.labels, q.tolist()))


def variance_factor(q, lmbd, kind):
    """R(q) = E_K[q_min^-K] of the variance bound (10); None when it is infinite."""
    q_min = min(q.values())
    if kind == "exp":
        return math.exp(lmbd*(1/q_min-1))
    return (1-lmbd)/(1-lmbd/q_min) if lmbd < q_min else None


def entry_statistics(ref, s1, s2, count):
    """Bias, variance and MSE of every Gram entry over the seeds.

    Relative values divide by ref^2 (|ref| for the bias); var_offdiag is the
    absolute pair variance, comparable with the bound (10).
    """
    if count < 2:
        return {}
    mean = s1/count
    var = np.maximum(s2/count-mean**2, 0)*count/(count-1)
    mse = np.maximum(s2/count-2*ref*mean+ref**2, 0)
    valid = np.abs(ref) > 1e-15
    offdiag = valid & ~np.eye(len(ref), dtype=bool)
    se = np.sqrt(var/count)
    tested = valid & (se > 0)
    t = np.abs(mean-ref)[tested]/se[tested]
    relative = lambda values, mask: float(np.mean(values[mask]/ref[mask]**2)) if mask.any() else None
    return {"rel_mse": relative(mse, valid), "rel_mse_offdiag": relative(mse, offdiag),
            "rel_var": relative(var, valid), "rel_var_offdiag": relative(var, offdiag),
            "rel_bias": float(np.mean(np.abs(mean-ref)[valid]/np.abs(ref[valid]))),
            "var_offdiag": float(np.mean(var[offdiag])) if offdiag.any() else None,
            "var_offdiag_max": float(np.max(var[offdiag])) if offdiag.any() else None,
            "bias_t_mean": float(np.mean(t)) if t.size else None,
            "bias_t_frac_above_3": float(np.mean(t > 3)) if t.size else None}


def mean_or_none(values):
    return float(np.mean(values)) if values else None


def seed_summary(records):
    """Means over the successful seeds: time, error, walk diagnostics and evaluation."""
    ok = [r for r in records if r["status"] == "ok"]
    times = [r["time_sec"] for r in ok]
    out = {"n_ok": len(ok), "n_failed": len(records)-len(ok), "time_mean": mean_or_none(times),
           "time_std": float(np.std(times)) if times else None,
           "error_mean": mean_or_none([r["errors"]["offdiagonal_mean_rel"] for r in ok
                                       if r["errors"]["offdiagonal_mean_rel"] is not None])}
    diags = [r["diagnostics"] for r in ok if r.get("diagnostics")]
    walks = sum(d["walks"] for d in diags)
    if walks:
        logs = [d["max_log_weight"] for d in diags if d["max_log_weight"] is not None]
        out.update(killed_fraction=sum(d["killed"] for d in diags)/walks,
                   skipped_fraction=sum(d["skipped"] for d in diags)/walks,
                   steps_per_walk=sum(d["steps"] for d in diags)/walks,
                   zero_feature_fraction=mean_or_none([d["zero_feature_fraction"] for d in diags]),
                   ess_fraction=mean_or_none([d["ess_fraction"] for d in diags if d["ess_fraction"] is not None]),
                   max_log_weight=max(logs) if logs else None)
    ridges = [r["ridge_rel"] for r in ok if "ridge_rel" in r]
    if ridges:
        out["ridge_rel"] = float(np.mean(ridges))
    for suffix in ("", "_biased", "_linear", "_biased_linear"):
        for key in ("mean_accuracy", "mean_rmse", "mean_mae", "mean_r2"):
            values = [r["evaluation"+suffix][key] for r in ok if key in (r.get("evaluation"+suffix) or {})]
            if values:
                out[f"{key}{suffix}_over_seeds"] = float(np.mean(values))
                out[f"{key}{suffix}_std_over_seeds"] = float(np.std(values))
    return out


def run_seeds(data, args, ref, base, estimate, tag):
    """estimate(seed) -> ({diagonal: Gram}, {diagonal: features}, extra) per repeat; records and aggregate.

    Errors and bias/variance use the unbiased Gram (GVoys: its only one). Every
    requested diagonal gets the kernel SVM on its Gram (evaluation<suffix>);
    diagonals with features (biased MCRWK, GVoys) also the linear SVM
    (evaluation<suffix>_linear), as in exps/tu_svm.py.
    """
    records, s1, s2, count = [], 0., 0., 0
    for repeat in range(args.n_repeats):
        seed = method_seed(args.seed+repeat, "mc")
        record = {**base, "repeat": repeat, "seed": seed, "status": "ok"}
        print(f"{tag} r={repeat} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            grams, features, extra = estimate(seed)
            record.update(time_sec=time.perf_counter()-t0, **extra)
            K = grams.get("unbiased", grams.get(None))
            record["errors"] = gram.matrix_errors(ref, K)
            print(f"{record['time_sec']:.4g}s, error {record['errors']['offdiagonal_mean_rel']:.3g}", flush=True)
            if "biased" in grams:
                record["ridge_rel"] = float(np.mean((np.diag(grams["biased"])-np.diag(K))/np.diag(ref)))
            if args.check_psd:
                record["psd"] = {d or "gram": common.psd_check(M) for d, M in grams.items()}
            for d, M in grams.items():
                if args.skip_evaluation or (d is not None and d not in args.mc_diagonals):
                    continue
                common.evaluate_record(record, M, features.get(d), data.y, data.task, args, args.seed,
                                       suffix="_biased" if d == "biased" else "")
            s1, s2, count = s1+K, s2+K*K, count+1
        except Exception as exc:
            print(f"failed: {exc}", flush=True)
            record.update(status="failed", error=str(exc), time_sec=record.get("time_sec", time.perf_counter()-t0))
            if args.fail_fast:
                raise
        records.append(record)
    aggregate = {**base, **seed_summary(records), **entry_statistics(ref, s1, s2, count)}
    if aggregate.get("rel_mse") is not None and aggregate["time_mean"] is not None:
        aggregate["mse_x_time"] = aggregate["rel_mse"]*aggregate["time_mean"]
    return records, aggregate


def mc_estimate(data, args, lmbd, spec, eps, lengths, n):
    """Closure for run_seeds: labeled MCRWK with this proposal; random q is redrawn per seed."""
    mu = utils.mu_func_gen(args.kind, lmbd)

    def estimate(seed):
        q = proposal(spec, data, eps, np.random.default_rng(seed))
        diagnostics = {}
        t0 = time.perf_counter()
        sample = mcrwk.random_walk_kernel_mc_features(data.Ps, data.vs, data.ws, mu, args.kind,
            n_length_samples=lengths, n_label_samples_per_length=n, n_walk_reps=1,
            q_sampling_kind=q, seed=seed, labeled=True, diagnostics=diagnostics)
        X = np.sqrt(sample.scale)*sample.features
        feature_time = time.perf_counter()-t0
        return common.mc_grams(sample), {"biased": X}, \
            {"feature_time_sec": feature_time, "diagnostics": diagnostics, "q_min": min(q.values()),
             "R_q": variance_factor(q, lmbd, args.kind)}
    return estimate


def gvoys_estimate(data, args, lmbd):
    def estimate(seed):
        t0 = time.perf_counter()
        X = gvoys.random_walk_kernel_gvoys_features(data.Ps, data.vs, data.ws, True, kind=args.kind,
            lambda_coeff=lmbd, p_halt=args.p_halt, nb_random_walks=args.n_samples_gvoys, seed=seed)
        return {None: X @ X.T}, {None: X}, {"feature_time_sec": time.perf_counter()-t0}
    return estimate


def with_theory(aggregate, q, lmbd, kind):
    """Deterministic proposals: q, q_min and R(q) belong to the aggregate as well."""
    aggregate.update(q=q, q_min=min(q.values()), R_q=variance_factor(q, lmbd, kind),
                     bound_finite=variance_factor(q, lmbd, kind) is not None)
    return aggregate


def run_experiment(args, experiment, configurations):
    """Loop datasets x lambdas; configurations(data, lmbd) yields (base, estimate, q or None)."""
    path = common.output_path(args)
    payload = {**common.new_payload(experiment, args), "aggregates": [], "references": []}
    for name in args.datasets:
        print(f"== {name}", flush=True)
        try:
            data = prepare(name, args)
            if data is None:
                print("  skipped: no edge labels", flush=True)
                payload["references"].append({"dataset": name, "status": "skipped", "reason": "no edge labels"})
                continue
            for lmbd in args.lambdas:
                ref, ref_record = reference(data, args, lmbd)
                payload["references"].append({**ref_record, "status": "ok"})
                print(f"  lambda={lmbd}: {ref_record['method']} reference {ref_record['time_sec']:.3g}s", flush=True)
                for base, estimate, q in configurations(data, lmbd):
                    base = {"dataset": name, "task": data.task, "lmbd": lmbd, **base}
                    tag = "  "+" ".join(f"{k}={base[k]}" for k in TAG_KEYS if k in base)
                    records, aggregate = run_seeds(data, args, ref, base, estimate, tag)
                    payload["records"] += records
                    payload["aggregates"].append(with_theory(aggregate, q, lmbd, args.kind) if q else aggregate)
                    common.save(payload, path)
        except Exception as exc:
            print(f"  failed: {exc}", flush=True)
            payload["references"].append({"dataset": name, "status": "failed", "error": str(exc)})
            if args.fail_fast:
                raise
        finally:
            common.save(payload, path)
    print(f"Saved {path}", flush=True)
    return int(common.failed(payload["records"]) or common.failed(payload["references"]))


# ----------------------------------------------------------------------
#  Experiment 5
# ----------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser, output_dir="results/exps/q_sampling", lambdas=[0.3, 0.7], repeats=5)
    option(parser,"proposals",type=proposal_spec,nargs="+",default=PROPOSALS)
    option(parser,"m_values",type=int,nargs="+",default=[100,1000,10000],help="Label sequences per replica (n=1).")
    args = parser.parse_args(argv)
    validate_arguments(parser, args)
    if any(m < 1 for m in args.m_values):
        parser.error("m values must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)

    def configurations(data, lmbd):
        for spec in args.proposals:
            # Mixing leaves uniform unchanged; random q is redrawn per seed.
            for eps in ([0.] if spec == "uniform" else args.mix_eps):
                q = None if spec == "random" else proposal(spec, data, eps, None)
                for m in args.m_values:
                    base = {"proposal": spec, "mix_eps": eps, "m": m, "lengths": m, "n": 1}
                    yield base, mc_estimate(data, args, lmbd, spec, eps, m, 1), q
        if args.with_gvoys:
            yield {"proposal": "gvoys", "n_samples_gvoys": args.n_samples_gvoys}, gvoys_estimate(data, args, lmbd), None

    return run_experiment(args, "q_sampling", configurations)


if __name__ == "__main__":
    raise SystemExit(main())
