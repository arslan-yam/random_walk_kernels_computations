"""Experiment 7: convergence of MCRWK and the tightness of its error bounds.

For --n-graphs synthetic graphs of each size N (BA with m=2; labeled graphs get
3 edge labels, each with probability 1/3), --n-repeats seeds of MCRWK are
compared with the exact kernel (direct below 128 vertices, CG otherwise) on
every off-diagonal pair k = k(G_i, G_j), the quantity the bounds are about:
  convergence  relative RMSE versus m on log-log axes has slope -1/2, and the
               curves of different N coincide (sample complexity does not
               depend on the graph size);
  variance     measured pair variance against the bound C^2 w^4/m (6) or,
               labeled, C^2 w^4/m (R(q)/n + 1/4) (10), as bound/measured;
  tails        P(|k_hat - k| > eps k) against Hoeffding (7) (unlabeled) and
               Chebyshev with the variance bound ((11) labeled);
  budget       the smallest m of the grid with P(|k_hat - k| > eps k) <= delta
               against the bounds (8) (unlabeled) and (12) (labeled).
w is the larger stopping weight of the pair (w_inf in (6)). Tail bounds hold
per pair; the reported bound is their mean over pairs, which bounds the pooled
empirical frequency. Labeled runs use uniform q over the d labels present, so
R(q) is finite only for lambda < q_min = 1/d (null otherwise).

Budgets are nested: every seed runs MCRWK once with the largest m, and the
first m feature columns give the estimate with m samples (the columns are
i.i.d.). Estimates of different m share walks within a seed, while every
statistic of a single m is taken over independent seeds.

Paper run:
    python exps/convergence.py
Quick check:
    python exps/convergence.py --sizes 8 16 --n-graphs 3 --n-repeats 5 --m-values 10 100 --lambdas 0.3
"""

import argparse
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from exps import common
from exps.q_sampling import variance_factor
from exps.scaling import add_graph_arguments, make_graphs, validate_graph_arguments
from src import mcrwk, utils
from src.benchmark import (CG_REFERENCE_MIN_NODES, DISTRIBUTIONS, KernelConfig, build_inputs,
                           compute_kernel, method_seed, option)


def exact_gram(Ps, vs, ws, args, lmbd, labeled, n_nodes):
    method = "direct" if n_nodes < min(args.direct_max_nodes, CG_REFERENCE_MIN_NODES) else "cg"
    config = KernelConfig(kind=args.kind, lmbd=lmbd, solver_tol=args.solver_tol, max_iter=args.max_iter).validate()
    t0 = time.perf_counter()
    return compute_kernel(method, Ps, vs, ws, config, labeled=labeled), method, time.perf_counter()-t0


def nested_estimates(Ps, vs, ws, args, lmbd, labeled, pairs):
    """(seeds x m-values x pairs) estimates from prefixes of one largest-budget run per seed."""
    mu = utils.mu_func_gen(args.kind, lmbd)
    C = mcrwk.kernel_normalizer(args.kind, mu)
    n, budget = args.n_label_samples_per_length, max(args.m_values)
    i, j = pairs
    estimates, times = np.empty((args.n_repeats, len(args.m_values), len(i))), []
    for repeat in range(args.n_repeats):
        t0 = time.perf_counter()
        sample = mcrwk.random_walk_kernel_mc_features(Ps, vs, ws, mu, args.kind, n_length_samples=budget//n,
            n_label_samples_per_length=n, seed=method_seed(args.seed+repeat, "mc"), labeled=labeled)
        times.append(time.perf_counter()-t0)
        running = np.cumsum(sample.features[i]*sample.features[j], axis=1)
        for a, m in enumerate(args.m_values):
            estimates[repeat, a] = C*running[:, m-1]/m
    return estimates, C, float(np.mean(times))


def statistics(estimates, exact, w, C, R, args):
    """Per-m records: errors, variance against its bound, tails against Hoeffding/Chebyshev."""
    n = args.n_label_samples_per_length
    rows = []
    for a, m in enumerate(args.m_values):
        est = estimates[:, a]
        relative = (est-exact)/exact
        variance = est.var(axis=0, ddof=1)
        # Unlabeled (6): C^2 w^4 / m; labeled (10): C^2 w^4 / m * (R(q)/n + 1/4).
        factor = 1. if R == "unlabeled" else (None if R is None else R/n+0.25)
        bound = C**2*w**4/m*factor if factor is not None else None
        measured = variance > 0
        row = {"m": m, "rel_rmse": float(np.sqrt(np.mean(relative**2))),
               "mean_rel_error": float(np.mean(np.abs(relative))),
               "abs_rmse": float(np.sqrt(np.mean((est-exact)**2))),
               "rel_bias": float(np.mean(np.abs(est.mean(axis=0)-exact)/exact)),
               "m_rel_var_median": float(np.median(m*variance/exact**2)),
               "tails": []}
        if bound is not None and measured.any():
            ratio = bound[measured]/variance[measured]
            row.update(variance_bound_ratio_median=float(np.median(ratio)),
                       variance_bound_ratio_min=float(np.min(ratio)),
                       variance_bound_ratio_max=float(np.max(ratio)))
        for eps in args.epsilons:
            tail = {"eps": eps, "empirical": float(np.mean(np.abs(relative) > eps))}
            gap = (eps*exact)**2
            if R == "unlabeled":
                tail["hoeffding"] = float(np.mean(np.minimum(1., 2*np.exp(-2*m*gap/(C**2*w**4)))))
            if bound is not None:
                tail["chebyshev"] = float(np.mean(np.minimum(1., bound/gap)))
            row["tails"].append(tail)
        rows.append(row)
    return rows


def required_budgets(rows, exact, w, C, R, args):
    """Smallest grid m with empirical tail <= delta, against bounds (8) and (12) per pair."""
    n = args.n_label_samples_per_length
    out = []
    for eps in args.epsilons:
        gap = (eps*exact)**2
        for delta in args.deltas:
            reached = [row["m"] for row in rows
                       if next(t for t in row["tails"] if t["eps"] == eps)["empirical"] <= delta]
            if R == "unlabeled":
                bound, name = C**2*w**4*math.log(2/delta)/(2*gap), "hoeffding (8)"
            elif R is not None:
                bound, name = C**2*w**4*(R/n+0.25)/(delta*gap), "chebyshev (12)"
            else:
                bound, name = None, "infinite (lambda >= q_min)"
            out.append({"eps": eps, "delta": delta, "m_empirical": min(reached) if reached else None,
                        "m_grid_max": max(args.m_values), "bound": name,
                        "m_bound_median": float(np.median(bound)) if bound is not None else None,
                        "m_bound_max": float(np.max(bound)) if bound is not None else None})
    return out


def run_case(args, payload, path, case, lmbd, n_nodes):
    labeled = case == "labeled"
    graphs = make_graphs(n_nodes, args, labeled, 0)
    Ps, vs, ws = build_inputs(graphs, args.u_w_distribution, labeled, args.seed)
    exact_matrix, method, reference_time = exact_gram(Ps, vs, ws, args, lmbd, labeled, n_nodes)
    pairs = np.triu_indices(len(graphs), k=1)
    exact = exact_matrix[pairs]
    w = np.maximum(*(np.array([float(ws[k].max()) for k in index]) for index in pairs))
    if labeled:
        labels = sorted({d["label"] for g in graphs for *_, d in g.edges(data=True)})
        R = variance_factor({label: 1/len(labels) for label in labels}, lmbd, args.kind)
    else:
        R = "unlabeled"
    estimates, C, mc_time = nested_estimates(Ps, vs, ws, args, lmbd, labeled, pairs)
    rows = statistics(estimates, exact, w, C, R, args)
    positive = [(row["m"], row["rel_rmse"]) for row in rows if row["rel_rmse"] > 0]
    slope = float(np.polyfit(*np.log(np.array(positive).T), 1)[0]) if len(positive) > 1 else None
    base = {"case": case, "lmbd": lmbd, "n_nodes": n_nodes}
    payload["records"] += [{**base, **row} for row in rows]
    payload["summaries"].append({**base, "slope": slope, "reference": method, "reference_time_sec": reference_time,
        "mc_time_largest_m_sec": mc_time, "C": C, "R_q": None if R == "unlabeled" else R,
        "n_pairs": len(exact), "exact_min": float(exact.min()), "exact_max": float(exact.max()),
        "w_inf_median": float(np.median(w)), **common.graph_stats(graphs, labeled),
        "required_m": required_budgets(rows, exact, w, C, R, args)})
    print(f"  N={n_nodes}: slope {slope:.3f}, rel RMSE {rows[0]['rel_rmse']:.3g} (m={rows[0]['m']}) -> "
          f"{rows[-1]['rel_rmse']:.3g} (m={rows[-1]['m']})", flush=True)
    common.save(payload, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_graph_arguments(parser)
    parser.set_defaults(n_graphs=8, n_repeats=50)
    option(parser,"output_dir",default="results/exps/convergence")
    option(parser,"cases",nargs="+",choices=common.CASES,default=list(common.CASES))
    option(parser,"sizes",type=int,nargs="+",default=[16,64,256,1024])
    option(parser,"m_values",type=int,nargs="+",default=[10,30,100,300,1000,3000,10000])
    option(parser,"kind",choices=["geom","exp"],default="geom")
    option(parser,"lambdas",type=float,nargs="+",default=[0.3,0.7],
           help="0.3 < 1/3 keeps the labeled bound finite for 3 labels; 0.7 is the paper default.")
    option(parser,"u_w_distribution",choices=DISTRIBUTIONS,default="normal")
    option(parser,"epsilons",type=float,nargs="+",default=[0.05,0.1,0.2],help="Relative errors for the tails.")
    option(parser,"deltas",type=float,nargs="+",default=[0.05])
    option(parser,"n_label_samples_per_length",type=int,default=1)
    option(parser,"solver_tol",type=float,default=1e-10)
    option(parser,"max_iter",type=int,default=5000)
    option(parser,"direct_max_nodes",type=int,default=128)
    option(parser,"seed",type=int,default=42)
    option(parser,"experiment_name",default="run")
    option(parser,"output",help="Exact JSON path; default is <output-dir>/<experiment-name>_<UTC stamp>.json.")
    args = parser.parse_args(argv)
    validate_graph_arguments(parser, args, args.sizes)
    if args.n_graphs < 2 or args.n_repeats < 2 or args.seed < 0:
        parser.error("need at least two graphs (one pair), two seeds and a nonnegative seed")
    n = args.n_label_samples_per_length
    if n < 1 or any(m < 1 or m % n for m in args.m_values):
        parser.error("m values must be positive multiples of n_label_samples_per_length")
    if any(not 0 < e for e in args.epsilons) or any(not 0 < d < 1 for d in args.deltas):
        parser.error("require eps > 0 and 0 < delta < 1")
    args.m_values = sorted(set(args.m_values))
    for lmbd in args.lambdas:
        try:
            utils.mu_func_gen(args.kind, lmbd)
        except ValueError as exc:
            parser.error(f"lambda={lmbd}: {exc}")
    return args


def main(argv=None):
    args = parse_args(argv)
    path = common.output_path(args)
    payload = {**common.new_payload("convergence", args), "summaries": []}
    for case in args.cases:
        for lmbd in args.lambdas:
            print(f"== {case} | lambda={lmbd}", flush=True)
            for n_nodes in args.sizes:
                run_case(args, payload, path, case, lmbd, n_nodes)
    print(f"Saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
