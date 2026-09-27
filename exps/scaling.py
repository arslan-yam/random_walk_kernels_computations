"""Experiment 1: runtime and relative error versus graph size.

For every size N (default 8, 16, ..., 8192) and repeat, a pair of sparse
synthetic graphs (BA with m=2 by default; labeled graphs get 3 edge labels,
each with probability 1/3) is built with normal boundary vectors. All methods
compute the same Gram matrix and are compared with the reference: direct for
N < 128 (its hard cap), CG otherwise. Sylvester runs only unlabeled.

MCRWK runs with m = c*N (mc_matched; c calibrated once per case at
--calibration-n so that it takes as long as GVoys) and with the fixed budgets
of --mc-fixed-m. GVoys uses a fixed number of samples, so its cost is linear
in N.

Paper run:
    python exps/scaling.py
Quick check:
    python exps/scaling.py --sizes 8 16 --n-repeats 1 --n-samples-gvoys 5 \\
        --mc-fixed-m 100 --calibration-n 8 --calibration-repeats 1
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exps import common
from src import utils
from src.benchmark import build_inputs, option

# Graph seeds of calibration graphs; repeats use graph_seed + 10000*repeat + i.
CALIBRATION_SEED_OFFSET = 1_000_000


def add_graph_arguments(parser):
    option(parser,"graph_type",choices=["er","ba","ws","sbm"],default="ba")
    option(parser,"ba_m",type=int,default=2,help="Fixed BA attachment count keeps graphs sparse at every size.")
    option(parser,"p_er",type=float,help="ER edge probability; default min(1, 2/N).")
    option(parser,"ws_k",type=int)
    option(parser,"n_labels",type=int,default=3,help="Edge labels, each drawn with probability 1/n_labels.")
    option(parser,"n_graphs",type=int,default=2,help="Graphs per Gram matrix; 2 is one pair.")
    option(parser,"n_repeats",type=int,default=3,help="Independent graphs, boundary vectors and estimator seeds.")
    option(parser,"graph_seed",type=int,default=0)


def validate_graph_arguments(parser, args, sizes):
    if min(sizes) < 2 or args.n_graphs < 1 or args.n_repeats < 1 or args.n_labels < 1 or args.graph_seed < 0:
        parser.error("require sizes >= 2, n_graphs, n_repeats, n_labels >= 1 and nonnegative graph_seed")
    if args.graph_type == "ba" and min(sizes) <= args.ba_m:
        parser.error("BA graphs need more vertices than --ba-m")


def make_graphs(n_nodes, args, labeled, seed_offset):
    kwargs = dict(kind=args.graph_type, p_er=args.p_er, ba_m=args.ba_m, ws_k=args.ws_k)
    seeds = [args.graph_seed+seed_offset+i for i in range(args.n_graphs)]
    if labeled:
        return [utils.graph_generator_labeled(n_nodes, n_labels=args.n_labels, seed=s, **kwargs) for s in seeds]
    return [utils.graph_generator(n_nodes, seed=s, **kwargs) for s in seeds]


def synthetic_calibration(args, config, labeled, n_nodes):
    """c for mc_matched, timed on separate graphs of n_nodes vertices."""
    def inputs():
        graphs = make_graphs(n_nodes, args, labeled, CALIBRATION_SEED_OFFSET)
        return build_inputs(graphs, args.u_w_distribution, labeled, args.seed)
    if "mc_matched" in args.methods and args.mc_c is None:
        print(f"calibrating c at N={n_nodes} ...", flush=True)
    return common.matched_c(args, config, labeled, inputs, n_nodes, args.seed)


def run_size(n_nodes, args, config, labeled, c, repeat, tag):
    """All methods on one Gram of n_graphs graphs with n_nodes vertices."""
    graphs = make_graphs(n_nodes, args, labeled, 10_000*repeat)
    seed = args.seed+repeat
    Ps, vs, ws, input_time = common.timed_inputs(graphs, args.u_w_distribution, labeled, seed)
    runs = common.expand_methods(args.methods, n_nodes, args.mc_fixed_m, c)
    records, _ = common.compute_methods(runs, Ps, vs, ws, config, labeled, seed,
                                        max_nodes=n_nodes, args=args, tag=tag)
    stats = {"n_nodes": n_nodes, "repeat": repeat, "labeled": labeled, "lmbd": config.lmbd,
             "input_time_sec": input_time, **common.graph_stats(graphs, labeled)}
    return [{**stats, **record} for record in records]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_kernel_arguments(parser, gvoys_samples=100, output_dir="results/exps/scaling")
    add_graph_arguments(parser)
    option(parser,"sizes",type=int,nargs="+",default=[2**p for p in range(3,14)])
    option(parser,"calibration_n",type=int,default=1024,help="Graph size at which c of mc_matched is calibrated.")
    args = parser.parse_args(argv)
    common.validate_kernel_arguments(parser, args)
    validate_graph_arguments(parser, args, args.sizes+[args.calibration_n])
    return args


def main(argv=None):
    args = parse_args(argv)
    path = common.output_path(args)
    payload = common.new_payload("scaling", args)
    config = common.kernel_config(args)
    for case in args.cases:
        labeled = case == "labeled"
        c, calibration = synthetic_calibration(args, config, labeled, args.calibration_n)
        if calibration:
            payload["calibrations"].append({"case": case, "lmbd": config.lmbd, **calibration})
        for n_nodes in args.sizes:
            for repeat in range(args.n_repeats):
                rows = run_size(n_nodes, args, config, labeled, c, repeat, f"[{case} N={n_nodes} r={repeat}]")
                payload["records"] += [{"case": case, **row} for row in rows]
                common.save(payload, path)
    common.save(payload, path)
    print(f"Saved {path}", flush=True)
    return int(common.failed(payload["records"]))


if __name__ == "__main__":
    raise SystemExit(main())
