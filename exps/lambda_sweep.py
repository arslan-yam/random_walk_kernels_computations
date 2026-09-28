"""Experiment 4: effect of the geometric-kernel lambda on runtime and accuracy.

For each lambda in --lambdas (default 0.1, ..., 0.9):
  synthetic  pairs of --n-nodes-vertex graphs as in exps/scaling.py; runtime and
             relative error against direct (< 128 vertices) or CG.
  tu         TU datasets as in exps/tu_svm.py; Gram time, error against direct
             and SVM accuracy (SVR metrics for regression datasets).
The MCRWK walk length has mean lambda/(1-lambda), so the cost ratio to GVoys
changes with lambda: c of mc_matched is calibrated anew for every lambda
(and dataset/case) unless --mc-c is given. --lmbd is ignored here.

Paper run:
    python exps/lambda_sweep.py
Quick check:
    python exps/lambda_sweep.py --lambdas 0.1 0.9 --n-nodes 8 --n-repeats 1 \\
        --datasets MUTAG --max-graphs 6 --n-splits 2 --inner-splits 2 \\
        --n-samples-gvoys 5 --mc-fixed-m 100 --calibration-graphs 3 --calibration-repeats 1
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exps import common
from exps.scaling import add_graph_arguments, run_size, synthetic_calibration, validate_graph_arguments
from exps.tu_svm import run_datasets
from src.benchmark import option

SETTINGS = ("synthetic", "tu")


def run_synthetic(args, payload, path):
    for case in args.cases:
        labeled = case == "labeled"
        for lmbd in args.lambdas:
            config = common.kernel_config(args, lmbd)
            print(f"== synthetic | {case} | lambda={lmbd}", flush=True)
            c, calibration = synthetic_calibration(args, config, labeled, args.n_nodes)
            if calibration:
                payload["calibrations"].append({"setting": "synthetic", "case": case, "lmbd": lmbd, **calibration})
            for repeat in range(args.n_repeats):
                rows = run_size(args.n_nodes, args, config, labeled, c, repeat,
                                f"[{case} lambda={lmbd} r={repeat}]")
                payload["records"] += [{"setting": "synthetic", "case": case, **row} for row in rows]
                common.save(payload, path)


def run_tu(args, payload, path):
    for lmbd in args.lambdas:
        run_datasets(args, payload, path, common.kernel_config(args, lmbd), evaluate=True,
                     extra={"setting": "tu"})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_kernel_arguments(parser, gvoys_samples=100, output_dir="results/exps/lambda_sweep",
                                mc_diagonals=("unbiased", "biased"))
    option(parser,"settings",nargs="+",choices=SETTINGS,default=list(SETTINGS))
    option(parser,"lambdas",type=float,nargs="+",default=[round(0.1*i,1) for i in range(1,10)])
    add_graph_arguments(parser)
    option(parser,"n_nodes",type=int,default=256,help="Synthetic graph size.")
    common.add_tu_arguments(parser, datasets=common.CLASSIFICATION_DATASETS)
    common.add_evaluation_arguments(parser)
    args = parser.parse_args(argv)
    common.validate_kernel_arguments(parser, args)
    validate_graph_arguments(parser, args, [args.n_nodes])
    for lmbd in args.lambdas:
        try:
            common.kernel_config(args, lmbd)
        except ValueError as exc:
            parser.error(f"lambda={lmbd}: {exc}")
    if args.n_splits < 2 or args.inner_splits < 2 or args.n_cv_repeats < 1 or args.calibration_graphs < 1:
        parser.error("invalid CV or calibration parameters")
    return args


def main(argv=None):
    args = parse_args(argv)
    path = common.output_path(args)
    payload = common.new_payload("lambda_sweep", args)
    if "synthetic" in args.settings:
        run_synthetic(args, payload, path)
    if "tu" in args.settings:
        run_tu(args, payload, path)
    common.save(payload, path)
    print(f"Saved {path}", flush=True)
    return int(common.failed(payload["records"]))


if __name__ == "__main__":
    raise SystemExit(main())
