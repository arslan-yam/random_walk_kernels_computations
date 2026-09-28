"""Experiment 3: Gram-matrix build time on TU datasets (no SVM).

Times every method on the full MUTAG, ENZYMES, PTC_MR, AIDS and NCI1,
unlabeled and edge-labeled; datasets without edge labels are skipped in the
labeled case. For GVoys and MCRWK the feature construction (feature_time_sec,
before any Gram) is recorded apart from the Gram product (gram_build_sec);
time_sec is their sum, comparable with the exact methods. Approximation errors
against direct are recorded as well. --n-repeats repeats each timing with new
estimator seeds (boundary vectors stay fixed). c of mc_matched is calibrated
once per dataset and case.

Paper run:
    python exps/gram_time.py
Quick check:
    python exps/gram_time.py --datasets MUTAG --max-graphs 3 --n-samples-gvoys 5 \\
        --mc-fixed-m 100 --calibration-graphs 3 --calibration-repeats 1
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exps import common
from exps.tu_svm import dataset_calibration, prepare_dataset, run_dataset
from src.benchmark import option


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_kernel_arguments(parser, gvoys_samples=200, output_dir="results/exps/gram_time")
    common.add_tu_arguments(parser, datasets=common.CLASSIFICATION_DATASETS)
    option(parser,"n_repeats",type=int,default=1,help="Timings per method with different estimator seeds.")
    args = parser.parse_args(argv)
    common.validate_kernel_arguments(parser, args)
    if args.n_repeats < 1 or args.calibration_graphs < 1:
        parser.error("require n_repeats >= 1 and calibration_graphs >= 1")
    return args


def main(argv=None):
    args = parse_args(argv)
    path = common.output_path(args)
    payload = common.new_payload("gram_time", args)
    config = common.kernel_config(args)
    for name in args.datasets:
        for case in args.cases:
            labeled = case == "labeled"
            print(f"== {name} | {case}", flush=True)
            try:
                prepared = prepare_dataset(name, args, labeled)
                if prepared is None:
                    print("  skipped: no edge labels", flush=True)
                    payload["records"].append({"dataset": name, "case": case, "status": "skipped",
                                               "reason": "dataset has no edge labels"})
                    continue
                graphs, y, task, _ = prepared
                c, calibration = dataset_calibration(args, config, labeled, graphs)
                if calibration:
                    payload["calibrations"].append({"dataset": name, "case": case, "lmbd": config.lmbd, **calibration})
                for repeat in range(args.n_repeats):
                    rows, _, _ = run_dataset(name, graphs, y, task, args, config, labeled, c,
                                             seed=args.seed+repeat, evaluate=False,
                                             tag=f"[{name} {case} r={repeat}]")
                    payload["records"] += [{"case": case, "repeat": repeat, **row} for row in rows]
                    common.save(payload, path)
            except Exception as exc:
                print(f"  failed: {exc}", flush=True)
                payload["records"].append({"dataset": name, "case": case, "status": "failed", "error": str(exc)})
                if args.fail_fast:
                    raise
            finally:
                common.save(payload, path)
    print(f"Saved {path}", flush=True)
    return int(common.failed(payload["records"]))


if __name__ == "__main__":
    raise SystemExit(main())
