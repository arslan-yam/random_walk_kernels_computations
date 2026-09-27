"""Experiment 2: SVM classification and SVR regression with RWK Gram matrices.

Classification datasets (MUTAG, ENZYMES, PTC_MR, AIDS, NCI1) are evaluated
with nested-CV SVC accuracy; datasets without class labels but with graph
attributes (ZINC_test: constrained solubility) with nested-CV SVR
(RMSE, MAE, R^2). The task is detected from the TU files. Gram time and
evaluation time are recorded separately, together with the approximation
error against direct (all graphs of these datasets have < 128 vertices).

The labeled case uses TU edge labels; datasets without them (ENZYMES, NCI1)
are skipped there. mc_matched uses m = c*N with N the mean graph size; c is
calibrated on --calibration-graphs random graphs of each dataset.

Paper run (full datasets):
    python exps/tu_svm.py
Quick check:
    python exps/tu_svm.py --datasets MUTAG --max-graphs 6 --n-splits 2 --inner-splits 2 \\
        --n-samples-gvoys 5 --mc-fixed-m 100 --calibration-graphs 3 --calibration-repeats 1
"""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from exps import common
from src.benchmark import build_inputs, option


def prepare_dataset(name, args, labeled):
    """(graphs, targets, task, selected indices), or None without edge labels."""
    graphs, y, task, has_edge_labels = common.load_tu(name, args.root_dir, edge_labels=labeled)
    if labeled and not has_edge_labels:
        return None
    graphs, y, indices = common.select_subset(graphs, y, task, args.max_graphs, args.max_nodes_per_graph, args.seed)
    return graphs, y, task, indices


def dataset_calibration(args, config, labeled, graphs):
    subset = common.calibration_subset(graphs, args.calibration_graphs, args.seed)
    n_ref = float(np.mean([len(g) for g in subset]))
    if "mc_matched" in args.methods and args.mc_c is None:
        print(f"calibrating c on {len(subset)} graphs ...", flush=True)
    return common.matched_c(args, config, labeled,
                            lambda: build_inputs(subset, args.u_w_distribution, labeled, args.seed),
                            n_ref, args.seed)


def run_dataset(name, graphs, y, task, args, config, labeled, c, *, seed, evaluate, tag):
    """All methods on one dataset; boundaries use args.seed, estimators and CV use seed."""
    Ps, vs, ws = build_inputs(graphs, args.u_w_distribution, labeled, args.seed)
    sizes = [len(g) for g in graphs]
    runs = common.expand_methods(args.methods, float(np.mean(sizes)), args.mc_fixed_m, c)
    records, matrices = common.compute_methods(runs, Ps, vs, ws, config, labeled, seed,
                                               max_nodes=max(sizes), args=args, tag=tag)
    for record in records:
        K = matrices.get(record["name"])
        if not evaluate or K is None:
            continue
        t0 = time.perf_counter()
        try:
            record["evaluation"] = common.evaluate(K, y, task, args, seed)
            record["eval_time_sec"] = time.perf_counter()-t0
        except Exception as exc:
            record.update(status="failed", error=f"{task}: {exc}")
            if args.fail_fast:
                raise
    stats = {"dataset": name, "task": task, "labeled": labeled, "lmbd": config.lmbd,
             "n_graphs": len(graphs), "mean_nodes": float(np.mean(sizes)), "max_nodes": max(sizes)}
    return [{**stats, **record} for record in records], matrices


def run_datasets(args, payload, path, config, *, evaluate, extra=None):
    """Loop datasets x cases; one calibration per pair. Used by experiments 2 and 4."""
    extra = extra or {}
    for name in args.datasets:
        for case in args.cases:
            labeled = case == "labeled"
            print(f"== {name} | {case} | lambda={config.lmbd}", flush=True)
            try:
                prepared = prepare_dataset(name, args, labeled)
                if prepared is None:
                    print("  skipped: no edge labels", flush=True)
                    payload["records"].append({**extra, "dataset": name, "case": case, "status": "skipped",
                                               "reason": "dataset has no edge labels"})
                    continue
                graphs, y, task, indices = prepared
                c, calibration = dataset_calibration(args, config, labeled, graphs)
                if calibration:
                    payload["calibrations"].append({**extra, "dataset": name, "case": case,
                                                    "lmbd": config.lmbd, **calibration})
                rows, matrices = run_dataset(name, graphs, y, task, args, config, labeled, c,
                                             seed=args.seed, evaluate=evaluate, tag=f"[{name} {case}]")
                payload["records"] += [{**extra, "case": case, **row} for row in rows]
                if getattr(args, "save_grams", False):
                    for method, K in matrices.items():
                        np.savez_compressed(path.with_name(f"{path.stem}_{name}_{case}_{method}.npz"),
                                            raw=K, indices=indices, y=y)
            except Exception as exc:
                print(f"  failed: {exc}", flush=True)
                payload["records"].append({**extra, "dataset": name, "case": case, "status": "failed",
                                           "error": str(exc)})
                if args.fail_fast:
                    raise
            finally:
                common.save(payload, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_kernel_arguments(parser, gvoys_samples=200, output_dir="results/exps/tu_svm")
    common.add_tu_arguments(parser, datasets=common.CLASSIFICATION_DATASETS+common.REGRESSION_DATASETS)
    common.add_evaluation_arguments(parser)
    option(parser,"save_grams",action="store_true",help="Save raw Gram matrices as NPZ next to the JSON.")
    args = parser.parse_args(argv)
    common.validate_kernel_arguments(parser, args)
    if args.n_splits < 2 or args.inner_splits < 2 or args.n_cv_repeats < 1 or args.calibration_graphs < 1:
        parser.error("invalid CV or calibration parameters")
    return args


def main(argv=None):
    args = parse_args(argv)
    path = common.output_path(args)
    payload = common.new_payload("tu_svm", args)
    run_datasets(args, payload, path, common.kernel_config(args), evaluate=True)
    print(f"Saved {path}", flush=True)
    return int(common.failed(payload["records"]))


if __name__ == "__main__":
    raise SystemExit(main())
