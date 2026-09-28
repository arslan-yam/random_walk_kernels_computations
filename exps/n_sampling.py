"""Experiment 6: label sequences per length n for labeled MCRWK on TU datasets.

Labeled MCRWK samples m lengths and n label sequences per length. Its variance
bound (10) is C^2 w^4 / m * (R(q)/n + 1/4). Two designs test it on labeled TU
datasets (default MUTAG, PTC_MR, AIDS, ZINC_test), each with --n-repeats seeds:
  fixed_budget   B = m*n walks per replica and graph, m = B/n (--budgets):
                 the paper predicts that n = 1 is best;
  fixed_lengths  m lengths, B = m*n grows with n (--lengths): the variance
                 drops with n only until about n = 4 R(q).
Records and aggregates have the same fields as exps/q_sampling.py, plus the
design, the bound and its two terms. The proposal is fixed (--proposal).

Paper run:
    python exps/n_sampling.py
Quick check:
    python exps/n_sampling.py --datasets MUTAG --max-graphs 8 --n-repeats 2 \\
        --n-values 1 4 --budgets 400 --lengths 100 --n-splits 2 --inner-splits 2
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exps.q_sampling import (add_arguments, mc_estimate, proposal, proposal_spec, run_experiment,
                             validate_arguments, variance_factor)
from src import mcrwk, utils
from src.benchmark import option


def bound(data, q, lmbd, kind, lengths, n):
    """Variance bound (10) with w_inf = largest stopping weight in the dataset."""
    R = variance_factor(q, lmbd, kind)
    C = mcrwk.kernel_normalizer(kind, utils.mu_func_gen(kind, lmbd))
    w4 = max(float(w.max()) for w in data.ws)**4
    if R is None:
        return {"R_q": None, "bound_finite": False}
    return {"R_q": R, "bound_finite": True, "variance_bound": C**2*w4/lengths*(R/n+0.25),
            "bound_label_term": C**2*w4/lengths*R/n, "bound_length_term": C**2*w4/lengths*0.25}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(parser, output_dir="results/exps/n_sampling", lambdas=[0.7], repeats=5)
    option(parser,"proposal",type=proposal_spec,default="uniform")
    option(parser,"n_values",type=int,nargs="+",default=[1,2,4,8,16])
    option(parser,"budgets",type=int,nargs="+",default=[1000,10000],help="Fixed B = m*n; empty list disables.")
    option(parser,"lengths",type=int,nargs="*",default=[1000],help="Fixed m; empty list disables.")
    args = parser.parse_args(argv)
    validate_arguments(parser, args)
    if any(v < 1 for v in args.n_values+args.budgets+args.lengths) or not args.budgets+args.lengths:
        parser.error("n values, budgets and lengths must be positive, and one design must be enabled")
    if len(args.mix_eps) != 1:
        parser.error("n_sampling uses a single --mix-eps value")
    return args


def main(argv=None):
    args = parse_args(argv)
    eps = args.mix_eps[0]

    def configurations(data, lmbd):
        q = None if args.proposal == "random" else proposal(args.proposal, data, eps, None)
        designs = [("fixed_budget", budget, None) for budget in args.budgets] + \
                  [("fixed_lengths", None, lengths) for lengths in args.lengths]
        for design, budget, fixed in designs:
            for n in args.n_values:
                lengths = fixed if fixed is not None else budget//n
                if lengths < 1:
                    continue
                base = {"design": design, "proposal": args.proposal, "mix_eps": eps,
                        "lengths": lengths, "n": n, "budget": lengths*n,
                        **(bound(data, q, lmbd, args.kind, lengths, n) if q else {})}
                yield base, mc_estimate(data, args, lmbd, args.proposal, eps, lengths, n), q

    return run_experiment(args, "n_sampling", configurations)


if __name__ == "__main__":
    raise SystemExit(main())
