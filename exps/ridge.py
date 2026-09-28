"""Experiment 8: ridge for every method, as one model on Grams and on features.

Same datasets, methods, cases, calibration and timings as exps/tu_svm.py, with
ridge in place of SVM, so that the kernel and the feature version of the same
method are the same model:
  evaluation         kernel ridge on the (diagonal-normalized) Gram of every
                     method: one model for all, the column for comparing them;
  evaluation_linear  ridge on the explicit features X of GVoys and of the
                     biased MCRWK diagonal (Gram X X^T; rows scaled to unit
                     length, which is the diagonal normalization of X X^T).
                     Its predictions equal those of kernel ridge on X X^T;
                     primal_dual_gap records the largest relative difference
                     on one split. With m features and n training graphs it
                     decomposes X^T X (m x m) when m <= n, O(n m^2 + m^3) and
                     so linear in the number of graphs; otherwise X X^T.
Classification is one-vs-rest least squares on +-1 codes (the ridge
classifier, predicting the largest score); regression uses standardized
targets. Targets are centred on each training fold, which leaves the
intercept unpenalized. alpha is chosen by the nested CV of exps/common.py,
with the folds of the SVM experiments.

The unbiased MCRWK Gram equals X X^T - diag(tau) with tau >= 0, so the biased
diagonal is kernel ridge with the graph-dependent ridge alpha + tau_i. The
unbiased Gram is evaluated only in kernel form: after normalization its
Woodbury form needs (alpha - tau_i/K_ii)^-1, and tau/K (about 0.3 on MUTAG)
lies inside the alpha grid, so that inverse is numerically unusable. The
eigenvalues s + alpha of an indefinite Gram that vanish are pseudo-inverted.

Paper run (full datasets):
    python exps/ridge.py
Quick check:
    python exps/ridge.py --datasets MUTAG ZINC_test --max-graphs 12 --n-splits 2 --inner-splits 2 \\
        --n-samples-gvoys 5 --mc-fixed-m 100 --calibration-graphs 3 --calibration-repeats 1
"""

import argparse
from collections import OrderedDict
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from exps import common
from exps.tu_svm import run_datasets
from src.benchmark import option
import dataset_bench as tu


class Ridge:
    """Ridge on a Gram K (dual) or on features X (primal when m <= n), with cached eigendecompositions.

    Eigendecompositions of a training block are reused over the alpha grid;
    the cache holds the inner folds plus the outer training set.
    """

    def __init__(self, task, y, *, K=None, X=None, cache_size=4, force_primal=False):
        self.classification = task == "classification"
        y = np.asarray(y)
        if self.classification:
            self.classes = np.unique(y)
            self.targets = np.where(y[:, None] == self.classes[None, :], 1., -1.)
        else:
            self.targets = y.astype(float)[:, None]
        self.K, self.X, self.force_primal = K, X, force_primal
        self.cache, self.cache_size = OrderedDict(), cache_size

    def _primal(self, n_train):
        return self.X is not None and (self.force_primal or self.K is None and self.X.shape[1] <= n_train)

    def _block(self, rows, cols):
        return self.K[np.ix_(rows, cols)] if self.K is not None else self.X[rows] @ self.X[cols].T

    def _decomposition(self, train):
        key = train.tobytes()
        if key not in self.cache:
            if self._primal(len(train)):
                Xtr = self.X[train]
                self.cache[key] = np.linalg.eigh(Xtr.T @ Xtr)
            else:
                self.cache[key] = np.linalg.eigh(self._block(train, train))
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def scores(self, train, test, alpha):
        T = self.targets[train]
        mean = T.mean(axis=0)
        scale = np.ones(T.shape[1]) if self.classification else T.std(axis=0)
        scale[scale == 0] = 1.
        s, U = self._decomposition(train)
        denominator = s+alpha
        # Pseudo-inverse where s + alpha vanishes (possible for the indefinite unbiased Gram).
        tolerance = 1e-12*max(np.abs(s).max(initial=0.), alpha)
        inverse = np.divide(1., denominator, out=np.zeros_like(denominator), where=np.abs(denominator) > tolerance)
        centred = (T-mean)/scale
        if self._primal(len(train)):
            Xtr = self.X[train]
            scores = self.X[test] @ (U @ ((U.T @ (Xtr.T @ centred))*inverse[:, None]))
        else:
            scores = self._block(test, train) @ (U @ ((U.T @ centred)*inverse[:, None]))
        return mean+scale*scores

    def __call__(self, train, test, params):
        alpha = params[0] if isinstance(params, tuple) else params
        scores = self.scores(train, test, alpha)
        return self.classes[np.argmax(scores, axis=1)] if self.classification else scores[:, 0]


def normalized(K=None, X=None, *, normalize=True):
    if not normalize:
        return K, X
    if K is not None:
        K = tu.normalize_gram_matrix(K)
    if X is not None:
        norms = np.linalg.norm(X, axis=1)
        if not np.isfinite(X).all() or np.any(norms <= 0):
            raise ValueError("feature normalization needs finite nonzero rows; use --no-normalize")
        X = X/norms[:, None]
    return K, X


def evaluate_ridge(y, task, args, seed, *, K=None, X=None):
    """Nested-CV ridge (alpha by inner accuracy or RMSE) on a Gram or on features."""
    K, X = normalized(K, X, normalize=not args.no_normalize)
    model = Ridge(task, y, K=K, X=X, cache_size=args.inner_splits+1)
    candidates = list(args.alphas) if task == "classification" else [(alpha,) for alpha in args.alphas]
    result = common.nested_cv(y, task, candidates, model, args.n_splits, args.n_cv_repeats, args.inner_splits, seed)
    selected = result.pop("selected_cs", None) or [p[0] for p in result.pop("selected_c_epsilon")]
    result["selected_alphas"] = selected
    result["solver"] = "primal" if model._primal(int(len(y)*(1-1/args.n_splits))) else "dual"
    return result


def primal_dual_gap(K, X, y, task, args, seed):
    """Largest relative difference of ridge scores from X X^T (dual) and from X (primal) on one split."""
    K, X = normalized(K, X, normalize=not args.no_normalize)
    y = np.asarray(y)
    splitter = (StratifiedKFold if task == "classification" else KFold)(n_splits=args.n_splits, shuffle=True,
                                                                         random_state=seed)
    train, test = next(splitter.split(np.zeros(len(y)), y))
    alpha = float(np.median(args.alphas))
    dual = Ridge(task, y, K=K).scores(train, test, alpha)
    primal = Ridge(task, y, X=X, force_primal=True).scores(train, test, alpha)
    return float(np.max(np.abs(dual-primal))/max(np.max(np.abs(dual)), 1e-300))


def evaluate_record(record, K, X, y, task, args, seed):
    """Kernel ridge on the Gram (evaluation) and ridge on features (evaluation_linear)."""
    for model, key, kwargs in (("kernel", "", {"K": K}), ("linear", "_linear", {"X": X})):
        if model in args.ridge and next(iter(kwargs.values())) is not None:
            t0 = time.perf_counter()
            record["evaluation"+key] = evaluate_ridge(y, task, args, seed, **kwargs)
            record["eval_time_sec"+key] = time.perf_counter()-t0
    if K is not None and X is not None and X.shape[1] <= args.gap_max_features:
        record["primal_dual_gap"] = primal_dual_gap(K, X, y, task, args, seed)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_kernel_arguments(parser, gvoys_samples=200, output_dir="results/exps/ridge",
                                mc_diagonals=("unbiased", "biased"))
    common.add_tu_arguments(parser, datasets=common.CLASSIFICATION_DATASETS+common.REGRESSION_DATASETS)
    option(parser,"alphas",type=float,nargs="+",default=[1e-3,1e-2,1e-1,1.,10.])
    option(parser,"ridge",nargs="+",choices=["kernel","linear"],default=["kernel","linear"],
           help="kernel: kernel ridge on the Gram of every method (evaluation). linear: ridge on the features "
                "of GVoys and the biased MCRWK diagonal (evaluation_linear), linear in the number of graphs.")
    option(parser,"gap_max_features",type=int,default=5000,
           help="Check primal = dual on one split when the feature dimension is at most this.")
    option(parser,"n_splits",type=int,default=5)
    option(parser,"inner_splits",type=int,default=3)
    option(parser,"n_cv_repeats",type=int,default=1)
    option(parser,"no_normalize",action="store_true",help="Disable diagonal Gram (feature row) normalization.")
    args = parser.parse_args(argv)
    common.validate_kernel_arguments(parser, args)
    if args.n_splits < 2 or args.inner_splits < 2 or args.n_cv_repeats < 1 or args.calibration_graphs < 1:
        parser.error("invalid CV or calibration parameters")
    if any(alpha <= 0 for alpha in args.alphas):
        parser.error("alphas must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    path = common.output_path(args)
    payload = common.new_payload("ridge", args)
    run_datasets(args, payload, path, common.kernel_config(args), evaluate=True, evaluator=evaluate_record)
    print(f"Saved {path}", flush=True)
    return int(common.failed(payload["records"]))


if __name__ == "__main__":
    raise SystemExit(main())
