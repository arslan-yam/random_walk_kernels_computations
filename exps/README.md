# Paper experiments

Eight CLI experiments and a plotting script. Run them from the repository root.
Each experiment writes one JSON file, by default to `results/exps/<experiment>/`.
The file is rewritten after every finished size or dataset, so an interrupted
run keeps its completed rows.

The JSON holds one record per method run with the following fields:

- `time_sec` (Gram time) and `input_time_sec` (building P, v, w);
- `errors` against the reference (every metric of `src.gram.matrix_errors`)
  and `diagonal_min`;
- the MCRWK budget `m` and, for `mc_matched`, `c`;
- graph statistics;
- for TU runs, `evaluation`: accuracy or RMSE/MAE/R², per-fold scores and the
  selected C/ε, plus `eval_time_sec`.

Gram matrices with at most `--json-gram-max-graphs` graphs (default 16, so
every scaling pair) are stored in the record as `gram`. The `calibrations` list
records how each c was chosen. `metadata` holds the CLI arguments, library
versions and a source hash.

All experiments share the same settings:

- Kernel: geometric with λ = 0.7.
- Boundary vectors v and w: `normal` (normalized |N(0,1)|). This avoids the
  degenerate uniform kernel.
- Cases: unlabeled and edge-labeled.
- Methods: Direct, CG, FP, Sylvester (unlabeled only), GVoys, and two MCRWK
  budgets:
  - `mc_matched` uses m = cN. By default c is calibrated so that MCRWK takes
    as long as GVoys (see `--mc-c`, `--calibration-*`).
  - `mc_fixed` uses one run per `--mc-fixed-m` value (default 100, 1000,
    10000), independent of graph size.
- MCRWK Grams exist in two versions built from the same walks. The unbiased
  version replaces the diagonal with the replica cross product. The biased
  version (`<name>_biased`) keeps the diagonal of the averaged-feature Gram;
  it is PSD, and its excess τ ≥ 0 acts as a ridge. `tu_svm`, `lambda_sweep`,
  `q_sampling` and `n_sampling` evaluate both; `scaling` and `gram_time` only
  the unbiased one. The runtime is the same for both. Set this with
  `--mc-diagonals`. `--check-psd` records the smallest eigenvalue of each Gram.
- Two SVM evaluations, kept apart (`--svm kernel linear`, default both):
  - `evaluation`: SVC (or SVR) on the Gram of every method, exact ones
    included. All methods share one classifier, so this is the column for
    comparing methods.
  - `evaluation_linear`: LinearSVC (or LinearSVR) on the explicit features X
    of GVoys and of the biased MCRWK diagonal, whose Gram is XXᵀ. It never
    builds the graphs × graphs matrix, so its cost is linear in the number of
    graphs. Rows are normalized to unit length, which matches the diagonal
    normalization of XXᵀ. The loss is squared hinge (squared ε-insensitive for
    regression) with liblinear's primal solver, so compare this column only
    with itself and with each method's own SVC result. The unbiased diagonal
    and the exact methods have no feature form and appear only in
    `evaluation`.
  - Timings: `time_sec` is the time to the Gram for every method.
    `feature_time_sec` (features only, the linear path) and `gram_build_sec`
    are recorded for GVoys and MCRWK. With `--svm linear`, no exact method,
    `--mc-diagonals biased` and no `--check-psd`, no Gram is built at all, and
    `time_sec` is the feature time (`time_basis`).
- Relative errors are measured against Direct for graphs with fewer than 128
  vertices, and against CG otherwise.

| Script | Experiment |
|---|---|
| `scaling.py` | Runtime and relative error for graph pairs with N = 8 … 8192 (BA, m = 2; labeled graphs have 3 labels, each with probability 1/3). Direct runs for N < 128, Sylvester (unlabeled only) for N ≤ 512. |
| `tu_svm.py` | Nested-CV SVM accuracy on MUTAG, ENZYMES, PTC_MR, AIDS and NCI1, and SVR regression (RMSE, MAE, R²) on ZINC_test. Records Gram time and evaluation time. |
| `gram_time.py` | Gram-matrix build time on the five full classification datasets. For GVoys and MCRWK the feature construction before the Gram (`feature_time_sec`) is recorded and plotted separately from the full time. |
| `lambda_sweep.py` | λ = 0.1 … 0.9. The synthetic setting measures runtime and error; the TU setting measures Gram time, error and SVM accuracy. c is recalibrated for every λ. |
| `q_sampling.py` | Labeled only (MUTAG, PTC_MR, AIDS, ZINC_test). Compares importance-sampling proposals q: uniform, q ∝ p^α (`freq:α`), `sq_mean`, `norm_l1`, `norm_fro`, `inverse` and `random`, optionally mixed with uniform (`--mix-eps`). Runs each for m ∈ {100, 1000, 10000} and λ ∈ {0.3, 0.7}, with n = 1 and 5 seeds per point. |
| `n_sampling.py` | Varies the number of label sequences per length n ∈ {1, 2, 4, 8, 16}, either at a fixed budget B = m·n or at a fixed number of lengths m. The measured variance is recorded next to the bound (10) and its two terms. |
| `convergence.py` | Theory check on synthetic graphs of several sizes, against the exact kernel of every pair: relative RMSE against m (slope −½, the same for every N), measured variance against the bounds (6)/(10), tail frequencies against Hoeffding (7) and Chebyshev (11), and the m actually needed for P(\|k̂−k\| > εk) ≤ δ against (8)/(12). Budgets are nested prefixes of one largest-m run per seed. |
| `ridge.py` | Same datasets and methods as `tu_svm.py`, with ridge in place of SVM: kernel ridge on every Gram (`evaluation`) and ridge on the features of GVoys and the biased MCRWK diagonal (`evaluation_linear`). Both are the same model, and `primal_dual_gap` checks that numerically. |
| `plot_results.py` | Writes a PNG for each figure, plus its numbers (mean ± std over repeats) as `.md` and `.json`. Results from several JSON files of one experiment are merged. |

`q_sampling.py` and `n_sampling.py` write three lists:

- `records`: one entry per seed, with Gram time, error, walk diagnostics from
  `src.mcrwk` (walks, steps, killed and skipped walks, feature ESS, largest log
  weight, q) and the SVM/SVR result;
- `aggregates`: one entry per configuration, with bias, variance and MSE of
  every Gram entry over the seeds, MSE × time, `ridge_rel` (the mean τ_i/K_ii
  of the biased diagonal), the SVM/SVR result for both diagonals, means of the diagnostics and of
  the accuracy, q_min, and R(q) together with whether the bound is finite
  (λ < q_min);
- `references`: the exact Gram per dataset and λ, with its timing, label
  frequencies and its own SVM/SVR result.

Seeds are shared across proposals, so all proposals use the same sampled
lengths. The SVM folds are fixed, so the spread of accuracy over seeds comes
only from kernel noise.

In the labeled case, datasets without edge labels (ENZYMES, NCI1) are skipped.
TU datasets are downloaded into `--root-dir` (default `tu_datasets/`). By default
all datasets are used in full. Exact methods on NCI1 and AIDS then take many
hours, so use `--max-graphs` for class-stratified subsets.

## Paper runs

```bash
python exps/scaling.py
python exps/tu_svm.py
python exps/gram_time.py
python exps/lambda_sweep.py
python exps/q_sampling.py
python exps/n_sampling.py
python exps/convergence.py
python exps/ridge.py
python exps/plot_results.py results/exps/*/*.json --out-dir fig/exps
```

For fair runtimes, pin BLAS threads, for example by prefixing each command with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. The thread settings
are recorded in the JSON metadata.

## Quick checks (seconds)

```bash
python exps/scaling.py --sizes 8 16 --n-repeats 1 --n-samples-gvoys 5 --mc-fixed-m 100 --calibration-n 8 --calibration-repeats 1
python exps/tu_svm.py --datasets MUTAG ZINC_test --max-graphs 12 --n-splits 2 --inner-splits 2 --n-samples-gvoys 5 --mc-fixed-m 100 --calibration-graphs 3 --calibration-repeats 1
python exps/gram_time.py --datasets MUTAG --max-graphs 3 --n-samples-gvoys 5 --mc-fixed-m 100 --calibration-graphs 3 --calibration-repeats 1
python exps/lambda_sweep.py --lambdas 0.1 0.9 --n-nodes 8 --n-repeats 1 --datasets MUTAG --max-graphs 6 --n-splits 2 --inner-splits 2 --n-samples-gvoys 5 --mc-fixed-m 100 --calibration-graphs 3 --calibration-repeats 1
python exps/q_sampling.py --datasets MUTAG --max-graphs 8 --n-repeats 2 --m-values 100 --lambdas 0.7 --n-splits 2 --inner-splits 2
python exps/n_sampling.py --datasets MUTAG --max-graphs 8 --n-repeats 2 --n-values 1 4 --budgets 400 --lengths 100 --n-splits 2 --inner-splits 2
```

```bash
python exps/convergence.py --sizes 8 16 --n-graphs 3 --n-repeats 5 --m-values 10 100 --lambdas 0.3
python exps/ridge.py --datasets MUTAG ZINC_test --max-graphs 12 --n-splits 2 --inner-splits 2 --n-samples-gvoys 5 --mc-fixed-m 100 --calibration-graphs 3 --calibration-repeats 1
```

`run_experiments.ipynb` has the commands for a cluster, one job per dataset and case.

Run `python exps/<script>.py --help` for all options. The smoke tests are in
`tests/test_exps.py`.
