# Paper experiments

Four CLI experiments and a plotting script. Run them from the repository root.
Each experiment writes one JSON file, by default to `results/exps/<experiment>/`.
The file is rewritten after every finished size or dataset, so an interrupted
run keeps its completed rows.

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
- Relative errors are measured against Direct for graphs with fewer than 128
  vertices, and against CG otherwise.

| Script | Experiment |
|---|---|
| `scaling.py` | Runtime and relative error for graph pairs with N = 8 … 8192 (BA, m = 2; labeled graphs have 3 labels, each with probability 1/3). Direct runs for N < 128, Sylvester for N ≤ 1024. |
| `tu_svm.py` | Nested-CV SVM accuracy on MUTAG, ENZYMES, PTC_MR, AIDS and NCI1, and SVR regression (RMSE, MAE, R²) on ZINC_test. Records Gram time and evaluation time. |
| `gram_time.py` | Gram-matrix build time on the five classification datasets. `--n-graphs-list` also times subsets with those numbers of graphs. |
| `lambda_sweep.py` | λ = 0.1 … 0.9. The synthetic setting measures runtime and error; the TU setting measures Gram time, error and SVM accuracy. c is recalibrated for every λ. |
| `plot_results.py` | Writes a PNG and a markdown table for each figure. Results from several JSON files of one experiment are merged. |

In the labeled case, datasets without edge labels (ENZYMES, NCI1) are skipped.
TU datasets are downloaded into `--root-dir` (default `tu_datasets/`). By default
all datasets are used in full. Exact methods on NCI1 and AIDS then take many
hours, so use `--max-graphs` for class-stratified subsets.

## Paper runs

```bash
python exps/scaling.py
python exps/tu_svm.py
python exps/gram_time.py --n-graphs-list 100 200 400 800
python exps/lambda_sweep.py
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
```

Run `python exps/<script>.py --help` for all options. The smoke tests are in
`tests/test_exps.py`.
