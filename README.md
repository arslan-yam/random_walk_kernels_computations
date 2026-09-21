# Monte Carlo Random Walk Kernels

Research code accompanying the Monte Carlo Random Walk Kernel (MCRWK) project.
The repository implements MCRWK, Graph Voyagers (GVoys), and deterministic
reference methods, with experiments on synthetic graphs and TU datasets.

## Installation

Python 3.11 or newer is required. Run the following commands from this directory:

```bash
python -m pip install -r requirements-bench.txt
python -m unittest discover -s tests -v
```

The experiment scripts require NumPy, SciPy, NetworkX, and scikit-learn.
The full `pyproject.toml` also includes dependencies used by the notebooks.

## Kernel definition

The input is a row-normalized adjacency matrix $P=D^{-1}A$, a starting
probability vector $v$, and nonnegative terminal weights $w$. The experiment
scripts normalize both boundary vectors to sum to one. For unlabeled graphs,

$$
h_G(k)=v_G^\top P_G^k w_G,\qquad
K(G,H)=\sum_{k=0}^{\infty}\mu_k h_G(k)h_H(k).
$$

Two coefficient families are supported:

| `--kind` | Coefficients | Parameter range | MCRWK length distribution |
|---|---|---|---|
| `geom` | $\mu_k=\lambda^k$ | $0\leq\lambda<1$ | $\Pr(L=k)=(1-\lambda)\lambda^k$ |
| `exp` | $\mu_k=\lambda^k/k!$ | $\lambda\geq0$ | Poisson with mean $\lambda$ |

For edge-labeled graphs, each $P_\ell$ restricts the **same** normalized matrix
$P$ to edges with label $\ell$. The kernel sums over matching label sequences;
the label matrices are not normalized separately. Vertex labels are not used.

GVoys and CG internally use the equivalent symmetric representation

$$
S=D^{1/2}PD^{-1/2}=D^{-1/2}AD^{-1/2},\qquad
v'=D^{-1/2}v,\quad w'=D^{1/2}w,
$$

so that $v^\top P^k w=v'^\top S^k w'$. The transformed vectors are weights,
not probability distributions. GVoys samples uniformly among available
transitions and applies the importance correction `number_of_choices * S[x,y]`.
The public API accepts **P, not S**; using S with unchanged boundary vectors
would define a different kernel. GVoys and CG require reversible inputs.

Isolated vertices are absorbing in the unlabeled representation and have zero
rows in the labeled representation. These conventions differ even if all real
edges have the same label.

### MCRWK replicas and GVoys features

MCRWK uses two conditionally independent trajectory replicas,
sharing the same outer random variables across graphs. With
$\bar F=(F_1+F_2)/2$, they estimate

$$
\widehat K_{ij}=\frac{C}{M}\sum_s\bar F_{is}\bar F_{js}\quad(i\ne j),
\qquad
\widehat K_{ii}=\frac{C}{M}\sum_s F_{1,is}F_{2,is},
$$

where $C=\sum_k\mu_k$. Averaging replicas
reduces off-diagonal noise; the cross product removes diagonal sampling bias.
The resulting raw estimate is unbiased and symmetric, but **not necessarily
positive semidefinite (PSD)**. Gram normalization or PSD projection generally
changes its expectation. No automatic PSD projection is applied.

GVoys uses one feature realization per graph and outer sample, with its
left/right walk constructions. Its dataset matrix is $FF^\top/M$, including
the diagonal, and is PSD up to numerical roundoff. There is no extra replica
or diagonal correction. The independent-graph pair estimator is unbiased;
squaring a reused random feature on the dataset diagonal is a distinct
construction and can add conditional sampling variance. The pair API uses
independent graph realizations even when both arguments are the same graph.

## Running experiments

Both main scripts expose their options through `--help`. Hyphenated flags and
historical underscore spellings are equivalent, for example
`--n-samples-mc` and `--n_samples_mc`.

### Synthetic graphs

A small unlabeled comparison with nonuniform boundary weights:

```bash
python synthetic_bench.py \
  --graph-type ba --ba-m 2 --n-nodes 32 --n-graphs 5 \
  --kind geom --lmbd 0.3 --u-w-distribution normal \
  --methods direct series cg gvoys mc \
  --mc-budget-mode total --n-samples-mc 2000 --n-samples-gvoys 100 \
  --graph-seed 7 --seed 41 --experiment-name ba_normal
```

An edge-labeled comparison with explicit nested sampling budgets:

```bash
python synthetic_bench.py \
  --graph-type er --n-nodes 32 --n-graphs 5 --p-er 0.1 \
  --labeled --n-labels 3 --u-w-distribution degree \
  --kind geom --lmbd 0.3 --methods direct cg gvoys mc \
  --n-length-samples 500 --n-label-samples-per-length 4 --n-walk-reps 2 \
  --q-sampling-kind norm_fro --n-samples-gvoys 100 \
  --graph-seed 7 --seed 41 --experiment-name er_labeled_degree
```

For an exponential kernel, use `--kind exp --lmbd 0.3`; compatible methods
are `direct`, `gvoys`, `mc`, and, without edge labels, `series`.

### TU datasets

Download/load a dataset, compute kernels, and run nested SVM cross-validation:

```bash
python dataset_bench.py \
  --datasets MUTAG --root-dir tu_datasets \
  --max-graphs 100 --max-nodes-per-graph 50 --labeled \
  --kind geom --lmbd 0.3 --u-w-distribution degree \
  --methods direct cg gvoys mc \
  --n-length-samples 1000 --n-label-samples-per-length 4 \
  --n-samples-gvoys 200 --block-size 64 \
  --n-splits 5 --inner-splits 3 --n-repeats 1 --c-values 0.1 1 10 \
  --check-psd --save-grams --seed 41 --experiment-name mutag_degree
```

Add `--skip-svm` to measure kernel approximation and runtime only. Remove
`--labeled` for unlabeled kernels. If requested edge labels are absent, the TU
loader reports the fallback and uses the unlabeled representation.

## Argument reference

### Methods and common settings

| Argument | Default | Meaning and choices |
|---|---|---|
| `--methods` | `direct cg fixed_point series gvoys mc` | Space-separated methods from the table below. For geom with a graph of at least 128 vertices, CG is automatically added as the reference if omitted. |
| `--kind` | `geom` | `geom` or `exp`; see the coefficient definitions above. |
| `--lmbd` | `0.1` | Kernel parameter $\lambda$, used when `--lambda-mode fixed`. |
| `--lambda-mode` | `fixed` | `fixed`: use `--lmbd`. `degree`: use $1/\max(2,d_{\max})^2$, where $d_{\max}$ is the maximum unweighted degree over the generated/selected graphs. This is independent of the boundary distribution. |
| `--u-w-distribution` | `uniform` | `uniform`, `random`, `normal`, or `degree`; definitions below. |
| `--labeled` | `0` | Enable edge labels. Accepts `--labeled`, `--labeled 1`, or `--labeled 0`. |
| `--seed` | `42` | Seed for boundary sampling, stochastic estimators, and TU subset selection/CV. |
| `--solver-tol` | `1e-10` | Relative residual tolerance for CG/GMRES/FPI; absolute entrywise truncation-error bound for `series`. Does not control `direct` or `sylvester`. |
| `--max-iter` | `5000` | Iteration limit for CG/GMRES/FPI; maximum number of terms for `series`. |
| `--direct-max-nodes` | `128` | Skip `direct` if any graph has **this many vertices or more**. May be lowered to 0 to disable direct; cannot exceed the hard safety cutoff of 128. |
| `--sylvester-max-nodes` | `512` | Corresponding size limit for `sylvester`. |
| `--fail-fast` | off | Stop at the first method failure instead of recording it and continuing. |

| Method | Supported kernels | Edge labels | Implementation |
|---|---|---|---|
| `mc` | geom, exp | Yes | MCRWK with shared lengths and independent replicas. |
| `gvoys` | geom, exp | Yes | GVoys with symmetric normalization, shared signs, and one feature realization per graph. |
| `direct` | geom, exp | Yes | Explicit product graph; sparse linear solve or matrix-exponential action. |
| `cg` | geom | Yes | Matrix-free CG on the equivalent symmetric system; reversible inputs only. |
| `gmres` | geom | Yes | Matrix-free GMRES, also supporting nonreversible inputs through the Python API. |
| `fixed_point` | geom | Yes | Matrix-free fixed-point iteration. |
| `sylvester` | geom | No | Dense Schur/Sylvester computation. |
| `series` | geom, exp | No | Deterministic per-graph walk features with a controlled tail bound. |

Incompatible methods are recorded as `skipped`. Reference selection uses the
largest graph in the generated dataset or selected TU subset:

| Largest graph | Direct computation | Reference for geom |
|---|---|---|
| Fewer than 128 vertices | Allowed, subject to `--direct-max-nodes`. | First available: `direct`, `series`, `cg`, `gmres`, `fixed_point`. |
| 128 vertices or more | **Always skipped**, before constructing a product graph. | **CG**, even if series was also computed. |

CG is a deterministic numerical reference, not an algebraically exact value.
It uses the symmetric system, `--solver-tol` (default `1e-10`), and an explicit
residual check. At or above 128 vertices, it runs even when omitted from
`--methods`. If it fails to converge, its status is `failed`, the reference and
approximation errors are `null`, and the process exits with a nonzero code;
another method is not silently substituted.

CG does not support exp. For exp the reference is direct, if computed within
the size limit, otherwise series for unlabeled inputs if requested. If neither
is available, errors are `null`. Product-graph methods may require substantial
memory even below the vertex-count limit. The hard cap applies to experiment
scripts; low-level solver functions remain available for explicit Python calls.

### Boundary distributions

The same option selects the construction of both $v$ and $w$, in graph node
order. All choices produce nonnegative vectors summing to one.

| Value | Construction | Relationship between $v$ and $w$ |
|---|---|---|
| `uniform` | Each coordinate equals $1/|V|$. | Identical. |
| `random` | Draw independent $x_i\sim U(0,1)$ and normalize by $\sum_i x_i$. | Independent draws. |
| `normal` | Draw independent $z_i\sim N(0,1)$, take $x_i=|z_i|$, then normalize. These are **half-normal weights**, ensuring valid probabilities. | Independent draws. |
| `degree` | $d_i/\sum_j d_j$, with $d_i=\sum_j A_{ij}$, using edge weights when present. Self-loops count once, consistently with P. | Identical. |

For `degree`, isolated vertices receive zero mass; a graph with no positive
edge weights falls back to `uniform`. Random choices are reproducible with
`--seed`, and every method in a run uses the same sampled boundary vectors.
**Compatibility:** older versions used `normal` as an alias for uniform random
weights. Use `random` to retain that distribution; `normal` now uses Gaussian draws.

Two unlabeled special cases are useful as controls:

- With `uniform`, $h_G(k)=1/|V_G|$, so $K(G,H)=C/(|V_G||V_H|)$.
- With `degree` on an undirected graph, $v$ is stationary and
  $h_G(k)=v^\top w=\sum_i(d_i/\sum_j d_j)^2$ for every length.

Both yield a rank-one dataset kernel, becoming all ones after diagonal
normalization. They should not be used alone to demonstrate structural
classification performance. Edge labels generally remove this simplification.
`random` and `normal` can provide nonconstant walk features, but a fixed realization
of random vertex weights is not automatically invariant to vertex relabeling.

### Sampling budgets

| Argument | Default | Meaning |
|---|---|---|
| `--n-samples-mc` | synthetic: `100`; TU: `200` | Base MCRWK feature budget. Synthetic experiments may scale it by graph size, as described below. |
| `--mc-budget-mode` | `per-node` | **Synthetic only:** `per-node` multiplies the base MC budget by `--n-nodes`; `total` uses it unchanged. TU always uses a total budget. |
| `--n-length-samples` | unset | **Labeled MC:** explicit number $m$ of independent lengths. Overrides derivation from the MC feature budget. |
| `--n-label-samples-per-length` | `1` | **Labeled MC:** number $n$ of independent label sequences conditional on each length. |
| `--n-walk-reps` | `1` | **Labeled MC:** trajectories averaged inside each of the two replicas, per sequence and graph. |
| `--q-sampling-kind` | `uniform` | **Labeled MC:** proposal for labels: `uniform`, `random`, `norm_fro`, or `norm_l1`. |
| `--n-samples-gvoys` | `200` | Number of outer GVoys samples. Each sample uses one left/right walk construction per graph, with walks from vertices with nonzero boundary weight. No extra replica is generated. |
| `--p-halt` | `0.2` | GVoys halt probability in $(0,1)$; expected sampled side length is $(1-p)/p$. |
| `--anchor-fraction` | `1.0` | GVoys anchor fraction in $(0,1]$: `max(1, floor(fraction * number_of_vertices))` anchors. |
| `--block-size` | `64` | Number of GVoys samples processed together. Bounds temporary feature storage; does not change sampled trajectories. |

Without labels, the effective MC budget is the number of shared lengths, with
two trajectory replicas per graph. The labeled-only options do not affect this
estimator. With labels, there are $m n$ feature columns. If `--n-length-samples`
is omitted, $m$ is the effective MC budget divided by
`--n-label-samples-per-length`; that division must be exact.
For example, `--n-length-samples 500 --n-label-samples-per-length 4` produces
2,000 feature columns but only **500 independent lengths**.

Label proposals use the union of labels in the dataset. `uniform` assigns equal
probabilities; `random` normalizes independent uniform draws; `norm_fro` and
`norm_l1` normalize the sums of per-graph label-matrix Frobenius norms and
entrywise L1 norms, respectively. A small uniform component preserves support
for nonuniform proposals. Label rarity and long lengths can produce large
importance weights; unbiasedness alone does not guarantee low or finite variance.

Equal `n_samples` values do not give equal computational budgets for MC and
GVoys. Compare error against elapsed time over several seeds. For synthetic
data, keep `--graph-seed` fixed to reuse the same graphs; changing `--seed` also
resamples `random`/`normal` boundary vectors. To study estimator noise with
fixed random boundaries, build inputs once using `src.benchmark.build_inputs`
and vary the seed passed to `src.benchmark.compute_kernel`.

### Synthetic graph options

| Argument | Default | Meaning and choices |
|---|---|---|
| `--n-graphs` | `10` | Number of graphs. |
| `--n-nodes` | `128` | Vertices per graph, at least 2. |
| `--graph-type` | `er` | `er`: Erdős–Rényi; `ba`: Barabási–Albert; `ws`: Watts–Strogatz; `sbm`: two-block stochastic block model. |
| `--graph-seed` | `0` | Graph $i$ is generated with seed `graph_seed + i`; also controls synthetic edge labels. |
| `--p-er` | `min(1, 2/n_nodes)` | ER edge probability in $[0,1]$. |
| `--ba-m` | `max(1, n_nodes // 20)` | BA attachment count, from 1 to `n_nodes - 1`. Fix it for sparse scaling experiments. |
| `--ws-k` | Largest even integer at most `min(n_nodes, max(2, 2*(n_nodes//20)))` | WS ring degree; must be even and between 0 and `n_nodes`. Rewiring probability is fixed at 0.1. |
| `--n-labels` | `3` | Number of uniformly sampled edge labels when `--labeled` is enabled. |

SBM uses two groups of sizes `n_nodes // 2` and the remainder, with within-group
probability 0.15 and between-group probability 0.02. Generator-specific options
are ignored by the other generators.

### TU data and classification options

| Argument | Default | Meaning |
|---|---|---|
| `--datasets` | `MUTAG ENZYMES NCI1 PTC_MR DD PROTEINS AIDS` | One or more TU dataset names, separated by spaces. |
| `--root-dir` | `tu_datasets` | Local dataset directory; missing datasets are downloaded. |
| `--max-graphs` | all | Maximum graphs after filtering; subsets are class-stratified. |
| `--max-nodes-per-graph` | unlimited | Exclude larger graphs before selecting the subset. |
| `--skip-svm` | off | Compute kernels and approximation errors without classification. |
| `--n-splits` | `5` | Outer stratified CV folds; reduced when class counts require it. |
| `--inner-splits` | `3` | Inner folds used to select SVM C; reduced for small classes. |
| `--n-repeats` | `1` | Repetitions of outer CV. |
| `--c-values` | `0.001 0.01 0.1 1 10 100` | Candidate SVM C values, separated by spaces. |
| `--no-normalize` | off | Disable diagonal Gram normalization before SVM. |
| `--check-psd` | off | Record the minimum eigenvalue and number of significantly negative eigenvalues; cubic cost, no projection. |

Subset selection requires at least two classes, with at least two eligible
graphs per class before subsampling. If an outer training fold has too few
examples for inner CV, the first C candidate is used. MCRWK can return
indefinite matrices; SVM accepting such a matrix does not restore
the usual PSD-kernel guarantees. Diagonal normalization requires strictly
positive diagonal entries and otherwise reports a failure.

## Results and reproducibility

Both scripts accept `--output-dir`, `--experiment-name` (default `run`), and
`--save-grams`. Default directories are `results_v2/synthetic` and
`results_v2/tu_benchmark`; TU results are grouped by dataset. Filenames include
the seed and a UTC timestamp. Synthetic experiments additionally accept
`--output path.json` and `--overwrite` for an explicit output filename.

JSON files record timings, raw-kernel errors (including separate diagonal and
off-diagonal errors), effective budgets, CLI arguments, seeds, library versions,
and a source-code hash. TU runs also record selected graph indices and SVM
metrics. Common input preparation is timed separately; method-specific
preparation is included in kernel computation time. Saved matrices are raw:
TU uses NPZ files containing `raw`, `indices`, and `y`; synthetic runs use a
pickle dictionary keyed by method.

Errors are measured before Gram normalization. Methods report `ok`, `skipped`,
or `failed`; failures normally allow remaining methods to run but produce a
nonzero exit code. Method seeds are independent of the ordering of `--methods`.
TU dataset seeds are `seed + dataset_index`; use one dataset per command when
reordering jobs should leave its inputs unchanged. Keep parameters, source
revision, and dependency versions fixed when comparing results.

## Repository layout

- `src/mcrwk.py`, `src/gvoys.py`: stochastic kernel estimators.
- `src/rwk.py`, `src/gram.py`: reference solvers, Gram construction, and error metrics.
- `src/normalization.py`, `src/_validation.py`: normalization and input checks.
- `src/benchmark.py`: shared experiment configuration and input preparation.
- `synthetic_bench.py`, `dataset_bench.py`: main experiment entry points.
- `fixed_samples.py`, `kernel_kmeans.py`, `graph_gp.py`: auxiliary sample-budget,
  clustering, and regression experiments, each with its own `--help`. Their
  boundary weights currently use independent normalized uniform draws.
- `tests/`: mathematical and experiment regression checks.
