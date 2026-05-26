# Monte Carlo Random Walk Kernel

This work offers new algorithm for computing the similarity characteristic between a pair of graphs, called Graph Random Walk Kernel (RWK). This algorithm is computable in linear time with respect to the graph dataset size and the size of graphs themselves. Suggested approach draws inspiration from the recently introduced linear algorithm called Graph Voyagers (GVoys) and offers a simpler way to compute RWK avoiding redundant random variables. We conduct a set of experiments and report, that our approach is faster than reference baseline GVoys while having same relative error rates.

# Repository

- `/src` contains implementations of baseline algorithms (`/src/rwk`, `/src/gvoys`) and code for the proposed Monte Carlo Random Walk Kernel at `/src/mcrwk`,
- `/notebooks` and contatins usage examples and examples of experiments

**Experiments**
- `synthetic_bench.py` conducts experiments on random graph generators
- `dataset_bench.py` conducts experiments on datasets from TUDatasets

Both scripts have cli parameters which can be viewed via `--help`, e.g `python synthetic_bench.py --help`. Both scripts compute MCRWK and baseline algorithms on chosen type of data and measure computation time. Additional to that, `synthetic_bench.py` measures acuracy of algorithms w.r.t exact RWK, if graph size is not grater than $2^7$, otherwise we use Conjugate Gradient method (CG) as precise enough reference.

# Results
The results we report in the paper support the main claim, MCRWK offers faster performance than GVoys, while having same relative error rates.

<figure style="text-align: center; margin: 0;">
  <div style="display: flex; gap: 15px; justify-content: center; margin-bottom: 10px;">
    <img src=fig/ba/uniform_labeled_time.png alt="Фото 1" style="width: 45%; max-width: 400px;">
    <img src=fig/ba/uniform_labeled_error.png alt="Фото 2" style="width: 45%; max-width: 400px;">
  </div>
  <figcaption>
    Fig. 1. Runtime and mean relative error of different methods on labeled Barabási–Albert graphs
for varying sizes of graphs, average over 3 
  </figcaption>
</figure>



<figure>
  <img src=fig/focused_aids_uniform_nci1_geom_normal_unlabeled.png alt="Dataset Benchmark">
  <figcaption>Fig. 2. Time to construct the Gram matrix for datasets, geometric kernel, unlabeled</figcaption>
</figure>
