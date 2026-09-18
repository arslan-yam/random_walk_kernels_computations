"""Shared CLI configuration, dispatch and reproducibility metadata."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
import networkx as nx
from importlib.metadata import version

from . import gram, gvoys, mcrwk, utils
from ._validation import positive_int

METHODS = ("direct", "cg", "gmres", "fixed_point", "sylvester", "series", "gvoys", "mc")
DISTRIBUTIONS = ("uniform", "random", "normal", "degree")
DIRECT_NODE_LIMIT = 128
CG_REFERENCE_MIN_NODES = 128


@dataclass(frozen=True)
class KernelConfig:
    kind: str = "geom"
    lmbd: float = 0.1
    n_samples_mc: int = 200
    n_samples_gvoys: int = 200
    n_length_samples: int | None = None
    n_label_samples_per_length: int = 1
    n_walk_reps: int = 1
    q_sampling_kind: str = "uniform"
    p_halt: float = 0.2
    anchor_fraction: float = 1.0
    block_size: int = 64
    solver_tol: float = 1e-10
    max_iter: int = 5000

    def validate(self):
        utils.mu_func_gen(self.kind, self.lmbd)
        for name in ("n_samples_mc", "n_samples_gvoys", "n_label_samples_per_length",
                     "n_walk_reps", "block_size", "max_iter"):
            positive_int(getattr(self,name),name)
        if self.n_length_samples is not None:
            positive_int(self.n_length_samples,"n_length_samples")
        elif self.n_samples_mc % self.n_label_samples_per_length:
            raise ValueError("n_samples_mc must be divisible by n_label_samples_per_length; or set n_length_samples")
        if not 0 < self.p_halt < 1 or not 0 < self.anchor_fraction <= 1:
            raise ValueError("require 0<p_halt<1 and 0<anchor_fraction<=1")
        if not np.isfinite(self.solver_tol) or not 0 < self.solver_tol < 1:
            raise ValueError("solver_tol must be between 0 and 1")
        if self.q_sampling_kind not in {"uniform","random","norm_fro","norm_l1"}:
            raise ValueError("unknown label proposal")
        return self

    @property
    def lengths(self):
        return self.n_length_samples if self.n_length_samples is not None else self.n_samples_mc//self.n_label_samples_per_length

    def metadata(self):
        return {**asdict(self), "effective_labeled_lengths":self.lengths,
                "effective_labeled_features":self.lengths*self.n_label_samples_per_length,
                "mc_replicas":2,"gvoys_replicas":2,"gvoys_sides_per_replica":2}


def compute_kernel(method, Ps,vs,ws,config,seed=42,labeled=False):
    config.validate()
    mu = utils.mu_func_gen(config.kind,config.lmbd)
    if method == "direct":
        return gram.gram_direct(Ps,vs,ws,mu,config.kind,labeled)
    if method in {"cg","gmres","fixed_point"}:
        if config.kind != "geom":
            raise ValueError(f"{method} supports only geom")
        return getattr(gram,"gram_"+method)(Ps,vs,ws,mu,labeled,
                    eps=config.solver_tol,max_iter=config.max_iter)
    if method == "sylvester":
        if labeled or config.kind != "geom":
            raise ValueError("sylvester supports only unlabeled geom")
        return gram.gram_sylvester(Ps,vs,ws,mu)
    if method == "series":
        if labeled:
            raise ValueError("series supports only unlabeled inputs")
        return gram.gram_series(Ps,vs,ws,mu,config.kind,eps=config.solver_tol,max_iter=config.max_iter)
    if method == "gvoys":
        fun = gvoys.random_walk_kernel_gvoys_labeled_dataset if labeled else gvoys.random_walk_kernel_gvoys_dataset
        return fun(Ps,vs,ws,kind=config.kind,lambda_coeff=config.lmbd,
                   p_halt=config.p_halt,anchor_fraction=config.anchor_fraction,
                   nb_random_walks=config.n_samples_gvoys,seed=seed,block_size=config.block_size)
    if method == "mc":
        if labeled:
            return mcrwk.random_walk_kernel_mc_labeled_dataset(Ps,vs,ws,mu,config.kind,
                n_length_samples=config.lengths,n_label_samples_per_length=config.n_label_samples_per_length,
                n_walk_reps=config.n_walk_reps,q_sampling_kind=config.q_sampling_kind,seed=seed)
        return mcrwk.random_walk_kernel_mc_dataset(Ps,vs,ws,mu,config.kind,config.n_samples_mc,seed)
    raise ValueError(f"unknown method: {method}")


def option(parser,name,**kwargs):
    """Accept both --hyphenated and historical --underscore flags."""
    flags = list(dict.fromkeys(["--"+name.replace("_","-"),"--"+name]))
    return parser.add_argument(*flags,dest=name,**kwargs)


def add_kernel_arguments(parser,default_mc=200):
    option(parser,"kind",choices=["exp","geom"],default="geom")
    option(parser,"lmbd",type=float,default=0.1,help="Kernel lambda; default 0.1. Use --lambda-mode degree for the old rule.")
    option(parser,"lambda_mode",choices=["fixed","degree"],default="fixed")
    option(parser,"n_samples_mc",type=int,default=default_mc,help="Shared MC feature budget before optional per-node scaling; each feature uses two replicas.")
    option(parser,"n_samples_gvoys",type=int,default=200,help="GVoys outer features per start vertex; two replicas and two sides each.")
    option(parser,"n_length_samples",type=int,help="Explicit labeled length count; overrides MC feature-budget derivation.")
    option(parser,"n_label_samples_per_length",type=int,default=1)
    option(parser,"n_walk_reps",type=int,default=1,help="Walks averaged inside EACH labeled replica.")
    option(parser,"q_sampling_kind",choices=["uniform","norm_fro","norm_l1","random"],default="uniform")
    option(parser,"p_halt",type=float,default=0.2)
    option(parser,"anchor_fraction",type=float,default=1.)
    option(parser,"block_size",type=int,default=64,help="GVoys temporary feature block; incomplete final blocks are supported.")
    option(parser,"solver_tol",type=float,default=1e-10,help="Relative iterative tolerance; absolute entrywise tail tolerance for series.")
    option(parser,"max_iter",type=int,default=5000)
    option(parser,"seed",type=int,default=42)
    option(parser,"u_w_distribution",choices=DISTRIBUTIONS,default="uniform",
           help="Boundary probabilities: uniform; normalized Uniform(0,1) draws (random); normalized abs(N(0,1)) draws (normal); normalized weighted degrees (degree).")


def config_from_args(args, graphs, sample_multiplier=1):
    lam = args.lmbd
    if args.lambda_mode == "degree":
        max_degree = max((d for G in graphs for _,d in G.degree()),default=0)
        # Avoid lambda=1 for matchings and division by zero for edgeless graphs.
        lam = 1/max(2,max_degree)**2
    fields = KernelConfig.__dataclass_fields__
    values = {key:getattr(args,key) for key in fields}
    values["lmbd"] = lam
    values["n_samples_mc"] *= sample_multiplier
    return KernelConfig(**values).validate()


def build_inputs(graphs,distribution="uniform",labeled=False,seed=42):
    """Build P and boundary probabilities using a shared, explicit RNG stream.

    random/normal draw v and w independently. uniform/degree set v=w;
    degree uses adjacency row sums, with uniform fallback on edgeless graphs.
    """
    if distribution not in DISTRIBUTIONS:
        raise ValueError(f"unknown boundary distribution: {distribution}")
    rng = np.random.default_rng(seed)
    Ps,vs,ws = [],[],[]
    for graph in graphs:
        n = len(graph)
        if n == 0:
            raise ValueError("empty graphs are not supported")
        Ps.append(utils.normalized_adj_matrix_labeled(graph) if labeled else utils.normalized_adj_matrix(graph))
        if distribution == "uniform":
            vs.append(utils.uniform_dist(n)); ws.append(utils.uniform_dist(n))
        elif distribution == "random":
            vs.append(utils.random_dist(n,rng)); ws.append(utils.random_dist(n,rng))
        elif distribution == "normal":
            vs.append(utils.normal_dist(n,rng)); ws.append(utils.normal_dist(n,rng))
        elif distribution == "degree":
            weights = utils.degree_dist(graph)
            vs.append(weights)
            ws.append(weights.copy())
    return Ps,vs,ws


def method_seed(seed,method):
    """Stable under reordering/subsetting the requested method list."""
    return seed+100*(METHODS.index(method)+1)


def planned_methods(methods, *, kind, max_nodes):
    """Ensure geometric runs at/above 128 vertices compute their CG reference."""
    planned = list(methods)
    if kind == "geom" and max_nodes >= CG_REFERENCE_MIN_NODES and "cg" not in planned:
        planned.append("cg")
    return planned


def method_skip_reason(method, *, kind, labeled, max_nodes,
                       direct_max_nodes=128, sylvester_max_nodes=512):
    """Skip direct at or above the cutoff in both experiment scripts.

    direct_max_nodes can lower the limit, but cannot bypass the 128-node cap,
    including when benchmarks are called from Python rather than the CLI.
    """
    limit = min(direct_max_nodes, DIRECT_NODE_LIMIT)
    if method == "direct" and max_nodes >= limit:
        return f"direct size limit: largest graph has {max_nodes} vertices; direct requires fewer than {limit}"
    if method == "sylvester" and max_nodes > sylvester_max_nodes:
        return "sylvester size limit"
    if kind != "geom" and method in {"cg", "gmres", "fixed_point", "sylvester"}:
        return "requires geom"
    if labeled and method in {"series", "sylvester"}:
        return "requires unlabeled inputs"
    return None


def reference_method(matrices, *, kind="geom", max_nodes=0):
    """Choose only successfully computed references; CG is required for large geom.

    A failed CG must not silently switch the requested reference to another
    method. Callers record null errors if no valid reference is available.
    """
    if kind == "geom" and max_nodes >= CG_REFERENCE_MIN_NODES:
        return "cg" if "cg" in matrices else None
    return next((m for m in ("direct","series","cg","gmres","fixed_point") if m in matrices),None)


def runtime_metadata():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for p in sorted(list((root/"src").glob("*.py"))+list(root.glob("*.py"))):
        digest.update(str(p.relative_to(root)).encode()); digest.update(p.read_bytes())
    return {"schema_version":2,"created_utc":datetime.now(timezone.utc).isoformat(),
            "python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,
            "networkx":nx.__version__,"scikit_learn":version("scikit-learn"),
            "platform":platform.platform(),"source_sha256":digest.hexdigest(),
            "target":"row_normalized_P","diagonal":"independent_replica_cross_product",
            "threads":{k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")}}
