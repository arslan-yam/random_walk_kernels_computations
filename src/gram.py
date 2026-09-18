"""Gram construction and approximation diagnostics."""

import numpy as np

from . import rwk
from ._validation import dataset_inputs, kernel_parameter, positive_int


def _pairwise(Ps,vs,ws,fun,**kwargs):
    if not (len(Ps) == len(vs) == len(ws)):
        raise ValueError("Ps, vs and ws must have matching lengths")
    G = np.zeros((len(Ps),len(Ps)))
    for i in range(len(Ps)):
        for j in range(i+1):
            G[i,j] = G[j,i] = fun(Ps[i],Ps[j],vs[i],vs[j],ws[i],ws[j],**kwargs)
    return G


def gram_direct(Ps,vs,ws,mu_func,kind,labeled=False):
    fun = rwk.random_walk_kernel_labeled if labeled else rwk.random_walk_kernel
    return _pairwise(Ps,vs,ws,fun,mu_func=mu_func,kind=kind)


def gram_sylvester(Ps,vs,ws,mu_func):
    return _pairwise(Ps,vs,ws,rwk.random_walk_kernel_sylvester,mu_func=mu_func)


def gram_fixed_point(Ps,vs,ws,mu_func,labeled=False,eps=1e-10,max_iter=5000):
    fun = rwk.random_walk_kernel_fixed_point_labeled if labeled else rwk.random_walk_kernel_fixed_point
    return _pairwise(Ps,vs,ws,fun,mu_func=mu_func,eps=eps,max_iter=max_iter)


def gram_cg(Ps,vs,ws,mu_func,labeled=False,eps=1e-10,max_iter=5000):
    fun = rwk.random_walk_kernel_cg_labeled if labeled else rwk.random_walk_kernel_cg
    return _pairwise(Ps,vs,ws,fun,mu_func=mu_func,eps=eps,max_iter=max_iter)


def gram_gmres(Ps,vs,ws,mu_func,labeled=False,eps=1e-10,max_iter=5000):
    return _pairwise(Ps,vs,ws,rwk.random_walk_kernel_gmres,mu_func=mu_func,
                     eps=eps,max_iter=max_iter,labeled=labeled)


def gram_series(Ps,vs,ws,mu_func,kind="geom",eps=1e-12,max_iter=5000):
    """Deterministic unlabeled features; absolute entrywise tail <= eps.

    Requires row-substochastic matrices and nonnegative boundary vectors.
    The geometric tail is analytic; the exponential tail uses Poisson survival.
    """
    from scipy.stats import poisson
    lam = kernel_parameter(kind,mu_func)
    positive_int(max_iter,"max_iter")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    data = dataset_inputs(Ps,vs,ws)
    if not data:
        return np.empty((0,0))
    states = [w.copy() for _,_,w in data]
    bound = max(np.max(w) for _,_,w in data)**2
    features = []
    coefficient = 1.0
    for k in range(max_iter):
        features.append([np.sqrt(coefficient)*float(v@z) for (_,v,_),z in zip(data,states)])
        tail = lam**(k+1)/(1-lam) if kind == "geom" else np.exp(lam)*poisson.sf(k,lam)
        if bound*tail <= eps:
            H = np.asarray(features).T
            return H@H.T
        states = [P@z for (P,_,_),z in zip(data,states)]
        coefficient *= lam if kind == "geom" else lam/(k+1)
    raise RuntimeError("series tail tolerance not reached; increase max_iter")


def matrix_errors(G_ref,G,zero_tol=1e-15):
    """Relative metrics omit near-zero references; absolute errors always include them."""
    ref,estimate = np.asarray(G_ref),np.asarray(G)
    if ref.shape != estimate.shape or ref.ndim != 2 or ref.shape[0] != ref.shape[1]:
        raise ValueError("expected matching square matrices")
    if not np.isfinite(ref).all() or not np.isfinite(estimate).all():
        raise ValueError("kernel matrices must be finite")
    error = np.abs(ref-estimate)
    valid = np.abs(ref)>zero_tol
    relative = np.zeros_like(error)
    np.divide(error,np.abs(ref),out=relative,where=valid)
    def mean(values):
        return float(np.mean(values)) if values.size else None
    def maximum(values):
        return float(np.max(values)) if values.size else None
    diagonal = np.eye(len(ref),dtype=bool)
    norm = np.linalg.norm(ref)
    return {"mean_abs":mean(error),"max_abs":maximum(error),
            "mean_rel":mean(relative[valid]),"max_rel":maximum(relative[valid]),
            "diagonal_mean_abs":mean(error[diagonal]),
            "offdiagonal_mean_abs":mean(error[~diagonal]),
            "diagonal_mean_rel":mean(relative[diagonal & valid]),
            "offdiagonal_mean_rel":mean(relative[~diagonal & valid]),
            "relative_frobenius":float(np.linalg.norm(error)/norm) if norm else None,
            "near_zero_reference_entries":int(np.sum(~valid))}
