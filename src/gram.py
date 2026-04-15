import numpy as np
import random as rnd
import scipy
from . import rwk


def gram_direct(Ps, vs, ws, mu_func, kind):
    n = len(Ps)
    G = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1):
            G[i, j] = rwk.random_walk_kernel(Ps[i], Ps[j], vs[i], vs[j], ws[i], ws[j], mu_func=mu_func, kind=kind)
            G[j, i] = G[i, j]
    return G


def gram_sylvester(Ps, vs, ws, mu_func):
    n = len(Ps)
    G = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1):
            G[i, j] = rwk.random_walk_kernel_sylvester(Ps[i], Ps[j], vs[i], vs[j], ws[i], ws[j], mu_func=mu_func)
            G[j, i] = G[i, j]
    return G


def gram_fixed_point(Ps, vs, ws, mu_func):
    n = len(Ps)
    G = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1):
            G[i, j] = rwk.random_walk_kernel_fixed_point(Ps[i], Ps[j], vs[i], vs[j], ws[i], ws[j], mu_func=mu_func, eps=1e-30, max_iter=5000)
            G[j, i] = G[i, j]
    return G


def gram_cg(Ps, vs, ws, mu_func):
    n = len(Ps)
    G = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1):
            G[i, j] = rwk.random_walk_kernel_cg(Ps[i], Ps[j], vs[i], vs[j], ws[i], ws[j], mu_func=mu_func, eps=1e-30, max_iter=5000)
            G[j, i] = G[i, j]
    return G


def matrix_errors(G_ref, G):
    abs_err = np.abs(G_ref - G)
    rel_err = abs_err / (np.abs(G_ref))
    mask = np.ones_like(G_ref, dtype=bool)
    mean_abs = abs_err[mask].mean()
    mean_rel = rel_err[mask].mean()
    max_abs = abs_err[mask].max()
    max_rel = rel_err[mask].max()
    return {
        "mean_abs": float(mean_abs),
        "mean_rel": float(mean_rel),
        "max_abs": float(max_abs),
        "max_rel": float(max_rel),
    }