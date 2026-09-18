"""Direct and iterative references for normalized random-walk kernels."""

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import cg, gmres, LinearOperator, spsolve, expm_multiply
from .normalization import symmetric_inputs
from ._validation import kernel_parameter, positive_int


def random_walk_kernel(P1, P2, v1, v2, w1, w2, mu_func, kind="general", max_iter=30):
    P1 = sp.csr_matrix(P1)
    P2 = sp.csr_matrix(P2)
    W = sp.kron(P1, P2, format="csr")
    v = np.kron(v1, v2)
    w = np.kron(w1, w2)

    if kind == "exp":
        lmbd = mu_func(1)
        return float(v @ expm_multiply(lmbd * W, w))

    if kind == "geom":
        lmbd = mu_func(1)
        I = sp.eye(W.shape[0], dtype=float, format="csr")
        x = spsolve(I - lmbd * W, w)
        return float(v @ x)

    y = w.copy()
    out = mu_func(0) * y
    for k in range(1, max_iter + 1):
        y = W @ y
        out += mu_func(k) * y
    return float(v @ out)

def random_walk_kernel_labeled(P1_labeled, P2_labeled, v1, v2, w1, w2, mu_func, kind="general", max_iter=30):
    n1, n2 = len(v1), len(v2)
    common_labels = set(P1_labeled.keys()) & set(P2_labeled.keys())

    W = sp.csr_matrix((n1 * n2, n1 * n2), dtype=float)
    for label in common_labels:
        P1_lab = sp.csr_matrix(P1_labeled[label])
        P2_lab = sp.csr_matrix(P2_labeled[label])
        W = W + sp.kron(P1_lab, P2_lab, format="csr")

    v = np.kron(v1, v2)
    w = np.kron(w1, w2)

    if kind == "exp":
        lmbd = mu_func(1)
        return float(v @ expm_multiply(lmbd * W, w))

    if kind == "geom":
        lmbd = mu_func(1)
        I = sp.eye(W.shape[0], dtype=float, format="csr")
        x = spsolve(I - lmbd * W, w)
        return float(v @ x)

    y = w.copy()
    out = mu_func(0) * y
    for k in range(1, max_iter + 1):
        y = W @ y
        out += mu_func(k) * y
    return float(v @ out)

# --- Sylvester ---
def random_walk_kernel_sylvester(P1, P2, v1, v2, w1, w2, mu_func):
    """
    geometric random-walk kernel via Schur-based Sylvester equation.
    """
    P1d = sp.csr_matrix(P1).toarray()
    P2d = sp.csr_matrix(P2).toarray()

    lmbd = mu_func(1)
    W0 = np.outer(w2, w1)
    V0 = np.outer(v2, v1)
    T2, U2 = la.schur(P2d, output="complex")
    T1, U1 = la.schur(P1d.T, output="complex")
    C = U2.conj().T @ W0 @ U1
    n2, n1 = C.shape
    Y = np.zeros((n2, n1), dtype=complex)

    for j in range(n1):
        rhs = C[:, j].copy()
        if j > 0:
            accum = np.zeros(n2, dtype=complex)
            for k in range(j):
                accum += Y[:, k] * T1[k, j]
            rhs += lmbd * (T2 @ accum)
        A = np.eye(n2, dtype=complex) - lmbd * T1[j, j] * T2
        Y[:, j] = la.solve_triangular(A, rhs, lower=False)

    M = U2 @ Y @ U1.conj().T
    val = np.sum(V0 * M)

    return float(np.real_if_close(val))

def _operator(P1, P2, n1, n2, lam, labeled):
    pairs = [(P1[l], P2[l]) for l in sorted(set(P1) & set(P2))] if labeled else [(P1,P2)]
    pairs = [(sp.csr_matrix(a).T.tocsr(), sp.csr_matrix(b)) for a,b in pairs]

    def product(x):
        X = x.reshape((n2,n1), order="F")
        Y = np.zeros_like(X)
        for a_transposed,b in pairs:
            Y += b @ X @ a_transposed
        return Y.reshape(-1, order="F")

    A = LinearOperator((n1*n2,n1*n2), matvec=lambda x: x-lam*product(x), dtype=float)
    return A, product


def _solve(P1,P2,v1,v2,w1,w2,mu_func,eps,max_iter,labeled,solver):
    """CG uses similarity-transformed symmetric inputs; GMRES uses P directly."""
    lam = kernel_parameter("geom", mu_func)
    positive_int(max_iter, "max_iter")
    if not np.isfinite(eps) or not 0 < eps < 1:
        raise ValueError("solver tolerance must be finite and between 0 and 1")
    if solver == "cg":
        P1,v1,w1 = symmetric_inputs(P1,v1,w1,labeled)
        P2,v2,w2 = symmetric_inputs(P2,v2,w2,labeled)
    n1,n2 = len(v1),len(v2)
    v,w = np.kron(v1,v2),np.kron(w1,w2)
    A,_ = _operator(P1,P2,n1,n2,lam,labeled)
    if not np.any(w):
        return 0.0
    if solver == "fixed_point":
        x = w.copy()
        for _ in range(max_iter):
            delta = w-A@x
            if np.linalg.norm(delta) <= eps*np.linalg.norm(w):
                break
            x += delta
        else:
            raise RuntimeError(f"FPI did not converge after {max_iter} iterations")
    elif solver == "cg":
        x,info = cg(A,w,rtol=eps,atol=0,maxiter=max_iter)
        if info:
            raise RuntimeError(f"symmetric CG did not converge, info={info}; try GMRES/FPI or more iterations")
    elif solver == "gmres":
        # callback_type='legacy' makes maxiter count inner iterations, not restarts.
        x,info = gmres(A,w,rtol=eps,atol=0,maxiter=max_iter,
                      callback=lambda _: None,callback_type="legacy")
        if info:
            raise RuntimeError(f"GMRES did not converge, info={info}")
    else:
        raise ValueError("unknown iterative solver")
    residual = np.linalg.norm(w-A@x)/np.linalg.norm(w)
    if not np.isfinite(residual) or residual > max(10*eps,1e-13):
        raise RuntimeError(f"true relative residual {residual:.3g} exceeds tolerance {eps:.3g}")
    return float(v@x)


def random_walk_kernel_fixed_point(P1,P2,v1,v2,w1,w2,mu_func,eps=1e-10,max_iter=1000):
    return _solve(P1,P2,v1,v2,w1,w2,mu_func,eps,max_iter,False,"fixed_point")


def random_walk_kernel_fixed_point_labeled(P1,P2,v1,v2,w1,w2,mu_func,eps=1e-10,max_iter=1000):
    return _solve(P1,P2,v1,v2,w1,w2,mu_func,eps,max_iter,True,"fixed_point")


def random_walk_kernel_cg(P1,P2,v1,v2,w1,w2,mu_func,eps=1e-10,max_iter=1000):
    return _solve(P1,P2,v1,v2,w1,w2,mu_func,eps,max_iter,False,"cg")


def random_walk_kernel_cg_labeled(P1,P2,v1,v2,w1,w2,mu_func,eps=1e-10,max_iter=1000):
    return _solve(P1,P2,v1,v2,w1,w2,mu_func,eps,max_iter,True,"cg")


def random_walk_kernel_gmres(P1,P2,v1,v2,w1,w2,mu_func,eps=1e-10,max_iter=1000,labeled=False):
    return _solve(P1,P2,v1,v2,w1,w2,mu_func,eps,max_iter,labeled,"gmres")
