import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import cg, LinearOperator, spsolve, expm as sparse_expm


def random_walk_kernel(P1, P2, v1, v2, w1, w2, mu_func, kind="general", max_iter=30):
    P1 = sp.csr_matrix(P1)
    P2 = sp.csr_matrix(P2)
    n1, n2 = len(v1), len(v2)
    W = sp.kron(P1, P2, format="csr")
    v = np.kron(v1, v2)
    w = np.kron(w1, w2)

    if kind == "exp":
        lmbd = mu_func(1)
        S = sparse_expm(lmbd * W)
        return float(v @ (S @ w))

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
        S = sparse_expm(lmbd * W)
        return float(v @ (S @ w))

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
        Y[:, j] = np.linalg.solve(A, rhs)

    M = U2 @ Y @ U1.conj().T
    val = np.sum(V0 * M)

    return float(np.real_if_close(val))

# --- Fixed Point ---
def random_walk_kernel_fixed_point(P1, P2, v1, v2, w1, w2, mu_func, eps=1e-30, max_iter=1000):
    P1 = sp.csr_matrix(P1)
    P2 = sp.csr_matrix(P2)

    lmbd = mu_func(1)
    w0 = np.outer(w2, w1)
    v0 = np.outer(v2, v1)
    x = w0.copy()

    P1t = P1.transpose().tocsr()

    for _ in range(max_iter):
        x_new = w0 + lmbd * (P2 @ x @ P1t)
        if np.linalg.norm(x_new - x, ord="fro") <= eps:
            x = x_new
            break
        x = x_new

    return float(np.sum(v0 * x))

def random_walk_kernel_fixed_point_labeled(P1_labeled, P2_labeled, v1, v2, w1, w2, mu_func, eps=1e-30, max_iter=1000):
    common_labels = set(P1_labeled.keys()) & set(P2_labeled.keys())
    w0 = np.outer(w2, w1)
    v0 = np.outer(v2, v1)
    x = w0.copy()
    lmbd = mu_func(1)

    P1t = {label: sp.csr_matrix(P1_labeled[label]).transpose().tocsr() for label in common_labels}
    P2s = {label: sp.csr_matrix(P2_labeled[label]) for label in common_labels}

    for _ in range(max_iter):
        x_new = w0.copy()
        for label in common_labels:
            x_new += lmbd * (P2s[label] @ x @ P1t[label])

        if np.linalg.norm(x_new - x, ord="fro") <= eps:
            x = x_new
            break

        x = x_new

    return float(np.sum(v0 * x))

# --- Conjugate Gradient ---
def random_walk_kernel_cg(P1, P2, v1, v2, w1, w2, mu_func, eps=1e-30, max_iter=1000):
    P1 = sp.csr_matrix(P1)
    P2 = sp.csr_matrix(P2)

    n1, n2 = P1.shape[0], P2.shape[0]
    v = np.kron(v1, v2)
    w = np.kron(w1, w2)
    lmbd = mu_func(1)
    P1t = P1.transpose().tocsr()

    def matvec(x):
        X = x.reshape((n2, n1), order="F")
        Y = X - lmbd * (P2 @ X @ P1t)
        return Y.reshape(-1, order="F")

    A = LinearOperator(shape=(n1 * n2, n1 * n2), matvec=matvec, dtype=float)
    x, info = cg(A, w, rtol=eps, maxiter=max_iter)
    if info != 0:
        raise RuntimeError(f"CG did not converge, info={info}")
    return float(v @ x)

def random_walk_kernel_cg_labeled(P1_labeled, P2_labeled, v1, v2, w1, w2, mu_func, eps=1e-30, max_iter=1000):
    common_labels = set(P1_labeled.keys()) & set(P2_labeled.keys())
    n1, n2 = len(v1), len(v2)
    v = np.kron(v1, v2)
    w = np.kron(w1, w2)
    lmbd = mu_func(1)

    P1t = {label: sp.csr_matrix(P1_labeled[label]).transpose().tocsr() for label in common_labels}
    P2s = {label: sp.csr_matrix(P2_labeled[label]) for label in common_labels}

    def matvec(x):
        X = x.reshape((n2, n1), order="F")
        Y = X.copy()
        for label in common_labels:
            Y -= lmbd * (P2s[label] @ X @ P1t[label])
        return Y.reshape(-1, order="F")

    operator = LinearOperator(shape=(n1 * n2, n1 * n2), matvec=matvec, dtype=float)
    x, info = cg(operator, w, rtol=eps, maxiter=max_iter)
    if info != 0:
        raise RuntimeError(f"CG did not converge, info={info}")
    return float(v @ x)