import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import cg, LinearOperator


def random_walk_kernel(P1, P2, v1, v2, w1, w2, mu_func, kind="general", max_iter=30):
    n1, n2 = len(v1), len(v2)
    W = np.kron(P1, P2)
    v = np.kron(v1, v2)
    w = np.kron(w1, w2)
    
    if kind == "exp":
        lmbd = mu_func(1)
        S = la.expm(lmbd * W)
        
    elif kind == "geom":
        lmbd = mu_func(1)
        I = np.eye(W.shape[0], dtype=float)
        S = np.linalg.inv(I - lmbd * W)
        
    else:
        Wk = np.eye(W.shape[0], dtype=float)
        S = mu_func(0) * Wk
        for k in range(1, max_iter + 1):
            Wk = Wk @ W
            S += mu_func(k) * Wk
            
    return float(v @ (S @ w))

def random_walk_kernel_labeled(P1_labeled, P2_labeled, v1, v2, w1, w2, mu_func, kind="general", max_iter=30):
    n1, n2 = len(v1), len(v2)
    W = np.zeros((n1 * n2, n1 * n2), dtype=float)
    v = np.kron(v1, v2)
    w = np.kron(w1, w2)

    for label in set(P1_labeled.keys()) & set(P2_labeled.keys()):
        W += np.kron(P1_labeled[label], P2_labeled[label])
        
    if kind == "exp":
        lmbd = mu_func(1)
        S = la.expm(lmbd * W)
        
    elif kind == "geom":
        lmbd = mu_func(1)
        I = np.eye(W.shape[0], dtype=float)
        S = np.linalg.inv(I - lmbd * W)
        
    else:
        Wk = np.eye(W.shape[0], dtype=float)
        S = mu_func(0) * Wk
        for k in range(1, max_iter + 1):
            Wk = Wk @ W
            S += mu_func(k) * Wk
            
    return float(v @ (S @ w))

# --- Sylvester ---
def random_walk_kernel_sylvester(P1, P2, v1, v2, w1, w2, mu_func):
    """
    geometric random-walk kernel via Schur-based Sylvester equation.
    """
    lmbd = mu_func(1)
    W0 = np.outer(w2, w1)
    V0 = np.outer(v2, v1)
    T2, U2 = la.schur(P2, output="complex")
    T1, U1 = la.schur(P1.T, output="complex")
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
    n1, n2 = P1.shape[0], P2.shape[0]
    lmbd = mu_func(1)
    w0 = np.outer(w2, w1) # vec(w0) = w1 \otimes w2
    v0 = np.outer(v2, v1)
    x = w0.copy()

    for i in range(max_iter):
        x_new = w0 + lmbd * (P2 @ x @ P1.T)
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

    for i in range(max_iter):
        x_new = w0.copy()
        for label in common_labels:
            x_new += lmbd * (P2_labeled[label] @ x @ P1_labeled[label].T)
        
        if np.linalg.norm(x_new - x, ord="fro") <= eps:
            x = x_new
            break
        
        x = x_new

    return float(np.sum(v0 * x))

# --- Conjugate Gradient ---
def random_walk_kernel_cg(P1, P2, v1, v2, w1, w2, mu_func, eps=1e-30, max_iter=1000):
    n1, n2 = P1.shape[0], P2.shape[0]
    v = np.kron(v1, v2)
    w = np.kron(w1, w2)
    lmbd = mu_func(1)
    
    def matvec(x):
        X = x.reshape((n2, n1), order="F")
        Y = X - lmbd * (P2 @ X @ P1.T)
        return Y.reshape(-1, order="F")

    A = LinearOperator(shape=(n1 * n2, n1 * n2),matvec=matvec, dtype=float)
    x, info = cg(A, w, rtol=eps, maxiter=max_iter)
    return float(v @ x)

def random_walk_kernel_cg_labeled(P1_labeled, P2_labeled, v1, v2, w1, w2, mu_func, tol=1e-30, max_iter=1000):
    common_labels = set(P1_labeled.keys()) & set(P2_labeled.keys())
    n1, n2 = len(v1),len(v2)
    v = np.kron(v1, v2)
    w = np.kron(w1, w2)
    lmbd = mu_func(1)

    def matvec(x):
        X = x.reshape((n2, n1), order="F")
        Y = X.copy()
        for label in common_labels:
            Y -= lmbd * (P1_labeled[label] @ X @ P2_labeled[label].T)
        return Y.reshape(-1, order="F")

    operator = LinearOperator(shape=(n1 * n2, n1 * n2), matvec=matvec, dtype=float)
    x, info = cg(operator, w, rtol=tol, maxiter=max_iter)
    return float(v @ x)