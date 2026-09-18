"""Equivalent symmetric representation of reversible normalized RWK inputs."""

import numpy as np
import scipy.sparse as sp

from ._validation import normalized_input


def symmetric_inputs(P, v, w, labeled=False):
    """Return S, v/sqrt(d), sqrt(d)*w with P = D^-1/2 S D^1/2.

    A positive reversible measure d is recovered from detailed balance on each
    connected component. Component-wise scaling of d cancels in the kernel.
    This accepts weighted undirected graphs and preserves absorbing isolates
    (unlabeled) or zero rows (labeled). Directed/non-reversible inputs are
    rejected rather than silently changing the target kernel.
    """
    P, v, w = normalized_input(P, v, w, labeled)
    n = len(v)
    total = sum(P.values(), sp.csr_matrix((n,n))) if labeled else P
    total = total.tocsr()
    total.sort_indices()
    transposed = total.T.tocsr()
    transposed.sort_indices()
    if not (np.array_equal(total.indptr, transposed.indptr) and
            np.array_equal(total.indices, transposed.indices)):
        raise ValueError("GVoys/symmetric CG require undirected support")
    # Fast path for ordinary unweighted graphs: reciprocal row weights recover
    # degrees. Validate it vectorially; weighted inputs fall back to propagation.
    candidate = np.ones(n)
    nonempty = np.diff(total.indptr) > 0
    candidate[nonempty] = 1 / total.data[total.indptr[:-1][nonempty]]
    row_measure = np.repeat(candidate, np.diff(total.indptr))
    fast = np.isfinite(candidate).all() and np.allclose(row_measure * total.data,
                       candidate[total.indices] * transposed.data, rtol=1e-8, atol=0)
    measure = candidate if fast else np.full(n, np.nan)
    for root in range(n):
        if np.isfinite(measure[root]):
            continue
        measure[root] = 1.0
        stack = [root]
        while stack:
            x = stack.pop()
            for pos in range(total.indptr[x], total.indptr[x+1]):
                y = total.indices[pos]
                target = measure[x] * total.data[pos] / transposed.data[pos]
                if not np.isfinite(target) or target <= 0:
                    raise ValueError("reversible measure has invalid dynamic range")
                if np.isnan(measure[y]):
                    measure[y] = target
                    stack.append(y)
                elif not np.isclose(measure[y], target, rtol=1e-8, atol=0):
                    raise ValueError("input is not reversible (detailed balance fails)")
    root_d = np.sqrt(measure)
    left, right = sp.diags(root_d), sp.diags(1/root_d)

    def convert(M):
        S = (left @ M @ right).tocsr()
        difference = S-S.T
        if difference.nnz and np.max(np.abs(difference.data)) > 1e-9:
            raise ValueError("each label matrix must be reversible with the same measure")
        # No symmetrization: preserve the exact similarity transformation.
        return S

    S = {lab: convert(M) for lab, M in P.items()} if labeled else convert(P)
    return S, v/root_d, w*root_d
