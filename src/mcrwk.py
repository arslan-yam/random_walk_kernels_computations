import numpy as np
import math
from scipy.special import factorial
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import cg, LinearOperator, spsolve, expm as sparse_expm



class TaylorGenerator(np.random.Generator):
    def exp_taylor(self, lmbd, size=None):
        """
        Генерирует СВ на основе ряда Тейлора для exp(lmbd).
        Эквивалентно rng.poisson(lmbd).
        """
        # 1. Определяем, сколько членов ряда нам нужно для точности.
        # Для малых lmbd (< 1) хватит 7-10 членов. 
        # Для больших — берем с запасом вокруг lmbd.
        k_max = int(max(10, lmbd + 5 * np.sqrt(lmbd) if lmbd > 0 else 10))
        ks = np.arange(0, k_max)
        
        # 2. Считаем члены ряда: lmbd^k / k!
        # Используем логарифмы, чтобы не "взорваться" на больших k или lmbd
        log_terms = ks * np.log(lmbd) - np.log(factorial(ks))
        terms = np.exp(log_terms)
        
        # 3. Нормируем. Сумма terms будет стремиться к exp(lmbd).
        probs = terms / terms.sum()
        
        # 4. Выбираем значение k (0, 1, 2...) согласно вероятностям
        return self.choice(ks, size=size, p=probs)
    
# 1. Define your custom PDF class
# class sin_taylor(rv_discrete):
#     def _pdf(self, x, lmbd):
#         # Example: A simple linear PDF (normalized)
#         k = x
#         return (-1) ** k * lmbd ** (2*k + 1) / math.factorial(k) * np.sin(lmbd)

def kernel_normalizer(kind, mu_func):
    if kind == "exp":
        lmbd = mu_func(1)
        return math.exp(lmbd)
    if kind == "geom":
        lmbd = mu_func(1)
        return 1.0 / (1.0 - lmbd)
    raise ValueError(f"unsupported kind: {kind}")

def sample_length(kind, mu_func, rng):
    if kind == "exp":
        lmbd = mu_func(1)
        return rng.poisson(lmbd)
    if kind == "geom":
        lmbd = mu_func(1)
        return rng.geometric(1.0 - lmbd) - 1
    # if kind == "log":
    #     return sin_taylor()
        
    raise ValueError(f"unsupported kind: {kind}")

def _prepare_sampling_rows(P):
    P = sp.csr_matrix(P)
    P.sort_indices()
    rows = []
    for i in range(P.shape[0]):
        start, end = P.indptr[i], P.indptr[i + 1]
        neigh = P.indices[start:end]
        probs = P.data[start:end]
        if neigh.size == 0:
            neigh = np.array([i], dtype=int)
            probs = np.array([1.0], dtype=float)
        rows.append((neigh, probs))
    return rows

def build_features(P, v, w, shared_random_variables, n_samples, rng):
    rows = _prepare_sampling_rows(P)
    samples = np.zeros(n_samples, dtype=float)
    for i in range(n_samples):
        len_walk = shared_random_variables[i]
        x = rng.choice(len(v), p=v)
        for _ in range(len_walk):
            neigh, probs = rows[x]
            x = rng.choice(neigh, p=probs)
        samples[i] = w[x]
    return samples

def random_walk_kernel_mc(P1, P2, v1, v2, w1, w2, mu_func, kind, n_samples=100, seed=42):
    rng = np.random.default_rng(seed)
    C = kernel_normalizer(kind, mu_func)
    shared_random_variables = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        shared_random_variables[i] = sample_length(kind, mu_func, rng)
    g1_samples = build_features(P1, v1, w1, shared_random_variables, n_samples, rng)
    g2_samples = build_features(P2, v2, w2, shared_random_variables, n_samples, rng)
    return C * (g1_samples * g2_samples).mean()

def random_walk_kernel_mc_dataset(Ps, vs, ws, mu_func, kind, n_samples, seed):
    rng = np.random.default_rng(seed)
    n_graphs = len(Ps)
    C = kernel_normalizer(kind, mu_func)

    shared_random_variables = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        shared_random_variables[i] = sample_length(kind, mu_func, rng)

    graph_features = np.zeros((n_graphs, n_samples), dtype=float)
    for i in range(n_graphs):
        graph_features[i] = build_features(Ps[i], vs[i], ws[i], shared_random_variables, n_samples, rng)

    gram_matrix = np.zeros((n_graphs, n_graphs), dtype=float)
    for i in range(n_graphs):
        for j in range(i + 1):
            gram_matrix[i, j] = C * (graph_features[i] * graph_features[j]).mean()
            gram_matrix[j, i] = gram_matrix[i, j]

    return gram_matrix

# --- Labeled case ---
def sample_label_seq(common_labels, q, K, n_label_samples_per_length, rng):
    label_seqs = []
    q_prods = np.ones(n_label_samples_per_length, dtype=float)

    for i in range(n_label_samples_per_length):
        ids = rng.choice(len(common_labels), size=K, p=q)
        seq = [common_labels[j] for j in ids]
        label_seqs.append(seq)
        q_prods[i] = float(np.prod(q[ids]))

    return label_seqs, q_prods

def _fro_norm(M):
    if sp.issparse(M):
        return float(np.sqrt(np.sum(M.data ** 2)))
    return float(np.linalg.norm(M, ord="fro"))

def _l1_sum(M):
    if sp.issparse(M):
        return float(np.sum(np.abs(M.data)))
    return float(np.sum(np.abs(M)))

def prepare_P(P):
    P_sampling = {}
    for label, P_label in P.items():
        P_label = sp.csr_matrix(P_label)
        P_label.sort_indices()
        dist_for_nodes = []
        for i in range(P_label.shape[0]):
            start, end = P_label.indptr[i], P_label.indptr[i + 1]
            neigh = P_label.indices[start:end]
            probs = P_label.data[start:end]
            row_sum = float(probs.sum())
            if row_sum > 0:
                probs = probs / row_sum
            else:
                neigh = np.array([], dtype=int)
                probs = np.array([], dtype=float)
            dist_for_nodes.append((neigh, probs, row_sum))
        P_sampling[label] = dist_for_nodes

    return P_sampling

def process_sequence_labeled(P_sampling, v, w, label_seq, n_reps, rng):
    total = 0.0
    for _ in range(n_reps):
        x = rng.choice(len(v), p=v)
        weight = 1.0
        for label in label_seq:
            if label not in P_sampling:
                weight = 0.0
                break
            neigh, probs, row_sum = P_sampling[label][x]
            if row_sum == 0.0:
                weight = 0.0
                break
            weight *= row_sum
            x = rng.choice(neigh, p=probs)
        total += weight * w[x]
    return total / n_reps

def build_features_labeled(P, v, w, shared_lengths, shared_label_seqs, shared_q_prods, n_length_samples, n_label_samples_per_length, n_walk_reps, rng):
    P_sampling = prepare_P(P)
    features = np.zeros((n_length_samples, n_label_samples_per_length), dtype=float)
    for i in range(n_length_samples):
        for j in range(n_label_samples_per_length):
            curr_seq = shared_label_seqs[i][j]
            curr_seq_prob = shared_q_prods[i][j]
            s = process_sequence_labeled(P_sampling, v, w, curr_seq, n_walk_reps, rng)
            features[i, j] = s / math.sqrt(curr_seq_prob)
    return features

def q_sampling(P1, P2, common_labels, q_sampling_kind="uniform"):
    d = len(common_labels)
    if d == 0:
        raise ValueError("no common labels")

    if q_sampling_kind == "uniform":
        return np.ones(d, dtype=float) / d

    if q_sampling_kind == "random":
        x = np.random.random(d)
        return x / x.sum()

    if q_sampling_kind == "norm_fro":
        scores = np.zeros(d, dtype=float)
        for i, lab in enumerate(common_labels):
            scores[i] = _fro_norm(P1[lab]) * _fro_norm(P2[lab])
        if np.all(scores == 0):
            return np.ones(d, dtype=float) / d
        return scores / scores.sum()

    if q_sampling_kind == "norm_l1":
        scores = np.zeros(d, dtype=float)
        for i, lab in enumerate(common_labels):
            scores[i] = _l1_sum(P1[lab]) * _l1_sum(P2[lab])
        if np.all(scores == 0):
            return np.ones(d, dtype=float) / d
        return scores / scores.sum()

    raise ValueError("unknown kind")

def random_walk_kernel_mc_labeled(P1, P2, v1, v2, w1, w2, mu_func, kind, n_length_samples=200, n_label_samples_per_length=50, n_walk_reps=10, q_sampling_kind="norm_fro", seed=42):
    rng = np.random.default_rng(seed)
    C = kernel_normalizer(kind, mu_func)
    common_labels = sorted(set(P1.keys()) & set(P2.keys()))
    d = len(common_labels)
    if d == 0:
        return 0.0

    q = q_sampling(P1, P2, common_labels, q_sampling_kind=q_sampling_kind)
    shared_lengths = np.zeros(n_length_samples, dtype=int)
    shared_label_seqs = []
    shared_q_prods = []

    for i in range(n_length_samples):
        K = sample_length(kind, mu_func, rng)
        shared_lengths[i] = K
        label_seqs, q_prods = sample_label_seq(common_labels, q, K, n_label_samples_per_length, rng)
        shared_label_seqs.append(label_seqs)
        shared_q_prods.append(q_prods)

    g1 = build_features_labeled(P1, v1, w1, shared_lengths, shared_label_seqs, shared_q_prods, n_length_samples, n_label_samples_per_length, n_walk_reps, rng)
    g2 = build_features_labeled(P2, v2, w2, shared_lengths, shared_label_seqs, shared_q_prods, n_length_samples, n_label_samples_per_length, n_walk_reps, rng)
    return C * (g1 * g2).mean(axis=1).mean()

def q_sampling_dataset(Ps, all_labels, q_sampling_kind="uniform"):
    d = len(all_labels)
    if d == 0:
        raise ValueError("no common labels")

    if q_sampling_kind == "uniform":
        return np.ones(d, dtype=float) / d

    if q_sampling_kind == "random":
        x = np.random.random(d)
        return x / x.sum()

    if q_sampling_kind == "norm_fro":
        scores = np.zeros(d, dtype=float)
        for i, lab in enumerate(all_labels):
            s = 0.0
            for P in Ps:
                if lab in P:
                    s += _fro_norm(P[lab])
            scores[i] = s
        if np.all(scores == 0):
            return np.ones(d, dtype=float) / d
        return scores / scores.sum()

    if q_sampling_kind == "norm_l1":
        scores = np.zeros(d, dtype=float)
        for i, lab in enumerate(all_labels):
            s = 0.0
            for P in Ps:
                if lab in P:
                    s += _l1_sum(P[lab])
            scores[i] = s
        if np.all(scores == 0):
            return np.ones(d, dtype=float) / d
        return scores / scores.sum()

    raise ValueError("unknown kind")

def random_walk_kernel_mc_labeled_dataset(Ps, vs, ws, mu_func, kind, n_length_samples=200, n_label_samples_per_length=50, n_walk_reps=1, q_sampling_kind="uniform", seed=42):
    rng = np.random.default_rng(seed)
    n_graphs = len(Ps)
    C = kernel_normalizer(kind, mu_func)

    all_labels = sorted(set().union(*[set(P.keys()) for P in Ps]))
    d = len(all_labels)

    if d == 0:
        return np.zeros((n_graphs, n_graphs), dtype=float)

    q = q_sampling_dataset(Ps, all_labels, q_sampling_kind=q_sampling_kind)
    shared_lengths = np.zeros(n_length_samples, dtype=int)
    shared_label_seqs = []
    shared_q_prods = []

    for i in range(n_length_samples):
        K = sample_length(kind, mu_func, rng)
        shared_lengths[i] = K
        label_seqs, q_prods = sample_label_seq(all_labels, q, K, n_label_samples_per_length, rng)
        shared_label_seqs.append(label_seqs)
        shared_q_prods.append(q_prods)

    graph_features = []
    for i in range(n_graphs):
        gi_feature = build_features_labeled(Ps[i], vs[i], ws[i], shared_lengths, shared_label_seqs, shared_q_prods, n_length_samples, n_label_samples_per_length, n_walk_reps, rng)
        graph_features.append(gi_feature)

    gram_matrix = np.zeros((n_graphs, n_graphs), dtype=float)
    for i in range(n_graphs):
        for j in range(i + 1):
            value = C * (graph_features[i] * graph_features[j]).mean(axis=1).mean()
            gram_matrix[i, j] = value
            gram_matrix[j, i] = value
    return gram_matrix