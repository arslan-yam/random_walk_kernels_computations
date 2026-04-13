import numpy as np
import math
from scipy.special import factorial


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
    
def get_samples(P, v, w, shared_random_variables, n_samples, rng):
    samples = np.zeros(n_samples, dtype=float)
    for i in range(n_samples):
        len_walk = shared_random_variables[i]
        x = rng.choice(len(v), p=v) #sample starting point from distribution v
        for j in range(len_walk):
            x = rng.choice(P.shape[1], p=P[x]) #random walk step
        samples[i] = w[x]
    return samples

def random_walk_kernel_mc(P1, P2, v1, v2, w1, w2, mu_func, kind, n_samples=100, seed=42):
    rng = np.random.default_rng(seed)
    C = kernel_normalizer(kind, mu_func)
    shared_random_variables = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        shared_random_variables[i] = sample_length(kind, mu_func, rng)  
    g1_samples = get_samples(P1, v1, w1, shared_random_variables, n_samples, rng)
    g2_samples = get_samples(P2, v2, w2, shared_random_variables, n_samples, rng)
    return C * (g1_samples * g2_samples).mean()

def random_walk_kernel_mc_dataset(Ps, vs, ws, mu_func, kind, n_samples, seed):
    '''
    "linear" time with respect to dataset size& we precompute samples for each graph and then we can compare each pair almost with constant time
    '''
    rng = np.random.default_rng(seed)
    C = kernel_normalizer(kind, mu_func)
    shared_random_variables = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        shared_random_variables[i] = sample_length(kind, mu_func, rng)
     
    n_graphs = len(Ps)
    graph_samples = np.zeros(n_graphs)
    for i in range(len(Ps)):
        graph_samples[i] = get_samples(Ps[i], vs[i], ws[i], shared_random_variables, n_samples)
        
    return graph_samples, C

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

def prepare_P(P):
    P_sampling = {}
    
    for label, P_label in P.items():
        dist_for_nodes = []
        for i in range(P_label.shape[0]):
            row = P_label[i]
            row_sum = row.sum()
            if row_sum > 0:
                neigh = np.where(row > 0)[0]
                probs = row[neigh] / row_sum
            else:
                neigh = np.array([], dtype=int)
                probs = np.array([], dtype=float)
            dist_for_nodes.append(((neigh, probs, row_sum)))
        P_sampling[label] = dist_for_nodes

    return P_sampling

def process_sequence_multi(P_sampling, v, w, label_seq, n_reps, rng):
    total = 0.0
    for _ in range(n_reps):
        x = rng.choice(len(v), p=v)
        weight = 1.0
        for label in label_seq:
            neigh, probs, row_sum = P_sampling[label][x]
            if row_sum == 0.0:
                weight = 0.0
                break
            weight *= row_sum
            x = rng.choice(neigh, p=probs)
        total += weight * w[x] if weight != 0.0 else 0.0
    return total / n_reps


def build_features_labeled(P, v, w, shared_lengths, shared_label_seqs, shared_q_prods, n_length_samples, n_label_samples_per_length, n_walk_reps, rng):
    P_sampling = prepare_P(P)
    features = np.zeros((n_length_samples, n_label_samples_per_length), dtype=float)
    for i in range(n_length_samples):
        for j in range(n_label_samples_per_length):
            curr_seq = shared_label_seqs[i][j]
            curr_seq_prob = shared_q_prods[i][j]
            s = process_sequence_multi(P_sampling, v, w, curr_seq, n_walk_reps, rng)
            features[i, j] = s / math.sqrt(curr_seq_prob)

    return features


def q_sampling(P1, P2, d, common_labels, q_sampling_kind="uniform"):
    if q_sampling_kind == "uniform":
        return np.ones(d, dtype=float) / d
    elif q_sampling_kind == "random":
        x = np.random.random(d)
        return x / x.sum()
    
    elif q_sampling_kind == "norm_fro":
        scores = np.zeros(d, dtype=float)
        for i, lab in enumerate(common_labels):
            scores[i] = np.linalg.norm(P1[lab], ord="fro") * np.linalg.norm(P2[lab], ord="fro")
        if np.all(scores == 0):
            return np.ones(d, dtype=float) / d
        return scores / scores.sum()

    elif q_sampling_kind == "norm_l1":
        scores = np.zeros(d, dtype=float)
        for i, lab in enumerate(common_labels):
            scores[i] = np.sum(np.abs(P1[lab])) * np.sum(np.abs(P2[lab]))
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
    q = q_sampling(P1, P2, d, common_labels, q_sampling_kind=q_sampling_kind)

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