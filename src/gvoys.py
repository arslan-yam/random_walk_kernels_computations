import numpy as np
import random as rnd
import scipy
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import cg, LinearOperator, spsolve, expm as sparse_expm



#  Original implementation uses global variables
SIGMA = 0.1
LAMBDA_COEFF = 0.1
P_HALT = 0.2
NB_RANDOM_WALKS = 1000
BIG_NUMBER = 10000

t_variables = np.random.uniform(size=(2 * BIG_NUMBER, 2 * BIG_NUMBER))
g_variables = np.where(np.random.normal(size=(2 * BIG_NUMBER, 2 * BIG_NUMBER)) > 0.0, 1.0, -1.0,)

def adj_matrix_to_lists(P):
    P = sp.csr_matrix(P)
    P.sort_indices()
    indptr = P.indptr
    indices = P.indices
    data = P.data

    adj_lists = []
    weight_lists = []
    for i in range(P.shape[0]):
        start, end = indptr[i], indptr[i + 1]
        adj_lists.append(indices[start:end].tolist())
        weight_lists.append(data[start:end].tolist())

    return adj_lists, weight_lists

def f_func_diffusion(i, lambda_coeff):
    # exponential kernel modulation
    return lambda_coeff ** i / (2 ** i * scipy.special.factorial(i))

def f_func_geometric(i, lambda_coeff):
    # geometric kernel modulation
    return lambda_coeff ** i

def create_pq_vectors(adj_lists, weight_lists, anchor_points_dict, p_halt, nb_random_walks, f, is_left, base_nb_walk_index,):
    n = len(adj_lists)
    s_matrix = np.zeros((nb_random_walks, len(anchor_points_dict), n))

    for w in range(nb_random_walks):
        for k in range(n):
            load = 1.0
            step_counter = 0
            current_vertex = k
            x_index = is_left * BIG_NUMBER + step_counter
            y_index = is_left * BIG_NUMBER + w + base_nb_walk_index

            if current_vertex in anchor_points_dict:
                add_term = load * np.sqrt(f(step_counter))
                add_term *= g_variables[x_index][y_index]
                s_matrix[w, anchor_points_dict[current_vertex], k] += add_term

            if not adj_lists[current_vertex]:
                continue

            while t_variables[x_index][y_index] > p_halt:
                if step_counter >= BIG_NUMBER - 1:
                    break
                rnd_index = int(rnd.uniform(0, 1) * len(adj_lists[current_vertex]))
                p_uv = weight_lists[current_vertex][rnd_index]
                load *= p_uv
                load *= 1.0 / np.sqrt(1.0 - p_halt)
                step_counter += 1
                current_vertex = adj_lists[current_vertex][rnd_index]
                x_index = is_left * BIG_NUMBER + step_counter
                y_index = is_left * BIG_NUMBER + w + base_nb_walk_index

                if current_vertex in anchor_points_dict:
                    add_term = load * np.sqrt(f(step_counter))
                    add_term *= g_variables[x_index][y_index]
                    s_matrix[w, anchor_points_dict[current_vertex], k] += add_term

                if not adj_lists[current_vertex]:
                    break

    return s_matrix

def approximate_graph_kernel_value(P1, P2, v1, v2, w1, w2, anchor_fraction=1.0, base_nb_walk_index=0, kind="exponential", lambda_coeff=LAMBDA_COEFF, p_halt=P_HALT, nb_random_walks=NB_RANDOM_WALKS):
    P1_adj_lists, P1_weight_lists = adj_matrix_to_lists(P1)
    P2_adj_lists, P2_weight_lists = adj_matrix_to_lists(P2)

    n1 = len(P1_adj_lists)
    n2 = len(P2_adj_lists)

    nb_anc1 = max(1, int(anchor_fraction * n1))
    nb_anc2 = max(1, int(anchor_fraction * n2))

    anc1 = np.random.choice(np.arange(n1), size=nb_anc1, replace=False)
    anc2 = np.random.choice(np.arange(n2), size=nb_anc2, replace=False)
    anc1 = np.sort(anc1)
    anc2 = np.sort(anc2)

    anc1_dict = dict(zip(anc1, np.arange(nb_anc1)))
    anc2_dict = dict(zip(anc2, np.arange(nb_anc2)))

    if kind == "exponential":
        f_function = lambda i: f_func_diffusion(i, lambda_coeff)
    elif kind == "geometric":
        f_function = lambda i: f_func_geometric(i, lambda_coeff)
    else:
        raise ValueError("kind must be 'exponential' or 'geometric'")

    p1 = create_pq_vectors(P1_adj_lists, P1_weight_lists, anc1_dict, p_halt, nb_random_walks, f_function, 0, base_nb_walk_index)
    p2 = create_pq_vectors(P2_adj_lists, P2_weight_lists, anc2_dict, p_halt, nb_random_walks, f_function, 0, base_nb_walk_index)
    
    q1 = create_pq_vectors(P1_adj_lists, P1_weight_lists, anc1_dict, p_halt, nb_random_walks, f_function, 1, base_nb_walk_index)
    q2 = create_pq_vectors(P2_adj_lists, P2_weight_lists, anc2_dict, p_halt, nb_random_walks, f_function, 1, base_nb_walk_index)
    
    P1_lat = np.einsum("br,br->br", np.einsum("brN,N->br", p1, v1), np.einsum("brN,N->br", q1, w1))
    P2_lat = np.einsum("br,br->br", np.einsum("brN,N->br", p2, v2), np.einsum("brN,N->br", q2, w2))
    
    final_batch = np.einsum("bx,by->bxy", P1_lat, P2_lat)
    return (1.0 / nb_random_walks) * np.sum(final_batch)


def approximate_graph_kernel_value_with_blocks(P1, P2, v1, v2, w1, w2, anchor_fraction=1.0, kind="exponential",
    lambda_coeff=LAMBDA_COEFF, p_halt=P_HALT, nb_random_walks=NB_RANDOM_WALKS, block_size=NB_RANDOM_WALKS):
    approx_val = 0.0
    if nb_random_walks % block_size != 0:
        raise ValueError("nb_random_walks must be divisible by block_size.")
    
    for i in range(nb_random_walks // block_size):
        approx_val += approximate_graph_kernel_value(P1, P2, v1, v2, w1, w2,
            anchor_fraction=anchor_fraction,
            base_nb_walk_index=i * block_size,
            kind=kind,
            lambda_coeff=lambda_coeff,
            p_halt=p_halt,
            nb_random_walks=block_size,
        )

    return approx_val * (block_size / nb_random_walks)

def build_gvoys_features(P, v, w, anchor_fraction=1.0, base_nb_walk_index=0, kind="exp", lambda_coeff=LAMBDA_COEFF, p_halt=P_HALT, nb_random_walks=NB_RANDOM_WALKS):
    P_adj_lists, P_weight_lists = adj_matrix_to_lists(P)
    n = len(P_adj_lists)
    nb_anc = max(1, int(anchor_fraction * n))
    anc = np.random.choice(np.arange(n), size=nb_anc, replace=False)
    anc = np.sort(anc)
    anc_dict = dict(zip(anc, np.arange(nb_anc)))

    if kind == "exp":
        f_function = lambda i: f_func_diffusion(i, lambda_coeff)
    elif kind == "geom":
        f_function = lambda i: f_func_geometric(i, lambda_coeff)
    else:
        raise ValueError("kind must be 'exp' or 'geom'")

    p_feat = create_pq_vectors(P_adj_lists, P_weight_lists, anc_dict, p_halt=p_halt, nb_random_walks=nb_random_walks, f=f_function, is_left=0, base_nb_walk_index=base_nb_walk_index)
    q_feat = create_pq_vectors(P_adj_lists, P_weight_lists, anc_dict, p_halt=p_halt, nb_random_walks=nb_random_walks, f=f_function, is_left=1, base_nb_walk_index=base_nb_walk_index)
    latent_embedding = np.einsum("br,br->br", np.einsum("brN,N->br", p_feat, v), np.einsum("brN,N->br", q_feat, w))
    return latent_embedding


def gvoys_kernel_from_features(feat1, feat2, nb_random_walks=NB_RANDOM_WALKS):
    final_batch = np.einsum("bx,by->bxy", feat1, feat2)
    return (1.0 / nb_random_walks) * np.sum(final_batch)


def random_walk_kernel_gvoys_dataset(Ps, vs, ws, anchor_fraction=1.0, kind="exponential", lambda_coeff=LAMBDA_COEFF, p_halt=P_HALT, nb_random_walks=NB_RANDOM_WALKS):
    n_graphs = len(Ps)
    graph_features = []
    for i in range(n_graphs):
        graph_features.append(build_gvoys_features(Ps[i], vs[i], ws[i], anchor_fraction=anchor_fraction, base_nb_walk_index=0, kind=kind, lambda_coeff=lambda_coeff, p_halt=p_halt, nb_random_walks=nb_random_walks))

    gram_matrix = np.zeros((n_graphs, n_graphs), dtype=float)
    for i in range(n_graphs):
        for j in range(i + 1):
            gram_matrix[i, j] = gvoys_kernel_from_features(graph_features[i], graph_features[j], nb_random_walks=nb_random_walks)
            gram_matrix[j, i] = gram_matrix[i, j]

    return gram_matrix

def random_walk_kernel_gvoys_dataset_block(Ps, vs, ws, anchor_fraction=1.0, kind="exponential",
                                           lambda_coeff=LAMBDA_COEFF, p_halt=P_HALT,
                                           nb_random_walks=NB_RANDOM_WALKS, block_size=NB_RANDOM_WALKS):
    """
    Вычисляет матрицу Грама для набора графов, используя блочный GVoys.
    
    Параметры:
        Ps: список матриц смежности (numpy array) для каждого графа
        vs, ws: списки векторов v и w для каждого графа (обычно uniform)
        anchor_fraction: доля вершин, используемых как якорные точки (0..1)
        kind: 'exponential' или 'geometric'
        lambda_coeff: коэффициент лямбда для ядра
        p_halt: вероятность остановки случайного блуждания
        nb_random_walks: общее число случайных блужданий на вершину
        block_size: размер блока (количество блужданий, обрабатываемых за раз)
    
    Возвращает:
        gram_matrix: матрица Грама размера (n_graphs, n_graphs)
    """
    if nb_random_walks % block_size != 0:
        raise ValueError("nb_random_walks must be divisible by block_size.")
    
    n_graphs = len(Ps)
    nb_blocks = nb_random_walks // block_size
    gram_matrix = np.zeros((n_graphs, n_graphs), dtype=float)
    
    for b in range(nb_blocks):
        base_idx = b * block_size
        # Для текущего блока строим признаки всех графов
        block_feats = []
        for i in range(n_graphs):
            feat = build_gvoys_features(
                Ps[i], vs[i], ws[i],
                anchor_fraction=anchor_fraction,
                base_nb_walk_index=base_idx,
                kind=kind,
                lambda_coeff=lambda_coeff,
                p_halt=p_halt,
                nb_random_walks=block_size
            )
            block_feats.append(feat)
        
        # Для каждой пары графов вычисляем вклад блока и накапливаем
        for i in range(n_graphs):
            for j in range(i, n_graphs):
                block_val = gvoys_kernel_from_features(
                    block_feats[i], block_feats[j],
                    nb_random_walks=block_size
                )
                gram_matrix[i, j] += block_val
                if i != j:
                    gram_matrix[j, i] += block_val  # симметрично, но можно только один раз
        # При желании можно удалить block_feats для экономии памяти, но Python GC справится
    
    # Усредняем по блокам (каждый блок даёт несмещённую оценку)
    gram_matrix /= nb_blocks
    return gram_matrix

# --- Labeled case ---
def adj_matrix_labeled_to_lists(P_labeled, n_nodes=None):
    labels = sorted(P_labeled.keys())
    if not labels:
        if n_nodes is None:
            return [], [], []
        return [[] for _ in range(n_nodes)], [[] for _ in range(n_nodes)], [[] for _ in range(n_nodes)]

    n = sp.csr_matrix(P_labeled[labels[0]]).shape[0]
    adj_lists = [[] for _ in range(n)]
    weight_lists = [[] for _ in range(n)]
    edge_label_lists = [[] for _ in range(n)]

    for label in labels:
        P_lab = sp.csr_matrix(P_labeled[label])
        P_lab.sort_indices()
        indptr = P_lab.indptr
        indices = P_lab.indices
        data = P_lab.data

        for i in range(n):
            start, end = indptr[i], indptr[i + 1]
            neigh = indices[start:end]
            probs = data[start:end]
            if neigh.size == 0:
                continue
            adj_lists[i].extend(neigh.tolist())
            weight_lists[i].extend(probs.tolist())
            edge_label_lists[i].extend([label] * neigh.size)

    return adj_lists, weight_lists, edge_label_lists


def _z_rademacher(edge_label, x_index, y_index):
    mixed = ((int(edge_label) + 1) * 1315423911 + (int(x_index) + 1) * 2654435761 + (int(y_index) + 1) * 97531)
    mixed ^= (mixed >> 16)
    mixed ^= (mixed >> 32)
    return 1.0 if (mixed & 1) == 0 else -1.0


def create_pq_vectors_labeled(adj_lists, weight_lists, edge_label_lists, anchor_points_dict, p_halt, nb_random_walks, f, is_left, base_nb_walk_index):
    n = len(adj_lists)
    s_matrix = np.zeros((nb_random_walks, len(anchor_points_dict), n))

    for w in range(nb_random_walks):
        for k in range(n):
            load = 1.0
            step_counter = 0
            current_vertex = k
            x_index = is_left * BIG_NUMBER + step_counter
            y_index = is_left * BIG_NUMBER + w + base_nb_walk_index

            if current_vertex in anchor_points_dict:
                add_term = load * np.sqrt(f(step_counter))
                add_term *= g_variables[x_index][y_index]
                s_matrix[w, anchor_points_dict[current_vertex], k] += add_term

            if not adj_lists[current_vertex]:
                continue

            while t_variables[x_index][y_index] > p_halt:
                if step_counter >= BIG_NUMBER - 1:
                    break

                rnd_index = int(rnd.uniform(0, 1) * len(adj_lists[current_vertex]))
                p_uv = weight_lists[current_vertex][rnd_index]
                edge_label = edge_label_lists[current_vertex][rnd_index]
                load *= p_uv
                load *= 1.0 / np.sqrt(1.0 - p_halt)

                step_counter += 1
                current_vertex = adj_lists[current_vertex][rnd_index]
                x_index = is_left * BIG_NUMBER + step_counter
                y_index = is_left * BIG_NUMBER + w + base_nb_walk_index

                load *= _z_rademacher(edge_label, x_index, y_index)

                if current_vertex in anchor_points_dict:
                    add_term = load * np.sqrt(f(step_counter))
                    add_term *= g_variables[x_index][y_index]
                    s_matrix[w, anchor_points_dict[current_vertex], k] += add_term

                if not adj_lists[current_vertex]:
                    break

    return s_matrix


def build_gvoys_features_labeled(P_labeled, v, w, anchor_fraction=1.0, base_nb_walk_index=0, kind="exp", lambda_coeff=LAMBDA_COEFF, p_halt=P_HALT, nb_random_walks=NB_RANDOM_WALKS):
    P_adj_lists, P_weight_lists, P_edge_label_lists = adj_matrix_labeled_to_lists(P_labeled, n_nodes=len(v))
    n = len(P_adj_lists)
    nb_anc = max(1, int(anchor_fraction * n))
    anc = np.random.choice(np.arange(n), size=nb_anc, replace=False)
    anc = np.sort(anc)
    anc_dict = dict(zip(anc, np.arange(nb_anc)))

    if kind == "exp":
        f_function = lambda i: f_func_diffusion(i, lambda_coeff)
    elif kind == "geom":
        f_function = lambda i: f_func_geometric(i, lambda_coeff)
    else:
        raise ValueError("kind must be 'exp' or 'geom'")

    p_feat = create_pq_vectors_labeled(P_adj_lists, P_weight_lists, P_edge_label_lists, anc_dict, p_halt=p_halt, nb_random_walks=nb_random_walks, f=f_function, is_left=0, base_nb_walk_index=base_nb_walk_index)
    q_feat = create_pq_vectors_labeled(P_adj_lists, P_weight_lists, P_edge_label_lists, anc_dict, p_halt=p_halt, nb_random_walks=nb_random_walks, f=f_function, is_left=1, base_nb_walk_index=base_nb_walk_index)
    latent_embedding = np.einsum("br,br->br", np.einsum("brN,N->br", p_feat, v), np.einsum("brN,N->br", q_feat, w))
    return latent_embedding


def random_walk_kernel_gvoys_labeled(P1_labeled, P2_labeled, v1, v2, w1, w2, anchor_fraction=1.0, kind="exp", lambda_coeff=LAMBDA_COEFF, p_halt=P_HALT, nb_random_walks=NB_RANDOM_WALKS):
    feat1 = build_gvoys_features_labeled(P1_labeled, v1, w1, anchor_fraction=anchor_fraction, base_nb_walk_index=0, kind=kind, lambda_coeff=lambda_coeff, p_halt=p_halt, nb_random_walks=nb_random_walks)
    feat2 = build_gvoys_features_labeled(P2_labeled, v2, w2, anchor_fraction=anchor_fraction, base_nb_walk_index=0, kind=kind, lambda_coeff=lambda_coeff, p_halt=p_halt, nb_random_walks=nb_random_walks)
    return gvoys_kernel_from_features(feat1, feat2, nb_random_walks=nb_random_walks)


def random_walk_kernel_gvoys_labeled_dataset(Ps_labeled, vs, ws, anchor_fraction=1.0, kind="exp", lambda_coeff=LAMBDA_COEFF, p_halt=P_HALT, nb_random_walks=NB_RANDOM_WALKS):
    n_graphs = len(Ps_labeled)
    graph_features = []
    for i in range(n_graphs):
        graph_features.append(build_gvoys_features_labeled(Ps_labeled[i], vs[i], ws[i], anchor_fraction=anchor_fraction, base_nb_walk_index=0, kind=kind, lambda_coeff=lambda_coeff, p_halt=p_halt, nb_random_walks=nb_random_walks))

    gram_matrix = np.zeros((n_graphs, n_graphs), dtype=float)
    for i in range(n_graphs):
        for j in range(i + 1):
            gram_matrix[i, j] = gvoys_kernel_from_features(graph_features[i], graph_features[j], nb_random_walks=nb_random_walks)
            gram_matrix[j, i] = gram_matrix[i, j]

    return gram_matrix

def random_walk_kernel_gvoys_labeled_dataset_block(
    Ps_labeled, vs, ws, anchor_fraction=1.0, kind="exp",
    lambda_coeff=LAMBDA_COEFF, p_halt=P_HALT,
    nb_random_walks=NB_RANDOM_WALKS, block_size=NB_RANDOM_WALKS
):
    """
    Вычисляет матрицу Грама для набора labeled графов, используя блочный GVoys.

    Параметры:
        Ps_labeled: список словарей, каждый словарь отображает метку ребра на матрицу смежности (numpy array)
        vs, ws: списки векторов v и w для каждого графа (обычно uniform)
        anchor_fraction: доля вершин, используемых как якорные точки (0..1)
        kind: 'exp' или 'geom' (соответствует exponential или geometric kernel)
        lambda_coeff: коэффициент лямбда для ядра
        p_halt: вероятность остановки случайного блуждания
        nb_random_walks: общее число случайных блужданий на вершину
        block_size: размер блока (количество блужданий, обрабатываемых за раз)

    Возвращает:
        gram_matrix: матрица Грама размера (n_graphs, n_graphs)
    """
    if nb_random_walks % block_size != 0:
        raise ValueError("nb_random_walks must be divisible by block_size.")

    n_graphs = len(Ps_labeled)
    nb_blocks = nb_random_walks // block_size
    gram_matrix = np.zeros((n_graphs, n_graphs), dtype=float)

    for b in range(nb_blocks):
        base_idx = b * block_size
        block_feats = []
        for i in range(n_graphs):
            feat = build_gvoys_features_labeled(
                Ps_labeled[i], vs[i], ws[i],
                anchor_fraction=anchor_fraction,
                base_nb_walk_index=base_idx,
                kind=kind,
                lambda_coeff=lambda_coeff,
                p_halt=p_halt,
                nb_random_walks=block_size
            )
            block_feats.append(feat)

        for i in range(n_graphs):
            for j in range(i, n_graphs):
                block_val = gvoys_kernel_from_features(
                    block_feats[i], block_feats[j],
                    nb_random_walks=block_size
                )
                gram_matrix[i, j] += block_val
                if i != j:
                    gram_matrix[j, i] += block_val

    gram_matrix /= nb_blocks
    return gram_matrix