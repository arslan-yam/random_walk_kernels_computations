import numpy as np
import random as rnd
import scipy


#  Original implementation uses global variables
SIGMA = 0.1
LAMBDA_COEFF = 0.1
P_HALT = 0.2
NB_RANDOM_WALKS = 1000
BIG_NUMBER = 2000

t_variables = np.random.uniform(size=(2 * BIG_NUMBER, 2 * BIG_NUMBER))
g_variables = np.where(np.random.normal(size=(2 * BIG_NUMBER, 2 * BIG_NUMBER)) > 0.0, 1.0, -1.0,)

def adj_matrix_to_lists(P):
    adj_lists = []
    weight_lists = []
    for i in range(P.shape[0]):
        neigh = np.where(P[i] > 0.0)[0].tolist()
        weights = P[i, neigh].tolist()
        adj_lists.append(neigh)
        weight_lists.append(weights)
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

            if adj_lists[current_vertex] == []:
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

                if adj_lists[current_vertex] == []:
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