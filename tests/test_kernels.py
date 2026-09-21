"""Mathematical regression checks; run: python -m unittest discover -s tests -v."""

import itertools
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_equal

from src import gram, gvoys, mcrwk, rwk, utils
from src._validation import corrected_gram
from src.benchmark import KernelConfig, build_inputs, compute_kernel
from src.normalization import symmetric_inputs


def inputs(labeled=False):
    graphs = [nx.path_graph(3), nx.star_graph(3)]
    for graph in graphs:
        for u, v in graph.edges:
            graph[u][v]["label"] = (u + v) % 2
            graph[u][v]["weight"] = 1 + (u + v) / 5
    return build_inputs(graphs, "random", labeled, seed=73)


class DeterministicTests(unittest.TestCase):
    def test_similarity_preserves_every_walk_word(self):
        for labeled in (False, True):
            Ps, vs, ws = inputs(labeled)
            for P, v, w in zip(Ps, vs, ws):
                S, left, right = symmetric_inputs(P, v, w, labeled)
                matrices = P if labeled else {0: P}
                symmetric = S if labeled else {0: S}
                for M in symmetric.values():
                    assert_allclose(M.toarray(), M.T.toarray(), atol=1e-14)
                for k in range(5):
                    for word in itertools.product(matrices, repeat=k):
                        x, y = w.copy(), right.copy()
                        for label in reversed(word):
                            x, y = matrices[label] @ x, symmetric[label] @ y
                        self.assertAlmostEqual(v @ x, left @ y, places=13)

    def test_geometric_and_exponential_convolution_coefficients(self):
        for kind, function in (("exp", gvoys.f_func_diffusion), ("geom", gvoys.f_func_geometric)):
            mu = utils.mu_func_gen(kind, 0.4)
            for k in range(15):
                self.assertAlmostEqual(sum(function(j, 0.4) * function(k-j, 0.4)
                                           for j in range(k+1)), mu(k), places=14)

    def test_references_agree_with_product_graph(self):
        for labeled in (False, True):
            Ps, vs, ws = inputs(labeled)
            for kind in ("exp", "geom"):
                config = KernelConfig(kind=kind, lmbd=0.6, solver_tol=1e-12)
                exact = compute_kernel("direct", Ps, vs, ws, config, labeled=labeled)
                methods = (["cg", "gmres", "fixed_point"] if kind == "geom" else [])
                if not labeled:
                    methods += ["series"] + (["sylvester"] if kind == "geom" else [])
                for method in methods:
                    with self.subTest(labeled=labeled, kind=kind, method=method):
                        estimate = compute_kernel(method, Ps, vs, ws, config, labeled=labeled)
                        assert_allclose(estimate, exact, rtol=1e-9, atol=2e-12)

    def test_row_stochastic_uniform_case_is_constant(self):
        graphs = [nx.path_graph(3), nx.star_graph(4), nx.empty_graph(6)]
        Ps, vs, ws = build_inputs(graphs)
        mu = utils.mu_func_gen("geom", 0.4)
        expected = 1 / np.outer([3, 5, 6], [3, 5, 6]) / 0.6
        assert_allclose(gram.gram_direct(Ps, vs, ws, mu, "geom"), expected)
        assert_allclose(mcrwk.random_walk_kernel_mc_dataset(Ps, vs, ws, mu, "geom", 50), expected)

    def test_weighted_self_loop_and_isolate_conventions(self):
        graph = nx.Graph()
        graph.add_nodes_from(range(3))
        graph.add_edge(0, 0, weight=2, label=0)
        graph.add_edge(0, 1, weight=3, label=0)
        P = utils.normalized_adj_matrix(graph)
        labeled = utils.normalized_adj_matrix_labeled(graph)[0]
        assert_allclose(P.toarray()[:2], labeled.toarray()[:2])
        self.assertEqual(P[2, 2], 1)
        self.assertEqual(labeled[2, 2], 0)

    def test_directed_input_uses_gmres_instead_of_symmetric_cg(self):
        P = sp.csr_matrix([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        v = np.array([0.2, 0.3, 0.5])
        w = np.array([0.1, 0.2, 0.7])
        mu = utils.mu_func_gen("geom", 0.7)
        exact = rwk.random_walk_kernel(P, P, v, v, w, w, mu, "geom")
        self.assertAlmostEqual(rwk.random_walk_kernel_gmres(P, P, v, v, w, w, mu), exact)
        with self.assertRaisesRegex(ValueError, "undirected support"):
            rwk.random_walk_kernel_cg(P, P, v, v, w, w, mu)

    def test_unconverged_solvers_fail_explicitly(self):
        Ps, vs, ws = inputs()
        args = (Ps[0], Ps[1], vs[0], vs[1], ws[0], ws[1], utils.mu_func_gen("geom", 0.9))
        for solver in (rwk.random_walk_kernel_cg, rwk.random_walk_kernel_fixed_point,
                       rwk.random_walk_kernel_gmres):
            with self.assertRaises(RuntimeError):
                solver(*args, max_iter=1, eps=1e-14)
        with self.assertRaises(RuntimeError):
            gram.gram_series(Ps, vs, ws, args[-1], max_iter=1)

    def test_corrected_gram_unbiasedness_and_variance_by_enumeration(self):
        # Four independent Bernoulli walks: exact finite enumeration, no MC tolerance.
        corrected, old = [], []
        for a, b, c, d in itertools.product((0., 1.), repeat=4):
            first, second = np.array([[a], [b]]), np.array([[c], [d]])
            corrected.append(corrected_gram(first, second))
            old.append(first @ first.T)
        corrected, old = np.asarray(corrected), np.asarray(old)
        assert_allclose(corrected.mean(axis=0), np.full((2, 2), 0.25))
        self.assertEqual(old[:, 0, 0].mean(), 0.5)
        self.assertLess(corrected[:, 0, 1].var(), old[:, 0, 1].var())
        self.assertLess(np.linalg.eigvalsh(corrected_gram(np.ones((2, 1)), np.zeros((2, 1))))[0], 0)

    def test_zero_lambda_and_absent_labels(self):
        for labeled in (False, True):
            Ps, vs, ws = inputs(labeled)
            expected = np.outer([v @ w for v, w in zip(vs, ws)], [v @ w for v, w in zip(vs, ws)])
            config = KernelConfig(lmbd=0, n_samples_mc=5, n_samples_gvoys=5)
            for method in ("mc", "gvoys", "direct", "cg"):
                assert_allclose(compute_kernel(method, Ps, vs, ws, config, labeled=labeled), expected, atol=1e-15)
        empty = [{} for _ in vs]
        for method in ("mc", "gvoys", "direct", "cg"):
            assert_allclose(compute_kernel(method, empty, vs, ws, KernelConfig(n_samples_gvoys=9), labeled=True), expected)
        P = sp.csr_matrix([[0, 1], [1, 0]])
        v = w = np.ones(2)/2
        self.assertEqual(mcrwk.random_walk_kernel_mc_labeled({0:P}, {1:P}, v,v,w,w,
                         utils.mu_func_gen("geom", 0.5), "geom"), 1/4)

    def test_gvoys_blocks_and_rng_independence(self):
        Ps, vs, ws = inputs(True)
        kwargs = dict(kind="geom", lambda_coeff=0.3, nb_random_walks=19,
                      anchor_fraction=0.5, seed=12)
        a = gvoys.random_walk_kernel_gvoys_labeled_dataset(Ps, vs, ws, block_size=4, **kwargs)
        np.random.seed(999)
        np.random.random(1000)
        b = gvoys.random_walk_kernel_gvoys_labeled_dataset(Ps, vs, ws, block_size=19, **kwargs)
        assert_allclose(a, b, rtol=1e-14, atol=1e-16)
        mu = utils.mu_func_gen("geom", 0.3)
        args = (Ps, vs, ws, mu, "geom")
        a = mcrwk.random_walk_kernel_mc_labeled_dataset(*args, n_length_samples=37,
                n_label_samples_per_length=3, q_sampling_kind="random", seed=12)
        np.random.seed(111)
        b = mcrwk.random_walk_kernel_mc_labeled_dataset(*args, n_length_samples=37,
                n_label_samples_per_length=3, q_sampling_kind="random", seed=12)
        assert_array_equal(a, b)

    def test_gvoys_uses_one_realization_and_a_psd_feature_gram(self):
        Ps, vs, ws = inputs(True)
        features = np.array([[1., -2., 3.], [4., 5., -6.]])

        def feature(graph, shared, seed, graph_id, walk_id, *args):
            return features[graph_id, walk_id]

        with patch.object(gvoys, "_feature", side_effect=feature) as sampler:
            result = gvoys.random_walk_kernel_gvoys_labeled_dataset(
                Ps, vs, ws, nb_random_walks=3, block_size=2,
            )
        self.assertEqual(sampler.call_count, len(Ps)*3)
        assert_allclose(result, features @ features.T / 3)
        self.assertGreaterEqual(np.linalg.eigvalsh(result).min(), -1e-14)
        metadata = KernelConfig().metadata()
        self.assertEqual(metadata["gvoys_replicas"], 1)
        self.assertEqual(metadata["gvoys_sides_per_replica"], 2)
        self.assertEqual(metadata["mc_replicas"], 2)

    def test_invalid_inputs_are_rejected(self):
        P = sp.eye(2)
        v = w = np.ones(2)/2
        mu = utils.mu_func_gen("geom", 0.1)
        for bad in (np.ones((2,2)), np.array([[1, -0.1], [0, 1]])):
            with self.assertRaises(ValueError):
                mcrwk.random_walk_kernel_mc_dataset([bad], [v], [w], mu, "geom")
        with self.assertRaises(ValueError):
            mcrwk.random_walk_kernel_mc_dataset([P], [v], [w], mu, "geom", n_samples=0)
        with self.assertRaises(ValueError):
            utils.mu_func_gen("geom", 1)

    def test_error_metrics_do_not_divide_by_zero(self):
        errors = gram.matrix_errors(np.zeros((2,2)), np.eye(2))
        self.assertEqual(errors["mean_abs"], 0.5)
        self.assertIsNone(errors["mean_rel"])
        self.assertIsNone(errors["relative_frobenius"])


class StatisticalTests(unittest.TestCase):
    def test_pair_estimators_and_mc_diagonal_against_exact(self):
        # Independent run-level SE: repeated labels within one length are not
        # incorrectly counted as independent observations. Fixed seeds avoid flakiness.
        for labeled in (False, True):
            Ps, vs, ws = inputs(labeled)
            for kind in ("exp", "geom"):
                config = KernelConfig(kind=kind, lmbd=0.3, n_samples_mc=180,
                    n_length_samples=180, n_label_samples_per_length=2,
                    n_walk_reps=2, n_samples_gvoys=180, anchor_fraction=0.5,
                    q_sampling_kind="norm_fro")
                exact = compute_kernel("direct", Ps, vs, ws, config, labeled=labeled)
                for method in ("mc", "gvoys"):
                    with self.subTest(labeled=labeled, kind=kind, method=method):
                        runs = np.array([compute_kernel(method, Ps, vs, ws, config,
                                         seed=seed, labeled=labeled) for seed in range(24)])
                        se = runs.std(axis=0, ddof=1)/np.sqrt(len(runs))
                        error = abs(runs.mean(axis=0)-exact)
                        # GVoys uses the original feature-square diagonal, not
                        # an independent self-pair estimate. Only MC corrects it.
                        checked = np.ones_like(exact, dtype=bool) if method == "mc" else ~np.eye(len(Ps), dtype=bool)
                        self.assertTrue(np.all(error[checked] <= 6*se[checked]+1e-5),
                            f"{method}/{kind}/labeled={labeled}: error={error}, SE={se}")
                        self.assertTrue(np.all(se > 0))
                        if method == "gvoys":
                            for K in runs:
                                self.assertGreaterEqual(np.linalg.eigvalsh(K).min(), -1e-12)

    def test_gvoys_independent_self_pair_against_exact(self):
        Ps, vs, ws = inputs(True)
        config = KernelConfig(lmbd=0.3, n_samples_gvoys=180, anchor_fraction=0.5)
        exact = compute_kernel("direct", Ps[:1], vs[:1], ws[:1], config, labeled=True)[0, 0]
        runs = np.array([
            gvoys.random_walk_kernel_gvoys_labeled(
                Ps[0], Ps[0], vs[0], vs[0], ws[0], ws[0],
                kind="geom", lambda_coeff=0.3, nb_random_walks=180,
                anchor_fraction=0.5, seed=seed,
            ) for seed in range(24)
        ])
        se = runs.std(ddof=1)/np.sqrt(len(runs))
        self.assertLessEqual(abs(runs.mean()-exact), 6*se+1e-5)


if __name__ == "__main__":
    unittest.main()
