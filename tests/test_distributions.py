"""Boundary-weight semantics shared by the synthetic and TU experiments."""

import unittest
from unittest.mock import Mock

import networkx as nx
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from dataset_bench import parse_args as parse_tu_args
from synthetic_bench import parse_args as parse_synthetic_args
from src import gram, utils
from src.benchmark import build_inputs


class BoundaryDistributionTests(unittest.TestCase):
    def test_normal_uses_absolute_gaussian_draws(self):
        rng = Mock()
        rng.standard_normal.return_value = np.array([-2., 1., 0.])
        assert_allclose(utils.normal_dist(3, rng), [2/3, 1/3, 0])
        rng.standard_normal.assert_called_once_with(3)
        rng.standard_normal.return_value = np.zeros(3)
        assert_allclose(utils.normal_dist(3, rng), np.ones(3)/3)

    def test_normal_boundaries_are_reproducible_independent_and_nonnegative(self):
        graph = nx.path_graph(20)
        for labeled in (False, True):
            nx.set_edge_attributes(graph, 0, "label")
            _, vs, ws = build_inputs([graph], "normal", labeled, seed=12)
            _, repeated_v, repeated_w = build_inputs([graph], "normal", labeled, seed=12)
            _, random_v, _ = build_inputs([graph], "random", labeled, seed=12)
            for vector in (vs[0], ws[0]):
                self.assertTrue(np.isfinite(vector).all())
                self.assertTrue((vector >= 0).all())
                self.assertAlmostEqual(vector.sum(), 1)
            assert_array_equal(vs[0], repeated_v[0])
            assert_array_equal(ws[0], repeated_w[0])
            self.assertFalse(np.array_equal(vs[0], ws[0]))
            self.assertFalse(np.array_equal(vs[0], random_v[0]))

    def test_degree_uses_weights_self_loops_and_node_order(self):
        graph = nx.Graph()
        graph.add_nodes_from(["hub", "leaf", "isolate"])
        graph.add_edge("hub", "hub", weight=2, label=0)
        graph.add_edge("hub", "leaf", weight=3, label=1)
        for labeled in (False, True):
            _, vs, ws = build_inputs([graph], "degree", labeled)
            assert_allclose(vs[0], [5/8, 3/8, 0])
            assert_array_equal(vs[0], ws[0])
            self.assertFalse(np.shares_memory(vs[0], ws[0]))

    def test_degree_edgeless_fallback_and_invalid_weights(self):
        graph = nx.empty_graph(3)
        assert_allclose(utils.degree_dist(graph), np.ones(3)/3)
        graph.add_edge(0, 1, weight=0)
        assert_allclose(utils.degree_dist(graph), np.ones(3)/3)
        graph[0][1]["weight"] = -1
        with self.assertRaises(ValueError):
            utils.degree_dist(graph)

    def test_degree_is_stationary_and_gives_rank_one_unlabeled_kernel(self):
        graphs = [nx.path_graph(4), nx.star_graph(4), nx.empty_graph(3)]
        Ps, vs, ws = build_inputs(graphs, "degree")
        for P, v in zip(Ps, vs):
            assert_allclose(v @ P, v)
        h = np.array([v @ w for v, w in zip(vs, ws)])
        exact = gram.gram_direct(Ps, vs, ws, utils.mu_func_gen("geom", 0.3), "geom")
        assert_allclose(exact, np.outer(h, h)/0.7)

    def test_both_cli_parsers_accept_new_distributions_and_flag_alias(self):
        for parse in (parse_synthetic_args, parse_tu_args):
            for flag in ("--u-w-distribution", "--u_w_distribution"):
                for distribution in ("normal", "degree"):
                    self.assertEqual(parse([flag, distribution]).u_w_distribution, distribution)


if __name__ == "__main__":
    unittest.main()
