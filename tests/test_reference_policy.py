"""Protect the product-graph cutoff and size-dependent numerical reference."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

import dataset_bench
import synthetic_bench
from src import utils
from src.benchmark import KernelConfig, method_skip_reason, planned_methods, reference_method


class ReferencePolicyTests(unittest.TestCase):
    def fake_compute(self, method, Ps, vs, ws, config, seed, labeled):
        values = {"direct": 1., "series": 3., "cg": 2., "mc": 2.2}
        return np.full((len(Ps), len(Ps)), values[method])

    def synthetic(self, size, *, kind="geom", methods=None, direct_limit=128):
        return synthetic_bench.bench(
            [nx.path_graph(size)], kind, utils.mu_func_gen(kind, 0.3),
            1, 3, 3, config=KernelConfig(kind=kind, lmbd=0.3),
            methods=methods or ["direct", "series", "mc"],
            direct_max_nodes=direct_limit,
        )

    def test_synthetic_127_128_129_boundary_and_auto_cg(self):
        for size, expected_ref in ((127, "direct"), (128, "cg"), (129, "cg")):
            with self.subTest(size=size), contextlib.redirect_stdout(io.StringIO()):
                with patch.object(synthetic_bench, "compute_kernel", side_effect=self.fake_compute) as compute:
                    data, _ = self.synthetic(size)
                called = [call.args[0] for call in compute.call_args_list]
                self.assertEqual(data["metadata"]["reference_method"], expected_ref)
                self.assertEqual("direct" in called, size < 128)
                self.assertEqual("cg" in called, size >= 128)
                expected_error = 0.2 if size >= 128 else 1.2
                self.assertAlmostEqual(data["mc"]["err"]["mean_abs"], expected_error)
                if size >= 128:
                    self.assertEqual(data["direct"]["status"], "skipped")

    def test_tu_uses_largest_graph_and_same_boundaries(self):
        for size, expected_ref in ((127, "direct"), (128, "cg"), (129, "cg")):
            # A mixed-size subset must apply the cutoff to its largest graph.
            graphs = [nx.path_graph(3), nx.path_graph(size), nx.path_graph(4), nx.path_graph(5)]
            for graph in graphs:
                nx.set_edge_attributes(graph, 0, "label")
            y = np.array([0, 0, 1, 1])
            with self.subTest(size=size), tempfile.TemporaryDirectory() as tmp:
                with contextlib.redirect_stdout(io.StringIO()), \
                     patch.object(dataset_bench, "load_tu_dataset", return_value=(graphs, y, y, True)), \
                     patch.object(dataset_bench, "compute_kernel", side_effect=self.fake_compute) as compute:
                    status = dataset_bench.run_tu_benchmark(
                        dataset_names=["Fixture"], kind="geom", methods=["direct", "mc"],
                        max_graphs=None, max_nodes_per_graph=None, n_samples_mc=3,
                        n_samples_gvoys=3, c_values=[1.], n_splits=2, n_repeats=1,
                        inner_splits=2, normalize_kernel=False, root_dir=tmp, seed=42,
                        output_dir=tmp, skip_svm=True, request_edge_labels=True,
                    )
                self.assertEqual(status, 0)
                data = json.loads(next(Path(tmp).glob("Fixture/*.json")).read_text())
                called = [call.args[0] for call in compute.call_args_list]
                self.assertEqual(data["metadata"]["reference_method"], expected_ref)
                self.assertEqual("direct" in called, size < 128)
                self.assertEqual("cg" in called, size >= 128)

    def test_hard_cap_cannot_be_bypassed_and_can_be_lowered(self):
        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(synthetic_bench, "compute_kernel", side_effect=self.fake_compute) as compute:
            self.synthetic(128, direct_limit=1000)
        self.assertNotIn("direct", [call.args[0] for call in compute.call_args_list])
        self.assertIsNotNone(method_skip_reason("direct", kind="geom", labeled=False,
                                               max_nodes=16, direct_max_nodes=16))
        for parser in (synthetic_bench.parse_args, dataset_bench.parse_args):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser(["--direct-max-nodes", "129"])

    def test_failed_cg_is_not_replaced_by_series(self):
        def compute(method, *args):
            if method == "cg":
                raise RuntimeError("CG did not converge")
            return self.fake_compute(method, *args)

        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(synthetic_bench, "compute_kernel", side_effect=compute):
            data, _ = self.synthetic(129)
        self.assertEqual(data["cg"]["status"], "failed")
        self.assertEqual(data["series"]["status"], "ok")
        self.assertIsNone(data["metadata"]["reference_method"])
        self.assertIsNone(data["mc"]["err"])

    def test_exp_does_not_add_or_select_cg(self):
        with contextlib.redirect_stdout(io.StringIO()), \
             patch.object(synthetic_bench, "compute_kernel", side_effect=self.fake_compute) as compute:
            data, _ = self.synthetic(129, kind="exp")
        self.assertEqual(data["metadata"]["reference_method"], "series")
        self.assertNotIn("cg", [call.args[0] for call in compute.call_args_list])
        self.assertEqual(data["direct"]["status"], "skipped")
        self.assertIsNone(reference_method({"mc": np.eye(1)}, kind="exp", max_nodes=129))

    def test_existing_cg_is_not_scheduled_twice(self):
        self.assertEqual(planned_methods(["mc", "cg"], kind="geom", max_nodes=128), ["mc", "cg"])


if __name__ == "__main__":
    unittest.main()
