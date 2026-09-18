"""Checks of experiment behavior that can silently invalidate comparisons."""

import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from dataset_bench import normalize_gram_matrix
from graph_gp import gp_predict
from kernel_kmeans import kernel_kmeans

ROOT = Path(__file__).resolve().parents[1]


class ExperimentTests(unittest.TestCase):
    def run_synthetic(self, output, methods, *extra):
        return subprocess.run(
            [sys.executable, str(ROOT / "synthetic_bench.py"),
             "--n-graphs", "3", "--n-nodes", "6", "--graph-type", "ba",
             "--u-w-distribution", "random", "--n-samples-mc", "35",
             "--mc-budget-mode", "total", "--n-samples-gvoys", "17",
             "--block-size", "8", "--methods", *methods,
             "--output", str(output), *extra],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )

    def test_cli_seed_independent_of_method_order_and_no_reference_is_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp)/"a.json", Path(tmp)/"b.json"
            for path, methods in ((first, ["mc", "gvoys"]), (second, ["gvoys", "mc"])):
                result = self.run_synthetic(path, methods, "--save-grams")
                self.assertEqual(result.returncode, 0, result.stderr)
                metadata = json.loads(path.read_text())
                self.assertIsNone(metadata["metadata"]["reference_method"])
                self.assertIsNone(metadata["mc"]["err"])
                self.assertEqual(metadata["metadata"]["kernel"]["mc_replicas"], 2)
            # Only read files created by this test.
            with first.with_suffix(".pickle").open("rb") as f:
                a = pickle.load(f)
            with second.with_suffix(".pickle").open("rb") as f:
                b = pickle.load(f)
            for method in a:
                assert_array_equal(a[method], b[method])

    def test_cli_records_solver_failure_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"failure.json"
            result = self.run_synthetic(output, ["cg", "mc"],
                                        "--max-iter", "1", "--solver-tol", "1e-14")
            self.assertEqual(result.returncode, 1, result.stderr)
            data = json.loads(output.read_text())
            self.assertEqual(data["cg"]["status"], "failed")
            self.assertEqual(data["mc"]["status"], "ok")
            self.assertIn("converge", data["cg"]["error"])

    def test_cli_runs_new_boundary_distributions_with_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            for distribution in ("normal", "degree"):
                output = Path(tmp)/f"{distribution}.json"
                result = self.run_synthetic(
                    output, ["direct", "mc", "gvoys"], "--labeled",
                    "--u-w-distribution", distribution,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                data = json.loads(output.read_text())
                self.assertEqual(data["metadata"]["distribution"], distribution)
                for method in ("direct", "mc", "gvoys"):
                    self.assertEqual(data[method]["status"], "ok")

    def test_normalization_does_not_hide_nonpositive_diagonal(self):
        for matrix in (np.zeros((2,2)), np.diag([-1., 1.])):
            with self.assertRaises(ValueError):
                normalize_gram_matrix(matrix)

    def test_kmeans_recovers_empty_clusters(self):
        labels = kernel_kmeans(np.zeros((6,6)), 4, max_iter=10)
        self.assertEqual(len(np.unique(labels)), 4)

    def test_gp_requires_positive_definite_regularized_covariance(self):
        with self.assertRaisesRegex(ValueError, "positive definite"):
            gp_predict(np.array([[0., 1.], [1., 0.]]), np.array([0., 1.]), np.zeros((1,2)))
        estimate = gp_predict(np.eye(2), np.array([0., 1.]), np.array([[1., 0.]]), alpha=0)
        self.assertAlmostEqual(float(estimate[0]), 0)


if __name__ == "__main__":
    unittest.main()
