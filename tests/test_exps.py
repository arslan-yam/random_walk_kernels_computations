"""Smoke and consistency checks for the paper experiments in exps/."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import networkx as nx
import numpy as np

from exps import common
from src.benchmark import KernelConfig, build_inputs

ROOT = Path(__file__).resolve().parents[1]
FAST = ["--n-samples-gvoys", "3", "--mc-fixed-m", "20", "40", "--calibration-repeats", "1"]


def write_tu(root, name, graphs, targets, regression=False):
    """Write graphs in TU format; both edge directions carry the same label."""
    directory = Path(root)/name
    directory.mkdir(parents=True)
    indicator, edges, labels, offset = [], [], [], 0
    for graph_id, graph in enumerate(graphs, start=1):
        ids = {u: offset+i+1 for i, u in enumerate(graph)}
        indicator += [graph_id]*len(graph)
        for u, v in graph.edges:
            edges += [(ids[u], ids[v]), (ids[v], ids[u])]
            labels += [(u+v) % 2]*2
        offset += len(graph)
    lines = lambda values: "\n".join(map(str, values))+"\n"
    (directory/f"{name}_graph_indicator.txt").write_text(lines(indicator))
    (directory/f"{name}_A.txt").write_text(lines(f"{a}, {b}" for a, b in edges))
    (directory/f"{name}_edge_labels.txt").write_text(lines(labels))
    target_file = "graph_attributes" if regression else "graph_labels"
    (directory/f"{name}_{target_file}.txt").write_text(lines(targets))


def toy_graphs():
    return [nx.cycle_graph(n) for n in range(4, 10)]+[nx.path_graph(n) for n in range(4, 10)]


class ExpsTests(unittest.TestCase):
    def run_script(self, script, *args):
        result = subprocess.run([sys.executable, str(ROOT/"exps"/script), *args],
                                cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout[-2000:]+result.stderr[-2000:])
        return result

    def test_scaling_records_every_method_and_matched_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"scaling.json"
            self.run_script("scaling.py", "--sizes", "8", "12", "--n-repeats", "1",
                            "--calibration-n", "8", "--output", str(output), *FAST)
            data = json.loads(output.read_text())
            self.assertEqual({c["case"] for c in data["calibrations"]}, {"unlabeled", "labeled"})
            c = {cal["case"]: cal["c"] for cal in data["calibrations"]}
            for record in data["records"]:
                expected = "skipped" if record["case"] == "labeled" and record["name"] == "sylvester" else "ok"
                self.assertEqual(record["status"], expected, record)
                self.assertEqual(record["reference"], "direct")
                self.assertIn("input_time_sec", record)
                if expected == "ok":
                    self.assertEqual(np.shape(record["gram"]), (2, 2))
                    self.assertIsNotNone(record["time_sec"])
                if record["name"] == "mc_cN":
                    self.assertEqual(record["m"], max(1, round(c[record["case"]]*record["n_nodes"])))
            names = {r["name"] for r in data["records"]}
            self.assertEqual(names, {"direct", "cg", "fixed_point", "sylvester", "gvoys", "mc_cN",
                                     "mc_m=20", "mc_m=40"})
            self.run_script("plot_results.py", str(output), "--out-dir", tmp)
            self.assertTrue((Path(tmp)/"scaling_labeled.png").exists())
            summary = json.loads((Path(tmp)/"scaling_labeled.json").read_text())
            self.assertEqual({row["x"] for row in summary}, {8, 12})

    def test_tu_scripts_on_local_classification_and_regression_data(self):
        graphs = toy_graphs()
        with tempfile.TemporaryDirectory() as tmp:
            write_tu(tmp, "TOYC", graphs, [0]*6+[1]*6)
            write_tu(tmp, "TOYR", graphs, [len(g)+0.5*(i % 2) for i, g in enumerate(graphs)], regression=True)
            common_args = ["--datasets", "TOYC", "TOYR", "--root-dir", tmp, "--calibration-graphs", "3",
                           "--methods", "direct", "gvoys", "mc_matched", "mc_fixed", *FAST]
            output = Path(tmp)/"tu.json"
            self.run_script("tu_svm.py", *common_args, "--n-splits", "2", "--inner-splits", "2",
                            "--c-values", "1", "10", "--epsilon-values", "0.1", "--output", str(output))
            records = json.loads(output.read_text())["records"]
            self.assertEqual({(r["dataset"], r["task"]) for r in records},
                             {("TOYC", "classification"), ("TOYR", "regression")})
            for record in records:
                self.assertEqual(record["status"], "ok", record)
                key = "mean_accuracy" if record["task"] == "classification" else "mean_rmse"
                self.assertIn(key, record["evaluation"])
                self.assertEqual(record["n_folds"], 2)
                self.assertEqual(np.shape(record["gram"]), (12, 12))
            output = Path(tmp)/"gram.json"
            self.run_script("gram_time.py", *common_args, "--n-graphs-list", "4", "--n-repeats", "2",
                            "--output", str(output))
            records = json.loads(output.read_text())["records"]
            self.assertEqual({r["n_graphs"] for r in records}, {4, 12})
            self.assertEqual(len([r for r in records if r["name"] == "gvoys"]), 2*2*2*2)

    def test_lambda_sweep_calibrates_each_lambda(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"lambda.json"
            self.run_script("lambda_sweep.py", "--settings", "synthetic", "--lambdas", "0.2", "0.8",
                            "--n-nodes", "8", "--n-repeats", "1", "--cases", "unlabeled",
                            "--output", str(output), *FAST)
            data = json.loads(output.read_text())
            self.assertEqual([c["lmbd"] for c in data["calibrations"]], [0.2, 0.8])
            self.assertEqual({r["lmbd"] for r in data["records"]}, {0.2, 0.8})

    def test_labeled_tu_without_edge_labels_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_tu(tmp, "TOYC", toy_graphs(), [0]*6+[1]*6)
            (Path(tmp)/"TOYC"/"TOYC_edge_labels.txt").unlink()
            output = Path(tmp)/"tu.json"
            self.run_script("gram_time.py", "--datasets", "TOYC", "--root-dir", tmp, "--cases", "labeled",
                            "--methods", "mc_fixed", "--output", str(output), *FAST)
            [record] = json.loads(output.read_text())["records"]
            self.assertEqual(record["status"], "skipped")

    def test_loader_reads_regression_targets_and_edge_labels(self):
        graphs = toy_graphs()
        targets = [0.25*i for i in range(len(graphs))]
        with tempfile.TemporaryDirectory() as tmp:
            write_tu(tmp, "TOYR", graphs, targets, regression=True)
            loaded, y, task, labeled = common.load_tu("TOYR", tmp, edge_labels=True)
        self.assertEqual((task, labeled), ("regression", True))
        np.testing.assert_allclose(y, targets)
        self.assertEqual([g.number_of_edges() for g in loaded], [g.number_of_edges() for g in graphs])
        self.assertTrue(all("label" in data for g in loaded for *_, data in g.edges(data=True)))

    def test_svr_fits_a_linear_kernel_target(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 3))
        y = X @ np.array([1., -2., 0.5])
        result = common.evaluate_svr_precomputed(X @ X.T, y, [10., 100.], [0.01], n_splits=4, inner_splits=2)
        self.assertGreater(result["mean_r2"], 0.95)

    def test_calibration_returns_positive_budget(self):
        graphs = [nx.cycle_graph(6), nx.path_graph(7)]
        Ps, vs, ws = build_inputs(graphs, "normal", False, 1)
        config = KernelConfig(kind="geom", lmbd=0.7, n_samples_gvoys=3).validate()
        calibration = common.calibrate_mc_c(Ps, vs, ws, config, False, n_ref=6.5, seed=1, repeats=1)
        self.assertGreaterEqual(calibration["m"], 1)
        self.assertAlmostEqual(calibration["c"], calibration["m"]/6.5)
        self.assertEqual(len(calibration["mc_probes"]), 3)

    def test_labeled_budget_counts_feature_columns(self):
        config = KernelConfig(n_label_samples_per_length=4).validate()
        budget = common.with_mc_budget(config, 100)
        self.assertEqual((budget.lengths, budget.lengths*budget.n_label_samples_per_length), (25, 100))


if __name__ == "__main__":
    unittest.main()
