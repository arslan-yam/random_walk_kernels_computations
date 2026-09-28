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
            by_name = {(r["dataset"], r["case"], r["name"]): r for r in records}
            for (dataset, case, name), record in by_name.items():
                if name.endswith("_biased"):
                    # Same walks and the same timings; both diagonals are timed up to their Gram.
                    twin = by_name[dataset, case, name.removesuffix("_biased")]
                    self.assertEqual(record["diagonal"], "biased")
                    for key in ("feature_time_sec", "gram_build_sec", "time_sec", "time_basis"):
                        self.assertEqual(record[key], twin[key])
            self.assertIn(("TOYC", "labeled", "mc_cN_biased"), by_name)
            for record in records:
                self.assertEqual(record["status"], "ok", record)
                key = "mean_accuracy" if record["task"] == "classification" else "mean_rmse"
                # SVC on the Gram for every method; linear SVM only where features exist.
                self.assertIn(key, record["evaluation"])
                has_features = record["name"] == "gvoys" or record["name"].endswith("_biased")
                self.assertEqual("evaluation_linear" in record, has_features, record["name"])
                if has_features:
                    self.assertIn(key, record["evaluation_linear"])
                    self.assertIn("convergence_warnings", record["evaluation_linear"])
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

    def test_q_and_n_sampling_on_local_data(self):
        graphs = toy_graphs()
        with tempfile.TemporaryDirectory() as tmp:
            write_tu(tmp, "TOYC", graphs, [0]*6+[1]*6)
            write_tu(tmp, "TOYR", graphs, [len(g)+0.5*(i % 2) for i, g in enumerate(graphs)], regression=True)
            shared = ["--datasets", "TOYC", "TOYR", "--root-dir", tmp, "--n-repeats", "2", "--lambdas", "0.3",
                      "--n-splits", "2", "--inner-splits", "2", "--c-values", "1", "--epsilon-values", "0.1"]
            output = Path(tmp)/"q.json"
            self.run_script("q_sampling.py", *shared, "--m-values", "20", "--mix-eps", "0", "0.1",
                            "--proposals", "uniform", "freq:2", "sq_mean", "random", "--with-gvoys",
                            "--n-samples-gvoys", "3", "--output", str(output))
            data = json.loads(output.read_text())
            self.assertEqual(len(data["references"]), 2)
            # uniform once, three proposals with two mixing weights, GVoys; per dataset.
            self.assertEqual(len(data["aggregates"]), 2*(1+3*2+1))
            for aggregate in data["aggregates"]:
                self.assertEqual(aggregate["n_ok"], 2)
                self.assertIn("rel_mse", aggregate)
                if aggregate["proposal"] != "gvoys":
                    self.assertGreater(aggregate["ridge_rel"], 0)
                    records = [r for r in data["records"] if r["proposal"] == aggregate["proposal"]]
                    for record in records:
                        self.assertTrue({"evaluation", "evaluation_biased", "evaluation_biased_linear"} <= set(record))
                        self.assertNotIn("evaluation_linear", record)
                    self.assertIn("mean_accuracy_biased_over_seeds" if aggregate["task"] == "classification"
                                  else "mean_rmse_biased_over_seeds", aggregate)
                if aggregate["proposal"] in ("uniform", "freq:2", "sq_mean"):
                    self.assertAlmostEqual(sum(aggregate["q"].values()), 1)
                    self.assertEqual(aggregate["bound_finite"], 0.3 < aggregate["q_min"])
            self.assertEqual(len(data["records"]), 2*len(data["aggregates"]))
            output = Path(tmp)/"n.json"
            self.run_script("n_sampling.py", *shared, "--n-values", "1", "4", "--budgets", "40",
                            "--lengths", "10", "--skip-evaluation", "--output", str(output))
            aggregates = json.loads(output.read_text())["aggregates"]
            designs = {(a["dataset"], a["design"], a["n"]): a for a in aggregates}
            self.assertEqual(designs["TOYC", "fixed_budget", 4]["lengths"], 10)
            self.assertEqual(designs["TOYC", "fixed_lengths", 4]["budget"], 40)
            self.assertIn("variance_bound", designs["TOYC", "fixed_budget", 1])
            self.run_script("plot_results.py", str(Path(tmp)/"q.json"), str(output), "--out-dir", tmp)
            self.assertTrue((Path(tmp)/"q_TOYR_lambda0.3.json").exists())
            self.assertTrue((Path(tmp)/"n_TOYC_lambda0.3.png").exists())

    def test_feature_normalization_matches_gram_normalization(self):
        from dataset_bench import normalize_gram_matrix
        rng = np.random.default_rng(1)
        X = rng.standard_normal((12, 30))
        rows = X/np.linalg.norm(X, axis=1)[:, None]
        np.testing.assert_allclose(rows @ rows.T, normalize_gram_matrix(X @ X.T), atol=1e-12)

    def test_linear_models_on_separable_features(self):
        import argparse
        rng = np.random.default_rng(0)
        y = np.repeat([0, 1, 2], 12)
        X = np.eye(3)[y]*5+0.1*rng.standard_normal((36, 3))+1
        args = argparse.Namespace(no_normalize=False, c_values=[1., 10.], epsilon_values=[0.1], n_splits=3,
                                  n_cv_repeats=1, inner_splits=2, linear_max_iter=10000)
        result = common.evaluate_features(X, y, "classification", args, 0)
        self.assertEqual((result["mean_accuracy"], len(result["scores"])), (1.0, 3))
        target = X @ np.array([1., -2., 0.5])
        result = common.evaluate_features(X, target, "regression", args, 0)
        self.assertGreater(result["mean_r2"], 0.8)

    def test_proposals_follow_label_frequencies(self):
        from exps.q_sampling import Labeled, proposal
        data = Labeled("toy", [], None, "classification", [], [], [], [0, 1], np.array([0.8, 0.2]),
                       np.array([0.5, 0.1]), {})
        q = lambda spec, eps=0.: np.array(list(proposal(spec, data, eps, None).values()))
        np.testing.assert_allclose(q("freq:2"), [16/17, 1/17])
        np.testing.assert_allclose(q("inverse"), [0.2, 0.8])
        np.testing.assert_allclose(q("sq_mean"), [5/6, 1/6])
        np.testing.assert_allclose(q("freq:1", 0.5), [0.65, 0.35])

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
