"""Figures, markdown tables and JSON summaries for the results of exps/*.py.

    python exps/plot_results.py results/exps/scaling/*.json --out-dir fig/exps

Files of one experiment are merged, so e.g. separate runs for small and large
sizes can be plotted together. Every figure is written with the plotted
numbers (mean +- std over repeats) as a markdown table and as a JSON list. Exact methods are omitted
from error panels: they agree with the reference to solver precision.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np

EXACT = {"direct", "cg", "fixed_point", "sylvester"}
MUTED, GRID, AXIS = "#52514e", "#e1e0d9", "#c3c2b7"
# Fixed color slot per method (validated palette); markers are the secondary encoding.
STYLE = {"mc_cN": ("#4a3aa7", "o", "MCRWK (m = cN)"),
         "gvoys": ("#eb6834", "s", "GVoys"),
         "cg": ("#1baf7a", "^", "CG"),
         "fixed_point": ("#eda100", "v", "FP"),
         "sylvester": ("#e87ba4", "D", "Sylvester"),
         "direct": ("#008300", "P", "Direct")}
# Fixed-budget MCRWK: one blue ordinal ramp, light (small m) to dark (large m).
MC_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.edgecolor": AXIS,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight"})


def fixed_m(name):
    return int(name.split("=")[1]) if name.startswith("mc_m=") else None


def method_order(names):
    fixed = sorted(n for n in names if fixed_m(n) is not None)
    fixed.sort(key=fixed_m)
    known = [n for n in ["mc_cN", *fixed, "gvoys", "cg", "fixed_point", "sylvester", "direct"] if n in names]
    return known+sorted(set(names)-set(known))


def style(name, names):
    if name in STYLE:
        color, marker, label = STYLE[name]
        return dict(color=color, marker=marker, label=label, linestyle="-")
    fixed = sorted({fixed_m(n) for n in names if fixed_m(n) is not None})
    if name.startswith("mc_m="):
        m = fixed_m(name)
        step = np.linspace(0, len(MC_RAMP)-1, len(fixed)).round().astype(int)[fixed.index(m)] if len(fixed) > 1 else 2
        return dict(color=MC_RAMP[step], marker="o", label=f"MCRWK (m = {m:,})", linestyle="--")
    return dict(color=MUTED, marker="x", label=name, linestyle=":")


# ----------------------------------------------------------------------
#  Record access and aggregation
# ----------------------------------------------------------------------
def load(paths):
    by_experiment = defaultdict(list)
    for path in paths:
        payload = json.loads(Path(path).read_text())
        by_experiment[payload["experiment"]] += payload["records"]
    return by_experiment


def runtime(record):
    return record.get("time_sec") if record.get("status") == "ok" else None


def error(metric):
    return lambda record: (record.get("errors") or {}).get(metric)


def evaluation(key):
    return lambda record: (record.get("evaluation") or {}).get(key) if record.get("status") == "ok" else None


def summarize(records, keys, value):
    groups = defaultdict(list)
    for record in records:
        v = value(record)
        if v is not None and np.isfinite(v):
            groups[tuple(record.get(k) for k in keys)].append(v)
    return {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in groups.items()}


def table(title, rows, columns, cells, fmt="{:.3g}"):
    """Markdown table; cells maps (row, column) to (mean, std, count)."""
    lines = [f"### {title}", "", "| method | "+" | ".join(map(str, columns))+" |",
             "|---|"+"---|"*len(columns)]
    for row in rows:
        values = []
        for column in columns:
            if (row, column) in cells:
                mean, std, n = cells[row, column]
                values.append(fmt.format(mean)+(f" ± {fmt.format(std)}" if n > 1 else ""))
            else:
                values.append("—")
        lines.append(f"| {row} | "+" | ".join(values)+" |")
    return "\n".join(lines)+"\n"


def log_axis(axis, set_scale):
    """Log scale labelled at 1-2-5 steps in plain notation."""
    set_scale("log")
    axis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))
    axis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_minor_formatter(ticker.NullFormatter())


def finish(ax, xlabel, ylabel, xlog=False, ylog=False):
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if xlog:
        ax.set_xscale("log", base=2)
    if ylog:
        log_axis(ax.yaxis, ax.set_yscale)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------
#  Line figures: metric versus graph size or lambda
# ----------------------------------------------------------------------
def line_figure(records, x_key, xlabel, panels, title, path, *, xlog):
    """panels: (ylabel, value function, log scale, include exact methods)."""
    names = method_order({r["name"] for r in records if r.get("name")})
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2*len(panels), 3.4), squeeze=False)
    tables = []
    for ax, (ylabel, value, ylog, with_exact) in zip(axes[0], panels):
        cells = summarize(records, ["name", x_key], value)
        shown = [n for n in names if (with_exact or n not in EXACT) and any(k[0] == n for k in cells)]
        xs = sorted({k[1] for k in cells})
        for rank, name in enumerate(shown):
            points = sorted((x, *cells[name, x][:2]) for x in xs if (name, x) in cells)
            points = [p for p in points if not ylog or p[1] > 0]
            if not points:
                continue
            x, mean, std = map(np.asarray, zip(*points))
            st = style(name, names)
            ax.errorbar(x, mean, yerr=std, color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                        label=st["label"], linewidth=2, markersize=6, capsize=2, elinewidth=1,
                        markeredgecolor="white", markeredgewidth=1, zorder=3+len(shown)-rank)
        finish(ax, xlabel, ylabel, xlog, ylog)
        tables.append((f"{title}: {ylabel}", shown, xs, cells))
    handles = {}
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=min(4, len(handles)),
               bbox_to_anchor=(0.5, 0.0))
    save(fig, path, tables)


def save(fig, path, tables):
    """PNG plus its numbers; tables are (title, methods, x values, cells)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    path.with_suffix(".md").write_text("\n".join(table(*t) for t in tables))
    summary = [{"table": title, "method": row, "x": column, "mean": mean, "std": std, "n": n}
               for title, rows, columns, cells in tables for row in rows for column in columns
               if (row, column) in cells for mean, std, n in [cells[row, column]]]
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(f"wrote {path} (+ .md, .json)")


def cases(records):
    return sorted({r["case"] for r in records if r.get("case")}, reverse=True)


def plot_scaling(records, out, metric):
    for case in cases(records):
        rows = [r for r in records if r.get("case") == case]
        line_figure(rows, "n_nodes", "vertices per graph N",
                    [("runtime, s", runtime, True, True), ("relative error", error(metric), True, False)],
                    f"Scaling, {case}", out/f"scaling_{case}.png", xlog=True)


def plot_lambda(records, out, metric):
    synthetic = [r for r in records if r.get("setting") == "synthetic"]
    for case in cases(synthetic):
        rows = [r for r in synthetic if r.get("case") == case]
        line_figure(rows, "lmbd", "λ", [("runtime, s", runtime, True, True),
                    ("relative error", error(metric), True, False)],
                    f"λ sweep, synthetic, {case}", out/f"lambda_synthetic_{case}.png", xlog=False)
    tu = [r for r in records if r.get("setting") == "tu" and r.get("name")]
    for dataset in sorted({r["dataset"] for r in tu}):
        for case in cases(tu):
            rows = [r for r in tu if r["dataset"] == dataset and r["case"] == case]
            if not rows:
                continue
            quality = ("accuracy", evaluation("mean_accuracy"), False, True) if rows[0]["task"] == "classification" \
                else ("RMSE", evaluation("mean_rmse"), False, True)
            line_figure(rows, "lmbd", "λ", [("Gram time, s", runtime, True, True), quality,
                        ("relative error", error(metric), True, False)],
                        f"λ sweep, {dataset}, {case}", out/f"lambda_{dataset}_{case}.png", xlog=False)


# ----------------------------------------------------------------------
#  TU figures: dot plots per dataset
# ----------------------------------------------------------------------
def dot_figure(records, datasets, value, xlabel, title, path, *, xlog, xerr=None):
    """One panel per dataset; methods on the y axis, one dot each."""
    cells = summarize(records, ["name", "dataset"], value)
    names = method_order({name for name, _ in cells})
    fig, axes = plt.subplots(1, len(datasets), figsize=(2.9*len(datasets)+1.2, 0.32*len(names)+1.2),
                             sharey=True, squeeze=False)
    spread = summarize(records, ["name", "dataset"], xerr) if xerr else {}
    for ax, dataset in zip(axes[0], datasets):
        for y, name in enumerate(names):
            if (name, dataset) not in cells:
                continue
            mean, std, n = cells[name, dataset]
            err = spread.get((name, dataset), (None,))[0] if xerr else (std if n > 1 else None)
            st = style(name, names)
            ax.errorbar([mean], [y], xerr=None if err is None else [err], color=st["color"], marker=st["marker"],
                        markersize=7, capsize=2, elinewidth=1, markeredgecolor="white", markeredgewidth=1)
        ax.set_title(dataset, fontsize=9)
        ax.set_yticks(range(len(names)), [style(n, names)["label"] for n in names])
        ax.set_ylim(len(names)-0.5, -0.5)
        finish(ax, xlabel, "", xlog=False, ylog=False)
        if xlog:
            log_axis(ax.xaxis, ax.set_xscale)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    save(fig, path, [(f"{title}: {xlabel}", names, datasets, cells)])


def full_size(records):
    """Keep only rows of the largest subset per dataset (gram_time --n-graphs-list)."""
    largest = defaultdict(int)
    for r in records:
        largest[r["dataset"]] = max(largest[r["dataset"]], r.get("n_graphs") or 0)
    return [r for r in records if (r.get("n_graphs") or 0) == largest[r["dataset"]]]


def plot_tu_svm(records, out, metric):
    rows = [r for r in records if r.get("name")]
    for case in cases(rows):
        in_case = [r for r in rows if r["case"] == case]
        for task, key, spread, label in [("classification", "mean_accuracy", "std_accuracy", "accuracy"),
                                         ("regression", "mean_rmse", "std_rmse", "RMSE")]:
            subset = [r for r in in_case if r["task"] == task]
            datasets = sorted({r["dataset"] for r in subset})
            if datasets:
                dot_figure(subset, datasets, evaluation(key), f"{label} (± fold std)", f"SVM {task}, {case}",
                           out/f"tu_{task}_{case}.png", xlog=False, xerr=evaluation(spread))
        datasets = sorted({r["dataset"] for r in in_case})
        dot_figure(in_case, datasets, runtime, "Gram time, s", f"Gram time, {case}",
                   out/f"tu_gram_time_{case}.png", xlog=True)


def plot_gram_time(records, out, metric):
    rows = [r for r in records if r.get("name")]
    for case in cases(rows):
        in_case = [r for r in rows if r["case"] == case]
        datasets = sorted({r["dataset"] for r in in_case})
        dot_figure(full_size(in_case), datasets, runtime, "Gram time, s", f"Gram time, {case}",
                   out/f"gram_time_{case}.png", xlog=True)
        for dataset in datasets:
            per_dataset = [r for r in in_case if r["dataset"] == dataset]
            if len({r["n_graphs"] for r in per_dataset}) > 1:
                line_figure(per_dataset, "n_graphs", "number of graphs", [("Gram time, s", runtime, True, True)],
                            f"Gram time vs dataset size, {dataset}, {case}",
                            out/f"gram_time_vs_graphs_{dataset}_{case}.png", xlog=True)


PLOTS = {"scaling": plot_scaling, "tu_svm": plot_tu_svm, "gram_time": plot_gram_time, "lambda_sweep": plot_lambda}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="JSON files written by exps/*.py")
    parser.add_argument("--out-dir", default="fig/exps")
    parser.add_argument("--error-metric", default="offdiagonal_mean_rel",
                        help="Key of src.gram.matrix_errors, e.g. mean_rel or relative_frobenius.")
    args = parser.parse_args(argv)
    for experiment, records in load(args.results).items():
        PLOTS[experiment](records, Path(args.out_dir), args.error_metric)


if __name__ == "__main__":
    main()
