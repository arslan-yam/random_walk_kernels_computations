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


def marker_colors(st):
    """Filled markers with a white ring, or hollow markers in the series color."""
    return dict(markerfacecolor="white", markeredgecolor=st["color"]) if st.get("hollow") \
        else dict(markerfacecolor=st["color"], markeredgecolor="white")


def fixed_m(name):
    return int(name.split("=")[1]) if name.startswith("mc_m=") else None


BIASED = "_biased"


def method_order(names):
    fixed = sorted({n.removesuffix(BIASED) for n in names if fixed_m(n.removesuffix(BIASED)) is not None},
                   key=lambda n: fixed_m(n))
    base = ["mc_cN", *fixed, "gvoys", "cg", "fixed_point", "sylvester", "direct"]
    known = [n for b in base for n in (b, b+BIASED) if n in names]
    return known+sorted(set(names)-set(known))


def style(name, names):
    """Line/marker style; the biased-diagonal MCRWK twin is hollow and dotted."""
    if name.endswith(BIASED):
        base = style(name.removesuffix(BIASED), [n.removesuffix(BIASED) for n in names])
        return {**base, "label": base["label"]+", biased diag", "linestyle": ":", "hollow": True}
    if name in STYLE:
        color, marker, label = STYLE[name]
        return dict(color=color, marker=marker, label=label, linestyle="-")
    fixed = sorted({fixed_m(n.removesuffix(BIASED)) for n in names if fixed_m(n.removesuffix(BIASED)) is not None})
    if name.startswith("mc_m="):
        m = fixed_m(name)
        step = np.linspace(0, len(MC_RAMP)-1, len(fixed)).round().astype(int)[fixed.index(m)] if len(fixed) > 1 else 2
        return dict(color=MC_RAMP[step], marker="o", label=f"MCRWK (m = {m:,})", linestyle="--")
    return dict(color=MUTED, marker="x", label=name, linestyle=":")


# ----------------------------------------------------------------------
#  Record access and aggregation
# ----------------------------------------------------------------------
def load(paths):
    """Merge records (and q/n aggregates and references) per experiment."""
    by_experiment = defaultdict(lambda: defaultdict(list))
    for path in paths:
        payload = json.loads(Path(path).read_text())
        for key in ("records", "aggregates", "references", "summaries"):
            by_experiment[payload["experiment"]][key] += payload.get(key, [])
    return by_experiment


def unbiased_only(value):
    """Both MCRWK diagonals share walks and time; plot the biased twin only for its own metrics."""
    return lambda record: None if record.get("name", "").endswith(BIASED) else value(record)


def runtime(record):
    return record.get("time_sec") if record.get("status") == "ok" else None


def error(metric):
    return lambda record: (record.get("errors") or {}).get(metric)


def evaluation(key, field="evaluation"):
    """Kernel-SVM result by default; field="evaluation_linear" for the linear SVM on features."""
    return lambda record: (record.get(field) or {}).get(key) if record.get("status") == "ok" else None


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
    """Log scale in plain notation: 1-2-5 steps over short ranges, decades otherwise."""
    set_scale("log")
    low, high = axis.get_view_interval()
    axis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5) if high <= 30*low else (1,)))
    axis.set_major_formatter(ticker.FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_minor_formatter(ticker.NullFormatter())


def finish(ax, xlabel, ylabel, xlog=False, ylog=False):
    """xlog=True: base-2 axis (graph sizes); xlog=10: decades labelled 1-2-5 (budgets)."""
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if xlog == 10:
        log_axis(ax.xaxis, ax.set_xscale)
    elif xlog:
        ax.set_xscale("log", base=2)
    if ylog:
        log_axis(ax.yaxis, ax.set_yscale)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------
#  Line figures: metric versus graph size or lambda
# ----------------------------------------------------------------------
def line_figure(records, x_key, xlabel, panels, title, path, *, xlog, style_fn=style, order_fn=method_order):
    """panels: (ylabel, value function, log scale, include exact methods)."""
    names = order_fn({r["name"] for r in records if r.get("name")})
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
            st = style_fn(name, names)
            ax.errorbar(x, mean, yerr=std, color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                        label=st["label"], linewidth=2, markersize=6, capsize=2, elinewidth=1,
                        markeredgewidth=1, zorder=3+len(shown)-rank, **marker_colors(st))
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
                    [("runtime, s", unbiased_only(runtime), True, True),
                     ("relative error", unbiased_only(error(metric)), True, False)],
                    f"Scaling, {case}", out/f"scaling_{case}.png", xlog=True)


def plot_lambda(records, out, metric):
    synthetic = [r for r in records if r.get("setting") == "synthetic"]
    for case in cases(synthetic):
        rows = [r for r in synthetic if r.get("case") == case]
        line_figure(rows, "lmbd", "λ", [("runtime, s", unbiased_only(runtime), True, True),
                    ("relative error", unbiased_only(error(metric)), True, False)],
                    f"λ sweep, synthetic, {case}", out/f"lambda_synthetic_{case}.png", xlog=False)
    tu = [r for r in records if r.get("setting") == "tu" and r.get("name")]
    for dataset in sorted({r["dataset"] for r in tu}):
        for case in cases(tu):
            rows = [r for r in tu if r["dataset"] == dataset and r["case"] == case]
            if not rows:
                continue
            key, label = ("mean_accuracy", "accuracy") if rows[0]["task"] == "classification" else ("mean_rmse", "RMSE")
            quality = [(f"{label}, SVC", evaluation(key), False, True)]
            if any(r.get("evaluation_linear") for r in rows):
                quality.append((f"{label}, linear SVM", evaluation(key, "evaluation_linear"), False, True))
            line_figure(rows, "lmbd", "λ", [("Gram time, s", unbiased_only(runtime), True, True), *quality,
                        ("relative error", unbiased_only(error(metric)), True, False)],
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
                        markersize=7, capsize=2, elinewidth=1, markeredgewidth=1, **marker_colors(st))
        ax.set_title(dataset, fontsize=9)
        ax.set_yticks(range(len(names)), [style(n, names)["label"] for n in names])
        ax.set_ylim(len(names)-0.5, -0.5)
        finish(ax, xlabel, "", xlog=False, ylog=False)
        if xlog:
            log_axis(ax.xaxis, ax.set_xscale)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    save(fig, path, [(f"{title}: {xlabel}", names, datasets, cells)])


def plot_tu_svm(records, out, metric, prefix="tu", models=("SVC on Gram", "linear SVM on features"),
                gram_time=True):
    """Kernel (every method) and linear (feature methods) results as separate figures."""
    rows = [r for r in records if r.get("name")]
    for case in cases(rows):
        in_case = [r for r in rows if r["case"] == case]
        for task, key, spread, label in [("classification", "mean_accuracy", "std_accuracy", "accuracy"),
                                         ("regression", "mean_rmse", "std_rmse", "RMSE")]:
            subset = [r for r in in_case if r["task"] == task]
            datasets = sorted({r["dataset"] for r in subset})
            # One model on every Gram (compare methods) and, separately, the same
            # family on the features of GVoys and the biased MCRWK diagonal.
            for field, model, suffix in (("evaluation", models[0], ""), ("evaluation_linear", models[1], "_linear")):
                if any(r.get(field) for r in subset):
                    dot_figure(subset, datasets, evaluation(key, field), f"{label} (± fold std)",
                               f"{task}, {model}, {case}", out/f"{prefix}_{task}{suffix}_{case}.png",
                               xlog=False, xerr=evaluation(spread, field))
        if gram_time:
            datasets = sorted({r["dataset"] for r in in_case})
            dot_figure(in_case, datasets, unbiased_only(runtime), "Gram time, s", f"Gram time, {case}",
                       out/f"{prefix}_gram_time_{case}.png", xlog=True)


def feature_time(record):
    return record.get("feature_time_sec") if record.get("status") == "ok" else None


def plot_gram_time(records, out, metric):
    """Full Gram time of every method and, separately, GVoys/MCRWK feature time before the Gram."""
    rows = [r for r in records if r.get("name")]
    for case in cases(rows):
        in_case = [r for r in rows if r["case"] == case]
        datasets = sorted({r["dataset"] for r in in_case})
        dot_figure(in_case, datasets, unbiased_only(runtime), "Gram time, s", f"Gram time (full), {case}",
                   out/f"gram_time_{case}.png", xlog=True)
        dot_figure(in_case, datasets, unbiased_only(feature_time), "feature time, s",
                   f"Feature construction before the Gram (GVoys, MCRWK), {case}",
                   out/f"gram_time_features_{case}.png", xlog=True)


# ----------------------------------------------------------------------
#  Proposal q and label sequences per length n (aggregates over seeds)
# ----------------------------------------------------------------------
def proposal_label(row):
    if row["proposal"] == "gvoys":
        return "GVoys"
    return row["proposal"]+(f" (ε={row['mix_eps']:g})" if row.get("mix_eps") else "")


def ramp(values, value):
    """Ordinal blue step for value among sorted values (light = small)."""
    values = sorted(values)
    if len(values) == 1:
        return MC_RAMP[2]
    return MC_RAMP[int(round(values.index(value)*(len(MC_RAMP)-1)/(len(values)-1)))]


QUALITY_LABELS = {"": "SVC, unbiased diag / GVoys", "_biased": "SVC, biased diag",
                  "_linear": "linear SVM on features"}


def quality_panel(task, suffix=""):
    """Evaluation panel over seeds: "" and "_biased" are SVC on the Gram, "_linear" the linear SVM."""
    key, name = ("mean_accuracy", "accuracy") if task == "classification" else ("mean_rmse", "RMSE")
    label = f"{name}, {QUALITY_LABELS[suffix]} (± std over seeds)"
    return (f"{key}{suffix}_over_seeds", label, False, f"{key}{suffix}_std_over_seeds", key)


def with_linear(row):
    """One linear-SVM field per row: biased MCRWK features, or GVoys features."""
    out = dict(row)
    for key in ("mean_accuracy", "mean_rmse"):
        for stat in ("", "_std"):
            value = row.get(f"{key}_biased_linear{stat}_over_seeds", row.get(f"{key}_linear{stat}_over_seeds"))
            if value is not None:
                out[f"{key}_linear{stat}_over_seeds"] = value
    return out


def proposal_figure(rows, panels, title, path, reference=None):
    """Proposals on the y axis, one dot per budget m (blue ramp), GVoys as its own row.

    panels: (key, label, log scale, std key or None, reference evaluation key or None).
    """
    labels = list(dict.fromkeys(proposal_label(r) for r in rows))
    ms = sorted({r["m"] for r in rows if r.get("m")})
    series = [f"m = {m:,}" for m in ms]+(["GVoys"] if "GVoys" in labels else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1*len(panels)+1.4, 0.34*len(labels)+1.6),
                             sharey=True, squeeze=False)
    tables = []
    for ax, (key, xlabel, xlog, std_key, ref_key) in zip(axes[0], panels):
        cells = {}
        for row in rows:
            value = row.get(key)
            if value is None or (xlog and value <= 0):
                continue
            label, m = proposal_label(row), row.get("m")
            name = f"m = {m:,}" if m else "GVoys"
            offset = (ms.index(m)-(len(ms)-1)/2)*0.18 if m else 0.
            color, marker = (ramp(ms, m), "o") if m else (STYLE["gvoys"][0], "s")
            std = row.get(std_key) if std_key else None
            ax.errorbar([value], [labels.index(label)+offset], xerr=None if not std else [std], color=color,
                        marker=marker, linestyle="none", markersize=6, capsize=2, elinewidth=1,
                        markeredgecolor="white", markeredgewidth=1, label=name)
            cells[label, name] = (value, std or 0., 1)
        if ref_key and reference and ref_key in (reference.get("evaluation") or {}):
            ax.axvline(reference["evaluation"][ref_key], color=MUTED, linewidth=1, label="exact kernel")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_ylim(len(labels)-0.5, -0.5)
        finish(ax, xlabel, "")
        if xlog:
            log_axis(ax.xaxis, ax.set_xscale)
        tables.append((f"{title}: {xlabel}", labels, series, cells))
    handles = {}
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.legend(handles.values(), handles.keys(), loc="upper center", ncol=min(5, len(handles)),
               bbox_to_anchor=(0.5, 0.0))
    save(fig, path, tables)


def reference_for(references, dataset, lmbd):
    return next((r for r in references if r.get("dataset") == dataset and r.get("lmbd") == lmbd
                 and r.get("status") == "ok"), None)


def plot_q(data, out, metric):
    aggregates = data["aggregates"]
    for dataset, lmbd in sorted({(a["dataset"], a["lmbd"]) for a in aggregates}):
        rows = [a for a in aggregates if a["dataset"] == dataset and a["lmbd"] == lmbd]
        panels = [("rel_mse", "relative MSE over seeds", True, None, None),
                  ("time_mean", "Gram time, s", True, "time_std", None),
                  ("mse_x_time", "relative MSE × time", True, None, None),
                  ("killed_fraction", "killed walks", False, None, None)]
        proposal_figure(rows, panels, f"Proposal q, {dataset}, λ = {lmbd}", out/f"q_{dataset}_lambda{lmbd}.png")
        task, rows = rows[0]["task"], [with_linear(r) for r in rows]
        svm = [panel for panel in (quality_panel(task), quality_panel(task, "_biased"), quality_panel(task, "_linear"),
                                   ("ridge_rel", "ridge of biased diag, τ / K_ii", False, None, None))
               if any(r.get(panel[0]) is not None for r in rows)]
        if svm:
            proposal_figure(rows, svm, f"Proposal q, SVM, {dataset}, λ = {lmbd}",
                            out/f"q_svm_{dataset}_lambda{lmbd}.png", reference_for(data["references"], dataset, lmbd))


def n_series(row):
    return f"B = {row['budget']:,} (fixed budget)" if row["design"] == "fixed_budget" \
        else f"m = {row['lengths']:,} (fixed lengths)"


def n_order(names):
    """Fixed-budget series by budget, then fixed-length series by m."""
    return sorted(names, key=lambda n: ("fixed lengths" in n, int(n.split()[2].replace(",", ""))))


def n_style(name, names):
    budgets = sorted(int(n.split()[2].replace(",", "")) for n in names if "fixed budget" in n)
    if "fixed budget" in name:
        return dict(color=ramp(budgets, int(name.split()[2].replace(",", ""))), marker="o",
                    label=name, linestyle="-")
    return dict(color=STYLE["mc_cN"][0], marker="s", label=name, linestyle="--")


def plot_n(data, out, metric):
    aggregates = [{**a, "name": n_series(a)} for a in data["aggregates"]]
    for dataset, lmbd in sorted({(a["dataset"], a["lmbd"]) for a in aggregates}):
        rows = [a for a in aggregates if a["dataset"] == dataset and a["lmbd"] == lmbd]
        rows = [with_linear(r) for r in rows]
        quality = [(label.split(" (")[0], lambda r, key=key: r.get(key), False, True)
                   for key, label, *_ in (quality_panel(rows[0]["task"], suffix) for suffix in QUALITY_LABELS)
                   if any(r.get(key) is not None for r in rows)]
        line_figure(rows, "n", "label sequences per length n",
                    [("pair variance", lambda r: r.get("var_offdiag"), True, True),
                     ("relative MSE", lambda r: r.get("rel_mse"), True, True),
                     ("Gram time, s", lambda r: r.get("time_mean"), True, True), *quality],
                    f"Label sequences per length, {dataset}, λ = {lmbd}",
                    out/f"n_{dataset}_lambda{lmbd}.png", xlog=True, style_fn=n_style, order_fn=n_order)


# ----------------------------------------------------------------------
#  Convergence and bounds (one series per graph size N)
# ----------------------------------------------------------------------
GUIDE = "∝ m^(−1/2)"


def size_of(name):
    return int(name.split(" = ")[1].split(",")[0])


def size_style(sizes):
    """Blue ramp by N; bounds dotted and hollow in the color of their N; the slope guide muted."""
    def style_fn(name, names):
        if name == GUIDE:
            return dict(color=MUTED, marker="", label=name, linestyle="-")
        base = dict(color=ramp(sizes, size_of(name)), marker="o", label=name, linestyle="-")
        return {**base, "linestyle": ":", "hollow": True} if name.endswith("bound") else base
    return style_fn


def size_order(names):
    return sorted(names, key=lambda n: (n == GUIDE, 0 if n == GUIDE else size_of(n), n.endswith("bound")))


def plot_convergence(data, out, metric):
    records = data["records"]
    for case, lmbd in sorted({(r["case"], r["lmbd"]) for r in records}, key=lambda k: (k[0] != "unlabeled", k[1])):
        rows = [r for r in records if r["case"] == case and r["lmbd"] == lmbd]
        sizes = sorted({r["n_nodes"] for r in rows})
        smallest = sorted((r for r in rows if r["n_nodes"] == sizes[0]), key=lambda r: r["m"])
        guide = [{"name": GUIDE, "m": r["m"], "rel_rmse": smallest[0]["rel_rmse"]*(r["m"]/smallest[0]["m"])**-0.5}
                 for r in smallest]
        series = [{**r, "name": f"N = {r['n_nodes']}"} for r in rows]
        style_fn = size_style(sizes)
        line_figure(series+guide, "m", "samples m",
                    [("relative RMSE", lambda r: r.get("rel_rmse"), True, True),
                     ("m · Var(k̂) / k² (median over pairs)", lambda r: r.get("m_rel_var_median"), True, True),
                     ("variance bound / measured (median)", lambda r: r.get("variance_bound_ratio_median"), True, True)],
                    f"Convergence, {case}, λ = {lmbd}", out/f"convergence_{case}_lambda{lmbd}.png",
                    xlog=10, style_fn=style_fn, order_fn=size_order)
        bound, bound_name = ("hoeffding", "Hoeffding (7)") if case == "unlabeled" else ("chebyshev", "Chebyshev (11)")
        tails = []
        for r in rows:
            for t in r["tails"]:
                tails.append({"name": f"N = {r['n_nodes']}", "m": r["m"], "eps": t["eps"], "value": t["empirical"]})
                if t.get(bound) is not None:
                    tails.append({"name": f"N = {r['n_nodes']}, bound", "m": r["m"], "eps": t["eps"],
                                  "value": t[bound]})
        epsilons = sorted({t["eps"] for t in tails})
        line_figure(tails, "m", "samples m",
                    [(f"P(|k̂ − k| > {eps:g}·k)", lambda r, e=eps: r["value"] if r["eps"] == e and r["value"] > 0
                      else None, True, True) for eps in epsilons],
                    f"Tail frequency against {bound_name}, {case}, λ = {lmbd}",
                    out/f"convergence_tails_{case}_lambda{lmbd}.png", xlog=10, style_fn=style_fn, order_fn=size_order)
    required_m_table(data["summaries"], out)


def required_m_table(summaries, out):
    """m needed for P(|k̂ − k| > eps k) <= delta: smallest grid m against the bound, as .md and .json."""
    rows = [{"case": s["case"], "lambda": s["lmbd"], "N": s["n_nodes"], "slope": s["slope"], **r,
             "bound_over_empirical": r["m_bound_median"]/r["m_empirical"]
             if r["m_bound_median"] is not None and r["m_empirical"] else None}
            for s in summaries for r in s["required_m"]]
    fmt = lambda v: "—" if v is None else f"{v:.3g}"
    lines = ["### m needed for P(|k̂ − k| > ε·k) ≤ δ: smallest grid m against the bound (median over pairs)", "",
             "| case | λ | N | slope | ε | δ | m, empirical | m, bound | bound / empirical | bound |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        empirical = r["m_empirical"] if r["m_empirical"] is not None else f"> {r['m_grid_max']}"
        lines.append(f"| {r['case']} | {r['lambda']:g} | {r['N']} | {fmt(r['slope'])} | {r['eps']:g} | {r['delta']:g} | "
                     f"{empirical} | {fmt(r['m_bound_median'])} | {fmt(r['bound_over_empirical'])} | {r['bound']} |")
    out.mkdir(parents=True, exist_ok=True)
    (out/"convergence_required_m.md").write_text("\n".join(lines)+"\n")
    (out/"convergence_required_m.json").write_text(json.dumps(rows, indent=2, allow_nan=False)+"\n")
    print(f"wrote {out/'convergence_required_m.md'} (+ .json)")


PLOTS = {"scaling": lambda d, o, m: plot_scaling(d["records"], o, m),
         "tu_svm": lambda d, o, m: plot_tu_svm(d["records"], o, m),
         "gram_time": lambda d, o, m: plot_gram_time(d["records"], o, m),
         "lambda_sweep": lambda d, o, m: plot_lambda(d["records"], o, m),
         "q_sampling": plot_q, "n_sampling": plot_n, "convergence": plot_convergence,
         "ridge": lambda d, o, m: plot_tu_svm(d["records"], o, m, prefix="ridge",
                                              models=("kernel ridge on Gram", "ridge on features"), gram_time=False)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", nargs="+", help="JSON files written by exps/*.py")
    parser.add_argument("--out-dir", default="fig/exps")
    parser.add_argument("--error-metric", default="offdiagonal_mean_rel",
                        help="Key of src.gram.matrix_errors, e.g. mean_rel or relative_frobenius.")
    args = parser.parse_args(argv)
    for experiment, data in load(args.results).items():
        PLOTS[experiment](data, Path(args.out_dir), args.error_metric)


if __name__ == "__main__":
    main()
