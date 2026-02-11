import argparse, os, json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from math import ceil
from pathlib import Path

def _load_json(path: str, verbose: bool = False):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        if verbose:
            print(f"[missing] {path}")
        return None

def _pick_results_json_file(dir_path: str, verbose: bool = False):
    if not os.path.exists(dir_path):
        if verbose:
            print(f"[missing dir] {dir_path}")
        return None
    files = [f for f in os.listdir(dir_path) if f.endswith(".json")]
    if not files:
        if verbose:
            print(f"[no json] {dir_path}")
        return None

    # Prefer aggregated file names if present
    prefer = [f for f in files if f.startswith("results_among_")]
    if prefer:
        fname = sorted(prefer)[-1]
    else:
        fname = sorted(files)[0] if len(files) == 1 else sorted(files)[-1]
        if verbose and len(files) > 1:
            print(f"[warn] multiple json files in {dir_path}, picking: {fname}")

    return os.path.join(dir_path, fname)

def get_result_json_B(dataset, encoder, model, root, attack: bool, verbose: bool = False):
    """Backward-compat path:
    {root}/{dataset}/{encoder}/{model}/{clean|B_eval}/{ts1}/{dataset}/{encoder}/{model}/{ts2}/results*.json
    """
    attack_ = "B_eval" if attack else "clean"
    log_dir = os.path.join(root, dataset, encoder, model, attack_)
    if not os.path.exists(log_dir):
        if verbose:
            print(f"[missing dir] {log_dir}")
        return None

    timestamps = sorted(os.listdir(log_dir))
    if not timestamps:
        if verbose:
            print(f"[no timestamps] {log_dir}")
        return None
    latest_timestamp = timestamps[-1]

    log_dir = os.path.join(log_dir, latest_timestamp, dataset, encoder, model)
    if not os.path.exists(log_dir):
        if verbose:
            print(f"[missing dir] {log_dir}")
        return None

    timestamps = sorted(os.listdir(log_dir))
    if not timestamps:
        if verbose:
            print(f"[no timestamps] {log_dir}")
        return None
    latest_timestamp = timestamps[-1]

    log_dir = os.path.join(log_dir, latest_timestamp)
    if not os.path.exists(log_dir):
        if verbose:
            print(f"[missing dir] {log_dir}")
        return None

    json_path = _pick_results_json_file(log_dir, verbose=verbose)
    if json_path is None:
        return None

    # Optional consistency check (seed dirs vs filename)
    try:
        files = os.listdir(log_dir)
        num_seeds = len([f for f in files if f.startswith("seed")])
        if num_seeds > 0 and f"{num_seeds}_seeds" not in os.path.basename(json_path):
            if verbose:
                print(f"[warn] seed count ({num_seeds}) not reflected in filename: {os.path.basename(json_path)}")
    except Exception:
        pass

    return _load_json(json_path, verbose=verbose)

def get_result_json(dataset, encoder, model, root, attack: bool = False, verbose: bool = False):
    """Common path:
    {root}/{dataset}/{encoder}/{model}/{timestamp}/results*.json

    If multiple jsons exist or the layout is different, falls back to get_result_json_B.
    """
    log_dir = os.path.join(root, dataset, encoder, model)
    if not os.path.exists(log_dir):
        if verbose:
            print(f"[missing dir] {log_dir}")
        return None

    timestamps = sorted(os.listdir(log_dir))
    if not timestamps:
        if verbose:
            print(f"[no timestamps] {log_dir}")
        return None
    latest_timestamp = timestamps[-1]

    log_dir = os.path.join(log_dir, latest_timestamp)
    if not os.path.exists(log_dir):
        if verbose:
            print(f"[missing dir] {log_dir}")
        return None

    # If there are multiple json files here, this might be a different layout
    json_files = [f for f in os.listdir(log_dir) if f.endswith(".json")]
    if len(json_files) != 1:
        return get_result_json_B(dataset, encoder, model, root, attack, verbose)

    json_path = os.path.join(log_dir, json_files[0])

    # Optional consistency check (seed dirs vs filename)
    try:
        files = os.listdir(log_dir)
        num_seeds = len([f for f in files if f.startswith("seed")])
        if num_seeds > 0 and f"{num_seeds}_seeds" not in os.path.basename(json_path):
            if verbose:
                print(f"[warn] seed count ({num_seeds}) not reflected in filename: {os.path.basename(json_path)}")
    except Exception:
        pass

    return _load_json(json_path, verbose=verbose)

def collect_rerun_results(datasets, encoders, models, path, attack_mode:str, verbose=False):
    lst_results = []
    for dataset in datasets:
        for encoder in encoders:
            for model in models:
                results = get_result_json(dataset, encoder, model, path, attack_mode != "no_attack", verbose)
                if results is None:
                    continue
                results['results']["dataset"] = dataset
                results['results']["encoder"] = encoder
                results['results']["model"] = model
                results['results']["attack_mode"] = attack_mode
                lst_results.append(results['results'])
    df_results = pd.DataFrame(lst_results).sort_values(by=["dataset", "encoder", "model"])
    return df_results

def concat_clean_and_attacked_results(df_clean:pd.DataFrame, df_attack:pd.DataFrame, attack_mode:str, include_ratio:bool=True):
    assert np.array_equal(
        df_clean[["dataset", "encoder"]].values,
        df_attack[["dataset", "encoder"]].values
    )
    numeric_cols = [col for col in df_clean.columns if df_clean[col].dtype in [np.float64, np.float32, np.int64, np.int32] and col not in ["dataset", "encoder", "model"]]
    if include_ratio:
        df_ratio = df_attack.copy()
        for col in numeric_cols:
            df_ratio[col] = df_attack[col] / df_clean[col]
        df_ratio[["dataset", "encoder", "model"]] = df_attack[["dataset", "encoder", "model"]]
        df_ratio["attack_mode"] = "ratio"
        df_concat = pd.concat([df_clean, df_attack, df_ratio], ignore_index=True)
        df_concat["attack_mode"] = pd.Categorical(df_concat["attack_mode"], categories=["none", attack_mode, "ratio"], ordered=True)
    else:
        df_concat = pd.concat([df_clean, df_attack], ignore_index=True)
        df_concat["attack_mode"] = pd.Categorical(df_concat["attack_mode"], categories=["none", attack_mode], ordered=True)
    df_concat = df_concat.sort_values(by=["dataset", "encoder", "model", "attack_mode"])
    df_concat = df_concat[["dataset", "encoder", "model", "attack_mode"] + numeric_cols]
    return df_concat

def collect_results_among_seeds(
    exp_dir: str,
    pattern: str = "results_among_*_seeds.json",
    verbose: bool = False,
):
    """Collect aggregated JSON results (e.g., results_among_10_seeds.json) under an experiment directory.

    Expected layout (most common):
        exp_dir/<method>/<dataset>/<encoder>/<model>/<timestamp>/results_among_*_seeds.json

    This function is robust to exp_dir depth by reading metadata from the last 5 folders above the JSON file.
    It also tolerates JSONs that either:
        - have a top-level key "results" (preferred), or
        - store metrics directly at the top level.
    """
    exp_dir = Path(exp_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"exp_dir not found: {exp_dir.resolve()}")
    rows = []

    for p in exp_dir.rglob(pattern):
        try:
            rel = p.relative_to(exp_dir)
        except Exception:
            rel = None

        # metadata from directory names (closest parents)
        timestamp = p.parent.name
        model = p.parent.parent.name
        encoder = p.parent.parent.parent.name
        dataset = p.parent.parent.parent.parent.name
        method = rel.parts[0] if (rel is not None and len(rel.parts) > 0) else p.parent.parent.parent.parent.parent.name

        obj = _load_json(str(p), verbose=verbose)
        if obj is None:
            continue
        res = obj.get("results", obj)
        if not isinstance(res, dict):
            if verbose:
                print(f"[skip] unexpected json format: {p}")
            continue

        row = dict(res)
        row["method"] = method
        row["dataset"] = dataset
        row["encoder"] = encoder
        row["model"] = model
        row["timestamp"] = timestamp

        # parse seed count from filename if present
        m = re.search(r"results_among_(\d+)_seeds", p.name)
        if m:
            row["n_seeds"] = int(m.group(1))

        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        if verbose:
            print(f"[empty] No files matched {pattern} under {exp_dir}")
        return df

    # keep a nice order if columns exist
    front = [c for c in ["method", "dataset", "encoder", "model", "timestamp", "n_seeds"] if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest].sort_values(by=[c for c in front if c in df.columns])
    return df

# ------------- VISUALIZATION UTILITIES -------------- #

# Helper to plot a pivot as a heatmap
def heatmap_from_pivot(
    pivot,
    title: str,
    xlabel="encoder",
    ylabel="dataset",
    vcenter=0.0,                 # center (0 for delta)
    symmetric=True,              # use symmetric range around vcenter
    neg_color="#2166AC",         # negative side color
    zero_color="#F7F7F7",        # center color
    pos_color="#B2182B",         # positive side color
    cbar_label=None,
):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    data = pivot.values.astype(float)

    # --- diverging colormap with two obvious colors + neutral center ---
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "neg_pos", [neg_color, zero_color, pos_color]
    )

    # --- normalization: map vcenter (0) to the neutral color ---
    if symmetric:
        absmax = np.nanmax(np.abs(data - vcenter))
        vmin, vmax = vcenter - absmax, vcenter + absmax
    else:
        vmin, vmax = np.nanmin(data), np.nanmax(data)

    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    fig = plt.figure(figsize=(5.5, 3.8), dpi=150)
    ax = plt.gca()

    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist())
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist())

    # annotate values
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=7)

    # colorbar tuning
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label is not None:
        cbar.set_label(cbar_label)
    cbar.set_ticks([vmin, vcenter, vmax])  # show neg / zero / pos anchors

    plt.tight_layout()
    plt.show()

def plot_metric_heatmap(pt: pd.DataFrame, title: str = "", annotate: bool = True):
    """
    pt: DataFrame shaped (rows = MultiIndex [dataset, method], cols = encoders),
        values are already in the display unit (e.g., percent).
    """
    # Ensure consistent ordering (optional but usually helpful)
    if isinstance(pt.index, pd.MultiIndex) and pt.index.nlevels >= 2:
        pt = pt.sort_index(level=list(range(pt.index.nlevels)))

    data = pt.to_numpy(dtype=float)
    nrows, ncols = data.shape

    fig_h = max(4, 0.45 * nrows)   # auto-ish sizing
    fig_w = max(6, 1.0 * ncols)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(data, aspect="auto")  # uses matplotlib default colormap
    cbar = fig.colorbar(im, ax=ax)
    if title:
        ax.set_title(title)

    # X ticks (encoders)
    ax.set_xticks(np.arange(ncols))
    ax.set_xticklabels([str(c).upper() for c in pt.columns], rotation=0)

    # Y tick labels: show dataset once per block
    if isinstance(pt.index, pd.MultiIndex) and pt.index.nlevels >= 2:
        labels = []
        prev_ds = None
        for ds, method in pt.index:
            if ds != prev_ds:
                labels.append(f"{ds} | {method}")
                prev_ds = ds
            else:
                labels.append(f"{'':{len(str(prev_ds))}} | {method}")
        ax.set_yticks(np.arange(nrows))
        ax.set_yticklabels(labels)
    else:
        ax.set_yticks(np.arange(nrows))
        ax.set_yticklabels([str(i) for i in pt.index])

    # Thin gridlines between cells (no manual colors)
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Thick separators between datasets
    if isinstance(pt.index, pd.MultiIndex) and pt.index.nlevels >= 2:
        ds_level = pt.index.get_level_values(0).to_numpy()
        breaks = np.where(ds_level[:-1] != ds_level[1:])[0] + 1
        for b in breaks:
            ax.axhline(b - 0.5, linewidth=2)

    # Cell annotations
    if annotate:
        # Choose text color based on background intensity
        norm = im.norm
        for i in range(nrows):
            for j in range(ncols):
                v = pt.iat[i, j]
                if pd.isna(v):
                    continue
                txt_color = "w" if norm(v) > 0.6 else "k"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color=txt_color)

    ax.set_xlabel("Encoder")
    ax.set_ylabel("Dataset | Method")
    plt.tight_layout()
    return fig, ax

def delta_vs_vanilla(pt_percent: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    pt_percent: index=[dataset, method], columns=[encoder], values in %
    returns: index=[dataset], columns=[encoder], values = (method - vanilla)
    """
    base = pt_percent.xs("vanilla", level="method")
    tgt  = pt_percent.xs(method, level="method")
    return (tgt - base).sort_index()

def plot_delta_heatmaps(pt_percent: pd.DataFrame, methods=("edge_adder", "vanilla_advtrain"), metric_name="DP_mean (%)"):
    deltas = [delta_vs_vanilla(pt_percent, m) for m in methods]

    # shared symmetric range around 0
    all_vals = np.concatenate([d.to_numpy().ravel() for d in deltas])
    vmax = np.nanmax(np.abs(all_vals))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    fig_h = max(4, 0.45 * deltas[0].shape[0])
    fig_w = max(10, 1.2 * deltas[0].shape[1] * len(methods))
    fig, axes = plt.subplots(1, len(methods), figsize=(fig_w, fig_h), sharey=True)

    if len(methods) == 1:
        axes = [axes]

    for ax, d, m in zip(axes, deltas, methods):
        im = ax.imshow(d.to_numpy(dtype=float), aspect="auto", norm=norm)  # default colormap
        ax.set_title(f"Δ {metric_name}\n({m} − vanilla)")

        # ticks
        ax.set_xticks(np.arange(d.shape[1]))
        ax.set_xticklabels([str(c).upper() for c in d.columns], rotation=0)
        ax.set_yticks(np.arange(d.shape[0]))
        ax.set_yticklabels([str(i) for i in d.index])

        # gridlines
        ax.set_xticks(np.arange(-0.5, d.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, d.shape[0], 1), minor=True)
        ax.grid(which="minor", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        # annotate cells
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                v = d.iat[i, j]
                if pd.isna(v):
                    continue
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9)

        ax.set_xlabel("Encoder")

    axes[0].set_ylabel("Dataset")

    # one shared colorbar
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label(f"Δ {metric_name}")

    plt.tight_layout()
    return fig, axes

def plot_dumbbell_vs_vanilla(
    pt_percent: pd.DataFrame,                 # index=[dataset, method], cols=[encoder], values in %
    target_method: str = "edge_adder",
    baseline_method: str = "vanilla",
    metric_label: str = "DP_mean (%)",
    ncols: int = 3,
    annotate_delta: bool = True,
    sort_encoders_by_delta: bool = False,
):
    assert isinstance(pt_percent.index, pd.MultiIndex), "pt_percent must have MultiIndex index=[dataset, method]"
    assert "method" in pt_percent.index.names and "dataset" in pt_percent.index.names, \
        f"Expect index names include 'dataset' and 'method', got {pt_percent.index.names}"

    datasets = list(pt_percent.index.get_level_values("dataset").unique())
    encoders = list(pt_percent.columns)

    # Optional: determine a consistent x-range across panels
    base = pt_percent.xs(baseline_method, level="method")
    tgt  = pt_percent.xs(target_method, level="method")
    all_vals = np.concatenate([base.to_numpy().ravel(), tgt.to_numpy().ravel()])
    xmin, xmax = np.nanmin(all_vals), np.nanmax(all_vals)
    pad = 0.05 * (xmax - xmin + 1e-9)
    xlim = (xmin - pad, xmax + pad)

    nrows = ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, max(3.5, 0.9 * nrows * len(encoders) / 3)), sharex=True)
    axes = np.array(axes).reshape(-1)

    # Legend proxies (no manual colors)
    proxy_v = plt.Line2D([0], [0], marker="o", linestyle="None", label=baseline_method)
    proxy_t = plt.Line2D([0], [0], marker="D", linestyle="None", label=target_method)

    for k, ds in enumerate(datasets):
        ax = axes[k]

        x0 = pt_percent.loc[(ds, baseline_method), encoders].to_numpy(dtype=float)
        x1 = pt_percent.loc[(ds, target_method),  encoders].to_numpy(dtype=float)

        # Optionally sort encoders by delta magnitude within each dataset
        order = np.arange(len(encoders))
        if sort_encoders_by_delta:
            order = np.argsort((x1 - x0))  # from most negative to most positive
        enc_ord = [encoders[i] for i in order]
        x0, x1 = x0[order], x1[order]

        y = np.arange(len(enc_ord))

        # Draw one dumbbell per encoder
        for i in range(len(enc_ord)):
            # line
            (ln,) = ax.plot([x0[i], x1[i]], [y[i], y[i]], linewidth=2)
            c = ln.get_color()  # reuse the same auto color for both endpoints

            # endpoints
            ax.scatter([x0[i]], [y[i]], marker="o", s=50, color=c, zorder=3)
            ax.scatter([x1[i]], [y[i]], marker="D", s=55, color=c, zorder=3)

            if annotate_delta and np.isfinite(x0[i]) and np.isfinite(x1[i]):
                dx = x1[i] - x0[i]
                ax.text(max(x0[i], x1[i]) + 0.01*(xlim[1]-xlim[0]), y[i], f"{dx:+.2f}",
                        va="center", fontsize=9)

        ax.set_title(str(ds))
        ax.set_yticks(y)
        ax.set_yticklabels([e.upper() for e in enc_ord])
        ax.set_xlim(*xlim)
        ax.grid(True, axis="x", linewidth=0.6)

        if k % ncols == 0:
            ax.set_ylabel("Encoder")
        ax.set_xlabel(metric_label)

    # Hide unused axes
    for j in range(len(datasets), len(axes)):
        axes[j].axis("off")

    fig.legend(handles=[proxy_v, proxy_t], loc="upper right", frameon=True)
    plt.tight_layout()
    return fig, axes


def plot_grouped_bars_per_dataset(
    pt_percent: pd.DataFrame,                      # index=[dataset, method], cols=[encoder]
    methods=("vanilla", "vanilla_advtrain", "edge_adder"),
    metric_label="DP_mean (%)",
    ncols=3,
    rotate_xticks=0,
    value_fmt="{:.2f}",            # <-- added
    label_padding=2,               # <-- added (points)
):
    assert isinstance(pt_percent.index, pd.MultiIndex), "pt_percent must have MultiIndex index=[dataset, method]"
    assert "dataset" in pt_percent.index.names and "method" in pt_percent.index.names, \
        f"Expect index names include 'dataset' and 'method', got {pt_percent.index.names}"

    datasets = list(pt_percent.index.get_level_values("dataset").unique())
    encoders = list(pt_percent.columns)

    # Shared y-limits across all panels
    vals = []
    for ds in datasets:
        for m in methods:
            if (ds, m) in pt_percent.index:
                vals.append(pt_percent.loc[(ds, m), encoders].to_numpy(dtype=float))
    all_vals = np.concatenate([v.ravel() for v in vals]) if vals else np.array([0.0])
    ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
    pad = 0.08 * (ymax - ymin + 1e-9)
    ylim = (ymin - pad, ymax + pad)

    nrows = ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), sharey=True)
    axes = np.array(axes).reshape(-1)

    x = np.arange(len(encoders))
    k = len(methods)
    width = min(0.22, 0.8 / k)  # keep bars readable

    for i, ds in enumerate(datasets):
        ax = axes[i]

        for j, m in enumerate(methods):
            if (ds, m) not in pt_percent.index:
                continue
            y = pt_percent.loc[(ds, m), encoders].to_numpy(dtype=float)

            bars = ax.bar(x + (j - (k - 1) / 2) * width, y, width=width, label=m)

            # ---- add value labels on top of each bar ----
            labels = [("" if np.isnan(v) else value_fmt.format(v)) for v in y]
            ax.bar_label(bars, labels=labels, padding=label_padding, fontsize=9)
            # --------------------------------------------

        ax.set_title(str(ds))
        ax.set_xticks(x)
        ax.set_xticklabels([e.upper() for e in encoders], rotation=rotate_xticks)
        ax.set_ylim(*ylim)
        ax.grid(True, axis="y", linewidth=0.6)
        ax.set_xlabel("Encoder")

        if i % ncols == 0:
            ax.set_ylabel(metric_label)

        # Keep legend only on first subplot (less clutter)
        if i == 0:
            ax.legend(frameon=True)

    # Hide unused axes
    for j in range(len(datasets), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig, axes


def plot_table_with_bars(
    pt_percent: pd.DataFrame,                # index=[dataset, method], cols=[encoder], values in %
    title: str = "",
    scale: str = "col",                      # "col" or "global"
    fmt: str = "{:.2f}",
):
    pt = pt_percent.copy()
    data = pt.to_numpy(dtype=float)
    nrows, ncols = data.shape

    # normalization
    if scale == "col":
        vmax = np.nanmax(data, axis=0)
        vmax[vmax == 0] = 1.0
        norm = data / vmax
    elif scale == "global":
        vmax = np.nanmax(data)
        vmax = 1.0 if (not np.isfinite(vmax) or vmax == 0) else vmax
        norm = data / vmax
    else:
        raise ValueError("scale must be 'col' or 'global'")

    # figure sizing
    fig_h = max(4, 0.42 * nrows)
    fig_w = max(7, 1.15 * ncols)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.invert_yaxis()

    # draw grid + bars + text
    for i in range(nrows):
        for j in range(ncols):
            v = pt.iat[i, j]
            # cell border
            ax.add_patch(Rectangle((j, i), 1, 1, fill=False, linewidth=0.6))
            if pd.isna(v):
                continue

            # bar inside cell (use default color, only set alpha)
            w = float(np.clip(norm[i, j], 0, 1)) * 0.92
            ax.add_patch(Rectangle((j + 0.04, i + 0.12), w, 0.76, alpha=0.35))

            # value text
            ax.text(j + 0.5, i + 0.5, fmt.format(v), ha="center", va="center", fontsize=9)

    # column labels (encoders)
    ax.set_xticks(np.arange(ncols) + 0.5)
    ax.set_xticklabels([str(c).upper() for c in pt.columns])

    # row labels (dataset | method)
    if isinstance(pt.index, pd.MultiIndex) and pt.index.nlevels >= 2:
        ds = pt.index.get_level_values(0).to_numpy()
        md = pt.index.get_level_values(1).to_numpy()

        # Fixed dataset-column width + extra gap so dataset and method don't feel cramped
        w_ds = max(len(str(x)) for x in ds)
        label_gap = 7                      # <-- increase if you want more separation
        sep = (" " * label_gap) # + "| "

        ylabels, prev = [], None
        for d, m in zip(ds, md):
            m = "advtrain" if m == "vanilla_advtrain" else m  # shorten method name for better fit
            if d != prev:
                ylabels.append(f"{str(d):<{w_ds}}{sep}{m}")
                prev = d
            else:
                ylabels.append(f"{'':<{w_ds}}{sep}{m}")

        ax.set_yticks(np.arange(nrows) + 0.5)
        ax.set_yticklabels(ylabels, fontfamily="monospace")  # monospace keeps spacing consistent

        # thick separators between datasets
        breaks = np.where(ds[:-1] != ds[1:])[0] + 1
        for b in breaks:
            ax.hlines(b, 0, ncols, linewidth=2)

        # Optional: push labels a bit away from the table
        ax.tick_params(axis="y", pad=8)
    else:
        ax.set_yticks(np.arange(nrows) + 0.5)
        ax.set_yticklabels([str(x) for x in pt.index])

    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return fig, ax


def _ensure_index_names(pt: pd.DataFrame):
    # Expect MultiIndex with 2 levels: dataset, method
    if not isinstance(pt.index, pd.MultiIndex) or pt.index.nlevels < 2:
        raise ValueError("pt must have a MultiIndex with levels [dataset, method].")
    names = list(pt.index.names)
    if names[0] is None: names[0] = "dataset"
    if names[1] is None: names[1] = "method"
    pt = pt.copy()
    pt.index = pt.index.set_names(names[:2])
    return pt

def win_counts(pt_percent: pd.DataFrame, better="min"):
    """
    pt_percent: index=[dataset, method], cols=[encoder], values
    better: "min" (lower is better) or "max" (higher is better)
    returns: DataFrame index=[dataset], cols=[method], values = #wins across encoders
    """
    pt = _ensure_index_names(pt_percent)
    datasets = pt.index.get_level_values("dataset").unique()
    methods  = pt.index.get_level_values("method").unique()
    encoders = pt.columns

    out = []
    for ds in datasets:
        block = pt.xs(ds, level="dataset")  # rows=method, cols=encoder
        # winner per encoder
        if better == "min":
            winners = block.idxmin(axis=0)
        elif better == "max":
            winners = block.idxmax(axis=0)
        else:
            raise ValueError("better must be 'min' or 'max'")
        counts = winners.value_counts().reindex(methods, fill_value=0)
        out.append(counts.rename(ds))

    return pd.DataFrame(out).astype(int)

def plot_win_counts(counts: pd.DataFrame, title="Wins per dataset (#encoders where method is best)"):
    # stacked bar
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(counts)), 4))
    bottom = np.zeros(len(counts))
    x = np.arange(len(counts))

    for m in counts.columns:
        ax.bar(x, counts[m].to_numpy(), bottom=bottom, label=str(m))
        bottom += counts[m].to_numpy()

    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in counts.index], rotation=0)
    ax.set_ylabel("#Wins")
    ax.set_title(title)
    ax.legend(frameon=True)
    ax.grid(True, axis="y", linewidth=0.6)
    plt.tight_layout()
    return fig, ax

from math import ceil

def ranks_per_dataset(pt_percent: pd.DataFrame, better="min"):
    pt = _ensure_index_names(pt_percent)
    datasets = pt.index.get_level_values("dataset").unique()
    out = {}
    for ds in datasets:
        block = pt.xs(ds, level="dataset")  # rows=method, cols=encoder
        # rank per column (encoder)
        ascending = (better == "min")
        r = block.rank(axis=0, method="min", ascending=ascending)  # 1 best
        out[ds] = r
    return out

def plot_bump_ranks(ranks_dict, ncols=3, title_prefix="Ranks across encoders (1 = best)"):
    datasets = list(ranks_dict.keys())
    nrows = ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2*ncols, 3.8*nrows), sharey=True)
    axes = np.array(axes).reshape(-1)

    for i, ds in enumerate(datasets):
        ax = axes[i]
        r = ranks_dict[ds]  # rows=method, cols=encoder
        x = np.arange(len(r.columns))

        for m in r.index:
            ax.plot(x, r.loc[m].to_numpy(), marker="o", label=str(m))

        ax.set_title(f"{ds}")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c).upper() for c in r.columns])
        ax.set_ylim(0.5, r.shape[0] + 0.5)
        ax.invert_yaxis()  # rank 1 at top
        ax.grid(True, axis="y", linewidth=0.6)
        if i % ncols == 0:
            ax.set_ylabel("Rank (1 best)")
        ax.set_xlabel("Encoder")

        if i == 0:
            ax.legend(frameon=True)

    for j in range(len(datasets), len(axes)):
        axes[j].axis("off")

    fig.suptitle(title_prefix, y=1.02)
    plt.tight_layout()
    return fig, axes

def pivot_to_long(pt: pd.DataFrame, value_name: str) -> pd.DataFrame:
    pt = pt.copy()
    if not isinstance(pt.index, pd.MultiIndex) or pt.index.nlevels < 2:
        raise ValueError("Expect MultiIndex index=[dataset, method].")
    pt.index = pt.index.set_names(["dataset", "method"][:pt.index.nlevels])
    df = pt.stack().rename(value_name).reset_index()
    df = df.rename(columns={df.columns[2]: "encoder"})  # stacked level name
    return df

def plot_fairness_utility_tradeoff(
    pt_fair: pd.DataFrame,
    pt_util: pd.DataFrame,
    fairness_label="DP_mean (%)",
    utility_label="F1 (%)",
    better_fairness="min",   # "min" for DP/EO; "max" otherwise
    ncols=3
):
    df = pivot_to_long(pt_fair, "fairness").merge(
        pivot_to_long(pt_util, "utility"),
        on=["dataset", "method", "encoder"],
        how="inner"
    ).dropna()

    # Make "higher is better" on y for fairness if lower is better (DP/EO)
    if better_fairness == "min":
        df["y"] = -df["fairness"]
        ylab = f"-{fairness_label} (higher is better)"
    else:
        df["y"] = df["fairness"]
        ylab = fairness_label

    datasets = list(df["dataset"].unique())
    methods  = list(df["method"].unique())
    encoders = list(df["encoder"].unique())

    # ✅ Fixed encoder→marker mapping (matches your legend expectation)
    fixed_marker_map = {
        "gat": "o", "gcn": "s", "gin": "D", "sage": "^", "sgc": "v",
        "GAT": "o", "GCN": "s", "GIN": "D", "SAGE": "^", "SGC": "v",
    }
    marker_pool = ["o", "s", "D", "^", "v", "P", "X", "<", ">", "*", "h"]
    marker_map = {}
    fallback_i = 0
    for e in encoders:
        if e in fixed_marker_map:
            marker_map[e] = fixed_marker_map[e]
        else:
            marker_map[e] = marker_pool[fallback_i % len(marker_pool)]
            fallback_i += 1

    # ✅ Consistent method→color mapping using Matplotlib’s DEFAULT cycle
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    method_color = {m: default_cycle[i % len(default_cycle)] for i, m in enumerate(methods)} if default_cycle else {}

    nrows = ceil(len(datasets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2*ncols, 4.2*nrows))
    axes = np.array(axes).reshape(-1)

    for i, ds in enumerate(datasets):
        ax = axes[i]
        sub = df[df["dataset"] == ds]

        for m in methods:
            sm = sub[sub["method"] == m]
            if sm.empty:
                continue
            for e in encoders:
                sme = sm[sm["encoder"] == e]
                if sme.empty:
                    continue
                ax.scatter(
                    sme["utility"].to_numpy(),
                    sme["y"].to_numpy(),
                    marker=marker_map[e],
                    color=method_color.get(m, None),
                    alpha=0.85,
                )

        ax.set_title(str(ds))
        ax.set_xlabel(utility_label)
        ax.set_ylabel(ylab)
        ax.grid(True, linewidth=0.6)

    for j in range(len(datasets), len(axes)):
        axes[j].axis("off")

    # Legends (built explicitly so they’re correct)
    method_handles = [
        Line2D([0],[0], marker="o", linestyle="None",
               color=method_color.get(m, None), label=m)
        for m in methods
    ]
    encoder_handles = [
        Line2D([0],[0], marker=marker_map[e], linestyle="None",
               color="black", label=str(e).upper())
        for e in encoders
    ]

    fig.legend(handles=encoder_handles, loc="upper left", frameon=True, title="Encoder")
    fig.legend(handles=method_handles,  loc="upper right", frameon=True, title="Method")

    plt.tight_layout()
    return fig, axes, df


def plot_utility_fairness(
    df: pd.DataFrame,
    dataset: str | None = None,
    backbone: str | None = None,
    lambda_param: str = "lambda_dp",            # or "lambda_eo"
    baseline_model: str = "vanilla",
    colors: dict | None = None,
    percent: bool = False,                      # format y-axes as percentages if metrics in [0,1]
    markers: bool = True,
    figsize=(7, 4.5),
):
    """
    Twin-y plot: left axis = AUROC/F1, right axis = DP/EO, with vanilla baselines at λ=0.

    Expected metric columns: 'AUC_mean', 'F1_mean', 'DP_mean', 'EO_mean'.
    Expected model column: 'model'. Lambda columns: 'lambda_dp'/'lambda_eo'.
    """
    # c1 = df["dataset"] == dataset
    # c2 = df["backbone"] == backbone
    # df_ = df[c1 & c2]
    # Make index columns accessible if needed
    df_ = df.reset_index()

    # Column detection
    model_col = "model" if "model" in df_.columns else None
    if model_col is None:
        raise ValueError("Column 'model' is required.")

    ds_col = next((c for c in ("dataset", "data") if c in df_.columns), None)
    bb_col = next((c for c in ("backbone", "bone", "encoder") if c in df_.columns), None)

    # Subset by dataset/backbone if provided
    mask = pd.Series(True, index=df_.index)
    if dataset is not None:
        if ds_col is None:
            raise ValueError("No dataset column found among ['dataset','data'], but 'dataset' arg was given.")
        mask &= df_[ds_col].astype(str).str.lower() == str(dataset).lower()
    if backbone is not None:
        if bb_col is None:
            raise ValueError("No backbone column found among ['backbone','bone','encoder'], but 'backbone' arg was given.")
        mask &= df_[bb_col].astype(str).str.lower() == str(backbone).lower()
    df_sub = df_.loc[mask].copy()
    if df_sub.empty:
        raise ValueError("No rows found after filtering by dataset/backbone.")

    # Validate lambda column
    if lambda_param not in df_sub.columns:
        raise ValueError(f"'{lambda_param}' not found in DataFrame columns.")

    # Regularized runs (λ > 0)
    df_reg = df_sub.loc[df_sub[lambda_param] > 0].sort_values(lambda_param)

    # Baseline row: model == vanilla
    base = df_sub.loc[df_sub[model_col].astype(str).str.lower() == baseline_model.lower()]
    if base.empty:
        # fallback: try λ == 0 regardless of model
        base = df_sub.loc[df_sub[lambda_param] == 0]
    if base.empty:
        raise ValueError("No baseline row found (model=='vanilla' or lambda==0).")
    base = base.iloc[0]

    # Colors
    if colors is None:
        colors = {
            "AUROC": "#1f77b4",  # blue
            "F1":    "#ff7f0e",  # orange
            "DP":    "#2ca02c",  # green
            "EO":    "#d62728",  # red
        }

    # Plot
    fig, ax1 = plt.subplots(figsize=figsize)
    mkw = dict(marker="o", markersize=4) if markers else {}

    l1, = ax1.plot(df_reg[lambda_param], df_reg["AUC_mean"], label="AUROC",
                   linewidth=2, color=colors["AUROC"], **mkw)
    l2, = ax1.plot(df_reg[lambda_param], df_reg["F1_mean"],  label="F1",
                   linewidth=2, color=colors["F1"], **mkw)
    ax1.set_ylabel("AUROC / F1")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    l3, = ax2.plot(df_reg[lambda_param], df_reg["DP_mean"], label="DP",
                   linewidth=2, color=colors["DP"], **mkw)
    l4, = ax2.plot(df_reg[lambda_param], df_reg["EO_mean"], label="EO",
                   linewidth=2, color=colors["EO"], **mkw)
    ax2.set_ylabel("DP / EO (lower is better)")

    if percent:
        ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0))

    # X label (pretty λ)
    if lambda_param.lower() == "lambda_dp":
        xl = r"$\lambda_{\mathrm{DP}}$"
    elif lambda_param.lower() == "lambda_eo":
        xl = r"$\lambda_{\mathrm{EO}}$"
    else:
        xl = lambda_param
    ax1.set_xlabel(xl)

    # Vanilla baselines (λ=0)
    x0 = 0.0
    ax1.axhline(base["AUC_mean"], color=colors["AUROC"], ls="--", lw=1.5, alpha=0.5)
    ax1.axhline(base["F1_mean"],  color=colors["F1"],    ls="--", lw=1.5, alpha=0.5)
    ax2.axhline(base["DP_mean"],  color=colors["DP"],    ls="--", lw=1.5, alpha=0.5)
    ax2.axhline(base["EO_mean"],  color=colors["EO"],    ls="--", lw=1.5, alpha=0.5)
    ax1.scatter([x0], [base["AUC_mean"]], color=colors["AUROC"], s=30, zorder=5)
    ax1.scatter([x0], [base["F1_mean"]],  color=colors["F1"],    s=30, zorder=5)
    ax2.scatter([x0], [base["DP_mean"]],  color=colors["DP"],    s=30, zorder=5)
    ax2.scatter([x0], [base["EO_mean"]],  color=colors["EO"],    s=30, zorder=5)

    # Make sure x=0 is visible
    if not df_reg.empty:
        xmin = min(0.0, float(df_reg[lambda_param].min()))
        xmax = float(df_reg[lambda_param].max())
        ax1.set_xlim(xmin, xmax)

    # Title
    title_parts = ["Utility and Fairness vs. Fairness Regularization ("]
    title_parts.append(xl)
    title_parts.append(")\n")
    ds_txt = f"on {dataset.upper()} " if dataset else ""
    bb_txt = f"with {backbone.upper()} Backbone" if backbone else ""
    title_parts.append(ds_txt + bb_txt)
    ax1.set_title("".join(title_parts))

    # Legend
    vanilla_handles = [
        Line2D([0],[0], color=colors["AUROC"], ls="--", lw=1.5, label="AUROC (vanilla)"),
        Line2D([0],[0], color=colors["F1"],    ls="--", lw=1.5, label="F1 (vanilla)"),
        Line2D([0],[0], color=colors["DP"],    ls="--", lw=1.5, label="DP (vanilla)"),
        Line2D([0],[0], color=colors["EO"],    ls="--", lw=1.5, label="EO (vanilla)"),
    ]
    handles = []
    for a, b in zip([l1, l2, l3, l4], vanilla_handles):
        handles += [a, b]
    labels = []
    for a, b in zip(
        [str(l.get_label()) for l in (l1, l2, l3, l4)],
        ["AUROC (vanilla)", "F1 (vanilla)", "DP (vanilla)", "EO (vanilla)"]
    ):
        labels += [a, b]
    ax2.legend(handles=handles, labels=labels, loc="best")

    plt.tight_layout()
    return fig, (ax1, ax2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Utility vs Fairness from CSV results.")
    parser.add_argument("--csv_file", type=str, help="Path to the CSV file")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to filter")
    parser.add_argument("--backbone", type=str, required=True, help="Backbone to filter")
    parser.add_argument("--lambda_param", type=str, required=True, help="Lambda parameter column name")
    parser.add_argument("--baseline_model", type=str, default="vanilla", help="Baseline model name")
    parser.add_argument("--save_dir", type=str, default="./figures/", help="Directory to save the plot image")
    parser.add_argument("--save_fn", type=str, default="auto", help="Path to save the plot image")
    args = parser.parse_args()

    if os.path.exists(args.save_dir) is False:
        os.makedirs(args.save_dir)

    if args.save_fn == "auto":
        save_path = os.path.join(args.save_dir, f"{args.dataset}_{args.backbone}_{args.lambda_param}.png")
    elif args.save_fn:
        save_path = os.path.join(args.save_dir, args.save_fn)
    else:
        save_path = None

    df = pd.read_csv(args.csv_file)
    plot_utility_fairness(df, dataset=args.dataset, backbone=args.backbone, lambda_param=args.lambda_param)
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
