import argparse, os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.lines import Line2D

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
