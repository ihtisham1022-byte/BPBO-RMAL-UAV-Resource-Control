"""
figures.py
==========
Regenerate every results figure from the experiment, uncertainty, and
robustness outputs. Vector PDF + PNG
are written to outputs/figures/. Colour-blind-safe (Okabe-Ito) palette.
"""
from __future__ import annotations

__author__ = "Ihtisham Ul Haq"
__creator__ = "Ihtisham Ul Haq"
__maintainer__ = "Ihtisham Ul Haq"
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "outputs")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

# Okabe-Ito colour-blind-safe palette
C_BASE, C_BPBO, C_HHO = "#999999", "#0072B2", "#E69F00"
C_TRUE, C_MEAS, C_ACC = "#009E73", "#D55E00", "#CC79A7"
METHOD_C = {"Baseline RMAL": C_BASE, "BPBO-RMAL": C_BPBO, "Q-learning+HHO": C_HHO}

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "figure.dpi": 130, "savefig.bbox": "tight", "axes.axisbelow": True,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name+".pdf"))
    fig.savefig(os.path.join(FIG, name+".png"), dpi=300)
    plt.close(fig)
    print("  saved", name)


def fig_comparison():
    df = pd.read_csv(os.path.join(OUT, "results_three_models.csv"))
    n = json.load(open(os.path.join(OUT, "config.json")))["N_eval"]
    metrics = [("Tail_reward", "Tail reward $R_{\\mathrm{tail}}$"),
               ("QoS_ratio", "QoS ratio"),
               ("Fairness_penalty", "Fairness penalty $P_{\\mathrm{fair}}$"),
               ("Fitness_F", "Fitness $F$")]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0))
    for ax, (col, title) in zip(axes, metrics):
        vals = df[col].values
        err = df[col+"_std"].values/np.sqrt(n)     # Type A std uncertainty of mean
        cols = [METHOD_C[m] for m in df.Method]
        ax.bar(range(len(df)), vals, yerr=err, capsize=4, color=cols,
               edgecolor="black", linewidth=0.6)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(["RMAL", "BPBO", "Q+HHO"], rotation=0)
        ax.set_title(title)
    axes[0].set_ylabel("value (mean $\\pm$ $u$)")
    fig.suptitle("Controller comparison (error bars: Type A standard uncertainty of the mean)",
                 y=1.04, fontsize=10)
    save(fig, "fig_comparison")


def fig_pareto():
    df = pd.read_csv(os.path.join(OUT, "results_three_models.csv"))
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    sizes = 300*(df.Fitness_F/df.Fitness_F.max())+120
    sc = ax.scatter(df.Tail_reward, df.Fairness_penalty, c=df.QoS_ratio,
                    s=sizes, cmap="viridis", edgecolor="black", linewidths=1.0,
                    zorder=3)
    for _, r in df.iterrows():
        ax.annotate(r.Method.replace(" RMAL", "").replace("Q-learning+", "Q+"),
                    (r.Tail_reward, r.Fairness_penalty),
                    xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Tail reward $R_{\\mathrm{tail}}$  (higher better $\\rightarrow$)")
    ax.set_ylabel("Fairness penalty $P_{\\mathrm{fair}}$  ($\\leftarrow$ lower better)")
    ax.invert_yaxis()
    cb = fig.colorbar(sc, ax=ax); cb.set_label("QoS ratio")
    ax.set_title("Reward–fairness trade-off (size $\\propto$ fitness)")
    save(fig, "fig_pareto")


def fig_convergence():
    d = np.load(os.path.join(OUT, "convergence.npz"))
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(range(1, len(d["bpbo"])+1), d["bpbo"], "-o", color=C_BPBO,
            ms=4, label="BPBO-RMAL")
    ax.plot(range(1, len(d["hho"])+1), d["hho"], "-s", color=C_HHO,
            ms=4, label="Q-learning+HHO")
    ax.set_xlabel("optimisation iteration")
    ax.set_ylabel("best fitness $F$")
    ax.set_title("Hyperparameter-search convergence")
    ax.legend()
    save(fig, "fig_convergence")


def fig_uncertainty_budget():
    df = pd.read_csv(os.path.join(OUT, "uncertainty_budget_sinr.csv"))
    df = df[~df.source.str.contains("Combined|Expanded")].copy()
    df["percent"] = df["percent"].astype(float)
    df = df.sort_values("percent")
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.barh(range(len(df)), df["percent"], color=C_BPBO, edgecolor="black",
            linewidth=0.5)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([s.replace("$", "") for s in df.source])
    ax.set_xlabel("contribution to SINR variance [%]")
    ax.set_title("SINR uncertainty budget (GUM law of propagation)")
    save(fig, "fig_uncertainty_budget")


def fig_mc_distribution():
    d = np.load(os.path.join(OUT, "uncertainty_mc.npz"))
    meas = pd.read_csv(os.path.join(OUT, "uncertainty_measurands.csv"))
    row = meas[meas.measurand == "SINR_db"].iloc[0]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.hist(d["sinr"], bins=80, color=C_BPBO, alpha=0.75, density=True,
            edgecolor="none")
    for x, ls, lab in [(row["mean"], "-", "mean"),
                       (row["ci95_lo"], "--", "95% coverage"),
                       (row["ci95_hi"], "--", None)]:
        ax.axvline(x, color=C_MEAS, ls=ls, lw=1.6, label=lab)
    ax.set_xlabel("SINR [dB]"); ax.set_ylabel("probability density")
    ax.set_title("Monte-Carlo SINR distribution (GUM Supplement 1)")
    ax.legend()
    save(fig, "fig_mc_distribution")


def fig_robustness():
    """Dataset threshold sensitivity: misclassification + believed QoS vs sigma."""
    ds = pd.read_csv(os.path.join(OUT, "robustness_dataset.csv"))
    fig, ax1 = plt.subplots(figsize=(5.6, 3.4), constrained_layout=True)
    ax1.plot(ds.sigma_est_db, 100*ds.misclass_mean, "-o", color=C_MEAS,
             label="QoS misclassification (MC)")
    ax1.plot(ds.sigma_est_db, 100*ds.misclass_analytic, "k--", lw=1, label="analytical")
    ax1b = ax1.twinx()
    ax1b.plot(ds.sigma_est_db, ds.qos_believed_mean, "-s", color=C_TRUE,
              label="believed QoS ratio")
    ax1b.axhline(ds.qos_true.iloc[0], color=C_BASE, ls=":", label="true QoS ratio")
    ax1.set_xlabel("SINR estimation error $\\sigma_\\gamma$ [dB]")
    ax1.set_ylabel("QoS misclassification [%]", color=C_MEAS)
    ax1b.set_ylabel("QoS ratio", color=C_TRUE)
    ax1.grid(False); ax1b.grid(False)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, fontsize=8, loc="center right")
    save(fig, "fig_robustness")


def fig_robustness_curves():
    """Robustness of QoS, Fitness and Jain vs SINR estimation error (3 panels)."""
    pol = pd.read_csv(os.path.join(OUT, "robustness_policy.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True)
    panels = [("qos", "experienced QoS ratio"), ("F", "fitness $F$"),
              ("jain", "Jain fairness index")]
    for ax, (col, ylab) in zip(axes, panels):
        for name in ["Baseline RMAL", "BPBO-RMAL", "Q-learning+HHO"]:
            s = pol[pol.method == name]
            yerr = s[col+"_u"] if col+"_u" in s else None
            ax.errorbar(s.sigma_est_db, s[col], yerr=yerr, marker="o", capsize=3,
                        color=METHOD_C[name], label=name)
        ax.set_xlabel("SINR estimation error $\\sigma_\\gamma$ [dB]")
        ax.set_ylabel(ylab)
    axes[1].legend(fontsize=8, loc="best")
    save(fig, "fig_robustness_curves")


def fig_fairness():
    df = pd.read_csv(os.path.join(OUT, "results_three_models.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), constrained_layout=True)
    cols = [("Jain", "Jain index $\\uparrow$"), ("Gini", "Gini coeff. $\\downarrow$"),
            ("Entropy", "Entropy fairness $\\uparrow$")]
    n = json.load(open(os.path.join(OUT, "config.json")))["N_eval"]
    for ax, (c, title) in zip(axes, cols):
        cols_c = [METHOD_C[m] for m in df.Method]
        ax.bar(range(len(df)), df[c], yerr=df[c+"_std"]/np.sqrt(n), capsize=4,
               color=cols_c, edgecolor="black", linewidth=0.6)
        ax.set_xticks(range(len(df))); ax.set_xticklabels(["RMAL", "BPBO", "Q+HHO"])
        ax.set_title(title)
        lo = min(df[c]); ax.set_ylim(max(0, lo-3*(df[c].max()-lo)-0.01), None)
    save(fig, "fig_fairness")


def fig_sensitivity():
    df = pd.read_csv(os.path.join(OUT, "sensitivity_indices.csv"))
    sub = df[df.output == "F"].copy().sort_values("ST")     # fitness drivers
    colors = [C_BPBO if k == "learn" else C_HHO for k in sub.kind]
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    ax.barh(range(len(sub)), sub.ST, xerr=sub.ST_conf, color=colors,
            edgecolor="black", linewidth=0.5, capsize=2)
    ax.set_yticks(range(len(sub))); ax.set_yticklabels(sub.parameter)
    ax.set_xlabel("total-order Sobol index $S_T$ (fitness)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_BPBO, label="learning (BPBO-tuned)"),
                       Patch(color=C_HHO, label="measurement")], fontsize=8, loc="lower right")
    save(fig, "fig_sensitivity")


def fig_mc_bounded():
    d = np.load(os.path.join(OUT, "mc_bounded.npz"))
    methods = ["Baseline RMAL", "BPBO-RMAL", "Q-learning+HHO"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
    for ax, (m, lab) in zip(axes, [("qos", "QoS ratio"), ("F", "fitness $F$")]):
        data = [d[f"{n}__{m}"] for n in methods]
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for pc, n in zip(parts["bodies"], methods):
            pc.set_facecolor(METHOD_C[n]); pc.set_alpha(0.6)
        ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["RMAL", "BPBO", "Q+HHO"])
        ax.set_ylabel(lab)
    fig.suptitle("Monte-Carlo distributions under input-quantity uncertainty ($M=10^4$)",
                 fontsize=10)
    save(fig, "fig_mc_bounded")


def fig_pipeline():
    """Schematic of the measurement model and uncertainty-propagation pipeline."""
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 40)
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3",
                     fc=fc, ec="black", lw=1))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=8.5)
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=12, lw=1.2, color="#444"))
    inp = "Input quantities\n$\\mathbf{p}_u,\\mathbf{p}_k,P_u,\\alpha,G_0,g,\\sigma^2$"
    box(2, 22, 20, 12, inp, "#DCE6F5")
    box(28, 22, 18, 12, "Measurement\nmodel $y=f(x)$\nSINR, RSS, $d$", "#FDECC8")
    box(52, 22, 18, 12, "Measurands\nQoS, $T_u$, fairness,\nreward, fitness", "#D8EDD8")
    box(78, 22, 20, 12, "Uncertainty-bounded\nperformance\n(mean $\\pm U$, CIs)", "#F5D9D9")
    arrow(22, 28, 28, 28); arrow(46, 28, 52, 28); arrow(70, 28, 78, 28)
    box(28, 4, 18, 10, "Type B: input\nuncertainties", "#EFEFEF")
    box(52, 4, 18, 10, "GUM LPU +\nMonte-Carlo (S1)", "#EFEFEF")
    box(78, 4, 20, 10, "Robustness +\nsensitivity", "#EFEFEF")
    arrow(37, 14, 37, 22); arrow(61, 14, 61, 22); arrow(88, 14, 88, 22)
    save(fig, "fig_pipeline")


def fig_validation():
    dist = np.load(os.path.join(OUT, "dataset_distributions.npz"))
    mc = np.load(os.path.join(OUT, "uncertainty_mc.npz"))
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.hist(dist["sinr_db"], bins=40, density=True, alpha=0.6, color=C_BASE,
            label="provided dataset", edgecolor="none")
    ax.hist(mc["sinr"], bins=80, density=True, histtype="step", lw=1.8,
            color=C_BPBO, label="reproducible simulator")
    ax.axvline(float(np.median(dist["sinr_db"])), color=C_BASE, ls=":", lw=1)
    ax.set_xlabel("SINR [dB]"); ax.set_ylabel("probability density")
    ax.set_xlim(-20, 40)
    ax.set_title("Model validation: SINR distribution")
    ax.legend()
    save(fig, "fig_validation")


def main():
    print("Regenerating figures ...")
    for f in [fig_comparison, fig_pareto, fig_convergence, fig_uncertainty_budget,
              fig_mc_distribution, fig_robustness, fig_validation,
              fig_robustness_curves, fig_fairness, fig_sensitivity,
              fig_mc_bounded, fig_pipeline]:
        try:
            f()
        except Exception as e:
            print(f"  [skip] {f.__name__}: {e}")
    print("Done ->", FIG)


if __name__ == "__main__":
    main()
