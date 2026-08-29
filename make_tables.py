"""
make_tables.py
==============
Emit LaTeX table fragments directly from the experiment / uncertainty /
robustness output files, so that every number in the manuscript is traceable to
the reproducible pipeline. Fragments are written to outputs/tables/*.tex and can
be \\input into the manuscript.
"""
from __future__ import annotations

__author__ = "Ihtisham Ul Haq"
__creator__ = "Ihtisham Ul Haq"
__maintainer__ = "Ihtisham Ul Haq"
import os, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "outputs")
TAB = os.path.join(OUT, "tables")
os.makedirs(TAB, exist_ok=True)


def w(name, s):
    with open(os.path.join(TAB, name), "w") as f:
        f.write(s)
    print("  wrote", name)


def sci(x, nd=2):
    """LaTeX scientific / fixed formatting choosing a sensible form."""
    ax = abs(x)
    if ax != 0 and (ax >= 1e4 or ax < 1e-2):
        m, e = f"{x:.{nd}e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{x:.{nd}f}"


def overall_and_improvement():
    df = pd.read_csv(os.path.join(OUT, "results_three_models.csv"))
    n = json.load(open(os.path.join(OUT, "config.json")))["N_eval"]
    df["Method"] = df["Method"]
    short = {"Baseline RMAL": "Baseline RMAL", "BPBO-RMAL": "\\textbf{BPBO-RMAL}",
             "Q-learning+HHO": "Q-learning+HHO"}

    def cell(v, s, bold=False):
        u = s/np.sqrt(n)
        txt = f"{v:.3f} $\\pm$ {u:.3f}" if abs(v) < 100 else f"{v:.2f} $\\pm$ {u:.2f}"
        return "\\textbf{"+txt+"}" if bold else txt

    best = {c: df[c].idxmax() for c in ["Tail_reward", "QoS_ratio", "Fitness_F"]}
    best["Fairness_penalty"] = df["Fairness_penalty"].idxmin()
    lines = []
    for i, r in df.iterrows():
        row = [short[r.Method]]
        for c in ["Tail_reward", "QoS_ratio", "Fairness_penalty", "Fitness_F"]:
            row.append(cell(r[c], r[c+"_std"], bold=(i == best[c])))
        lines.append(" & ".join(row) + " \\\\")
    body = "\n".join(lines)
    tbl = ("\\begin{tabular}{lcccc}\n\\hline\n"
           "\\textbf{Method} & \\textbf{Tail reward} & \\textbf{QoS ratio} & "
           "\\textbf{Fairness penalty} & \\textbf{Fitness $F$} \\\\\n\\hline\n"
           f"{body}\n\\hline\n\\end{{tabular}}")
    w("table_overall.tex", tbl)

    # percentage improvement vs baseline
    b = df.set_index("Method")
    base = b.loc["Baseline RMAL"]
    rows = []
    for m in ["BPBO-RMAL", "Q-learning+HHO"]:
        r = b.loc[m]
        dr = 100*(r.Tail_reward/base.Tail_reward - 1)
        dq = 100*(r.QoS_ratio/base.QoS_ratio - 1)
        dfp = 100*(r.Fairness_penalty/base.Fairness_penalty - 1)
        name = "\\textbf{BPBO-RMAL}" if m == "BPBO-RMAL" else m
        rows.append(f"{name} & {dr:+.1f} & {dq:+.1f} & {dfp:+.1f} \\\\")
    tbl2 = ("\\begin{tabular}{lccc}\n\\hline\n"
            "\\textbf{Method} & \\textbf{Reward (\\%)} & \\textbf{QoS (\\%)} & "
            "\\textbf{Fairness (\\%)} \\\\\n\\hline\n"
            + "\n".join(rows) + "\n\\hline\n\\end{tabular}")
    w("table_improvement.tex", tbl2)
    # also dump a small json of headline numbers for the abstract
    head = dict(reward_gain=float(100*(b.loc["BPBO-RMAL"].Tail_reward/base.Tail_reward-1)),
                qos_gain=float(100*(b.loc["BPBO-RMAL"].QoS_ratio/base.QoS_ratio-1)),
                fair_change=float(100*(b.loc["BPBO-RMAL"].Fairness_penalty/base.Fairness_penalty-1)),
                bpbo=b.loc["BPBO-RMAL"].to_dict(), base=base.to_dict(),
                qhho=b.loc["Q-learning+HHO"].to_dict(), N=n)
    json.dump(head, open(os.path.join(OUT, "headline.json"), "w"), indent=2, default=float)


def hyperparams():
    cfg = json.load(open(os.path.join(OUT, "config.json")))
    tb = cfg["theta_bpbo"]
    base = dict(eps_init=0.5, eps_min=0.1, eps_decay=0.99, C_beta=1.0, phi_beta=0.5,
                w_sinr=1.0, w_fair=1.0, w_power=1.0)
    names = [("eps_init", "$\\epsilon_{\\text{init}}$"),
             ("eps_min", "$\\epsilon_{\\min}$"),
             ("eps_decay", "$\\epsilon_{\\text{decay}}$"),
             ("C_beta", "$C_\\beta$"), ("phi_beta", "$\\phi_\\beta$"),
             ("w_sinr", "$w_{\\text{sinr}}$"), ("w_fair", "$w_{\\text{fair}}$"),
             ("w_power", "$w_{\\text{power}}$")]
    rows = [f"{lbl} & {base[k]:.3f} & {tb[k]:.3f} \\\\" for k, lbl in names]
    tbl = ("\\begin{tabular}{lcc}\n\\hline\n\\textbf{Parameter} & "
           "\\textbf{Baseline RMAL} & \\textbf{BPBO-RMAL} \\\\\n\\hline\n"
           + "\n".join(rows) + "\n\\hline\n\\end{tabular}")
    w("table_hyper_rmal.tex", tbl)

    tq = cfg["theta_qhho"]
    qn = [("alpha_init", "$\\alpha_{\\text{init}}$"), ("alpha_min", "$\\alpha_{\\min}$"),
          ("alpha_decay", "$\\alpha_{\\text{decay}}$"), ("gamma", "$\\gamma$"),
          ("eps_init", "$\\epsilon_{\\text{init}}$"), ("eps_min", "$\\epsilon_{\\min}$"),
          ("eps_decay", "$\\epsilon_{\\text{decay}}$")]
    rows = [f"{lbl} & {tq[k]:.4f} \\\\" for k, lbl in qn]
    tbl = ("\\begin{tabular}{lc}\n\\hline\n\\textbf{Parameter} & \\textbf{Value} \\\\\n"
           "\\hline\n" + "\n".join(rows) + "\n\\hline\n\\end{tabular}")
    w("table_hyper_qhho.tex", tbl)


def uncertainty_tables():
    # input Type B
    js = json.load(open(os.path.join(OUT, "uncertainty_summary.json")))
    iu = js["input_u"]
    labels = [("u_P_db", "Transmit power $P$", "dB", "normal", "RSS, reward"),
              ("u_pos_uav", "UAV position (per axis)", "m", "normal", "distance, assoc."),
              ("u_pos_user", "User position (per axis)", "m", "normal", "distance, assoc."),
              ("u_alpha", "Path-loss exponent $\\alpha$", "--", "normal", "channel gain, SINR"),
              ("u_G0_db", "Reference gain $G_0$", "dB", "normal", "RSS, SINR"),
              ("u_noise_db", "Noise-floor level $\\sigma^2$", "dB", "normal", "SINR, QoS"),
              ("u_fading_db", "Channel (fading) estimate", "dB", "normal", "SINR, policy robustness")]
    rows = [f"{lbl} & {iu[k]:.2f} & {unit} & {dist} & {eff} \\\\"
            for k, lbl, unit, dist, eff in labels]
    w("table_input_uncertainty.tex",
      "\\begin{tabular}{lcccl}\n\\hline\n\\textbf{Input quantity} & "
      "\\textbf{$u(x_i)$} & \\textbf{Unit} & \\textbf{PDF} & "
      "\\textbf{Affects (original metric)} \\\\\n"
      "\\hline\n" + "\n".join(rows) + "\n\\hline\n\\end{tabular}")

    # LPU budget
    bud = pd.read_csv(os.path.join(OUT, "uncertainty_budget_sinr.csv"))
    rows = []
    for _, r in bud.iterrows():
        if "Combined" in str(r.source) or "Expanded" in str(r.source):
            rows.append("\\hline")
            rows.append(f"{r.source} & & & & {r['u_contribution_dB']} & \\\\")
        else:
            rows.append(f"{r.source} & {float(r.u_i):.3g}~{r.unit} & "
                        f"{float(r['c_i (dB/unit)']):+.3g} & "
                        f"{float(r['u_contribution_dB']):.3f} & "
                        f"{float(r.percent):.1f} \\\\")
    # rebuild cleanly with 5 columns
    bud2 = pd.read_csv(os.path.join(OUT, "uncertainty_budget_sinr.csv"))
    body = []
    for _, r in bud2.iterrows():
        s = str(r.source)
        if "Combined" in s:
            body.append("\\hline\n\\textbf{Combined std. uncertainty } $u_c(\\mathrm{SINR})$"
                         f" & & & {float(r['u_contribution_dB']):.3f} & 100.0 \\\\")
        elif "Expanded" in s:
            body.append("\\textbf{Expanded uncertainty } $U=k\\,u_c$ ($k=2$)"
                         f" & & & {float(r['u_contribution_dB']):.3f} & \\\\")
        else:
            body.append(f"{s} & {float(r.u_i):.3g}~{r.unit} & "
                        f"{float(r['c_i (dB/unit)']):+.3g} & "
                        f"{float(r['u_contribution_dB']):.3f} & {float(r.percent):.1f} \\\\")
    w("table_budget.tex",
      "\\begin{tabular}{lcccc}\n\\hline\n\\textbf{Uncertainty source} & "
      "\\textbf{$u(x_i)$} & \\textbf{$c_i$ [dB/unit]} & "
      "\\textbf{$|c_i|u(x_i)$ [dB]} & \\textbf{\\%} \\\\\n\\hline\n"
      + "\n".join(body) + "\n\\hline\n\\end{tabular}")

    # MC measurands
    meas = pd.read_csv(os.path.join(OUT, "uncertainty_measurands.csv"))
    pretty = {"SINR_db": ("SINR", "dB"), "RSS_dbm": ("RSS", "dBm"),
              "throughput_Mbps": ("Throughput", "Mbit/s"),
              "QoS_ratio": ("QoS ratio", "--"), "Fairness": ("Fairness penalty", "--"),
              "Reward": ("Reward", "--")}
    rows = []
    for _, r in meas.iterrows():
        nm, unit = pretty.get(r.measurand, (r.measurand, ""))
        rows.append(f"{nm} & {unit} & {float(r['mean']):.3g} & {float(r.u_std):.3g} & "
                    f"[{float(r.ci95_lo):.3g}, {float(r.ci95_hi):.3g}] \\\\")
    w("table_measurands.tex",
      "\\begin{tabular}{llccc}\n\\hline\n\\textbf{Measurand} & \\textbf{Unit} & "
      "\\textbf{Value} & \\textbf{$u_c$} & \\textbf{95\\% coverage interval} \\\\\n"
      "\\hline\n" + "\n".join(rows) + "\n\\hline\n\\end{tabular}")


def dataset_and_robustness():
    ds = json.load(open(os.path.join(OUT, "dataset_summary.json")))
    base = pd.read_csv(os.path.join(OUT, "dataset_baseline_metrics.csv"))
    rows = []
    for _, r in base.iterrows():
        rows.append(f"{r.epsilon:.2f} & {r.R_tail:.2f} $\\pm$ {r.u_R_tail:.2f} & "
                    f"{r.qos:.3f} $\\pm$ {r.u_qos:.3f} & "
                    f"{r.P_fair:.2f} $\\pm$ {r.u_P_fair:.2f} & "
                    f"{r.mean_sinr:.2f} $\\pm$ {r.u_mean_sinr:.2f} \\\\")
    w("table_dataset_baseline.tex",
      "\\begin{tabular}{lcccc}\n\\hline\n$\\epsilon$ & \\textbf{Tail reward} & "
      "\\textbf{QoS ratio} & \\textbf{Fairness} & \\textbf{Mean SINR [dB]} \\\\\n"
      "\\hline\n" + "\n".join(rows) + "\n\\hline\n\\end{tabular}")

    rob = pd.read_csv(os.path.join(OUT, "robustness_dataset.csv"))
    rows = []
    for _, r in rob.iterrows():
        rows.append(f"{r.sigma_est_db:.1f} & {r.qos_true:.3f} & "
                    f"{r.qos_believed_mean:.3f} & {100*r.misclass_mean:.2f} & "
                    f"{100*r.misclass_analytic:.2f} \\\\")
    w("table_robustness_dataset.tex",
      "\\begin{tabular}{lcccc}\n\\hline\n"
      "$\\sigma_{\\text{est}}$ [dB] & \\textbf{True QoS} & \\textbf{Believed QoS} & "
      "\\textbf{Misclass. \\% (MC)} & \\textbf{Misclass. \\% (anal.)} \\\\\n\\hline\n"
      + "\n".join(rows) + "\n\\hline\n\\end{tabular}")

    pol = pd.read_csv(os.path.join(OUT, "robustness_policy.csv"))
    short = {"Baseline RMAL": "Baseline", "BPBO-RMAL": "\\textbf{BPBO-RMAL}",
             "Q-learning+HHO": "Q+HHO"}
    rows = []
    for name in ["Baseline RMAL", "BPBO-RMAL", "Q-learning+HHO"]:
        sub = pol[pol.method == name]
        first = True
        for _, r in sub.iterrows():
            lbl = short[name] if first else ""
            first = False
            rows.append(f"{lbl} & {r.sigma_est_db:.0f} & {r.qos:.3f} & {r.jain:.3f} & "
                        f"{r.F:.3f} & {r.qos_degr_pct:+.1f} \\\\")
        rows.append("\\hline")
    w("table_robustness_policy.tex",
      "\\begin{tabular}{lccccc}\n\\hline\n\\textbf{Controller} & "
      "$\\sigma_\\gamma$ [dB] & \\textbf{QoS} & \\textbf{Jain} & \\textbf{Fitness} & "
      "\\textbf{QoS degr. \\%} \\\\\n\\hline\n" + "\n".join(rows) + "\n\\end{tabular}")


def fairness_table():
    df = pd.read_csv(os.path.join(OUT, "results_three_models.csv"))
    n = json.load(open(os.path.join(OUT, "config.json")))["N_eval"]
    short = {"Baseline RMAL": "Baseline RMAL", "BPBO-RMAL": "\\textbf{BPBO-RMAL}",
             "Q-learning+HHO": "Q-learning+HHO"}
    best = dict(Fairness_penalty=df.Fairness_penalty.idxmin(), Jain=df.Jain.idxmax(),
                Gini=df.Gini.idxmin(), Entropy=df.Entropy.idxmax())
    def c(v, s, i, key, fmt="{:.3f}"):
        u = s/np.sqrt(n); t = f"{fmt.format(v)} $\\pm$ {fmt.format(u)}"
        return "\\textbf{"+t+"}" if i == best[key] else t
    rows = []
    for i, r in df.iterrows():
        rows.append(" & ".join([short[r.Method],
            c(r.Fairness_penalty, r.Fairness_penalty_std, i, "Fairness_penalty", "{:.2f}"),
            c(r.Jain, r.Jain_std, i, "Jain"), c(r.Gini, r.Gini_std, i, "Gini"),
            c(r.Entropy, r.Entropy_std, i, "Entropy")]) + " \\\\")
    w("table_fairness.tex",
      "\\begin{tabular}{lcccc}\n\\hline\n\\textbf{Method} & "
      "\\textbf{Fairness penalty $\\downarrow$} & \\textbf{Jain $\\uparrow$} & "
      "\\textbf{Gini $\\downarrow$} & \\textbf{Entropy $\\uparrow$} \\\\\n\\hline\n"
      + "\n".join(rows) + "\n\\hline\n\\end{tabular}")


def mc_bounded_table():
    df = pd.read_csv(os.path.join(OUT, "mc_bounded_summary.csv"))
    pretty = {"qos": "QoS ratio", "R_tail": "Tail reward",
              "P_fair": "Fairness penalty", "F": "Fitness $F$"}
    short = {"Baseline RMAL": "Baseline", "BPBO-RMAL": "\\textbf{BPBO-RMAL}",
             "Q-learning+HHO": "Q+HHO"}
    rows = []
    for m in ["qos", "R_tail", "P_fair", "F"]:
        rows.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{pretty[m]}}}}} \\\\")
        for name in ["Baseline RMAL", "BPBO-RMAL", "Q-learning+HHO"]:
            r = df[(df.Method == name) & (df.metric == m)].iloc[0]
            rows.append(f"\\quad {short[name]} & {r['mean']:.3f} & {r.u:.3f} & "
                        f"{r.U_k2:.3f} & [{r.boot_lo:.3f}, {r.boot_hi:.3f}] \\\\")
        rows.append("\\hline")
    w("table_mc_bounded.tex",
      "\\begin{tabular}{lcccc}\n\\hline\n\\textbf{Measurand / Controller} & "
      "\\textbf{Mean} & \\textbf{$u_c$} & \\textbf{$U$ ($k{=}2$)} & "
      "\\textbf{Bootstrap 95\\% CI} \\\\\n\\hline\n"
      + "\n".join(rows) + "\n\\end{tabular}")


def sensitivity_table():
    df = pd.read_csv(os.path.join(OUT, "sensitivity_indices.csv"))
    outs = [("qos", "QoS"), ("P_fair", "Fairness"), ("R_tail", "Reward"), ("F", "Fitness")]
    # top-5 parameters by ST for each output
    def esc(s):
        return s.replace("_", "\\_")
    rows = []
    for key, lbl in outs:
        sub = df[df.output == key].sort_values("ST", ascending=False).head(5)
        drivers = ", ".join(f"{esc(r.parameter)} ({r.ST:.2f})" for _, r in sub.iterrows())
        rows.append(f"{lbl} & {drivers} \\\\")
    w("table_sensitivity.tex",
      "\\begin{tabular}{ll}\n\\hline\n\\textbf{Indicator} & "
      "\\textbf{Dominant parameters (total-order Sobol index $S_T$)} \\\\\n\\hline\n"
      + "\n".join(rows) + "\n\\hline\n\\end{tabular}")


def main():
    for fn in [overall_and_improvement, hyperparams, uncertainty_tables,
               dataset_and_robustness, fairness_table, mc_bounded_table,
               sensitivity_table]:
        try:
            fn()
        except Exception as e:
            print(f"  [skip] {fn.__name__}: {e}")
    print("Tables ->", TAB)


if __name__ == "__main__":
    main()
