"""
mc_bounded.py
=============
Monte-Carlo, uncertainty-bounded validation of the original controller
comparison (GUM Supplement 1).

For each controller (Baseline RMAL, BPBO-RMAL, Q-learning+HHO) the policy is
trained once and then FROZEN. The frozen policy is evaluated over M trials in
which the true input quantities of the measurement model (UAV/user positions,
transmit power, path-loss exponent, reference gain, noise floor, and the
small-scale fading) are sampled from their Type-B distributions (Table of input
uncertainties). This propagates the measurement uncertainty into the
performance indicators, so that the original comparison is reported as
uncertainty-bounded rather than as point estimates.

Outputs: outputs/mc_bounded_summary.csv, outputs/mc_bounded.npz
"""
from __future__ import annotations

__author__ = "Ihtisham Ul Haq"
__creator__ = "Ihtisham Ul Haq"
__maintainer__ = "Ihtisham Ul Haq"
import os, json
import numpy as np
import pandas as pd
import mm_core as mm
import controllers as ctl

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "outputs")
M = 10000            # Monte-Carlo trials per controller (GUM S1)
T_EVAL = 80          # evaluation horizon per trial
B_BOOT = 2000        # bootstrap resamples for the CI of the mean

# Type-B input-quantity uncertainties (same as the uncertainty budget)
PERTURB = dict(u_pos=1.5, u_P_db=0.5, u_alpha=0.10, u_G0_db=1.0,
               u_noise_db=0.5, sigma_est_db=0.0)


def load_params():
    cfg = json.load(open(os.path.join(OUT, "config.json")))
    p = mm.SysParams()
    p.gamma_th_db = cfg["gamma_th_db"]
    p.noise_floor_dbm = cfg["sys_params"].get("noise_floor_dbm")
    return p, cfg


def controllers_specs(cfg):
    tq = cfg["theta_qhho"]
    return [
        ("Baseline RMAL", ctl.RMAL_BASELINE, dict()),
        ("BPBO-RMAL", cfg["theta_bpbo"], dict()),
        ("Q-learning+HHO", ctl._qhho_theta_to_rmal(tq),
         dict(alpha_sched=(tq["alpha_init"], tq["alpha_min"], tq["alpha_decay"]),
              gamma_q=tq["gamma"], reward_modulated=False)),
    ]


def boot_ci(x, rng, B=B_BOOT):
    """Bootstrap 95% CI of the mean."""
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    p, cfg = load_params()
    refs = ctl.Refs()
    rng = np.random.default_rng(20240705)
    metrics = ["qos", "R_tail", "P_fair", "F", "jain", "gini", "entropy"]
    dists = {}
    rows = []
    print(f"[mc-bounded] M={M} trials per controller, T={T_EVAL}")

    for name, theta, kw in controllers_specs(cfg):
        Q = ctl.train_policy(p, theta, seed=1,
                             alpha_sched=kw.get("alpha_sched"),
                             gamma_q=kw.get("gamma_q", 0.9),
                             reward_modulated=kw.get("reward_modulated", True))
        acc = {m: np.empty(M) for m in metrics}
        for i in range(M):
            r = ctl.eval_policy(p, theta, Q, rng, T=T_EVAL,
                                alpha_sched=kw.get("alpha_sched"),
                                gamma_q=kw.get("gamma_q", 0.9), perturb=PERTURB)
            r["F"] = ctl.fitness(r["R_tail"], r["qos"], r["P_fair"], refs)
            for m in metrics:
                acc[m][i] = r[m]
        dists[name] = acc
        for m in metrics:
            v = acc[m]
            lo, hi = boot_ci(v, rng)
            rows.append(dict(Method=name, metric=m, mean=float(v.mean()),
                             u=float(v.std(ddof=1)), U_k2=float(2*v.std(ddof=1)),
                             ci95_lo=float(np.percentile(v, 2.5)),
                             ci95_hi=float(np.percentile(v, 97.5)),
                             boot_lo=lo, boot_hi=hi))
        print(f"  {name:16s} QoS={acc['qos'].mean():.3f}+/-{acc['qos'].std():.3f}  "
              f"F={acc['F'].mean():.3f}+/-{acc['F'].std():.3f}  "
              f"Jain={acc['jain'].mean():.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "mc_bounded_summary.csv"), index=False)
    np.savez(os.path.join(OUT, "mc_bounded.npz"),
             **{f"{n}__{m}": dists[n][m] for n in dists for m in metrics})

    # verdict: does BPBO retain the best QoS and fitness under uncertainty?
    piv = df.pivot(index="Method", columns="metric", values="mean")
    best_qos = piv["qos"].idxmax(); best_F = piv["F"].idxmax()
    verdict = dict(best_qos=best_qos, best_F=best_F,
                   bpbo_qos_ci=[float(df.loc[(df.Method=="BPBO-RMAL") & (df.metric=="qos"), "ci95_lo"].iloc[0]),
                                float(df.loc[(df.Method=="BPBO-RMAL") & (df.metric=="qos"), "ci95_hi"].iloc[0])],
                   base_qos_ci=[float(df.loc[(df.Method=="Baseline RMAL") & (df.metric=="qos"), "ci95_lo"].iloc[0]),
                                float(df.loc[(df.Method=="Baseline RMAL") & (df.metric=="qos"), "ci95_hi"].iloc[0])])
    json.dump(verdict, open(os.path.join(OUT, "mc_bounded_verdict.json"), "w"), indent=2)
    print(f"[verdict] best QoS under uncertainty: {best_qos}; best fitness: {best_F}")
    print("[done] outputs/mc_bounded_summary.csv, mc_bounded.npz")


if __name__ == "__main__":
    main()
