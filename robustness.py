"""
robustness.py
=============
Robustness of the controller findings under SINR measurement error.
SINR is treated as an estimated quantity,
    SINR_hat_dB = SINR_dB + e,  e ~ N(0, sigma_gamma^2),
and the estimation error sigma_gamma is swept over {0, 1, 2, 3, 5} dB.

(1) Data-driven threshold sensitivity on the provided dataset (MC + analytical).
(2) Closed-loop robustness of each FROZEN trained controller: the policy is
    trained nominally, then evaluated while it observes noisy SINR. For every
    controller and sigma we report QoS, tail reward, fairness penalty, Jain,
    Gini, entropy, fitness and the degradation w.r.t. the perfect-knowledge case.

Outputs: outputs/robustness_dataset.csv, outputs/robustness_policy.csv,
         outputs/robustness.npz
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
SIGMAS = [0.0, 1.0, 2.0, 3.0, 5.0]     # SINR estimation-error std [dB]
K_TRIALS = 40                          # evaluation trials per (controller, sigma)
T_EVAL = 80


def load_params():
    cfg = json.load(open(os.path.join(OUT, "config.json")))
    p = mm.SysParams()
    p.gamma_th_db = cfg["gamma_th_db"]
    p.noise_floor_dbm = cfg["sys_params"].get("noise_floor_dbm")
    return p, cfg


def dataset_threshold_sensitivity(gamma=3.0, n_mc=2000, seed=11):
    path = None
    candidates = [os.path.join(OUT, "provided_dataset.csv"),
                  os.path.join(HERE, "rmal_micro_manuscript_grade.xlsx")]
    env_path = os.environ.get("RMAL_DATASET_PATH")
    if env_path:
        candidates.insert(0, env_path)
    for c in candidates:
        if os.path.exists(c):
            path = c; break
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    sinr_true = df.sinr_db.values
    qos_true = float((sinr_true >= gamma).mean())
    rng = np.random.default_rng(seed)
    rows = []
    for s in SIGMAS:
        if s == 0:
            believed, mis = np.full(n_mc, qos_true), np.zeros(n_mc)
        else:
            believed = np.empty(n_mc); mis = np.empty(n_mc)
            for i in range(n_mc):
                est = sinr_true + rng.normal(0, s, size=sinr_true.shape)
                believed[i] = (est >= gamma).mean()
                mis[i] = np.mean((est >= gamma) != (sinr_true >= gamma))
        ana = float(np.mean(mm.qos_misclassification_prob(sinr_true, gamma, s)))
        rows.append(dict(sigma_est_db=s, qos_true=qos_true,
                         qos_believed_mean=float(believed.mean()),
                         qos_believed_u=float(believed.std(ddof=1)) if s > 0 else 0.0,
                         misclass_mean=float(mis.mean()),
                         misclass_analytic=ana))
    return pd.DataFrame(rows)


def policy_robustness(p, cfg, seed_base=5000):
    tq = cfg["theta_qhho"]
    specs = [
        ("Baseline RMAL", ctl.RMAL_BASELINE, dict()),
        ("BPBO-RMAL", cfg["theta_bpbo"], dict()),
        ("Q-learning+HHO", ctl._qhho_theta_to_rmal(tq),
         dict(alpha_sched=(tq["alpha_init"], tq["alpha_min"], tq["alpha_decay"]),
              gamma_q=tq["gamma"], reward_modulated=False)),
    ]
    refs = ctl.Refs()
    metrics = ["qos", "R_tail", "P_fair", "jain", "gini", "entropy", "F"]
    rows = []
    for name, theta, kw in specs:
        Q = ctl.train_policy(p, theta, seed=1, alpha_sched=kw.get("alpha_sched"),
                             gamma_q=kw.get("gamma_q", 0.9),
                             reward_modulated=kw.get("reward_modulated", True))
        base_vals = {}
        for s in SIGMAS:
            rng = np.random.default_rng(seed_base + int(10*s))
            acc = {m: [] for m in metrics}
            for k in range(K_TRIALS):
                r = ctl.eval_policy(p, theta, Q, rng, T=T_EVAL,
                                    alpha_sched=kw.get("alpha_sched"),
                                    gamma_q=kw.get("gamma_q", 0.9),
                                    perturb=dict(sigma_est_db=s))
                r["F"] = ctl.fitness(r["R_tail"], r["qos"], r["P_fair"], refs)
                for m in metrics:
                    acc[m].append(r[m])
            row = dict(method=name, sigma_est_db=s)
            for m in metrics:
                v = np.array(acc[m])
                row[m] = float(v.mean()); row[m+"_u"] = float(v.std(ddof=1)/np.sqrt(K_TRIALS))
            if s == 0:
                base_vals = {m: row[m] for m in metrics}
            # degradation (%) of QoS and fitness vs perfect knowledge
            row["qos_degr_pct"] = 100*(row["qos"]-base_vals["qos"])/base_vals["qos"]
            row["F_degr_pct"] = 100*(row["F"]-base_vals["F"])/base_vals["F"] if base_vals["F"] else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    p, cfg = load_params()
    print(f"[robustness] gamma_th={p.gamma_th_db} dB, floor={p.noise_floor_dbm:.2f} dBm, "
          f"sigmas={SIGMAS}")

    ds = dataset_threshold_sensitivity(gamma=p.gamma_th_db)
    ds.to_csv(os.path.join(OUT, "robustness_dataset.csv"), index=False)
    print("\n(1) Dataset threshold sensitivity:")
    for _, r in ds.iterrows():
        print(f"   sigma={r.sigma_est_db:.1f}  believed QoS={r.qos_believed_mean:.3f}  "
              f"misclass={100*r.misclass_mean:5.2f}% (anal {100*r.misclass_analytic:.2f}%)")

    pol = policy_robustness(p, cfg)
    pol.to_csv(os.path.join(OUT, "robustness_policy.csv"), index=False)
    print("\n(2) Closed-loop frozen-policy robustness:")
    for name in pol.method.unique():
        sub = pol[pol.method == name]
        print(f"   {name}:")
        for _, r in sub.iterrows():
            print(f"     sigma={r.sigma_est_db:.1f}  QoS={r.qos:.3f}  F={r.F:.3f}  "
                  f"Jain={r.jain:.3f}  QoS_degr={r.qos_degr_pct:+.1f}%  F_degr={r.F_degr_pct:+.1f}%")

    np.savez(os.path.join(OUT, "robustness.npz"),
             sigmas=np.array(SIGMAS), ds_sigma=ds.sigma_est_db.values,
             ds_believed=ds.qos_believed_mean.values,
             ds_misclass=ds.misclass_mean.values, ds_qos_true=ds.qos_true.values)
    print("\n[done] robustness_dataset.csv, robustness_policy.csv, robustness.npz")


if __name__ == "__main__":
    main()
