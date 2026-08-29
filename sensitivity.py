"""
sensitivity.py
==============
Variance-based (Sobol) global sensitivity analysis of the performance
indicators (QoS ratio, fairness penalty, tail reward, fitness) with respect to
both the measurement-uncertainty parameters and the learning hyperparameters
that BPBO optimises. This provides metrological support for the BPBO design: it
shows that the parameters BPBO tunes are among the dominant contributors to the
variability of QoS and fitness.

Model per sample: build an RMAL theta from the learning parameters, set the QoS
threshold, train a policy, freeze it, and evaluate it under the sampled
measurement uncertainties; return the mean of each performance indicator.

Outputs: outputs/sensitivity_indices.csv, outputs/sensitivity.npz
"""
from __future__ import annotations

__author__ = "Ihtisham Ul Haq"
__creator__ = "Ihtisham Ul Haq"
__maintainer__ = "Ihtisham Ul Haq"
import os, json
import numpy as np
import pandas as pd
from dataclasses import replace
import mm_core as mm
import controllers as ctl
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "outputs")
N_SOBOL = 64          # base sample size -> N*(2D+2) model evaluations
N_EVAL_TRIALS = 3
T_TRAIN, T_EVAL = 120, 80

# (name, low, high, kind)  kind in {meas, learn}
PARAMS = [
    ("u_SINR (dB)",     0.0, 3.0,   "meas"),
    ("u_RSS/power (dB)", 0.0, 2.0,  "meas"),
    ("u_location (m)",  0.0, 5.0,   "meas"),
    ("u_channel (dB)",  0.0, 3.0,   "meas"),
    ("QoS threshold",   1.0, 6.0,   "meas"),
    ("eps_init",        0.05, 1.0,  "learn"),
    ("eps_min",         0.0, 0.30,  "learn"),
    ("eps_decay",       0.90, 0.999, "learn"),
    ("C_beta",          0.3, 3.0,   "learn"),
    ("phi_beta",        0.1, 0.9,   "learn"),
    ("w_sinr",          0.5, 2.0,   "learn"),
    ("w_fair",          0.2, 2.0,   "learn"),
    ("w_power",         0.2, 2.0,   "learn"),
]
NAMES = [p[0] for p in PARAMS]
PROBLEM = dict(num_vars=len(PARAMS), names=NAMES,
               bounds=[[p[1], p[2]] for p in PARAMS])
OUTPUTS = ["qos", "P_fair", "R_tail", "F"]


def load_params():
    cfg = json.load(open(os.path.join(OUT, "config.json")))
    p = mm.SysParams()
    p.gamma_th_db = cfg["gamma_th_db"]
    p.noise_floor_dbm = cfg["sys_params"].get("noise_floor_dbm")
    return p


def model(p, x, refs, rng):
    """Evaluate the four performance indicators for one parameter vector x."""
    (uS, uR, uL, uC, gth, e0, emin, edec, Cb, pb, ws, wf, wp) = x
    theta = dict(eps_init=e0, eps_min=min(emin, e0), eps_decay=edec,
                 C_beta=Cb, phi_beta=pb, w_sinr=ws, w_fair=wf, w_power=wp)
    pp = replace(p, gamma_th_db=float(gth))
    _, Q = ctl.run_episode(pp, theta, np.random.default_rng(1), T=T_TRAIN)
    perturb = dict(sigma_est_db=uS, u_P_db=uR, u_pos=uL, u_G0_db=uC)
    acc = {m: [] for m in OUTPUTS}
    for k in range(N_EVAL_TRIALS):
        r = ctl.eval_policy(pp, theta, Q, rng, T=T_EVAL, perturb=perturb)
        r["F"] = ctl.fitness(r["R_tail"], r["qos"], r["P_fair"], refs)
        for m in OUTPUTS:
            acc[m].append(r[m])
    return {m: float(np.mean(acc[m])) for m in OUTPUTS}


def main():
    p = load_params()
    refs = ctl.Refs()
    rng = np.random.default_rng(77)
    X = sobol_sample.sample(PROBLEM, N_SOBOL, calc_second_order=False)
    print(f"[sensitivity] {X.shape[0]} model evaluations, {len(PARAMS)} params")
    Y = {m: np.empty(X.shape[0]) for m in OUTPUTS}
    for i, x in enumerate(X):
        out = model(p, x, refs, rng)
        for m in OUTPUTS:
            Y[m][i] = out[m]
        if (i+1) % 200 == 0:
            print(f"   {i+1}/{X.shape[0]}")

    rows = []
    for m in OUTPUTS:
        Si = sobol_analyze.analyze(PROBLEM, Y[m], calc_second_order=False,
                                   print_to_console=False)
        for j, nm in enumerate(NAMES):
            rows.append(dict(output=m, parameter=nm, kind=PARAMS[j][3],
                             S1=float(Si["S1"][j]), ST=float(Si["ST"][j]),
                             ST_conf=float(Si["ST_conf"][j])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "sensitivity_indices.csv"), index=False)
    np.savez(os.path.join(OUT, "sensitivity.npz"), X=X, **{f"Y_{m}": Y[m] for m in OUTPUTS})

    print("\n=== Total-order Sobol indices (ST), top drivers per output ===")
    for m in OUTPUTS:
        sub = df[df.output == m].sort_values("ST", ascending=False).head(4)
        drivers = ", ".join(f"{r.parameter}({r.ST:.2f})" for _, r in sub.iterrows())
        print(f"  {m:8s}: {drivers}")
    print("[done] sensitivity_indices.csv, sensitivity.npz")


if __name__ == "__main__":
    main()
