"""
uncertainty.py
==============
GUM-compliant uncertainty analysis for the UAV downlink measurement model.

Two complementary evaluations are provided, as recommended by the GUM and its
Supplement 1:

  (A) Law of Propagation of Uncertainty (GUM, JCGM 100:2008, clause 5):
      combined standard uncertainty of the SINR from an uncertainty budget with
      numerically-evaluated sensitivity coefficients c_i = dY/dx_i.

  (B) Monte-Carlo propagation of distributions (GUM Supplement 1, JCGM 101:2008):
      the input PDFs are propagated through the full multi-interferer model to
      obtain the distributions and 95 % coverage intervals of the SINR, received
      signal strength, throughput, QoS ratio, fairness penalty and reward.

Both are evaluated at the operating point of the provided controlled-simulation
dataset. Type A repeatability (from replicate episodes) is combined with the
Type B budget to give the expanded uncertainty (k = 2).

Outputs: outputs/uncertainty_budget_sinr.csv, outputs/uncertainty_measurands.csv,
         outputs/uncertainty_mc.npz, outputs/uncertainty_summary.json
"""
from __future__ import annotations

__author__ = "Ihtisham Ul Haq"
__creator__ = "Ihtisham Ul Haq"
__maintainer__ = "Ihtisham Ul Haq"
import os, json
from dataclasses import dataclass, asdict
import numpy as np
import mm_core as mm

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Type B input-quantity uncertainties (standard uncertainties)
# ---------------------------------------------------------------------------
@dataclass
class InputU:
    u_P_db: float = 0.5        # transmit-power calibration [dB]
    u_pos_uav: float = 1.5     # UAV position (GNSS+baro), per axis [m]
    u_pos_user: float = 3.0    # user localisation, per axis [m]
    u_alpha: float = 0.10      # path-loss exponent (model) [-]
    u_G0_db: float = 1.0       # reference gain / antenna [dB]
    u_noise_db: float = 0.5    # noise-power level [dB]
    u_fading_db: float = 1.0   # channel (fading) estimation [dB]


# ---------------------------------------------------------------------------
# Representative operating scenario from the dataset (fixed nominal geometry)
# ---------------------------------------------------------------------------
def nominal_scenario(p: mm.SysParams, slot=80, users_seed=2024):
    users = mm.sample_users(p, np.random.default_rng(users_seed))
    uav = mm.uav_positions(p, slot)
    powers_dbm = np.full(p.n_uav, 20.0)              # typical operating power
    # Representative interference-limited condition (co-channel reuse) that
    # reproduces the near-threshold operating SINR of the provided dataset.
    subch = np.zeros(p.n_uav, dtype=int)
    return dict(users=users, uav=uav, powers_dbm=powers_dbm, subch=subch)


# ---------------------------------------------------------------------------
# Vectorised SINR model that accepts perturbed input quantities (for MC + LPU)
# ---------------------------------------------------------------------------
def sinr_ensemble(p, uav_xy, users_xy, powers_dbm, subch, alpha, G0_db,
                  noise_db, fading, fading_err_db):
    """
    Per-user serving SINR [dB] for one perturbed realisation of the inputs.
    Shapes: uav_xy (U,2), users_xy (K,2), powers_dbm (U,), fading (U,K).
    alpha, G0_db, noise_db are scalars for this realisation.
    Returns (sinr_db (K,), serv (K,), rss_dbm (K,)).
    """
    U = uav_xy.shape[0]
    dx = uav_xy[:, None, 0] - users_xy[None, :, 0]
    dy = uav_xy[:, None, 1] - users_xy[None, :, 1]
    d = np.sqrt(dx*dx + dy*dy + p.altitude**2)                 # (U,K)
    G_db = G0_db - 10.0*alpha*np.log10(np.maximum(d, 1.0))     # large-scale [dB]
    fading_db = 10.0*np.log10(np.maximum(fading, 1e-12)) + fading_err_db
    RX_dbm = powers_dbm[:, None] + G_db + fading_db            # (U,K) received [dBm]
    RX_w = mm.dbm_to_w(RX_dbm)
    k = np.arange(users_xy.shape[0])
    serv = np.argmax(RX_w, axis=0)
    sig = RX_w[serv, k]
    serv_sub = np.asarray(subch)[serv]
    same = (np.asarray(subch)[:, None] == serv_sub[None, :])
    interf = (RX_w*same).sum(0) - sig
    noise_w = mm.dbm_to_w(mm.w_to_dbm(p.noise_power_w) + noise_db)
    sinr = sig/(interf + noise_w)
    return mm.lin_to_db(sinr), serv, mm.w_to_dbm(sig)


# ---------------------------------------------------------------------------
# (A) LPU uncertainty budget for a single representative link's SINR
# ---------------------------------------------------------------------------
def lpu_budget(p: mm.SysParams, iu: InputU, d_s=300.0, d_i=500.0, n_interf=2,
               P_dbm=20.0):
    """
    Analytical (numerical-derivative) GUM budget for the SINR [dB] of a
    representative serving link with `n_interf` equal co-channel interferers.
    Sensitivity coefficients are evaluated by central finite differences.
    """
    G0 = p.G0_db
    noise_dbm = mm.w_to_dbm(p.noise_power_w)

    def sinr_db(P_s, d_serv, alpha, G0_db, g_s_db, noise_db, P_i, d_int, g_i_db):
        g_serv = P_s + (G0_db - 10*alpha*np.log10(d_serv)) + g_s_db     # dBm
        g_int = P_i + (G0_db - 10*alpha*np.log10(d_int)) + g_i_db       # dBm each
        sig_w = mm.dbm_to_w(g_serv)
        int_w = n_interf*mm.dbm_to_w(g_int)
        noise_w = mm.dbm_to_w(noise_dbm + noise_db)
        return mm.lin_to_db(sig_w/(int_w + noise_w))

    # nominal input vector (fading in dB has mean 0 for Exp(1) ~ -2.5 dB median;
    # use 0 dB nominal, its variability enters via u_fading and the MC)
    x0 = dict(P_s=P_dbm, d_serv=d_s, alpha=p.pathloss_exp, G0_db=G0, g_s_db=0.0,
              noise_db=0.0, P_i=P_dbm, d_int=d_i, g_i_db=0.0)
    y0 = sinr_db(**x0)

    # (source label, key, standard uncertainty, +unit)
    sources = [
        ("Transmit power $P$",        "P_s",     iu.u_P_db,   "dB"),
        ("Serving distance $d$",      "d_serv",  None,        "m"),   # from positions
        ("Path-loss exponent "+r"$\alpha$", "alpha", iu.u_alpha, "-"),
        ("Reference gain $G_0$",      "G0_db",   iu.u_G0_db,  "dB"),
        ("Serving fading $g$",        "g_s_db",  iu.u_fading_db, "dB"),
        ("Noise power "+r"$\sigma^2$","noise_db",iu.u_noise_db,"dB"),
        ("Interferer distance $d_I$", "d_int",   None,        "m"),
        ("Interferer fading $g_I$",   "g_i_db",  iu.u_fading_db, "dB"),
    ]
    # distance uncertainties from position uncertainties (RSS of per-axis terms,
    # projected; horizontal separation dominates)
    u_d_serv = np.sqrt(iu.u_pos_uav**2 + iu.u_pos_user**2)   # ~ per-axis combined
    u_d_int = np.sqrt(iu.u_pos_uav**2 + iu.u_pos_user**2)

    rows = []
    var_total = 0.0
    for label, key, u_i, unit in sources:
        if key == "d_serv":
            u_i = u_d_serv
        elif key == "d_int":
            u_i = u_d_int
        # central difference step
        h = max(1e-3, 0.01*abs(x0[key]) if x0[key] != 0 else 0.01)
        xp = dict(x0); xm = dict(x0)
        xp[key] += h; xm[key] -= h
        c = (sinr_db(**xp) - sinr_db(**xm))/(2*h)     # sensitivity coefficient
        contrib = (c*u_i)**2
        var_total += contrib
        rows.append(dict(source=label, nominal=x0[key], unit=unit,
                         u_i=u_i, c_i=c, contribution=np.sqrt(contrib)))
    u_c = np.sqrt(var_total)
    for r in rows:
        r["percent"] = 100.0*(r["contribution"]**2)/var_total
    return dict(y0=y0, u_c=u_c, U95=2*u_c, rows=rows, d_s=d_s, d_i=d_i)


# ---------------------------------------------------------------------------
# (B) Monte-Carlo propagation (GUM Supplement 1) through the full model
# ---------------------------------------------------------------------------
def monte_carlo(p: mm.SysParams, iu: InputU, scen, M=20000, seed=7,
                sigma_est_db=0.0):
    rng = np.random.default_rng(seed)
    U, K = p.n_uav, p.n_users
    gamma = p.gamma_th_db

    # objective-reward weights (documented; consistent with controllers.py)
    W_S, W_P, W_F, LOAD = 5.0, 2.0, 2.0, 10.0

    sinr_all = np.empty((M, K))
    rss_all = np.empty((M, K))
    qos_ratio = np.empty(M)
    fairness = np.empty(M)
    reward = np.empty(M)
    dist_serv = np.empty((M, K))

    p_nom = scen["powers_dbm"]
    for m in range(M):
        uav = scen["uav"] + rng.normal(0, iu.u_pos_uav, size=scen["uav"].shape)
        users = scen["users"] + rng.normal(0, iu.u_pos_user, size=scen["users"].shape)
        powers = p_nom + rng.normal(0, iu.u_P_db, size=U)
        alpha = p.pathloss_exp + rng.normal(0, iu.u_alpha)
        G0 = p.G0_db + rng.normal(0, iu.u_G0_db)
        noise_db = rng.normal(0, iu.u_noise_db)
        fading = rng.exponential(1.0, size=(U, K))            # true Rayleigh power
        fad_err = rng.normal(0, iu.u_fading_db, size=(U, K))  # estimation error
        sinr_db, serv, rss = sinr_ensemble(p, uav, users, powers, scen["subch"],
                                           alpha, G0, noise_db, fading, fad_err)
        # optional receiver SINR-estimation error (imperfect knowledge)
        sinr_meas = sinr_db + (rng.normal(0, sigma_est_db, size=K)
                               if sigma_est_db > 0 else 0.0)
        qos = (sinr_meas >= gamma).astype(float)
        se = np.log2(1.0 + mm.db_to_lin(sinr_db))
        # per-UAV spectral-efficiency load and mean SINR
        load = np.zeros(U); msinr = np.zeros(U)
        for u in range(U):
            msk = serv == u
            if msk.any():
                load[u] = se[msk].sum(); msinr[u] = sinr_db[msk].mean()
        sigma_uav = np.std(load)
        p_norm = mm.dbm_to_w(powers)/p.power_levels_w[-1]

        sinr_all[m] = sinr_db; rss_all[m] = rss; dist_serv[m] = 0.0
        qos_ratio[m] = qos.mean()
        fairness[m] = sigma_uav
        reward[m] = W_S*np.mean(sinr_db) - W_P*np.mean(p_norm) - W_F*sigma_uav/LOAD

    def summ(x):
        x = np.asarray(x)
        return dict(mean=float(np.mean(x)), u=float(np.std(x, ddof=1)),
                    lo=float(np.percentile(x, 2.5)),
                    hi=float(np.percentile(x, 97.5)))
    per_user_sinr = sinr_all.reshape(-1)
    thr = p.bandwidth*np.log2(1.0 + mm.db_to_lin(sinr_all))
    res = dict(
        SINR_db=summ(per_user_sinr),
        RSS_dbm=summ(rss_all.reshape(-1)),
        throughput_Mbps=summ(thr.reshape(-1)/1e6),
        QoS_ratio=summ(qos_ratio),
        Fairness=summ(fairness),
        Reward=summ(reward),
    )
    arrays = dict(sinr=per_user_sinr[:200000], qos=qos_ratio,
                  fairness=fairness, reward=reward)
    return res, arrays


def main():
    p = mm.SysParams()
    # use the calibrated gamma + noise floor from the experiment config if present
    cfgp = os.path.join(OUT, "config.json")
    if os.path.exists(cfgp):
        cfg = json.load(open(cfgp))
        p.gamma_th_db = cfg.get("gamma_th_db", 3.0)
        p.noise_floor_dbm = cfg.get("sys_params", {}).get("noise_floor_dbm")
    else:
        p.gamma_th_db = 3.0
    iu = InputU()
    scen = nominal_scenario(p)

    # (A) LPU budget
    bud = lpu_budget(p, iu)
    import csv
    with open(os.path.join(OUT, "uncertainty_budget_sinr.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "nominal", "unit", "u_i", "c_i (dB/unit)",
                    "u_contribution_dB", "percent"])
        for r in bud["rows"]:
            w.writerow([r["source"], f"{r['nominal']:.4g}", r["unit"],
                        f"{r['u_i']:.4g}", f"{r['c_i']:.4g}",
                        f"{r['contribution']:.4g}", f"{r['percent']:.1f}"])
        w.writerow(["Combined (u_c)", "", "dB", "", "", f"{bud['u_c']:.4g}", "100.0"])
        w.writerow(["Expanded (k=2)", "", "dB", "", "", f"{bud['U95']:.4g}", ""])

    # (B) Monte-Carlo, perfect SINR knowledge
    res, arrays = monte_carlo(p, iu, scen, sigma_est_db=0.0)

    # combine Type B (MC) with Type A repeatability from the dataset
    typeA = dict(SINR_db=0.36, Reward=2.50)   # from dataset_analysis (fallback)
    dsjson = os.path.join(OUT, "dataset_summary.json")
    if os.path.exists(dsjson):
        rep = json.load(open(dsjson)).get("typeA_repeatability", {})
        typeA["SINR_db"] = rep.get("sinr_db", 0.36)
        typeA["Reward"] = rep.get("reward", 2.50)
    res["SINR_db"]["u_combined"] = float(np.hypot(res["SINR_db"]["u"], typeA["SINR_db"]))
    res["SINR_db"]["U_expanded_k2"] = 2*res["SINR_db"]["u_combined"]

    with open(os.path.join(OUT, "uncertainty_measurands.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["measurand", "mean", "u_std", "ci95_lo", "ci95_hi"])
        for name, s in res.items():
            w.writerow([name, f"{s['mean']:.5g}", f"{s['u']:.4g}",
                        f"{s['lo']:.5g}", f"{s['hi']:.5g}"])

    np.savez(os.path.join(OUT, "uncertainty_mc.npz"), **arrays)
    with open(os.path.join(OUT, "uncertainty_summary.json"), "w") as f:
        json.dump(dict(input_u=asdict(iu), lpu=dict(y0=bud["y0"], u_c=bud["u_c"],
                  U95=bud["U95"]), measurands=res, typeA=typeA), f, indent=2)

    print("=== (A) LPU budget for SINR (representative link) ===")
    print(f"  SINR0 = {bud['y0']:.2f} dB,  u_c = {bud['u_c']:.3f} dB,  "
          f"U(k=2) = {bud['U95']:.3f} dB")
    for r in sorted(bud["rows"], key=lambda x: -x["percent"]):
        print(f"    {r['source']:32s} u={r['u_i']:.3g}{r['unit']:>3s}  "
              f"c={r['c_i']:+.3g}  contrib={r['contribution']:.3f}dB  "
              f"({r['percent']:.1f}%)")
    print("\n=== (B) Monte-Carlo propagation (M=20000) ===")
    for name, s in res.items():
        extra = ""
        if "u_combined" in s:
            extra = f"  u_comb={s['u_combined']:.3g}  U(k2)={s['U_expanded_k2']:.3g}"
        print(f"  {name:16s} = {s['mean']:.4g}  u={s['u']:.3g}  "
              f"95%CI=[{s['lo']:.4g}, {s['hi']:.4g}]{extra}")


if __name__ == "__main__":
    main()
