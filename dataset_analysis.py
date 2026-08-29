"""
dataset_analysis.py
===================
Characterise the *provided* controlled-simulation dataset
(rmal_micro_manuscript_grade.xlsx) that underpins the study, and extract:

  * transparent provenance and structural statistics;
  * genuine Baseline-RMAL performance metrics per exploration setting, computed
    with the manuscript's own definitions (reproducible from the raw data);
  * Type A (repeatability) standard uncertainties of the measurands, estimated
    from the replicate episodes at matched conditions (GUM 4.2);
  * empirical SINR / distance / reward distributions used to validate the
    reproducible simulator.

Outputs: outputs/dataset_summary.json, outputs/dataset_baseline_metrics.csv,
         outputs/dataset_distributions.npz
"""
from __future__ import annotations

__author__ = "Ihtisham Ul Haq"
__creator__ = "Ihtisham Ul Haq"
__maintainer__ = "Ihtisham Ul Haq"
import os, json, glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
GAMMA_TH = 3.0   # QoS threshold [dB] used in the reference dataset


def find_dataset():
    candidates = [os.path.join(HERE, "rmal_micro_manuscript_grade.xlsx")]
    env_path = os.environ.get("RMAL_DATASET_PATH")
    if env_path:
        candidates.insert(0, env_path)
    for c in candidates:
        if os.path.exists(c):
            return c
    hits = glob.glob(os.path.join(HERE, "..", "**", "rmal_micro*.xlsx"),
                     recursive=True)
    return hits[0] if hits else None


def episode_metrics(g: pd.DataFrame, gamma=GAMMA_TH):
    """Reference performance metrics for one (episode, epsilon) group."""
    qos = float((g.sinr_db >= gamma).mean())
    tmin, tmax = g.time_slot.min(), g.time_slot.max()
    tcut = tmax - 0.15*(tmax - tmin)
    R_tail = float(g[g.time_slot >= tcut].reward.mean())
    P_fair = float(g.groupby("uav_id").reward.sum().std())
    return dict(qos=qos, R_tail=R_tail, P_fair=P_fair,
                mean_sinr=float(g.sinr_db.mean()))


def main():
    path = find_dataset()
    if path is None:
        raise FileNotFoundError("rmal_micro_manuscript_grade.xlsx not found")
    df = pd.read_excel(path)
    # normalise a copy into the reproducible folder for packaging
    df.to_csv(os.path.join(OUT, "provided_dataset.csv"), index=False)

    summary = dict(
        source_file=os.path.basename(path), n_rows=int(df.shape[0]),
        columns=list(df.columns),
        n_episodes=int(df.episode.nunique()),
        epsilons=sorted(map(float, df.epsilon.unique())),
        n_uav=int(df.uav_id.nunique()), n_users=int(df.user_id.nunique()),
        n_subchannels=int(df.subchannel.nunique()),
        power_levels_dbm=sorted(map(float, df.power_level_dbm.unique())),
        time_slots=int(df.time_slot.nunique()),
        time_slot_range=[int(df.time_slot.min()), int(df.time_slot.max())],
        distance_m=dict(min=float(df.distance_m.min()),
                        max=float(df.distance_m.max()),
                        mean=float(df.distance_m.mean()),
                        std=float(df.distance_m.std())),
        sinr_db=dict(min=float(df.sinr_db.min()), max=float(df.sinr_db.max()),
                     mean=float(df.sinr_db.mean()), std=float(df.sinr_db.std())),
        reward=dict(min=float(df.reward.min()), max=float(df.reward.max()),
                    mean=float(df.reward.mean()), std=float(df.reward.std())),
        overall_qos_ratio=float((df.sinr_db >= GAMMA_TH).mean()),
    )

    # --- per-epsilon baseline metrics + Type A repeatability across episodes ---
    rows = []
    for eps, ge in df.groupby("epsilon"):
        per_ep = [episode_metrics(gep) for _, gep in ge.groupby("episode")]
        n = len(per_ep)
        agg = {}
        for key in ["qos", "R_tail", "P_fair", "mean_sinr"]:
            vals = np.array([m[key] for m in per_ep], float)
            agg[key] = float(vals.mean())
            # Type A standard uncertainty of the mean = s / sqrt(n)
            agg["u_"+key] = float(vals.std(ddof=1)/np.sqrt(n)) if n > 1 else 0.0
        agg["epsilon"] = float(eps); agg["n_episodes"] = n
        rows.append(agg)
    base = pd.DataFrame(rows)[["epsilon", "n_episodes", "R_tail", "u_R_tail",
                               "qos", "u_qos", "P_fair", "u_P_fair",
                               "mean_sinr", "u_mean_sinr"]]
    base.to_csv(os.path.join(OUT, "dataset_baseline_metrics.csv"), index=False)

    # Best operating point (max tail reward) as the representative baseline
    best = base.loc[base.R_tail.idxmax()].to_dict()
    summary["baseline_best_operating_point"] = {k: float(v) for k, v in best.items()}

    # --- Type A repeatability pooled across all matched conditions ---
    # within-condition (episode-to-episode) pooled std of the per-slot SINR
    def pooled_repeatability(col):
        resid = []
        for _, g in df.groupby(["epsilon", "time_slot", "uav_id"]):
            if len(g) > 1:
                resid.append(g[col].values - g[col].mean())
        resid = np.concatenate(resid) if resid else np.array([0.0])
        return float(np.std(resid, ddof=1))
    summary["typeA_repeatability"] = dict(
        sinr_db=pooled_repeatability("sinr_db"),
        reward=pooled_repeatability("reward"),
        distance_m=pooled_repeatability("distance_m"))

    with open(os.path.join(OUT, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    np.savez(os.path.join(OUT, "dataset_distributions.npz"),
             sinr_db=df.sinr_db.values, distance_m=df.distance_m.values,
             reward=df.reward.values, power_dbm=df.power_level_dbm.values)

    print("=== Reference dataset ===")
    print(f"  file: {summary['source_file']}  rows: {summary['n_rows']}")
    print(f"  {summary['n_uav']} UAVs, {summary['n_users']} users, "
          f"{summary['n_subchannels']} subchannels, "
          f"powers {summary['power_levels_dbm']} dBm")
    print(f"  epsilons: {summary['epsilons']}, {summary['n_episodes']} episodes")
    print(f"  SINR {summary['sinr_db']['min']:.2f}..{summary['sinr_db']['max']:.2f} dB "
          f"(mean {summary['sinr_db']['mean']:.2f})")
    print(f"  overall QoS ratio (>= {GAMMA_TH} dB): {summary['overall_qos_ratio']:.4f}")
    print("\n=== Per-epsilon baseline metrics (mean +/- Type A u) ===")
    for _, r in base.iterrows():
        print(f"  eps={r.epsilon:.2f}: R_tail={r.R_tail:6.2f}+/-{r.u_R_tail:4.2f}  "
              f"QoS={r.qos:.3f}+/-{r.u_qos:.3f}  "
              f"P_fair={r.P_fair:6.2f}+/-{r.u_P_fair:5.2f}")
    print(f"\n  representative baseline (max R_tail) at eps={best['epsilon']:.2f}: "
          f"R_tail={best['R_tail']:.2f}, QoS={best['qos']:.3f}, "
          f"P_fair={best['P_fair']:.2f}")
    print(f"  Type A repeatability: u(SINR)={summary['typeA_repeatability']['sinr_db']:.3f} dB, "
          f"u(reward)={summary['typeA_repeatability']['reward']:.3f}")


if __name__ == "__main__":
    main()
