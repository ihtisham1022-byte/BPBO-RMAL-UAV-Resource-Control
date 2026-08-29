"""
mm_core.py
==========
Formal measurement model for micro-level UAV downlink resource control.

This module implements the physical signal chain that maps controllable and
environmental *input quantities* to the *measurands* used by the reinforcement
learning controllers and by the performance evaluation:

    positions ---> distance ---> large-scale gain ---> RSS ---> interference
                                        |                          |
                                     fading -----------------------+---> SINR
                                                                        |
                                        throughput <--- QoS <----------- +
                                             |
                                        fairness, reward

Everything is deterministic given (parameters, RNG seed), so the whole study is
reproducible. The model is written so that the same functions are reused by:
  * run_experiment.py   (dataset generation + controller training/evaluation)
  * uncertainty.py      (GUM LPU + Monte-Carlo propagation)
  * robustness.py       (imperfect-SINR-knowledge sweep)

"""

from __future__ import annotations

__author__ = "Ihtisham Ul Haq"
__creator__ = "Ihtisham Ul Haq"
__maintainer__ = "Ihtisham Ul Haq"
from dataclasses import dataclass, field, asdict
import numpy as np

C_LIGHT = 299_792_458.0  # speed of light [m/s]


# ---------------------------------------------------------------------------
# System parameters (measurement conditions)
# ---------------------------------------------------------------------------
@dataclass
class SysParams:
    # --- deployment geometry ---
    R_disk: float = 600.0        # service-area radius [m]
    n_users: int = 80            # ground users K
    n_uav: int = 3               # UAVs U
    n_subch: int = 4             # subchannels
    n_power: int = 3             # discrete power levels
    altitude: float = 80.0       # UAV altitude H [m]
    uav_speed: float = 40.0      # radial speed [m/s]
    slot_time: float = 0.1       # time-slot duration [s]

    # --- radio parameters ---
    fc: float = 2.0e9            # carrier frequency [Hz]
    bandwidth: float = 180e3     # per-subchannel bandwidth B [Hz] (one PRB)
    p_min_dbm: float = 0.0       # lowest transmit power level [dBm]
    p_max_dbm: float = 23.0      # highest transmit power level [dBm]

    # large-scale channel: G(d) = G0 * d^(-alpha), G0 referenced to 1 m (FSPL)
    pathloss_exp: float = 2.5    # path-loss exponent alpha
    extra_gain_db: float = 0.0   # aggregate antenna gain offset [dB]

    # noise floor: sigma^2 = k T B F, OR an explicit effective noise-plus-external
    # -interference floor [dBm] representing out-of-network interference. When set,
    # it overrides the thermal computation and fixes the operating SINR regime.
    noise_figure_db: float = 7.0
    temperature_k: float = 290.0
    noise_floor_dbm: float = None

    # --- QoS / reward ---
    gamma_th_db: float = 3.0     # SINR QoS threshold [dB] (calibrated at runtime)
    w_sinr: float = 1.0          # reward weight on utility term
    w_fair: float = 1.0          # reward weight on fairness penalty
    w_power: float = 1.0         # reward weight on power penalty

    def __post_init__(self):
        self.init_angles = np.linspace(0.0, 2*np.pi, self.n_uav, endpoint=False)

    # power levels in dBm and Watt
    @property
    def power_levels_dbm(self) -> np.ndarray:
        return np.linspace(self.p_min_dbm, self.p_max_dbm, self.n_power)

    @property
    def power_levels_w(self) -> np.ndarray:
        return dbm_to_w(self.power_levels_dbm)

    # reference gain at 1 m (free-space) folded with antenna gain
    @property
    def G0_db(self) -> float:
        fspl_1m = 20*np.log10(4*np.pi*self.fc/C_LIGHT)  # FSPL at d0 = 1 m
        return -fspl_1m + self.extra_gain_db

    @property
    def noise_power_w(self) -> float:
        if self.noise_floor_dbm is not None:
            return dbm_to_w(self.noise_floor_dbm)
        kB = 1.380649e-23
        n_w = kB*self.temperature_k*self.bandwidth*db_to_lin(self.noise_figure_db)
        return n_w


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------
def dbm_to_w(dbm):
    return 10.0**((np.asarray(dbm, dtype=float) - 30.0)/10.0)

def w_to_dbm(w):
    return 10.0*np.log10(np.maximum(np.asarray(w, dtype=float), 1e-300)) + 30.0

def db_to_lin(db):
    return 10.0**(np.asarray(db, dtype=float)/10.0)

def lin_to_db(x):
    return 10.0*np.log10(np.maximum(np.asarray(x, dtype=float), 1e-300))


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def sample_users(p: SysParams, rng: np.random.Generator) -> np.ndarray:
    """Uniformly distributed ground users in a disk of radius R_disk. (K,2)."""
    theta = rng.uniform(0.0, 2*np.pi, size=p.n_users)
    rad = p.R_disk*np.sqrt(rng.uniform(0.0, 1.0, size=p.n_users))
    return np.c_[rad*np.cos(theta), rad*np.sin(theta)]


def uav_positions(p: SysParams, slot: int) -> np.ndarray:
    """UAVs fly radially inward from the cell edge toward the centre. (U,2)."""
    d_travel = min(p.uav_speed*p.slot_time*slot, p.R_disk)
    pos = np.zeros((p.n_uav, 2))
    for u in range(p.n_uav):
        start = np.array([p.R_disk*np.cos(p.init_angles[u]),
                          p.R_disk*np.sin(p.init_angles[u])])
        nrm = np.linalg.norm(start)
        direction = -start/nrm if nrm > 0 else np.zeros(2)
        pos[u] = start + direction*d_travel
    return pos


def distance_3d(uav_xy: np.ndarray, user_xy: np.ndarray, H: float) -> np.ndarray:
    """3-D UAV-user distance matrix d[u,k] [m] from horizontal positions + H."""
    dx = uav_xy[:, None, 0] - user_xy[None, :, 0]
    dy = uav_xy[:, None, 1] - user_xy[None, :, 1]
    return np.sqrt(dx*dx + dy*dy + H*H)


# ---------------------------------------------------------------------------
# Channel, RSS, interference, SINR  (the core measurement equations)
# ---------------------------------------------------------------------------
def large_scale_gain(p: SysParams, d: np.ndarray) -> np.ndarray:
    """Deterministic power gain G(d) = G0 * d^(-alpha) [linear]. Eq. (channel)."""
    return db_to_lin(p.G0_db)*np.power(np.maximum(d, 1.0), -p.pathloss_exp)


def channel_gain(p: SysParams, d: np.ndarray, fading: np.ndarray) -> np.ndarray:
    """h = G(d) * |g|^2, with |g|^2 the small-scale (Rayleigh) power fading."""
    return large_scale_gain(p, d)*fading


def rayleigh_fading(shape, rng: np.random.Generator) -> np.ndarray:
    """Rayleigh small-scale fading power |g|^2 ~ Exponential(mean 1)."""
    return rng.exponential(scale=1.0, size=shape)


def received_power(p_tx_w, h):
    """RSS in Watt: P_rx = P_tx * h.  Eq. (received_power)."""
    return np.asarray(p_tx_w)*np.asarray(h)


def sinr_linear(p_serv_w, h_serv, interference_w, noise_w):
    """SINR = (P* h*) / (sum_interf + noise).  Eq. (sinr)."""
    return (p_serv_w*h_serv)/(interference_w + noise_w)


# ---------------------------------------------------------------------------
# Performance measurands
# ---------------------------------------------------------------------------
def throughput(p: SysParams, sinr_lin) -> np.ndarray:
    """Shannon throughput B log2(1+SINR) [bit/s]. Eq. (throughput)."""
    return p.bandwidth*np.log2(1.0 + np.asarray(sinr_lin))


def qos_indicator(p: SysParams, sinr_lin) -> np.ndarray:
    """Binary QoS satisfaction 1[SINR >= gamma_th]. Eq. (qos)."""
    return (np.asarray(sinr_lin) >= db_to_lin(p.gamma_th_db)).astype(float)


def fairness_std(per_uav_load: np.ndarray) -> float:
    """Load imbalance = std of per-UAV throughput/load. Eq. (fairness)."""
    return float(np.std(np.asarray(per_uav_load)))


# --- standard fairness indicators (supporting the fairness-penalty metric) ---
def jain_index(load: np.ndarray) -> float:
    """Jain's fairness index J = (sum x)^2 / (U sum x^2), in (0,1]; higher=fairer."""
    x = np.asarray(load, dtype=float)
    s2 = np.sum(x)**2
    d = len(x)*np.sum(x*x)
    return float(s2/d) if d > 0 else 1.0


def gini_coefficient(load: np.ndarray) -> float:
    """Gini coefficient in [0,1]; lower = fairer."""
    x = np.asarray(load, dtype=float)
    if np.all(x <= 0):
        return 0.0
    U = len(x)
    diff = np.abs(x[:, None] - x[None, :]).sum()
    return float(diff/(2*U*np.sum(x))) if np.sum(x) > 0 else 0.0


def entropy_fairness(load: np.ndarray) -> float:
    """Normalised Shannon entropy of the load share in [0,1]; higher = fairer."""
    x = np.asarray(load, dtype=float)
    x = np.clip(x, 0, None)
    tot = x.sum()
    if tot <= 0 or len(x) <= 1:
        return 1.0
    p = x/tot
    p = p[p > 0]
    return float(-np.sum(p*np.log(p))/np.log(len(x)))


# ---------------------------------------------------------------------------
# Imperfect SINR knowledge
# ---------------------------------------------------------------------------
def estimate_sinr_db(true_sinr_db, sigma_est_db, rng: np.random.Generator,
                     bias_db: float = 0.0):
    """
    Model the *measured* SINR as an estimate of the true SINR.

    A practical receiver estimates SINR from a finite number of pilot/reference
    symbols; the resulting estimate is well modelled, in the logarithmic domain,
    as Gaussian around the true value:

        SINR_hat_dB = SINR_dB + b + e,   e ~ N(0, sigma_est_dB^2)

    where sigma_est_dB decreases with the number of averaged pilots
    (sigma_est ∝ 1/sqrt(N_pilot)) and b is a residual estimation bias.
    Setting sigma_est_dB = 0 recovers the perfect-knowledge assumption used in
    the reference formulation.
    """
    true_sinr_db = np.asarray(true_sinr_db, dtype=float)
    noise = rng.normal(0.0, sigma_est_db, size=true_sinr_db.shape) if sigma_est_db > 0 else 0.0
    return true_sinr_db + bias_db + noise


def qos_misclassification_prob(true_sinr_db, gamma_th_db, sigma_est_db):
    """
    Analytical probability that an imperfect SINR estimate flips the QoS
    decision at threshold gamma_th, given Gaussian dB-domain estimation error.
    P(misclassify) = Phi(-(|SINR-gamma|)/sigma_est) = 0.5*erfc(|Δ|/(sqrt2 sigma)).
    """
    from math import erfc, sqrt
    d = np.abs(np.asarray(true_sinr_db, dtype=float) - gamma_th_db)
    if sigma_est_db <= 0:
        return np.zeros_like(d)
    vfun = np.vectorize(lambda x: 0.5*erfc(x/(sqrt(2.0)*sigma_est_db)))
    return vfun(d)


# ---------------------------------------------------------------------------
# One full physical evaluation of a slot given chosen actions
# ---------------------------------------------------------------------------
def evaluate_slot_assoc(p: SysParams, uav_xy, users_xy, powers_w, subch, fading):
    """
    Faithful slot evaluation with max-RSS user association (Eq. association) and
    subchannel-aware interference.

    Each UAV u transmits at power powers_w[u] on subchannel subch[u]. A user
    associates with the UAV giving the strongest received power; interference at
    that user comes only from *other* UAVs sharing the serving UAV's subchannel.
    UAVs on orthogonal subchannels do not interfere, so spreading across
    subchannels is a genuinely learnable interference-avoidance action.

    Inputs
      uav_xy   : (U,2) UAV horizontal positions
      users_xy : (K,2) user positions
      powers_w : (U,)  transmit power per UAV [W]
      subch    : (U,)  subchannel index per UAV
      fading   : (U,K) small-scale power fading |g|^2
    """
    subch = np.asarray(subch)
    d = distance_3d(uav_xy, users_xy, p.altitude)         # (U,K)
    H = large_scale_gain(p, d)*fading                     # channel gain (U,K)
    RX = np.asarray(powers_w)[:, None]*H                  # received power (U,K)
    k_idx = np.arange(p.n_users)
    serv = np.argmax(RX, axis=0)                          # serving UAV per user
    sig = RX[serv, k_idx]                                 # desired power (K,)
    serv_sub = subch[serv]                                # (K,)
    same_sub = (subch[:, None] == serv_sub[None, :])      # (U,K) co-channel mask
    interf = (RX*same_sub).sum(axis=0) - sig              # co-channel interf (K,)
    sinr = sig/(interf + p.noise_power_w)                 # (K,)
    return dict(sinr_lin=sinr, sinr_db=lin_to_db(sinr), serv=serv,
                serv_sub=serv_sub, rss_dbm=w_to_dbm(sig), interf_w=interf,
                distance=d[serv, k_idx],
                throughput=throughput(p, sinr), qos=qos_indicator(p, sinr))


def evaluate_slot(p: SysParams, uav_xy, users_xy, chosen_user, chosen_subch,
                  chosen_pidx, fading_serv, fading_interf):
    """
    Compute the true per-UAV SINR, throughput and QoS for one time slot.

    Each UAV u serves its chosen_user[u] on subchannel chosen_subch[u] at power
    level chosen_pidx[u]. Interference at that user comes from every *other* UAV
    transmitting on the same subchannel. Returns dict of per-UAV arrays.
    """
    U = p.n_uav
    pw = p.power_levels_w
    noise_w = p.noise_power_w

    # distances from every UAV to every chosen user (needed for interference)
    d_all = distance_3d(uav_xy, users_xy, p.altitude)     # (U, K_users)

    sinr_lin = np.zeros(U)
    rss_w = np.zeros(U)
    interf_w = np.zeros(U)
    for u in range(U):
        k = chosen_user[u]
        g_serv = large_scale_gain(p, d_all[u, k])*fading_serv[u]
        p_serv = pw[chosen_pidx[u]]*g_serv
        rss_w[u] = p_serv

        I = 0.0
        for j in range(U):
            if j == u:
                continue
            if chosen_subch[j] == chosen_subch[u]:
                g_j = large_scale_gain(p, d_all[j, k])*fading_interf[u, j]
                I += pw[chosen_pidx[j]]*g_j
        interf_w[u] = I
        sinr_lin[u] = p_serv/(I + noise_w)

    thr = throughput(p, sinr_lin)
    qos = qos_indicator(p, sinr_lin)
    return dict(sinr_lin=sinr_lin, sinr_db=lin_to_db(sinr_lin),
                rss_w=rss_w, rss_dbm=w_to_dbm(rss_w),
                interf_w=interf_w, throughput=thr, qos=qos,
                distance=np.array([d_all[u, chosen_user[u]] for u in range(U)]))


if __name__ == "__main__":
    # quick self-test / calibration probe
    p = SysParams()
    rng = np.random.default_rng(0)
    users = sample_users(p, rng)
    print("G0_db =", round(p.G0_db, 2), "dB   noise =",
          round(w_to_dbm(p.noise_power_w), 2), "dBm")
    sinrs = []
    for slot in range(0, 80, 8):
        uav = uav_positions(p, slot)
        cu = rng.integers(0, p.n_users, size=p.n_uav)
        cs = rng.integers(0, p.n_subch, size=p.n_uav)
        cp = np.full(p.n_uav, p.n_power-1)
        fs = rayleigh_fading(p.n_uav, rng)
        fi = rayleigh_fading((p.n_uav, p.n_uav), rng)
        out = evaluate_slot(p, uav, users, cu, cs, cp, fs, fi)
        sinrs.append(out["sinr_db"])
    sinrs = np.concatenate(sinrs)
    print("SINR dB: min/median/max = %.1f / %.1f / %.1f" %
          (sinrs.min(), np.median(sinrs), sinrs.max()))
