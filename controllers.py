"""
controllers.py
==============
Decentralised RL controllers evaluated in the *same* physically-grounded UAV
environment (mm_core). All three controllers share the identical environment,
QoS threshold and reward definition, so the comparison is fair and reproducible.

  * Baseline RMAL      : reward-modulated epsilon-greedy tabular Q-learning,
                         fixed hyperparameters.
  * BPBO-RMAL          : Beta-Parameter Bounded Optimisation of the RMAL
                         hyperparameter vector theta (Beta-distributed move
                         toward the incumbent + Gaussian exploration).
  * Q-learning + HHO   : Harris-Hawks Optimisation of a Q-learning schedule
                         (learning-rate + exploration), same environment/fitness.

The environment returns full per-link logs so that run_experiment.py can emit a
transparent controlled-simulation dataset.
"""

from __future__ import annotations

__author__ = "Ihtisham Ul Haq"
__creator__ = "Ihtisham Ul Haq"
__maintainer__ = "Ihtisham Ul Haq"
import numpy as np
from dataclasses import dataclass, replace
import mm_core as mm


# ---------------------------------------------------------------------------
# Hyperparameter containers
# ---------------------------------------------------------------------------
RMAL_KEYS = ["eps_init", "eps_min", "eps_decay", "C_beta", "phi_beta",
             "w_sinr", "w_fair", "w_power"]

RMAL_BOUNDS = {
    "eps_init":  (0.05, 1.00),
    "eps_min":   (0.00, 0.30),
    "eps_decay": (0.90, 0.999),
    "C_beta":    (0.30, 3.00),
    "phi_beta":  (0.10, 0.90),
    "w_sinr":    (0.50, 2.00),
    "w_fair":    (0.20, 2.00),
    "w_power":   (0.20, 2.00),
}

RMAL_BASELINE = dict(eps_init=0.50, eps_min=0.10, eps_decay=0.99,
                     C_beta=1.00, phi_beta=0.50,
                     w_sinr=1.0, w_fair=1.0, w_power=1.0)

QHHO_KEYS = ["alpha_init", "alpha_min", "alpha_decay", "gamma",
             "eps_init", "eps_min", "eps_decay"]

QHHO_BOUNDS = {
    "alpha_init":  (0.05, 1.00),
    "alpha_min":   (0.001, 0.20),
    "alpha_decay": (0.90, 0.999),
    "gamma":       (0.80, 0.999),
    "eps_init":    (0.10, 1.00),
    "eps_min":     (0.01, 0.50),
    "eps_decay":   (0.90, 0.999),
}


# ---------------------------------------------------------------------------
# Reward reference scales for the fitness index (fixed, documented constants)
# ---------------------------------------------------------------------------
@dataclass
class Refs:
    # Reference scales for the fitness index, set to the baseline-RMAL operating
    # magnitudes so that reward, QoS and fairness all contribute comparably and
    # the optimiser cannot ignore the fairness objective.
    R_ref: float = 15.0      # reference tail (objective) reward
    F_ref: float = 6.0       # reference fairness penalty (per-UAV imbalance)

# Fixed, documented weights for the OBJECTIVE evaluation reward and fairness.
# They are NOT tuned by any optimiser, so the fitness cannot be gamed by
# dropping the shaping penalties. Chosen so the tail reward lands on the same
# O(10) scale as the provided controlled-simulation dataset.
W_S_EVAL = 5.0     # objective weight on mean SINR [dB]
W_P_EVAL = 2.0     # objective weight on normalised transmit power
W_F_EVAL = 2.0     # objective weight on load imbalance
LOAD_SCALE = 10.0  # normalisation of the per-UAV spectral-efficiency load


def fitness(R_tail, qos, P_fair, refs: Refs):
    """Expert fitness F = 0.5 R_tail/R_ref + 0.3 QoS - 0.2 P_fair/F_ref."""
    return 0.5*(R_tail/refs.R_ref) + 0.3*qos - 0.2*(P_fair/refs.F_ref)


# ---------------------------------------------------------------------------
# Core episode roll-out (shared by all controllers)
# ---------------------------------------------------------------------------
def _state_of(sinr_db, edges):
    """Discretise a UAV's own SINR (dB) into a small state index."""
    return int(np.digitize(sinr_db, edges))


def run_episode(p: mm.SysParams, theta: dict, rng: np.random.Generator,
                *, T=160, n_states=4, alpha_sched=None, gamma_q=0.9,
                reward_modulated=True, Q_init=None, log=False, users_seed=2024,
                sigma_est_db=0.0):
    """
    Run one episode of decentralised tabular Q-learning in the UAV environment.

    theta supplies the exploration schedule (eps_init/min/decay), the RMAL
    learning-rate schedule (C_beta, phi_beta) and the reward weights
    (w_sinr, w_fair, w_power). If alpha_sched=(a_init,a_min,a_decay) is given
    (Q-learning+HHO), a per-slot learning rate schedule is used instead of the
    RMAL 1/((t+C_beta)^phi_beta) rule. Returns metrics and (optionally) logs.
    """
    U = p.n_uav
    A = p.n_subch*p.n_power             # action = (subchannel, power level)
    a_sub = np.repeat(np.arange(p.n_subch), p.n_power)   # decode: subchannel
    a_pidx = np.tile(np.arange(p.n_power), p.n_subch)    # decode: power level
    edges = p.gamma_th_db + np.array([-6.0, 0.0, 6.0])[:n_states-1]  # state bins

    # Fixed user deployment (measurement condition): geometry is held constant
    # across episodes/controllers so that metric variability reflects channel
    # fading and policy exploration, not a re-randomised layout.
    users_xy = mm.sample_users(p, np.random.default_rng(users_seed))
    Q = [np.zeros((n_states, A)) for _ in range(U)] if Q_init is None \
        else [q.copy() for q in Q_init]
    state = np.zeros(U, dtype=int)

    eps = theta["eps_init"]
    agg_reward_ts = np.zeros(T)
    qos_hits = qos_tot = 0
    cum_load = np.zeros(U)              # per-UAV cumulative SE-load (for fairness)
    logs = [] if log else None

    for t in range(T):
        uav_xy = mm.uav_positions(p, t)
        if alpha_sched is None:
            lr = 1.0/((t + theta["C_beta"])**theta["phi_beta"])   # RMAL rule
        else:
            a_init, a_min, a_dec = alpha_sched
            lr = max(a_min, a_init*(a_dec**t))

        # epsilon-greedy power-level selection per UAV
        act = np.zeros(U, dtype=int)
        for u in range(U):
            if rng.random() < eps:
                act[u] = rng.integers(0, A)
            else:
                act[u] = int(np.argmax(Q[u][state[u]]))
        sub = a_sub[act]
        powers_w = p.power_levels_w[a_pidx[act]]
        p_norm = powers_w/p.power_levels_w[-1]

        # physical slot evaluation with association + subchannel-aware interference
        fading = mm.rayleigh_fading((U, p.n_users), rng)
        out = mm.evaluate_slot_assoc(p, uav_xy, users_xy, powers_w, sub, fading)
        sinr_db, serv, se = out["sinr_db"], out["serv"], np.log2(1.0+out["sinr_lin"])

        # imperfect SINR knowledge: the controller OBSERVES a noisy estimate and
        # bases its state and shaped reward on it; the true SINR still governs the
        # actually-experienced throughput and QoS.
        if sigma_est_db > 0:
            sinr_obs = sinr_db + rng.normal(0.0, sigma_est_db, size=sinr_db.shape)
        else:
            sinr_obs = sinr_db
        se_obs = np.log2(1.0 + mm.db_to_lin(sinr_obs))

        # per-UAV spectral-efficiency load, mean observed SE and mean observed SINR
        load = np.zeros(U); se_u = np.zeros(U); msinr_u = np.full(U, edges[0]-6)
        for u in range(U):
            m = serv == u
            if m.any():
                load[u] = se[m].sum()               # true SE-load (fairness)
                se_u[u] = se_obs[m].mean()           # observed (drives learning)
                msinr_u[u] = sinr_obs[m].mean()      # observed (drives state)
        sigma_uav = float(np.std(load))

        # (a) SHAPED per-UAV reward (depends on theta): drives the Q-update
        r_learn = (theta["w_sinr"]*se_u
                   - theta["w_power"]*p_norm
                   - theta["w_fair"]*sigma_uav/LOAD_SCALE)
        # (b) OBJECTIVE per-UAV reward (fixed weights) and SE-load for fairness
        cum_load += load
        r_eval = float(W_S_EVAL*np.mean(sinr_db) - W_P_EVAL*np.mean(p_norm)
                       - W_F_EVAL*sigma_uav/LOAD_SCALE)
        agg_reward_ts[t] = r_eval

        # next state (per-UAV mean SINR bucket) + Q-update on the shaped reward
        nstate = np.array([_state_of(msinr_u[u], edges) for u in range(U)])
        for u in range(U):
            s, a, sn = state[u], act[u], nstate[u]
            Q[u][s, a] += lr*(r_learn[u] + gamma_q*np.max(Q[u][sn]) - Q[u][s, a])
        state = nstate

        qos_hits += float(out["qos"].sum()); qos_tot += p.n_users

        # reward-modulated exploration (RMAL) or scheduled decay (Q-learning)
        if reward_modulated:
            phi = 1.0/(1.0 + np.exp(-np.mean(r_learn)))     # phi_beta in (0,1)
            eps = max(theta["eps_min"], eps*np.exp(-theta["C_beta"]*phi*0.05))
        else:
            eps = max(theta["eps_min"], eps*theta["eps_decay"])

        if log:
            # log one representative served link per UAV (transparent micro-dataset)
            for u in range(U):
                m = np.where(serv == u)[0]
                if len(m) == 0:
                    continue
                k = int(m[rng.integers(len(m))])
                logs.append((t, u, k, int(sub[u]),
                             float(p.power_levels_dbm[a_pidx[act[u]]]),
                             float(out["distance"][k]), float(sinr_db[k]),
                             float(out["rss_dbm"][k]), float(out["qos"][k]),
                             float(W_S_EVAL*se_u[u] - W_P_EVAL*p_norm[u])))

    tail = max(1, int(0.15*T))
    load_avg = cum_load/T
    metrics = dict(
        R_tail=float(agg_reward_ts[-tail:].mean()),
        qos=float(qos_hits/qos_tot),
        # load imbalance = std across UAVs of time-averaged SE-load (T-invariant)
        P_fair=float(np.std(load_avg)),
        # standard fairness indicators (support the fairness-penalty metric)
        jain=mm.jain_index(load_avg),
        gini=mm.gini_coefficient(load_avg),
        entropy=mm.entropy_fairness(load_avg),
        reward_ts=agg_reward_ts,
    )
    return (metrics, Q, logs) if log else (metrics, Q)


def eval_policy(p, theta, Q, rng, *, T=80, n_states=4, alpha_sched=None,
                gamma_q=0.9, users_seed=2024, perturb=None):
    """
    Evaluate a FROZEN policy (Q-table) for one trial, optionally propagating
    measurement uncertainty by sampling the true input quantities from their
    distributions (GUM Supplement 1). `perturb` is a dict with standard
    uncertainties: u_pos, u_P_db, u_alpha, u_G0_db, u_noise_db, sigma_est_db.
    The frozen policy acts on the observed SINR; metrics use the true SINR.
    Returns a metrics dict (no learning is performed).
    """
    U = p.n_uav
    A = p.n_subch*p.n_power
    a_sub = np.repeat(np.arange(p.n_subch), p.n_power)
    a_pidx = np.tile(np.arange(p.n_power), p.n_subch)
    edges = p.gamma_th_db + np.array([-6.0, 0.0, 6.0])[:n_states-1]
    pe = perturb or {}
    u_pos = pe.get("u_pos", 0.0); u_P = pe.get("u_P_db", 0.0)
    sig_est = pe.get("sigma_est_db", 0.0)

    # per-trial perturbation of the scalar input quantities (Type B)
    pp = p
    if pe:
        pp = replace(p,
                     pathloss_exp=p.pathloss_exp + rng.normal(0, pe.get("u_alpha", 0.0)),
                     extra_gain_db=p.extra_gain_db + rng.normal(0, pe.get("u_G0_db", 0.0)),
                     noise_floor_dbm=(None if p.noise_floor_dbm is None
                                      else p.noise_floor_dbm + rng.normal(0, pe.get("u_noise_db", 0.0))))
    users0 = mm.sample_users(pp, np.random.default_rng(users_seed))
    state = np.zeros(U, dtype=int)
    eps = theta.get("eps_min", 0.02)          # near-greedy evaluation
    qos_hits = qos_tot = 0
    agg = np.zeros(T); cum_load = np.zeros(U)

    for t in range(T):
        uav_xy = mm.uav_positions(pp, t)
        if u_pos > 0:
            uav_xy = uav_xy + rng.normal(0, u_pos, size=uav_xy.shape)
            users_xy = users0 + rng.normal(0, u_pos, size=users0.shape)
        else:
            users_xy = users0
        act = np.array([int(np.argmax(Q[u][state[u]])) if rng.random() >= eps
                        else rng.integers(0, A) for u in range(U)])
        sub = a_sub[act]
        pw = pp.power_levels_w[a_pidx[act]]
        if u_P > 0:
            pw = mm.dbm_to_w(mm.w_to_dbm(pw) + rng.normal(0, u_P, size=U))
        p_norm = pw/pp.power_levels_w[-1]
        fading = mm.rayleigh_fading((U, pp.n_users), rng)
        out = mm.evaluate_slot_assoc(pp, uav_xy, users_xy, pw, sub, fading)
        sinr_db, serv = out["sinr_db"], out["serv"]
        sinr_obs = sinr_db + (rng.normal(0, sig_est, size=sinr_db.shape) if sig_est > 0 else 0.0)
        se = np.log2(1.0 + out["sinr_lin"])
        load = np.zeros(U); msinr = np.full(U, edges[0]-6)
        for u in range(U):
            m = serv == u
            if m.any():
                load[u] = se[m].sum(); msinr[u] = sinr_obs[m].mean()
        cum_load += load
        sigma_uav = float(np.std(load))
        agg[t] = W_S_EVAL*np.mean(sinr_db) - W_P_EVAL*np.mean(p_norm) - W_F_EVAL*sigma_uav/LOAD_SCALE
        qos_hits += float(out["qos"].sum()); qos_tot += pp.n_users
        state = np.array([_state_of(msinr[u], edges) for u in range(U)])

    tail = max(1, int(0.15*T)); load_avg = cum_load/T
    return dict(R_tail=float(agg[-tail:].mean()), qos=float(qos_hits/qos_tot),
                P_fair=float(np.std(load_avg)), jain=mm.jain_index(load_avg),
                gini=mm.gini_coefficient(load_avg), entropy=mm.entropy_fairness(load_avg))


def train_policy(p, theta, seed=1, alpha_sched=None, gamma_q=0.9,
                 reward_modulated=True):
    """Train a controller once and return its frozen Q-table."""
    _, Q = run_episode(p, theta, np.random.default_rng(seed),
                       alpha_sched=alpha_sched, gamma_q=gamma_q,
                       reward_modulated=reward_modulated)
    return Q


def _agg_eval(rows, refs):
    """Aggregate per-seed metric dicts into mean/std incl. fairness indices."""
    def col(k):
        return np.array([r[k] for r in rows], float)
    R, Q, Fp = col("R_tail"), col("qos"), col("P_fair")
    Fi = np.array([fitness(r["R_tail"], r["qos"], r["P_fair"], refs) for r in rows])
    out = dict(R_tail=R.mean(), qos=Q.mean(), P_fair=Fp.mean(), F=Fi.mean(),
               R_tail_s=R.std(), qos_s=Q.std(), P_fair_s=Fp.std(), F_s=Fi.std())
    for k in ("jain", "gini", "entropy"):
        if k in rows[0]:
            v = col(k); out[k] = v.mean(); out[k+"_s"] = v.std()
    return out


def evaluate_theta(p, theta, refs, seeds, **kw):
    """Average metrics + fitness of an RMAL theta over several seeds."""
    rows = [run_episode(p, theta, np.random.default_rng(sd), **kw)[0] for sd in seeds]
    return _agg_eval(rows, refs)


# ---------------------------------------------------------------------------
# BPBO: Beta-Parameter Bounded Optimisation over the RMAL theta
# ---------------------------------------------------------------------------
def _clip(theta, bounds):
    return {k: float(np.clip(theta[k], *bounds[k])) for k in theta}


def bpbo_optimise(p, refs, *, n_birds=10, n_iter=18, eval_seeds=(1, 2, 3),
                  beta_a=2.0, beta_b=2.0, sigma=0.15, seed=100, verbose=False):
    """
    Beta-Parameter Bounded Optimisation. Each particle is an RMAL theta; the
    move rule is theta <- theta + beta*(theta_best - theta) + (1-beta)*N(0,sigma),
    with beta ~ Beta(a,b) drawn per particle (Eq. particle evolution).
    """
    rng = np.random.default_rng(seed)
    pop = []
    for _ in range(n_birds):
        th = {k: rng.uniform(*RMAL_BOUNDS[k]) for k in RMAL_KEYS}
        pop.append(th)
    pop[0] = dict(RMAL_BASELINE)   # seed one particle at the baseline
    span = {k: RMAL_BOUNDS[k][1]-RMAL_BOUNDS[k][0] for k in RMAL_KEYS}

    def fit(th):
        e = evaluate_theta(p, th, refs, eval_seeds)
        return e["F"], e
    scored = [fit(th) for th in pop]
    F_hist = []
    best_i = int(np.argmax([s[0] for s in scored]))
    best_th, best_F = dict(pop[best_i]), scored[best_i][0]

    for it in range(n_iter):
        for i in range(n_birds):
            beta = rng.beta(beta_a, beta_b)
            th_new = {}
            for k in RMAL_KEYS:
                gauss = rng.normal(0.0, sigma*span[k])
                th_new[k] = pop[i][k] + beta*(best_th[k]-pop[i][k]) + (1-beta)*gauss
            th_new = _clip(th_new, RMAL_BOUNDS)
            F_new, e_new = fit(th_new)
            if F_new > scored[i][0]:      # greedy retention
                pop[i], scored[i] = th_new, (F_new, e_new)
                if F_new > best_F:
                    best_F, best_th = F_new, dict(th_new)
        F_hist.append(best_F)
        if verbose:
            print(f"  BPBO it {it+1:02d}: best F = {best_F:.4f}")
    return best_th, best_F, np.array(F_hist)


# ---------------------------------------------------------------------------
# HHO: Harris-Hawks Optimisation over a Q-learning schedule
# ---------------------------------------------------------------------------
def _qhho_theta_to_rmal(tq):
    """Embed a Q-learning schedule into the shared environment's theta."""
    return dict(eps_init=tq["eps_init"], eps_min=tq["eps_min"],
                eps_decay=tq["eps_decay"], C_beta=1.0, phi_beta=0.5,
                w_sinr=1.0, w_fair=1.0, w_power=1.0)


def evaluate_qhho(p, tq, refs, seeds, **kw):
    th = _qhho_theta_to_rmal(tq)
    rows = [run_episode(p, th, np.random.default_rng(sd),
                        alpha_sched=(tq["alpha_init"], tq["alpha_min"], tq["alpha_decay"]),
                        gamma_q=tq["gamma"], reward_modulated=False, **kw)[0]
            for sd in seeds]
    return _agg_eval(rows, refs)


def hho_optimise(p, refs, *, n_hawks=10, n_iter=18, eval_seeds=(1, 2, 3),
                 seed=200, verbose=False):
    """Standard HHO (exploration + soft/hard besiege) maximising fitness."""
    rng = np.random.default_rng(seed)
    keys = QHHO_KEYS
    lo = np.array([QHHO_BOUNDS[k][0] for k in keys])
    hi = np.array([QHHO_BOUNDS[k][1] for k in keys])

    def vec2t(v):
        return {k: float(v[i]) for i, k in enumerate(keys)}

    def fit(v):
        return evaluate_qhho(p, vec2t(v), refs, eval_seeds)["F"]

    X = rng.uniform(lo, hi, size=(n_hawks, len(keys)))
    Fv = np.array([fit(x) for x in X])
    rb = int(np.argmax(Fv)); rabbit = X[rb].copy(); rabbit_F = Fv[rb]
    F_hist = []

    for t in range(n_iter):
        E1 = 2*(1 - t/n_iter)
        for i in range(n_hawks):
            E0 = 2*rng.random()-1
            E = E1*E0
            if abs(E) >= 1:                                   # exploration
                if rng.random() < 0.5:
                    xr = X[rng.integers(n_hawks)]
                    Xn = xr - rng.random()*np.abs(xr - 2*rng.random()*X[i])
                else:
                    Xn = (rabbit - X.mean(0)) - rng.random()*(lo + rng.random()*(hi-lo))
            else:                                             # exploitation
                dX = rabbit - X[i]
                if rng.random() >= 0.5:
                    J = 2*(1-rng.random())
                    Xn = rabbit - E*np.abs(J*rabbit - X[i])   # soft besiege
                else:
                    Xn = rabbit - E*np.abs(dX)                # hard besiege
            Xn = np.clip(Xn, lo, hi)
            Fn = fit(Xn)
            if Fn > Fv[i]:
                X[i], Fv[i] = Xn, Fn
                if Fn > rabbit_F:
                    rabbit_F, rabbit = Fn, Xn.copy()
        F_hist.append(rabbit_F)
        if verbose:
            print(f"  HHO it {t+1:02d}: best F = {rabbit_F:.4f}")
    return vec2t(rabbit), rabbit_F, np.array(F_hist)
