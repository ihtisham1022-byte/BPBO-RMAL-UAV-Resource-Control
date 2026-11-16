"""Matched-budget search on RMAL, using the repository's controller."""
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

import controllers as ctl
import mm_core as mm

BUDGET = 128
INITIAL = 8
T = 160
N_REPEATS = 5
N_VALID = 4
N_TEST = 24
KEYS = [key for key in ctl.RMAL_KEYS if key != "eps_decay"]
LOW, HIGH = np.array([ctl.RMAL_BOUNDS[key] for key in KEYS]).T
BASE = (np.array([ctl.RMAL_BASELINE[key] for key in KEYS]) - LOW)/(HIGH - LOW)
SYSTEM = mm.SysParams(noise_floor_dbm=-91.71697497367859)
REFS = ctl.Refs()


@dataclass(frozen=True)
class Scenario:
    layout: int
    training_action: int
    training_channel: int


def scenarios(split, count):
    children = np.random.SeedSequence(split).spawn(count)
    return [Scenario(*map(int, child.generate_state(3, dtype=np.uint64))) for child in children]


VALIDATION = scenarios(71001, N_VALID)
TEST = scenarios(71002, N_TEST)


def evaluate(theta, case):
    _, q = ctl.run_episode(
        SYSTEM, theta, np.random.default_rng(case.training_action), T=T,
        users_seed=case.layout, channel_rng=np.random.default_rng(case.training_channel),
    )
    # Greedy scoring restarts the same stream for each layout.
    score = ctl.eval_policy(
        SYSTEM, dict(theta, eps_min=0.0), q, np.random.default_rng(0), T=T,
        users_seed=case.layout,
    )
    score["fitness"] = ctl.fitness(score["R_tail"], score["qos"], score["P_fair"], REFS)
    return score


def decode(x):
    return dict(zip(KEYS, LOW + x*(HIGH - LOW)))


def search(job):
    method, repeat = job
    rng_init = np.random.default_rng(76000 + repeat)
    rng = np.random.default_rng(77000 + repeat)
    points = list(rng_init.random((INITIAL, len(KEYS))))
    points[0] = BASE.copy()
    fitnesses = []

    def score(x):
        theta = decode(x)
        return float(np.mean([evaluate(theta, case)["fitness"] for case in VALIDATION]))

    for x in points:
        fitnesses.append(score(x))
    pop = np.array(points)
    pop_scores = np.array(fitnesses)
    for i in range(INITIAL, BUDGET):
        best = points[int(np.argmax(fitnesses))]
        if method == "BPBO":
            index = (i - INITIAL) % INITIAL
            beta = rng.beta(2, 2)
            x = np.clip(pop[index] + beta*(best - pop[index])
                        + (1 - beta)*rng.normal(0, 0.15, len(KEYS)), 0, 1)
        elif method == "Random search":
            x = rng.random(len(KEYS))
        else:
            gp = GaussianProcessRegressor(
                kernel=ConstantKernel(1.0, "fixed")*Matern(0.5, "fixed", nu=2.5),
                alpha=1e-6, normalize_y=True, optimizer=None,
            ).fit(np.array(points), np.array(fitnesses))
            candidates = rng.random((1024, len(KEYS)))
            mean, sd = gp.predict(candidates, return_std=True)
            delta = mean - max(fitnesses)
            z = delta/np.maximum(sd, 1e-12)
            ei = delta*ndtr(z) + sd*np.exp(-0.5*z*z)/np.sqrt(2*np.pi)
            x = candidates[int(np.argmax(ei))]
        value = score(x)
        points.append(x)
        fitnesses.append(value)
        if method == "BPBO" and value > pop_scores[index]:
            pop[index], pop_scores[index] = x, value

    best = int(np.argmax(fitnesses))
    theta = decode(points[best])
    runs = [dict(method=method, repeat=repeat, test_case=j, **evaluate(theta, case))
            for j, case in enumerate(TEST)]
    history = [dict(method=method, repeat=repeat, trial=j + 1, fitness=f,
                    **decode(x)) for j, (x, f) in enumerate(zip(points, fitnesses))]
    print(method, repeat, round(fitnesses[best], 4), flush=True)
    return runs, history


def main():
    methods = ["BPBO", "Random search", "Bayesian optimization"]
    jobs = [(method, repeat) for repeat in range(N_REPEATS) for method in methods]
    with ProcessPoolExecutor(max_workers=8) as pool:
        result = list(pool.map(search, jobs))
    for index, name in enumerate(("held_out_metrics", "validation_trials")):
        rows = [row for entry in result for row in entry[index]]
        with Path(f"{name}.csv").open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
