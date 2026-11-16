"""Extend the saved matched-search benchmark with HHO and resource baselines."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import csv

import numpy as np
from scipy.stats import wilcoxon

import optimizer_comparison as exp
import controllers as ctl
import mm_core as mm
from instantaneous import allocate

HERE = Path(__file__).resolve().parent
METRICS = ('R_tail', 'qos', 'P_fair', 'fitness')

def read_csv(name):
    with (HERE / name).open(newline='') as stream:
        return list(csv.DictReader(stream))

def write_csv(name, rows):
    with (HERE / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def hho(repeat):
    initial_rng = np.random.default_rng(76000 + repeat)
    rng = np.random.default_rng(77000 + repeat)
    population = initial_rng.random((exp.INITIAL, len(exp.KEYS)))
    population[0] = exp.BASE
    history = []

    def score(point):
        theta = exp.decode(point)
        value = float(np.mean([exp.evaluate(theta, case)['fitness'] for case in exp.VALIDATION]))
        history.append(dict(repeat=repeat, trial=len(history)+1, fitness=value, **theta))
        return value

    scores = np.array([score(point) for point in population])
    best = population[scores.argmax()].copy()
    best_score = scores.max()
    generations = (exp.BUDGET - exp.INITIAL) // exp.INITIAL
    for generation in range(generations):
        energy_scale = 2 * (1 - generation / generations)
        for index, point in enumerate(population):
            energy = energy_scale * rng.uniform(-1, 1)
            if abs(energy) >= 1:
                if rng.random() < 0.5:
                    peer = population[rng.integers(exp.INITIAL)]
                    proposal = peer - rng.random() * abs(peer - 2*rng.random()*point)
                else:
                    proposal = best - population.mean(axis=0) - rng.random()*rng.random()
            elif rng.random() < 0.5:
                jump = 2 * (1 - rng.random())
                proposal = best - energy*abs(jump*best - point)
            else:
                proposal = best - energy*abs(best - point)
            proposal = np.clip(proposal, 0, 1)
            value = score(proposal)
            if value > scores[index]:
                population[index], scores[index] = proposal, value
                if value > best_score:
                    best, best_score = proposal.copy(), value
    assert len(history) == exp.BUDGET
    theta = exp.decode(best)
    rows = [dict(method='HHO-RMAL', repeat=repeat, test_case=index, **exp.evaluate(theta, case))
            for index, case in enumerate(exp.TEST)]
    return rows, history

def resource_episode(case, assignment=None):
    p = exp.SYSTEM
    rng = np.random.default_rng(0)
    random_actions = np.random.default_rng(case.training_action)
    users = mm.sample_users(p, np.random.default_rng(case.layout))
    scores, hits, loads = [], [], []
    for slot in range(exp.T):
        # eval_policy consumes one action RNG draw per UAV before drawing fading,
        # even for greedy evaluation. Reproduce that exact scoring fade sequence.
        rng.random(p.n_uav)
        fading = mm.rayleigh_fading((p.n_uav, p.n_users), rng)
        if assignment is None:
            actions = random_actions.integers(p.n_subch*p.n_power, size=p.n_uav)
        elif callable(assignment):
            actions = assignment(p, mm.uav_positions(p, slot), users, fading)
        else:
            actions = assignment
        powers = p.power_levels_w[actions % p.n_power]
        physical = mm.evaluate_slot_assoc(p, mm.uav_positions(p, slot), users,
                                          powers, actions // p.n_power, fading)
        se = np.log2(1 + physical['sinr_lin'])
        load = np.array([se[physical['serv'] == u].sum() for u in range(p.n_uav)])
        scores.append(ctl.W_S_EVAL*physical['sinr_db'].mean()
                      - ctl.W_P_EVAL*(powers/p.power_levels_w[-1]).mean()
                      - ctl.W_F_EVAL*load.std()/ctl.LOAD_SCALE)
        hits.append(physical['qos'].mean())
        loads.append(load)
    tail = np.mean(scores[-max(1, int(0.15*exp.T)):])
    qos = np.mean(hits)
    average_load = np.mean(loads, axis=0)
    penalty = average_load.std()
    return dict(R_tail=tail, qos=qos, P_fair=penalty, jain=mm.jain_index(average_load),
                gini=mm.gini_coefficient(average_load), entropy=mm.entropy_fairness(average_load),
                fitness=ctl.fitness(tail, qos, penalty, exp.REFS))

def summary(rows):
    means = []
    for name in dict.fromkeys(row['method'] for row in rows):
        matching = [row for row in rows if row['method'] == name]
        means.append(dict(method=name, n_rows=len(matching),
                          **{metric: np.mean([float(row[metric]) for row in matching]) for metric in METRICS}))
    write_csv('comment4_means.csv', means)

    # Conditional inference: pair the 24 held-out scenarios after averaging
    # each learning method's five search repetitions within each scenario.
    def scenario_means(name, metric):
        return np.array([np.mean([float(row[metric]) for row in rows
                                  if row['method'] == name and int(row['test_case']) == j])
                         for j in range(exp.N_TEST)])

    rng = np.random.default_rng(3081)
    indices = rng.integers(exp.N_TEST, size=(10000, exp.N_TEST))
    contrasts = []
    comparisons = ['HHO-RMAL', 'Uniform random allocation', 'Orthogonal mid-power', 'Orthogonal maximum power', 'Centralized slot optimizer']
    for name in comparisons:
        for metric in METRICS:
            delta = scenario_means('BPBO', metric) - scenario_means(name, metric)
            low, high = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
            p = wilcoxon(delta, zero_method='pratt').pvalue if np.any(delta) else 1.0
            contrasts.append(dict(comparator=name, metric=metric, difference=delta.mean(), lower_95=low,
                                  upper_95=high, p_wilcoxon=p))
    # One Holm family covering all 20 pairwise tests.
    running = 0
    for rank, idx in enumerate(np.argsort([row['p_wilcoxon'] for row in contrasts])):
        running = max(running, min(1., (len(contrasts)-rank)*contrasts[idx]['p_wilcoxon']))
        contrasts[idx]['p_holm'] = running
    write_csv('comment4_paired.csv', contrasts)

def main():
    saved = read_csv('held_out_metrics.csv')
    histories = read_csv('validation_trials.csv')
    # Confirm the carried-forward BPBO outcomes can be reproduced by this benchmark.
    for repeat in range(exp.N_REPEATS):
        chosen = max((r for r in histories if r['method']=='BPBO' and int(r['repeat'])==repeat),
                     key=lambda row: float(row['fitness']))
        observed = exp.evaluate({key:float(chosen[key]) for key in exp.KEYS}, exp.TEST[0])
        expected = next(r for r in saved if r['method']=='BPBO' and int(r['repeat'])==repeat and int(r['test_case'])==0)
        np.testing.assert_allclose([observed[k] for k in METRICS], [float(expected[k]) for k in METRICS],
                                   atol=1e-12, rtol=1e-12)
    with ProcessPoolExecutor(max_workers=6) as pool:
        pending = [pool.submit(hho, repeat) for repeat in range(exp.N_REPEATS)]
        orthogonal = np.arange(exp.SYSTEM.n_uav)*exp.SYSTEM.n_power + exp.SYSTEM.n_power // 2
        rows = []
        for name, assignment in [('Uniform random allocation', None), ('Orthogonal mid-power', orthogonal),
                                 ('Orthogonal maximum power', orthogonal + 1), ('Centralized slot optimizer', allocate)]:
            rows.extend(dict(method=name, repeat=-1, test_case=j, **resource_episode(case, assignment))
                        for j,case in enumerate(exp.TEST))
        for future in pending:
            held_out, search_history = future.result()
            rows.extend(held_out)
            write_csv(f'hho_validation_{search_history[0]["repeat"]}.csv', search_history)
            print('HHO repeat', search_history[0]['repeat'], 'completed', flush=True)
    write_csv('comment4_new_held_out.csv', rows)
    summary(saved + rows)

if __name__ == '__main__':
    main()
