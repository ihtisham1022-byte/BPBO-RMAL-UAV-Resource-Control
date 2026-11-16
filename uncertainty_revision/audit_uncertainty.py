"""Check nominal Monte Carlo outputs and sensitivity to the assigned scales."""
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import fields, replace
import json
from pathlib import Path

import numpy as np

import mm_core as mm
from uncertainty import InputU, monte_carlo, nominal_scenario

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / 'outputs/config.json').read_text())
SYSTEM = mm.SysParams(**CONFIG['sys_params'])
INPUTS = InputU()
NAMES = [f.name for f in fields(INPUTS)]
SCALES = (0.5, 2.0)
LINKS = ('Serving', 'Interferer 1', 'Interferer 2')

def mc(job):
    name, factor = job
    keys = NAMES if name == 'all' else ([] if name == 'nominal' else [name])
    inputs = replace(INPUTS, **{key: getattr(INPUTS, key) * factor for key in keys})
    summary, _ = monte_carlo(SYSTEM, inputs, nominal_scenario(SYSTEM), M=20_000, seed=7)
    return [dict(variation=name, factor=factor, output=k, mean=v['mean'], sd=v['u'],
                 q025=v['lo'], q975=v['hi']) for k,v in summary.items()]

def covariance_for_ranges(distances, azimuths):
    """Propagate independent isotropic position errors into a shared-user range block."""
    scale = np.sqrt(1 - (SYSTEM.altitude/distances)**2)
    jac = scale[:,None] * np.column_stack((np.cos(azimuths), np.sin(azimuths)))
    var_uav = np.diag((INPUTS.u_pos_uav * scale)**2)
    return var_uav + INPUTS.u_pos_user**2 * jac @ jac.T

def local_budget():
    d = np.array([300., 500., 500.])
    input_names = ('d_s','d_i1','d_i2','P_s','P_i1','P_i2','alpha','G0','g_s','g_i1','g_i2','N')
    tilt = np.deg2rad([0.,120.,240.])
    power_dbm = 20.
    x = np.r_[d, [power_dbm]*3, SYSTEM.pathloss_exp, SYSTEM.G0_db, [0.]*4]
    def f(v):
        distances, power, alpha, gain, link_delta, noise_delta = (
            v[:3],v[3:6],v[6],v[7],v[8:11],v[11])
        received = mm.dbm_to_w(power + gain - 10*alpha*np.log10(distances) + link_delta)
        noise = SYSTEM.noise_power_w * mm.db_to_lin(noise_delta)
        return float(mm.lin_to_db(received[0] / (received[1:].sum() + noise)))

    h = 1e-4
    c = np.array([(f(x+h*e)-f(x-h*e))/(2*h) for e in np.eye(x.size)])
    ranges = covariance_for_ranges(d,tilt)
    scales = np.r_[np.sqrt(np.diag(ranges)), [INPUTS.u_P_db]*3, INPUTS.u_alpha,
                   INPUTS.u_G0_db, [INPUTS.u_fading_db]*3, INPUTS.u_noise_db]
    cov = np.diag(scales**2)
    cov[:3,:3] = ranges
    diagonal = (c * scales)**2
    rows = [dict(source=n, standard_deviation=u, coefficient=v, component_dB=abs(u*v),
                 variance_dB2=w) for n,u,v,w in zip(input_names,scales,c,diagonal)]
    rows.append(dict(source='Range covariance', standard_deviation='', coefficient='',
                     component_dB='', variance_dB2=c @ cov @ c-diagonal.sum()))
    cases = [('Independent ranges', np.diag(scales**2)),
             ('Shared user; azimuths 0,120,240 deg', cov)]
    aligned = cov.copy()
    aligned[:3,:3] = covariance_for_ranges(d,np.zeros(3))
    cases.append(('Shared user; aligned azimuths',aligned))
    for rho in (-0.5,0.5):
        correlated = cov.copy()
        correlated[6,7] = correlated[7,6] = rho * scales[6]*scales[7]
        cases.append((f'Shared user and alpha-G0 correlation {rho:+.1f}',correlated))
    for rho in (0.5,1.0):
        correlated = cov.copy()
        correlated[8:11,8:11] += rho * INPUTS.u_fading_db**2 * (np.ones((3,3))-np.eye(3))
        cases.append((f'Shared user and residual gain correlation {rho:+.1f}',correlated))
    outputs = []
    for name, matrix in cases:
        assert np.linalg.eigvalsh(matrix).min() >= -1e-10
        u = np.sqrt(c @ matrix @ c)
        outputs.append(dict(case=name, u_sinr_db=u, expanded_k2_db=2*u))
    assert abs(sum(row['variance_dB2'] for row in rows) - c@cov@c) < 1e-12
    return rows, outputs

def write_csv(name, rows):
    with (ROOT/'outputs'/name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    jobs = [('nominal',1.)] + [(name,factor) for name in [*NAMES,'all'] for factor in SCALES]
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = [row for case in pool.map(mc,jobs) for row in case]
    saved = json.loads((ROOT/'outputs/uncertainty_summary.json').read_text())['measurands']
    for row in rows[:6]:
        expected = saved[row['output']]
        np.testing.assert_allclose([row[q] for q in ('mean','sd','q025','q975')],
                                   [expected[q] for q in ('mean','u','lo','hi')],atol=1e-10,rtol=0)
    budget, covariance = local_budget()
    write_csv('uq_scale_sensitivity.csv', rows)
    write_csv('uq_local_budget.csv', budget)
    write_csv('uq_covariance_sensitivity.csv', covariance)
    print('Nominal Monte Carlo reproduced for all six outputs.', flush=True)
    for row in covariance:
        print(row)

if __name__ == '__main__':
    main()
