"""Exact slot-score maximization for three UAVs on four exchangeable channels."""
from itertools import product

import numpy as np

import controllers as ctl
import mm_core as mm
import optimizer_comparison as exp

# The five possible partitions of three UAVs across exchangeable subchannels.
CHANNELS = np.array([(0,1,2),(0,0,1),(0,1,0),(0,1,1),(0,0,0)])
POWER_INDEX = np.array(list(product(range(exp.SYSTEM.n_power), repeat=3)))
SAME_CHANNEL = CHANNELS[:, :, None] == CHANNELS[:, None, :]

def allocate(p, uav, users, fading):
    gains = mm.large_scale_gain(p, mm.distance_3d(uav, users, p.altitude)) * fading
    powers = p.power_levels_w[POWER_INDEX]
    received = powers[:, :, None] * gains[None, :, :]
    association = received.argmax(axis=1)
    serving = np.take_along_axis(received, association[:, None, :], axis=1).squeeze(axis=1)
    total = np.einsum('cij,pjk->cpik', SAME_CHANNEL, received)
    associated_total = np.take_along_axis(total, association[None, :, None, :], axis=2).squeeze(axis=2)
    sinr = serving[None, :, :] / (associated_total - serving + p.noise_power_w)
    loads = np.einsum('cpk,puk->cpu', np.log2(1 + sinr), association[:, None, :] == np.arange(3)[None, :, None])
    score = (ctl.W_S_EVAL * mm.lin_to_db(sinr).mean(axis=-1)
             - ctl.W_P_EVAL * (powers / p.power_levels_w[-1]).mean(axis=-1)
             - ctl.W_F_EVAL * loads.std(axis=-1) / ctl.LOAD_SCALE)
    channel, power = np.unravel_index(score.argmax(), score.shape)
    return CHANNELS[channel] * p.n_power + POWER_INDEX[power]

def check():
    p = exp.SYSTEM
    assert p.n_uav == 3 and p.n_subch >= 3
    rng = np.random.default_rng(126)
    users = mm.sample_users(p, rng)
    for slot in (0, 80, 159):
        uav = mm.uav_positions(p, slot)
        fading = mm.rayleigh_fading((p.n_uav, p.n_users), rng)
        def score(actions):
            powers = p.power_levels_w[np.asarray(actions) % p.n_power]
            out = mm.evaluate_slot_assoc(p, uav, users, powers, np.asarray(actions) // p.n_power, fading)
            loads = np.bincount(out['serv'], weights=np.log2(1+out['sinr_lin']), minlength=3)
            return (ctl.W_S_EVAL*out['sinr_db'].mean()
                    -ctl.W_P_EVAL*(powers/p.power_levels_w[-1]).mean()
                    -ctl.W_F_EVAL*loads.std()/ctl.LOAD_SCALE)
        best = max(map(score, product(range(p.n_subch*p.n_power), repeat=p.n_uav)))
        np.testing.assert_allclose(score(allocate(p, uav, users, fading)), best, atol=1e-12, rtol=0)

if __name__ == '__main__':
    check()
    print('Exact slot optimizer agrees with brute-force enumeration on all three checks.')
