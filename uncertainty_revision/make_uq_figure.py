"""Render the revised local variance budget."""
import csv
from pathlib import Path

import matplotlib.pyplot as plt

def draw():
    root = Path(__file__).resolve().parent
    with (root/'outputs/uq_local_budget.csv').open() as file:
        variance = {r['source']: float(r['variance_dB2']) for r in csv.DictReader(file)}
    groups = [
        ('Residual link-gain perturbations', ('g_s', 'g_i1', 'g_i2')),
        ('Transmit powers', ('P_s', 'P_i1', 'P_i2')),
        ('Slant ranges (diagonal)', ('d_s', 'd_i1', 'd_i2')),
        ('Reference gain', ('G0',)),
        ('Path-loss exponent', ('alpha',)),
        ('Shared-user range covariance', ('Range covariance',)),
        ('Noise-floor level', ('N',)),
    ]
    total = sum(variance.values())
    values = [100*sum(variance[k] for k in keys)/total for _, keys in groups]
    fig, ax = plt.subplots(figsize=(7.0, 3.35), constrained_layout=True)
    ax.barh(range(len(groups)), values, color='#315e83')
    ax.set_yticks(range(len(groups)), [name for name,_ in groups], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Share of linearized SINR variance (%)')
    ax.set_xlim(0, 88)
    for index, value in enumerate(values):
        ax.text(value+0.8, index, f'{value:.1f}%', va='center', fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    return fig

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    draw().savefig(root/'fig_uncertainty_budget.png', dpi=300, bbox_inches='tight')
