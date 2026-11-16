# BPBO-RMAL for UAV Downlink Resource Control

This package contains the simulator, resource-control methods, simulated data, and numerical results used in the study. It covers nominal comparisons, matched hyperparameter searches, resource-allocation baselines, and post-training uncertainty evaluation.

## Setup

Use Python 3.12 and install the dependencies from the repository root. All computations use a CPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with `.\.venv\Scripts\Activate.ps1` after creating it. The scripts below regenerate files in place. Keep a copy of the included outputs if you want to compare them with a rerun.

## Reproduce the analyses

| Analysis | Command from the repository root | Main files |
| --- | --- | --- |
| Nominal training, optimization, and controller comparison | `python run_experiment.py` | `outputs/config.json`, `results_three_models.csv`, `convergence.npz`, `uav_micro_dataset.csv` |
| Reference workbook analysis | `python dataset_analysis.py` | `outputs/dataset_summary.json`, `dataset_baseline_metrics.csv` |
| Operating-point Monte Carlo | `python uncertainty.py` | `outputs/uncertainty_measurands.csv`, `uncertainty_mc.npz` |
| Covariance budget and input-scale sensitivity | `python -m uncertainty_revision.audit_uncertainty` | `uncertainty_revision/outputs/uq_local_budget.csv`, `uq_covariance_sensitivity.csv`, `uq_scale_sensitivity.csv` |
| Frozen-policy Monte Carlo | `python mc_bounded.py` | `outputs/mc_bounded_summary.csv`, `mc_bounded.npz` |
| SINR-error stress tests | `python robustness.py` | `outputs/robustness_dataset.csv`, `robustness_policy.csv` |
| Stochastic sensitivity screening | `python sensitivity.py` | `outputs/sensitivity_indices_recomputed.csv`, `sensitivity_recomputed.npz` |
| Numerical tables and results figures | `python make_tables.py` and `python figures.py` | `outputs/tables/`, `outputs/figures/` |
| System schematics and workflow | `python draw_schematics.py` | `figures/fig_uav_geometry.pdf`, `fig_uav_coverage.pdf`, `fig_controller_workflow.pdf` |

Run the nominal experiment before regenerating analyses that use its configuration. The plotting and table scripts can also run directly on the included results. The directory `figures/manuscript/` contains the exact assets used for the remaining figures in the compiled study. The sensitivity script reuses the archived design in `outputs/sensitivity.npz` and writes a separate copy of its results. `--resample` generates a different, explicitly seeded design. Screening scores are used for relative ranking. The recomputed bootstrap errors use an explicit seed and may differ from the archived ones.

Run the matched benchmarks from their own directory. The initial command evaluates all 15 optimizer runs and may take substantial CPU time. The second command can be run directly on the included benchmark CSVs. The third command checks the carried-forward BPBO results, reruns HHO-RMAL and the four allocation rules, and regenerates the paired statistics.

```bash
cd benchmarks
python optimizer_comparison.py
python summarize_comparison.py
python comment4.py
cd ..
```

The benchmark outputs include per-candidate fitness in `validation_trials.csv` and `hho_validation_*.csv`, per-scenario results in `held_out_metrics.csv` and `comment4_new_held_out.csv`, and summary estimates in `paired_intervals.csv`, `comment4_means.csv`, and `comment4_paired.csv`.

## Protocol settings

- Each UAV has four mean-SINR observation bins with edges at −3, 3, and 9 dB. Its Q-table has four rows and 12 actions. The actions pair four subchannels with transmit powers of 0, 11.5, and 23 dBm. UAV trajectories are prescribed.
- The nominal BPBO and HHO searches each use eight candidates and 15 update rounds, for 128 evaluations including initialization. Each candidate uses one fresh 160-slot learning episode per seed (seeds 1–14). The nominal comparison reuses those seed values. Shared seed values do not guarantee identical fading because exploration and channel sampling share a stream.
- The matched searches use five search runs per optimizer, 128 candidate evaluations per run, and the same seven active RMAL parameters. Four validation scenarios are used for selection and 24 held-out layouts for testing. A fresh 160-slot training episode precedes each 160-slot greedy scoring episode. Training uses separate action and fading streams. All scoring scenarios use the same fixed stream seeded with 0.
- BPBO draws its mixing coefficient from Beta(2, 2). The Gaussian proposal scale is 0.15 times the parameter range. Fitness uses reference scales of 15 for reward and 6 for the fairness penalty. The tail score averages the final 15% of slots after rounding down, with a minimum of one slot.
- Operating-point Monte Carlo uses 20,000 draws. The frozen-policy controller comparison uses 10,000 trials of 80 slots per controller. The SINR-error stress test uses 40 trials of 80 slots per controller and error level. These experiments have different evaluation conditions and aggregation windows.

## Data and citation

`rmal_micro_manuscript_grade.xlsx` contains the 3,060-row reference simulation dataset. `outputs/uav_micro_dataset.csv` contains the separately generated 2,320-row nominal controller trace. Neither file contains field measurements. The remaining CSV and NPZ files contain simulation outcomes, configuration records, and samples or statistics derived from them. Use `CITATION.cff` to cite the software.
