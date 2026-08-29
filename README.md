# Measurement-Aware UAV Resource Control — Reproducible RMAL Pipeline

A reproducible research repository for measurement-aware reinforcement-learning-based UAV downlink resource control. The code implements the physical measurement model, Baseline RMAL, BPBO-RMAL, Q-learning + HHO, uncertainty propagation, robustness evaluation, sensitivity analysis, table generation, and publication-quality figures.

## Owner and creator

**Ihtisham Ul Haq**  
Repository owner, creator, and maintainer.

## What this repository contains

| File / folder | Purpose |
|---|---|
| `mm_core.py` | Physical and measurement model: geometry, channel gain, RSS, interference, SINR, throughput, QoS, fairness, and reward. |
| `controllers.py` | Baseline RMAL, BPBO-RMAL, and Q-learning + HHO controllers. |
| `run_experiment.py` | End-to-end controller optimisation and evaluation. |
| `dataset_analysis.py` | Dataset statistics, baseline metrics, and Type A repeatability analysis. |
| `uncertainty.py` | GUM law of propagation of uncertainty and Monte Carlo uncertainty propagation. |
| `mc_bounded.py` | Bounded Monte Carlo controller-performance analysis. |
| `robustness.py` | Robustness analysis under imperfect SINR knowledge. |
| `sensitivity.py` | Sobol global sensitivity analysis. |
| `make_tables.py` | Recreates LaTeX result tables from saved outputs. |
| `figures.py` | Recreates publication-ready PDF and PNG figures. |
| `rmal_micro_manuscript_grade.xlsx` | Reference controlled-simulation dataset. |
| `outputs/` | Reproducible numerical results and generated LaTeX tables. |

## Requirements

- Python 3.11 or newer is recommended.
- A GPU is **not required**.

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Beginner setup

### 1. Download or clone the repository

After creating the repository on GitHub, you can clone it with:

```bash
git clone YOUR_PRIVATE_REPOSITORY_URL
cd YOUR_REPOSITORY_FOLDER
```

If you do not use Git from the command line, GitHub Desktop can clone the repository for you.

### 2. Create a virtual environment (recommended)

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Run a quick test

```bash
python run_experiment.py --quick
```

### 4. Run the analysis pipeline

```bash
python run_experiment.py
python dataset_analysis.py
python uncertainty.py
python mc_bounded.py
python robustness.py
python make_tables.py
python figures.py
```

The Sobol sensitivity analysis is separate because it is more computationally demanding:

```bash
python sensitivity.py
```

## Dataset location

By default, the scripts use:

```text
rmal_micro_manuscript_grade.xlsx
```

from the repository root. If the dataset is stored elsewhere, define the optional `RMAL_DATASET_PATH` environment variable instead of editing the code.

Windows PowerShell example:

```powershell
$env:RMAL_DATASET_PATH = "D:\Research\rmal_micro_manuscript_grade.xlsx"
python dataset_analysis.py
```

## Reproducibility

The pipeline uses fixed random seeds and common evaluation conditions so that controller comparisons can be reproduced. Key generated artifacts are retained under `outputs/`, including CSV summaries, NumPy archives, JSON configuration/results, and LaTeX tables.

## Main methodological components

- Controlled-simulation UAV downlink environment.
- Subchannel-aware interference and estimated SINR.
- Baseline RMAL and BPBO-RMAL optimisation.
- Q-learning + Harris Hawks Optimisation comparator.
- GUM-compatible uncertainty budget.
- Monte Carlo propagation of measurement uncertainty.
- Closed-loop robustness to SINR estimation error.
- Global sensitivity analysis using Sobol indices.
- Reproducible table and figure generation.

## Citation and authorship

If this repository is used in academic work, please cite the associated publication when available and acknowledge the repository creator:

**Ihtisham Ul Haq**

Machine-readable citation metadata are provided in `CITATION.cff`.

## Repository visibility

For unpublished research, keep this repository **Private** on GitHub. Only collaborators whom you explicitly invite to the private repository can access its contents.

## Contact

For repository-related questions, please contact **Ihtisham Ul Haq** through the contact information associated with the corresponding research publication or GitHub profile.
