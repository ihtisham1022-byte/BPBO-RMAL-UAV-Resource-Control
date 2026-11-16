"""Summarize held-out results and paired bootstrap intervals."""
import numpy as np
import pandas as pd

runs = pd.read_csv("held_out_metrics.csv")
metrics = ["R_tail", "qos", "P_fair", "fitness"]
print(runs.groupby("method")[metrics].mean().to_string(float_format="%.6f"))
rng = np.random.default_rng(3081)
repeat = rng.integers(0, 5, (10000, 5, 1))
scenario = rng.integers(0, 24, (10000, 1, 24))
rows = []
for metric in metrics:
    by_method = runs.pivot(index=["repeat", "test_case"], columns="method", values=metric)
    for other in ("Random search", "Bayesian optimization"):
        delta = (by_method["BPBO"] - by_method[other]).to_numpy().reshape(5, 24)
        lo, hi = np.percentile(delta[repeat, scenario].mean(axis=(1, 2)), [2.5, 97.5])
        rows.append(dict(metric=metric, comparator=other, bpbo_minus_comparator=delta.mean(),
                         lower_95=lo, upper_95=hi))
pd.DataFrame(rows).to_csv("paired_intervals.csv", index=False)
