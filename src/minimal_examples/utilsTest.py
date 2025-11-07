import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(".."))
from utils import (
    error_consistency,
    estimate_copy_model_parameters,
    fast_cohen,
    fast_cohen_norm,
    simulate_trials_from_copy_model,
)


def report_copy_probability_simulation_results(ec: float, acc1: float, acc2: float):

    print(f"From     ec    = {   ec:.2f}, acc1  = {acc1:.2f}, and acc2  = {acc2:.2f},")

    pcopy, uacc1, uacc2 = estimate_copy_model_parameters(ec, acc1, acc2)
    print(
        f"we infer pcopy = {pcopy:.2f}, uacc1 = {uacc1:.2f}, and uacc2 = {uacc2:.2f}."
    )
    print("")

    print(
        "We then sample with these inferred parameters and should obtain similar results as before."
    )
    n_trials = 10000
    trials1, trials2 = simulate_trials_from_copy_model(ec, acc1, acc2, n_trials)

    # Observed marginals 1
    simulated_acc1 = pd.DataFrame(trials1).value_counts() / len(trials1)
    print(
        f"Simulated accuracy 1 should be roughly {acc1} and is {simulated_acc1[1]:.2f}"
    )

    # Observed marginals 2
    simulated_acc2 = pd.DataFrame(trials2).value_counts() / len(trials2)
    print(
        f"Simulated accuracy 2 should be roughly {acc2} and is {simulated_acc2[1]:.2f}"
    )

    # Contingency table
    simulated_ec = fast_cohen(trials1, trials2)
    print(f"Error consistency    should be roughly {ec} and is {simulated_ec:.2f}")
    print("\n\n")


ec = 0.5
acc1 = 0.5
acc2 = 0.5
report_copy_probability_simulation_results(ec, acc1, acc2)

ec = 0
acc1 = 0.5
acc2 = 0.5
report_copy_probability_simulation_results(ec, acc1, acc2)

ec = -0.2
acc1 = 0.5
acc2 = 0.5
report_copy_probability_simulation_results(ec, acc1, acc2)

ec = -0.2
acc1 = 0.6
acc2 = 0.8
report_copy_probability_simulation_results(ec, acc1, acc2)

ec = -0.99
acc1 = 0.5
acc2 = 0.5
report_copy_probability_simulation_results(ec, acc1, acc2)

ec = -1
acc1 = 0.5
acc2 = 0.5
report_copy_probability_simulation_results(ec, acc1, acc2)

ec = -0.3
acc1 = 0.6
acc2 = 0.2
report_copy_probability_simulation_results(ec, acc1, acc2)
