import EstimateKappa as ek
import EstimateUnbiasedKappa as euk

# import EstimateCopyProb as cp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

# Define examples --------------------------------------------------------------
example = 1

if example == 1:  # Kristof's first example

    # fmt: off
    observed1 = np.asarray([0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1,
            1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1])

    observed2 =  np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1])
    # fmt: on

if example == 2:  # Unbalanced first ratings, pu1 = .4
    # 50% copy probability, cp = .5
    # (Underlying) balanced second ratings, pu2 = .5
    observed1 = np.repeat(
        np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 5
    )
    observed2 = np.repeat(
        np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1]), 5
    )

if example == 3:  # pu1 = .5, cp = .5, pu2 = .6
    observed1 = np.repeat(
        np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), 5
    )
    observed2 = np.repeat(
        np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]), 5
    )

# Evaluate ---------------------------------------------------------------------
kappa = ek.calculate_kappa(observed1, observed2)
meanKappa, lower, upper, kappas = ek.bootstrap_kappa(observed1, observed2)

ukappa = euk.calculate_unbiased_kappa(observed1, observed2)
meanUKappa, ukappa_CI_lower, ukappa_CI_upper, ukappas = euk.bootstrap_unbiased_kappa(
    observed1, observed2
)

# copyprob, pu1, pu2 = cp.calculate_cp(observed1, observed2)
# mean_cp, cp_CI_lower, cp_CI_upper = cp.bootstrap_cp(observed1, observed2)

cm = confusion_matrix(observed1, observed2, labels=[0, 1])

# Print ------------------------------------------------------------------------
print(cm / len(observed1))

print(
    f"""
Kappa from orig. data = {kappa:.3f}
Bootstrapping yields mean Kappa = {meanKappa:.3f}, 
  with 95% CI [{lower:.2f},{upper:.2f}].
"""
)

print(
    f"""
Unbiased Kappa from orig. data = {ukappa:.3f}
Bootstrapping yields mean unbiased Kappa = {meanUKappa:.3f}, 
  with 95% CI [{ukappa_CI_lower:.2f},{ukappa_CI_upper:.2f}].
"""
)

# print(f"""
# CopyProb from orig. data = {copyprob:.3f}
# Underlying marignals = {pu1[0]:.3f} and {pu2[0]:.3f}
# Bootstrapping yields mean CopyProb = {mean_cp:.3f},
#   with 95% CI [{cp_CI_lower:.2f},{cp_CI_upper:.2f}].
# """)
