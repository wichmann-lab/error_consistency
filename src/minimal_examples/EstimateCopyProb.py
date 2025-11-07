import EstimateKappa as ek
import numpy as np
from sklearn.metrics import confusion_matrix


def table(x):
    unique, counts = np.unique(x, return_counts=True)
    return np.asarray(counts)


# Copy probability
def calculate_cp(observed1, observed2):
    kappa = ek.calculate_kappa(observed1, observed2)
    p1 = table(observed1) / len(observed1)  # Observed marginal 1
    p2 = table(observed2) / len(observed2)  # Observed marginal 2
    cp = kappa * sum(p1 * p2) / sum(p2 * p2)  # Copy probability
    pu1 = p1  # Underlying marginal 1
    if cp == 1:
        pu2 = pu1
    else:
        pu2 = (p2 - (cp * pu1)) / (1 - cp)  # Underlying marignal 2
    return cp, pu1, pu2


def bootstrap_cp(observed1, observed2, num_bootstrap=1000, alpha=0.05):
    n = len(observed1)
    cp_bootstrap = np.zeros(num_bootstrap)

    for i in range(num_bootstrap):
        # Sample indices with replacement
        indices = np.random.choice(n, n, replace=True)
        # Create resampled obs1 and obs2
        resampled_obs1 = observed1[indices]
        resampled_obs2 = observed2[indices]
        # Calculate kappa for the resampled set
        cp_bootstrapped = calculate_cp(resampled_obs1, resampled_obs2)
        cp_bootstrap[i] = cp_bootstrapped[0]

    # Calculate confidence intervals
    lower_bound = np.percentile(cp_bootstrap, 100 * (alpha / 2))
    upper_bound = np.percentile(cp_bootstrap, 100 * (1 - alpha / 2))

    return np.mean(cp_bootstrap), lower_bound, upper_bound
