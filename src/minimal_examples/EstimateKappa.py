import numpy as np


def calculate_kappa(observed1, observed2):
    observedEqual = np.mean(observed1 == observed2)
    acc1 = np.mean(observed1)
    acc2 = np.mean(observed2)
    cexpTrue = acc1 * acc2 + (1 - acc1) * (1 - acc2)
    kappa = (observedEqual - cexpTrue) / (1 - cexpTrue)
    return kappa


def bootstrap_kappa(observed1, observed2, num_bootstrap=1000, alpha=0.05):
    n = len(observed1)
    kappa_bootstrap = np.zeros(num_bootstrap)

    for i in range(num_bootstrap):
        # Sample indices with replacement
        indices = np.random.choice(n, n, replace=True)
        # Create resampled obs1 and obs2
        resampled_obs1 = observed1[indices]
        resampled_obs2 = observed2[indices]
        # Calculate kappa for the resampled set
        kappa_bootstrap[i] = calculate_kappa(resampled_obs1, resampled_obs2)

    # Calculate confidence intervals
    lower_bound = np.percentile(kappa_bootstrap, 100 * (alpha / 2))
    upper_bound = np.percentile(kappa_bootstrap, 100 * (1 - alpha / 2))

    return np.mean(kappa_bootstrap), lower_bound, upper_bound, kappa_bootstrap
