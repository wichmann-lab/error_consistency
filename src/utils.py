"""
Utility functions for the Error Consistency project.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from numba import njit

rng = np.random.default_rng()


def is_binary(arr: np.array) -> bool:
    assert isinstance(arr, np.ndarray), "Expected a numpy array."
    return np.all((arr == 0) | (arr == 1))


@njit
def calculate_kappa(obs: float, exp: float) -> float:
    """
    Calculates Cohen's kappa from the agreements.

    :param obs: the observed agreement
    :param exp: the expected agreement

    :returns: kappa
    """

    assert 0.0 <= obs <= 1.0, "Observed agreement not within bounds."
    assert 0.0 <= exp <= 1.0, "Expected agreement not within bounds."

    if exp == 1.0:
        return np.nan
    return (obs - exp) / (1.0 - exp)


def fast_cohen(trials1: np.array, trials2: np.array, unbiased: bool = False) -> float:
    """
    A fast implementation of Cohen's kappa for binary sequences.
    This is a wrapper around fast_cohen_core that checks types and sizes.

    :param trials1: np array with binary values
    :param trials2: np array with binary values
    :param unbiased: whether to use the unbiased version of Cohen's kappa

    :returns: Cohen's Kappa score, potentially nan
    """

    assert isinstance(trials1, np.ndarray), "Expected a numpy array."
    assert isinstance(trials2, np.ndarray), "Expected a numpy array."
    assert is_binary(trials1) and is_binary(
        trials2
    ), "At least one set of trials was not binary"
    assert trials1.shape == trials2.shape, "Arrays need to match!"
    assert trials1.ndim == 1, "Expects 1d arrays!"

    return fast_cohen_core(trials1, trials2, unbiased)


@njit
def fast_cohen_core(
    trials1: np.array, trials2: np.array, unbiased: bool = False
) -> float:
    """
    A fast implementation of Cohen's kappa for binary sequences.
    This assumes that type checks have already been performed.
    For simulations, we typically call this directly.

    :param trials1: np array with binary values
    :param trials2: np array with binary values
    :param unbiased: whether to use the unbiased version of Cohen's kappa

    :returns: Cohen's Kappa score, potentially nan
    """

    c1 = np.mean(trials1)
    c2 = np.mean(trials2)
    obs = np.mean(trials1 == trials2)

    exp = c1 * c2 + (1.0 - c1) * (1.0 - c2)

    kappa_val = calculate_kappa(obs, exp)

    if unbiased:
        n = len(trials1)
        kappa_val = n * kappa_val / ((n - 1) + kappa_val)

    return kappa_val


@njit
def kappa_max(acc1: float, acc2: float) -> float:
    """
    Get the maximum possible kappa between two observers with accuracies acc1 and acc2.

    :param acc1: the accuracy of the first observer
    :param acc2: the accuracy of the second observer

    :returns: the maximum possible kappa between the two observers
    """

    assert 0 < acc1 < 1, "Accuracy1 had an illegitimate value"
    assert 0 < acc2 < 1, "Accuracy2 had an illegitimate value"

    exp = acc1 * acc2 + (1.0 - acc1) * (1.0 - acc2)
    obs = min(acc1, acc2) + min(1.0 - acc1, 1.0 - acc2)

    return calculate_kappa(obs, exp)


@njit
def kappa_min(acc1: float, acc2: float) -> float:
    """
    Get the minimum possible kappa between two observers with accuracies acc1 and acc2.
    The idea is to do the inverse from kappa_max, by pushing as much weight onto the other diagonal as possible.

    :param acc1: the accuracy of the first observer
    :param acc2: the accuracy of the second observer

    :returns: the minimum possible kappa between the two observers
    """

    assert 0 < acc1 < 1, "Accuracy1 had an illegitimate value"
    assert 0 < acc2 < 1, "Accuracy2 had an illegitimate value"

    exp = acc1 * acc2 + (1.0 - acc1) * (1.0 - acc2)
    off_diag = min(acc1, 1 - acc2) + min(1.0 - acc1, acc2)
    obs = 1.0 - off_diag

    return calculate_kappa(obs, exp)


def estimate_copy_model_parameters(
    ec: float,
    acc1: float,
    acc2: float,
) -> Tuple[float, float, float]:
    """
    Estimate the copy probability and underlying marginals of two classifiers
    with the given error consistency (EC) and marginals (which are here the
    accuracies).

    :param ec: the observed EC between the two observers
    :param acc1: the accuracy of the first observer, who has higher accuracy
    :param acc2: the accuracy of the second observer, who has lower accuracy

    :returns: copy probability and the two observers' underlying accuracies
    """

    # Calculate the copy probability
    numerator = 1 - (acc1 * acc2 + (1 - acc1) * (1 - acc2))
    denominator = 1 - (acc1**2 + (1 - acc1) ** 2)
    f = numerator / denominator  # if acc1 == acc2, this is 1
    pcopy = ec * f

    # We assume that the second observer copies from the first, so the
    # underlying accuracy of the first observer is the observed one
    uacc1 = acc1

    # calculate the latent marginal for obs2, i.e. the probability with which it's sampling when not copying

    # Positive case
    if pcopy >= 0:
        if pcopy == 1:
            uacc2 = uacc1  # avoid division by zero
        else:
            uacc2 = (acc2 - pcopy * acc1) / (1 - pcopy)  # if acc1 == acc2, this is acc2

    # Negative case (where the marginal must be inverted in the copy cases)
    if pcopy < 0:
        if pcopy == -1:
            uacc2 = -uacc1  # avoid division by zero
        else:
            uacc2 = (acc2 - abs(pcopy) * (1 - acc1)) / (1 - abs(pcopy))

    return pcopy, uacc1, uacc2


def simulate_trials_from_copy_model(
    kappa: float, acc1: float, acc2: float, n_trials: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly samples n_trials binary trials (0,1) for two observers with accuracies
    acc1 and acc2, respectively, assuming a ground-truth EC between the two of kappa.
    As n_trials converges to infinity, the empirical EC of the simulated trials will converge to kappa.

    :param ec: the (latent) true EC between the two observers
    :param acc1: the accuracy of the first observer, who has higher accuracy
    :param acc2: the accuracy of the second observer, who has lower accuracy
    :param n_trials: the number of trials to be drawn

    :returns: two binary np-arrays of length n_trials
    """

    assert 0 <= acc1 <= 1, "Accuracy 1 not within bounds."
    assert 0 <= acc2 <= 1, "Accuracy 2 not within bounds."
    assert n_trials > 0, "Number of trials has to be greater than zero."

    kmax = kappa_max(acc1, acc2)
    kmin = kappa_min(acc1, acc2)

    assert (
        kappa <= kmax
    ), f"Requested kappa ({kappa}) is greater than maximum possible ({kmax}), given marginals ({acc1, acc2})!"

    assert (
        kappa >= kmin
    ), f"Requested kappa ({kappa}) is smaller than minimum possible ({kmin}), given marginals ({acc1, acc2})!"

    # estimate the underlying parameters of the copy probability model: copy
    # probability and the two observers' underlying accuracies
    pcopy, uacc1, uacc2 = estimate_copy_model_parameters(kappa, acc1, acc2)

    # draw the trials for obs1 from a binomial distribution
    trials1 = rng.choice(a=[0, 1], size=n_trials, replace=True, p=[1 - uacc1, uacc1])

    # get trials for obs2 by copying from obs1
    n_copy = abs(int(np.round(pcopy * n_trials)))  # how many trials should be copied
    copied_trials = trials1[0:n_copy]
    if pcopy < 0:
        copied_trials = np.logical_not(copied_trials)
    trials2 = copied_trials

    # draw the remaining trials and append them
    if n_trials - n_copy > 0:
        remainder = rng.choice(
            a=[0, 1], size=n_trials - n_copy, replace=True, p=[1 - uacc2, uacc2]
        )
        trials2 = np.concatenate((copied_trials, remainder))

    return trials1, trials2


def simulate_trials_exact(
    kappa: float, acc1: float, acc2: float, n_trials: int, shuffle: bool = True
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Generates n_trials binary trials for two observers with accuracies
    acc1 and acc2, respectively, with a ground-truth EC between the two of kappa.
    This guarantees that the generated trials fulfill the requirements,
    up to the precision limits imposed by n_trials.
    This also returns the targeted EC, because the exact provided EC may be impossible
    at the given number of trials.

    :param ec: the (latent) true EC between the two observers
    :param acc1: the accuracy of the first observer, who has higher accuracy
    :param acc2: the accuracy of the second observer, who has lower accuracy
    :param n_trials: the number of trials to be drawn
    :param shuffle: whether to shuffle the values

    :returns: triple of target-EC and two binary np-arrays of length n_trials
    """

    assert 0.0 <= acc1 <= 1.0, "Accuracy 1 not within bounds."
    assert 0.0 <= acc2 <= 1.0, "Accuracy 2 not within bounds."
    assert n_trials > 0, "Can only generate at least one trial."

    # canonical form: classifier 2 is the better one
    if acc1 > acc2:
        temp = acc1
        acc1 = acc2
        acc2 = temp
        idx1 = 0
        idx0 = 1
    else:
        idx0 = 0
        idx1 = 1

    kmax = kappa_max(acc1, acc2)
    kmin = kappa_min(acc1, acc2)

    assert (
        kappa <= kmax
    ), "Requested kappa is greater than maximum possible, given marginals!"
    assert (
        kappa >= kmin
    ), "Requested kappa is smaller than minimum possible, given marginals!"

    # calculate into how many correct trials the accuracy translates
    n_correct_trials1 = int(np.round(n_trials * acc1))
    n_correct_trials2 = int(np.round(n_trials * acc2))

    # revert the accuracies back to proportions, assuring realistic values
    acc1 = n_correct_trials1 / n_trials
    acc2 = n_correct_trials2 / n_trials
    exp = acc1 * acc2 + (1.0 - acc1) * (1.0 - acc2)

    # the consistency matrix will be
    # [TT, TF]
    # [FT, FF]

    # first, get starting matrix, which has maximum kappa:
    cell_TT = n_correct_trials1  # correct trials of the worse observer
    cell_FF = n_trials - n_correct_trials2  # incorrect trials of the better observer
    cell_FT = (
        n_correct_trials2 - n_correct_trials1
    )  # additional correct trials of the second observer
    cell_TF = 0  # empty for now

    # second, figure out how many trials we need to move
    needed_observed = (kappa * (1 - exp) + exp) * n_trials
    n_move = (cell_TT + cell_FF - needed_observed) / 2

    # third, move half of the needed ones from TT to TF and the other half from FF to FT
    cell_TT -= int(np.ceil(n_move))
    cell_TF += int(np.ceil(n_move))
    cell_FF -= int(np.floor(n_move))
    cell_FT += int(np.floor(n_move))

    trials = (
        [[1, 1]] * cell_TT
        + [[1, 0]] * cell_TF
        + [[0, 1]] * cell_FT
        + [[0, 0]] * cell_FF
    )
    trials = np.array(trials)

    if shuffle:
        new_idx = rng.choice(np.arange(0, n_trials), size=n_trials, replace=False)
        trials = trials[new_idx, :]

    trials1 = trials[:, idx0]
    trials2 = trials[:, idx1]
    target_kappa = fast_cohen_core(trials1, trials2)

    return target_kappa, trials1, trials2


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robert does not use all experiments and conditions in his analysis, but removes those that make no sense.

    :param df: the raw df containing all experiments
    :returns: the filtered df, containing only experiments and conditions that should actually be included
    """

    # list all experiment + condition combos that should be excluded
    # we just need to adjust the naming, because I call the eidolon conditions differently
    # see https://github.com/bethgelab/model-vs-human/blob/79ed7cd1d9ac3a114e71b6638f0a98bdd627316c/modelvshuman/plotting/plot.py#L53
    exclude_conditions = {
        "colour": ["cr"],
        "contrast": ["c100", "c03", "c01"],
        "high-pass": ["inf", "0.55", "0.45", "0.4"],
        "low-pass": ["0", "15", "40"],
        "phase-scrambling": ["0", "150", "180"],
        "power-equalisation": ["0"],
        "false-colour": ["True"],
        "rotation": ["0"],
        "eidolonI": [
            str(int(np.log2(i))) for i in [1, 64, 128]
        ],  # ["1-10-10", "64-10-10", "128-10-10"],
        "eidolonII": [
            str(int(np.log2(i))) for i in [1, 32, 64, 128]
        ],  # ["1-3-10", "32-3-10", "64-3-10", "128-3-10"],
        "eidolonIII": [
            str(int(np.log2(i))) for i in [1, 16, 32, 64, 128]
        ],  # ["1-0-10", "16-0-10", "32-0-10", "64-0-10", "128-0-10"],
        "uniform-noise": ["0.0", "0.6", "0.9"],
    }

    for ex_exp, ex_cons in exclude_conditions.items():
        for ex_con in ex_cons:
            sub = df[(df["experiment"] == ex_exp) & (df["condition"] == ex_con)]
            assert len(sub) > 0, "Mismatch in exclude conditions naming"

    mask = df.apply(
        lambda row: row["condition"]
        not in exclude_conditions.get(row["experiment"], []),
        axis=1,
    )
    filtered_df = df[mask]
    return filtered_df


def calculate_ec_pvalue(
    ec: float, n_correct_1: float, n_correct_2: float, n_trials: int, n_bootstraps: int
) -> float:
    """
    Test whether the empirical EC found between two observers is significant, by
    comparing to a null hypothesis of independent binomial observers.

    :param ec: the empirical EC
    :param n_correct_1: number of correct trials of the first classifier
    :param n_correct_2: number of correct trials of the second classifier
    :param n_trials: the number of trials that were conducted to get ec
    :param n_bootstraps: how many samples from independent binomial observers we need

    :returns: the p-value of this EC measurement
    """

    assert -1 <= ec <= 1, "EC not within bounds."
    assert (
        n_correct_1 <= n_trials
    ), "Impossible to get that many trials correct for total number of trials."
    assert (
        n_correct_2 <= n_trials
    ), "Impossible to get that many trials correct for total number of trials."
    assert n_bootstraps > 0, "Do at least one bootstrap."

    p1_posterior_samples = rng.beta(
        n_correct_1, n_trials - n_correct_1, size=n_bootstraps
    )
    p2_posterior_samples = rng.beta(
        n_correct_2, n_trials - n_correct_2, size=n_bootstraps
    )

    # simulate ECs under the null hypothesis
    ecs_under_null = np.empty(shape=(n_bootstraps))
    for i in range(n_bootstraps):
        p1 = p1_posterior_samples[i]
        p2 = p2_posterior_samples[i]
        independent_trials1 = rng.choice(
            a=[0, 1], size=n_trials, replace=True, p=[1 - p1, p1]
        )
        independent_trials2 = rng.choice(
            a=[0, 1], size=n_trials, replace=True, p=[1 - p2, p2]
        )
        ecs_under_null[i] = fast_cohen(independent_trials1, independent_trials2)

    # find number of times the null hypothesis produced ECs at least as extreme as ec

    ecs_under_null = np.abs(np.array(ecs_under_null))
    n = len(ecs_under_null[ecs_under_null > np.abs(ec)])

    return n / n_bootstraps


def error_consistency(
    trials1: np.ndarray, trials2: np.ndarray, n_bootstraps: int = 10000
) -> Tuple[float, float]:
    """
    Calculates the Error Consistency and p-value for two sets of trials.

    :param trials1: 1d array of trials of classifier 1
    :param trials2: 1d array of trials of classifier 2
    :param n_bootstraps: the number of bootstraps to conduct to estimate p-value

    :returns: a tuple of EC and p-value
    """

    assert isinstance(trials1, np.ndarray), "Expected a numpy array."
    assert isinstance(trials2, np.ndarray), "Expected a numpy array."
    assert len(trials1) == len(trials2), "Sequences must be of equal length."
    assert is_binary(trials1) and is_binary(trials2), "Sequences must be binary."
    assert n_bootstraps > 0, "Must do at least one bootstrap."

    if np.all(trials1) or np.all(trials2):
        if np.all(trials1) and np.all(trials2):
            print(
                "WARNING: EC is undefined because both observers had perfect accuracy!"
            )
        else:
            print("WARNING: EC is 0 because one observer was perfect!")

    if not np.any(trials1) or not np.any(trials2):
        if not np.any(trials1) and not np.any(trials2):
            print("WARNING: EC is undefined because both observers had accuracy of 0!")
        else:
            print("WARNING: EC is 0 because one observer had accuracy of 0!")

    ec = fast_cohen(trials1, trials2)
    p_value = calculate_ec_pvalue(
        ec, np.sum(trials1), np.sum(trials2), len(trials1), n_bootstraps
    )

    return ec, p_value


def calc_accuracy_bounds_from_kappa(acc1: float, kappa: float) -> Tuple[float, float]:
    """
    Calculates the lower and upper bound on the accuracy that observer 2 can have,
    given that observer 1 has accuracy acc1 and their EC is kappa.

    :param acc1: the accuracy of observer 1
    :param kappa: the EC between the two

    :returns: tuple of lower and upper bound
    """

    assert 0.0 <= acc1 <= 1.0, "Accuracy 1 not within bounds."
    assert -1.0 <= kappa <= 1.0, "Kappa not within bounds."

    def calc_upper_acc_bound(a1, k):
        a2_max = a1 * (2 - k) / (k + 2 * a1 * (1 - k))
        return a2_max

    def calc_lower_acc_bound(a1, k):
        a2_min = a1 * k / (2 * a1 * k - 2 * a1 - k + 2)
        return a2_min

    bounds = (
        calc_lower_acc_bound(acc1, kappa),
        calc_upper_acc_bound(acc1, kappa),
    )
    return bounds
