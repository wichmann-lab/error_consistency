"""
Bootstrapping the Error Consistency data between humans (see plot_model_differences.ipynb).

"""

import os
import sys
import warnings
from argparse import ArgumentParser
from os.path import join as pjoin
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(".."))
from utils import fast_cohen, filter_df


def calc_human_human_ecs(
    human_trials: np.ndarray,
    ec_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Calculates all pairwise error consistencies between a set of humans.

    :param human_trials: the n x l matrix of human responses
    :param ec_func: the vectorized function that takes a list of binary responses and calculates EC

    :returns: a 1d array of length n * ( n - 2 ), containing human-human error consistencies
    """

    n_humans = human_trials.shape[1]
    result = np.full((n_humans, n_humans), np.nan)

    for i in range(n_humans):
        for j in range(i, n_humans):
            # here, we have different options for handling NaN-entries:
            trials1 = human_trials[:, i]
            trials2 = human_trials[:, j]

            # Removing trials were response of either human was na.
            # This only has an effect when args.ignore_nan is true, otherwise trials never contains nan
            indices = ~(np.isnan(trials1) | np.isnan(trials2))
            trials1 = trials1[indices]
            trials2 = trials2[indices]
            result[i, j] = ec_func(trials1, trials2)

    return result[np.triu_indices(n_humans, k=1)]


def bootstrap_core(
    n_bootstrap_trials: int, ec_type: str, real_trials: np.ndarray
) -> np.ndarray:
    """
    Bootstraps error consistencies for N observers, given their real trials.

    :param n_bootstrap_trials: the number of times we want to bootstrap
    :param ec_type: what kind of Error Consistency to calculate (standard or with normalization)
    :param real_trials: the real trials of the observers, 2d array (trial_id x subject)

    :return: a 1d array of length n_bootstrap_trials containing the bootstrapped
        average human-human EC values
    """

    n_real_trials, n_humans = real_trials.shape

    # holds all results, shape: bootstraps x pairs_of_humans
    # not strictly necessary, but having this might be useful if I want to make changes in aggregation later
    result = np.full((n_bootstrap_trials, n_humans * (n_humans - 1) // 2), np.nan)

    assert ec_type in ["standard"], "Unknown EC type"
    ec_func = fast_cohen  # if at some point in the future we do something else

    # for debugging and sanity checks, the first entry is always the real one
    result[0, :] = calc_human_human_ecs(real_trials, ec_func)

    # loop over remaining trials
    for i in range(1, n_bootstrap_trials):

        row_indices = np.random.choice(n_real_trials, size=n_real_trials, replace=True)

        resampled = real_trials[row_indices]

        # calculate human-human ECs for all pairs of humans
        result[i, :] = calc_human_human_ecs(resampled, ec_func)

    # take the mean across the humans, resulting in 1d array of length n_bootstrap_trials
    # if a bootstrap resulted in a perfect score, the EC to all humans will be NaN, which is fine
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        mean = np.nanmean(result, axis=1)
    return mean


def bootstrap(df: pd.DataFrame, n_bootstrap_trials: int, ec_type: str) -> None:

    # this will create a new dataframe with the following columns:
    experiments = []
    conditions = []
    bootstrap_ids = []  # for every bootstrapped similarity, we give an index
    human_human_ecs = []

    # loop over all experiments
    for experiment, exp_df in tqdm(df.groupby("experiment", observed=True)):

        # loop over all conditions
        for condition, cond_df in tqdm(exp_df.groupby("condition", observed=True)):

            # idea: unmelt the df, to get like:
            # image_id subj1 subj2 subj3
            #        0     0     1     1
            # ...
            # then, randomly select n_rows rows with replacement
            # to speed this up, we do this over the raw numpy array of values
            pivot_df = cond_df.pivot(
                index="img_identifier", columns="subj", values="correct"
            ).reset_index()

            trimmed = pivot_df.drop(columns="img_identifier")
            resp_array = trimmed.to_numpy(dtype=float)

            ecs = bootstrap_core(n_bootstrap_trials, ec_type, resp_array)

            experiments.extend([experiment] * n_bootstrap_trials)
            conditions.extend([condition] * n_bootstrap_trials)
            bootstrap_ids.extend(np.arange(n_bootstrap_trials).tolist())
            human_human_ecs.extend(ecs.tolist())

    ec_df = pd.DataFrame(
        {
            "experiment": experiments,
            "condition": conditions,
            "bootstrap_id": bootstrap_ids,
            "human-human-ec": human_human_ecs,
        }
    )

    ec_df["experiment"] = ec_df["experiment"].astype("category")
    ec_df["condition"] = ec_df["condition"].astype("category")

    return ec_df


def main(all_df: pd.DataFrame, n_bootstrap_trials: int, ec_type: str) -> None:

    # 1. filter the df with raw data to include only those conditions that Robert also includes in his analysis.
    filtered_df = filter_df(all_df)

    # 2. for every condition, bootstrap the data
    ec_df = bootstrap(filtered_df, n_bootstrap_trials, ec_type)

    # 3. save the results
    save_path = pjoin(
        "data", f"bootstrapped_human_ecs_{ec_type}_{n_bootstrap_trials}.parquet"
    )
    print("Created df with", len(ec_df), "lines!")
    print(ec_df.groupby("experiment", observed=True).head(20))
    print(ec_df.info())
    print("Saving to", save_path)
    ec_df.to_parquet(save_path, engine="pyarrow")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n-bootstrap-trials", "-n", type=int, default=1_000)
    parser.add_argument("--ec-type", type=str, choices=["standard"], default="standard")
    # Whether rows with NA-responses should be dropped, as if they never happened.
    # Otherwise, they are just treated as incorrect responses.
    # Default is to treat them as incorrect responses, i.e. don't pass this argument.
    parser.add_argument("--ignore-nan", action="store_true")
    args = parser.parse_args()

    # load Robert's data but keep only humans
    all_df = pd.read_parquet("data/roberts_raw_data.parquet", engine="pyarrow")
    all_df = all_df[all_df["subj"].str.startswith("subject-")]

    if args.ignore_nan:
        all_df["correct"] = all_df.apply(
            lambda row: np.nan if row["object_response"] == "na" else row["correct"],
            axis=1,
        )

    main(all_df, args.n_bootstrap_trials, args.ec_type)
