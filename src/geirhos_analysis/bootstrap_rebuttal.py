"""
Bootstrapping the Error Consistency data to humans for every individual model (see plot_model_differences.ipynb).
Created for testing significance of difference between two models during the rebuttal.
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


def calc_machine_human_ecs(
    machine_trials: np.ndarray,
    human_trials: np.ndarray,
    ec_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """
    Calculates all error consistencies between a set of machines and a set of humans.

    :param machine_trials: the n x k matrix of machine responses
    :param human_trials: the n x l matrix of human responses
    :param ec_func: the vectorized function that takes a list of binary responses and calculates EC

    :returns: an k x l matrix of human-machine error consistencies
    """

    n_machines = machine_trials.shape[1]
    n_humans = human_trials.shape[1]
    result = np.full((n_machines, n_humans), np.nan)

    for i in range(n_machines):
        for j in range(n_humans):
            # here, we have different options for handling NaN-entries:
            mtrials = machine_trials[:, i]
            htrials = human_trials[:, j]

            # Removing trials were response was na.
            # This only has an effect when args.ignore_nan is true, otherwise htrials is never nan
            indices = ~np.isnan(htrials)
            htrials = htrials[indices]
            mtrials = mtrials[indices]
            result[i, j] = ec_func(mtrials, htrials)

    return result


def bootstrap_core(
    n_bootstrap_trials: int, ec_type: str, real_trials: np.ndarray, is_human: np.ndarray
) -> np.ndarray:
    """
    Bootstraps error consistencies for N observers, given their real trials.

    :param n_bootstrap_trials: the number of times we want to bootstrap
    :param ec_type: what kind of Error Consistency to calculate (standard or with normalization)
    :param real_trials: the real trials of the observers, 2d array (trial_id x subject)
    :param is_human: an array of length n_subjects, indicating whether the subject is human or not

    :return: a 2d array models x n_bootstrap_trials containing the bootstrapped
        average EC values
    """

    # holds all results, shape: models x humans x bootstraps
    # this is not strictly necessary, we could just sum up in a 2d array and divide by n afterwards
    # but I feel like having this might be useful if I want to make changes in aggregation later
    result = np.full(
        (np.sum(is_human == 0), np.sum(is_human == 1), n_bootstrap_trials), np.nan
    )

    n_real_trials = real_trials.shape[0]

    # for debugging, the first entry is always the real one
    machine_trials = real_trials[:, is_human == 0]
    human_trials = real_trials[:, is_human == 1]

    assert ec_type in ["standard"], "Unknown EC type"
    ec_func = fast_cohen  # in case we want to support something else in the future

    result[:, :, 0] = calc_machine_human_ecs(machine_trials, human_trials, ec_func)

    # loop over remaining trials
    for i in range(1, n_bootstrap_trials):

        row_indices = np.random.choice(n_real_trials, size=n_real_trials, replace=True)

        resampled = real_trials[row_indices]

        # calculate machine-human ECs for all models and all humans
        human_trials = resampled[:, is_human == 1]
        machine_trials = resampled[:, is_human == 0].copy()

        # now, the manipulation for modelling the null hypothesis: Randomly exchange the responses of the two models
        flip_mask = np.random.rand(machine_trials.shape[0]) < 0.5
        machine_trials[flip_mask] = machine_trials[flip_mask][:, ::-1]

        result[:, :, i] = calc_machine_human_ecs(machine_trials, human_trials, ec_func)

    # take the mean across the humans, resulting in shape models x n_bootstrap_trials
    # if a bootstrap resulted in a perfect score, the EC to all humans will be NaN, which is fine
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        mean = np.nanmean(result, axis=1)
    return mean


def bootstrap(df: pd.DataFrame, n_bootstrap_trials: int, ec_type: str) -> None:

    # this will create a new dataframe with the following columns:
    experiments = []
    conditions = []
    models = []
    bootstrap_ids = []  # for every bootstrapped similarity, we give an index
    model_human_ecs = []

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

            human = np.array(["subject-" in col for col in trimmed.columns])
            ecs = bootstrap_core(n_bootstrap_trials, ec_type, resp_array, human)

            model_idx = 0
            for model, is_human in zip(trimmed.columns, human):

                if is_human:  # just skip humans
                    continue

                experiments.extend([experiment] * n_bootstrap_trials)
                conditions.extend([condition] * n_bootstrap_trials)
                bootstrap_ids.extend(np.arange(n_bootstrap_trials).tolist())
                models.extend([model] * n_bootstrap_trials)
                model_human_ecs.extend(ecs[model_idx, :])
                model_idx += 1

    ec_df = pd.DataFrame(
        {
            "experiment": experiments,
            "condition": conditions,
            "bootstrap_id": bootstrap_ids,
            "model": models,
            "model-human-ec": model_human_ecs,
        }
    )

    ec_df["experiment"] = ec_df["experiment"].astype("category")
    ec_df["condition"] = ec_df["condition"].astype("category")
    ec_df["model"] = ec_df["model"].astype("category")

    return ec_df


def main(
    all_df: pd.DataFrame,
    n_bootstrap_trials: int,
    ec_type: str,
    out_file: str,
    model1: str,
    model2: str,
) -> None:

    # 1. filter the df with raw data to include only those conditions that Robert also includes in his analysis.
    filtered_df = filter_df(all_df)

    # 1.5 filter out the irrelevant models
    filtered_df = filtered_df[
        (filtered_df["subj"].isin([model1, model2]))
        | (filtered_df["subject_type"] == "human")
    ]

    # 2. for every condition, bootstrap the data
    ec_df = bootstrap(filtered_df, n_bootstrap_trials, ec_type)

    # 3. save the results
    save_path = pjoin(
        "data", f"{out_file}_{ec_type}_{n_bootstrap_trials}_{model1}_{model2}.parquet"
    )
    print("Created df with", len(ec_df), "lines!")
    print(ec_df.groupby("model", observed=True).head(20))
    print(ec_df.info())
    print("Saving to", save_path)
    ec_df.to_parquet(save_path, engine="pyarrow")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input-file", "-i", type=str, default="data/roberts_raw_data.parquet"
    )
    parser.add_argument(
        "--output-file", "-o", type=str, default="rebuttal_model_wise_bootstrapped_ecs"
    )
    parser.add_argument("--n-bootstrap-trials", "-n", type=int, default=1_000)
    parser.add_argument("--ec-type", type=str, choices=["standard"], default="standard")
    parser.add_argument("--model1", "-m1", type=str, default="First model")
    parser.add_argument("--model2", "-m2", type=str, default="Second model")
    # Whether rows with NA-responses should be dropped, as if they never happened.
    # Otherwise, they are just treated as incorrect responses.
    # Default is to treat them as incorrect responses, i.e. don't pass this argument.
    parser.add_argument("--ignore-nan", action="store_true")
    args = parser.parse_args()

    all_df = pd.read_parquet(args.input_file, engine="pyarrow")

    if args.ignore_nan:
        all_df["correct"] = all_df.apply(
            lambda row: np.nan if row["object_response"] == "na" else row["correct"],
            axis=1,
        )

    main(
        all_df,
        args.n_bootstrap_trials,
        args.ec_type,
        args.output_file,
        args.model1,
        args.model2,
    )
