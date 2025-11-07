"""
Bootstrapping the Error Consistency data, to give CIs for Robert's figure of ECs for every condition.

I'm doing this outside of the jupyter kernel, because I thought it might be necessary
to speed this up using numba / mypyc, but seems like numpy is fast enough.
"""

import os
import sys
import warnings
from argparse import ArgumentParser
from os.path import join as pjoin
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(".."))
from utils import fast_cohen


def split_ecs(
    values: np.ndarray, id: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates three sets of pairwise error consistencies

    :param values: a N x N array of error consistency values
    :param id: a N array indicating whether this column / row is a human or a machine
    """

    N = values.shape[0]
    upper_tri_mask = np.triu(np.ones_like(values, dtype=bool), k=0)

    # Broadcast row and column masks
    id_row = id[:, None]  # column vector (N x 1)
    id_col = id[None, :]  # row vector    (1 x N)

    # Create the three masks
    mask_11 = id_row & id_col
    mask_00 = ~id_row & ~id_col
    mask_xor = id_row ^ id_col

    # Apply the masks to the values array
    hhec = values[mask_11 & upper_tri_mask]
    mmec = values[mask_00 & upper_tri_mask]
    hmec = values[mask_xor & upper_tri_mask]

    return hhec, mmec, hmec


def calc_pairwise_ecs(responses: np.ndarray, ec_type: str) -> np.ndarray:
    n, m = responses.shape
    result = np.full((m, m), np.nan)

    for i in range(m):
        for j in range(i + 1, m):  # only upper triangle excluding diagonal
            if ec_type == "standard":
                result[i, j] = fast_cohen(responses[:, i], responses[:, j])
            else:
                raise NotImplementedError(f"Unrecognized EC type! {ec_type}")

    return result


def bootstrap(
    n_bootstrap_trials: int, ec_type: str, real_trials: np.ndarray, human: np.ndarray
) -> np.ndarray:
    """
    Bootstraps error consistencies for N observers, given their real trials.

    :param n_bootstrap_trials: the number of times we want to bootstrap
    :param ec_type: what kind of Error Consistency to calculate (standard or with normalization)
    :param real_trials: the real trials of the observers, 2d array (trial_id x subject)
    :param human: an array of length n_subjects, indicating whether the subject is human or not

    :return: a dict mapping comparison type to array of mean EC values
    """
    # a 3 x N array of humanhuman, machine-machine, and human-machine ECs
    mean_ecs = np.full((3, n_bootstrap_trials), np.nan)

    n, m = real_trials.shape
    for i in range(n_bootstrap_trials):

        row_indices = np.random.choice(n, size=n, replace=True)

        resampled = real_trials[row_indices]

        pairwise_ecs = calc_pairwise_ecs(resampled, ec_type)

        hhec, mmec, hmec = split_ecs(pairwise_ecs, human)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Mean of empty slice"
            )  # can happen, is fine
            mean_ecs[0, i] = np.nanmean(hhec)
            mean_ecs[1, i] = np.nanmean(mmec)
            mean_ecs[2, i] = np.nanmean(hmec)

    return {
        "human-human": mean_ecs[0, :],
        "machine-machine": mean_ecs[1, :],
        "human-machine": mean_ecs[2, :],
    }


def main(all_df: pd.DataFrame, n_bootstrap_trials: int, ec_type: str) -> None:

    # this will create a new dataframe with the following columns:
    experiments = []
    conditions = []
    bootstrap_ids = []  # for every bootstrapped similarity, we give an index
    human_human_ecs = []
    machine_machine_ecs = []
    human_machine_ecs = []

    # loop over all experiments
    for experiment, exp_df in tqdm(all_df.groupby("experiment", observed=True)):

        # within one experiment, loop over all conditions
        for condition, cond_df in exp_df.groupby("condition", observed=True):

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
            resp_array = trimmed.to_numpy(dtype=bool)
            human = np.array(
                ["subject-" in col for col in trimmed.columns]
            )  # whether this column is a human or not
            ecs = bootstrap(n_bootstrap_trials, ec_type, resp_array, human)

            experiments.extend([experiment] * n_bootstrap_trials)
            conditions.extend([condition] * n_bootstrap_trials)
            bootstrap_ids.extend(np.arange(n_bootstrap_trials).tolist())
            human_human_ecs.extend(ecs["human-human"].tolist())
            machine_machine_ecs.extend(ecs["machine-machine"].tolist())
            human_machine_ecs.extend(ecs["human-machine"].tolist())

    ec_df = pd.DataFrame(
        {
            "experiment": experiments,
            "condition": conditions,
            "bootstrap_id": bootstrap_ids,
            "human-human-ec": human_human_ecs,
            "machine-machine-ec": machine_machine_ecs,
            "human-machine-ec": human_machine_ecs,
        }
    )

    ec_df["experiment"] = ec_df["experiment"].astype("category")
    ec_df["condition"] = ec_df["condition"].astype("category")
    save_path = pjoin("data", f"bootstrapped_ecs_{ec_type}.parquet")

    print("Created df with", len(ec_df), "lines!")
    print(ec_df.head())
    print(ec_df.info())
    print("Saving to", save_path)

    ec_df.to_parquet(save_path, engine="pyarrow")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n-bootstrap-trials", "-n", type=int, default=1_000)
    parser.add_argument("--ec-type", type=str, choices=["standard"], required=True)
    args = parser.parse_args()

    all_df = pd.read_parquet("data/roberts_raw_data.parquet", engine="pyarrow")

    main(all_df, args.n_bootstrap_trials, args.ec_type)
