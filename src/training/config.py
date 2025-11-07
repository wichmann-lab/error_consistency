"""
Getting config
"""

import logging
import pprint
import sys
from argparse import ArgumentParser
from datetime import datetime as dt

import numpy as np
import torch

pp = pprint.PrettyPrinter(indent=2, depth=10)


def prettyprint(mydict):
    """Prints a dictionary with prettier styling."""
    pp.pprint(mydict)


# set print config
np.set_printoptions(suppress=True)
torch.set_printoptions(precision=10)
torch.set_printoptions(sci_mode=False)

# set logging config
logging.basicConfig(
    format="%(levelname)s:  %(asctime)s %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)


def get_config():
    """Reads command line arguments and builds config dict."""

    # parse command line arguments
    parser = ArgumentParser()
    parser.add_argument(
        "--wandb_user",
        "-wu",
        type=str,
        default="none",
        choices=["none", "thoklei"],
        help="Which wandb user to be used. Default: not using wandb.",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Whether to start from pretrained weights.",
    )
    parser.add_argument(
        "--ffcv_path",
        "-ffcvp",
        type=str,
        default="/scratch_local/datasets/ImageNet-ffcv",  # "/mnt/lustre/datasets/ffcv_imagenet_data"  #"/scratch_local/datasets/ffcv_imagenet_data"
    )
    parser.add_argument(
        "--save_path",
        "-sp",
        type=str,
        default="/mnt/lustre/work/bethge/tklein16/projects/ec2/train_out",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Whether to use dev-settings, i.e. no model saving etc.",
    )
    parser.add_argument(
        "--no_tqdm", action="store_true", help="Whether to silence tqdm output."
    )
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # get device and show available GPUs
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logging.error("No GPUs found")
        sys.exit(1)
    logging.info(f"Found device {device} and {num_gpus} GPUs.")
    gpu_factor = 12  # how many CPUs to use for feeding 1 GPU - FFCV training is CPU-bound, use 12 or more

    # build configuration

    hardcoded_args = dict(
        # basics
        model="resnet18",
        wandb=args.wandb_user != "none",
        epochs=90,
        num_gpus=num_gpus,  # storing how many gpus are available
        gpu_factor=gpu_factor,  # how many cpus to use as workers per GPU
        num_workers=gpu_factor
        * num_gpus,  # how many workers the dataloaders should use
        device=device,
        start_time=str(dt.now().strftime("%Y-%m-%d_%H-%M-%S")),
        log_interval=100,  # every n minibatches, data is logged to wandb
        test_batch_size=args.batch_size,
        # model validation / saving choices
        validate_interval=1,  # every n-th epoch, we also validate
        validate_first=True,
        model_checkpoint_interval=5,
        save_initial=True,
        # learning rate
        n_plateau_epochs=8,  # for how many epochs the LR should plateau at its max
        lr_decay=0.95,  # exponential decay factor for LR
        # image resolution - just doing full resolution for now
        validation_res=224,
        train_res_min=192,  # 160
        train_res_max=192,
    )

    # join the two dictionaries
    config = {**hardcoded_args, **vars(args)}

    if config["dev"]:
        config["validate_first"] = False
        config["save_initial"] = False
        config["validate_interval"] = 10

    print("Config: ")
    prettyprint(config)

    return config
