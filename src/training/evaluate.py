"""
Evaluates a directory of model checkpoints, by loading each checkpoint and evaluating it on the ImageNet validation set.
"""

import os
import pickle
from argparse import ArgumentParser
from os.path import join as pjoin

import numpy as np
import pandas as pd
import torch
import webdataset as wds
from PIL import Image
from torch.amp import autocast
from torchvision import transforms as T
from torchvision.models import resnet18
from tqdm import tqdm


def get_wds_loader(
    split: str,
    batch_size: int,
    transform,
    datadir: str = "/mnt/lustre/datasets/imagenet-1k-wds",
):
    """
    Get a WebDataset loader for the ImageNet validation set.

    Args:
        split: str, either "val" or "train"
        batch_size: int
        transform: torchvision.transforms.Compose
        datadir: str

    Returns:
        dataloader: WebDataset loader
    """
    assert batch_size > 0, "Batch Size muste be greater than 0"

    if "lustre" in datadir:
        print("WARNING: Using Lustre dataset")

    if split == "val":
        datadir = pjoin(datadir, "imagenet1k-validation-{00..63}.tar")
    elif split == "train":
        datadir = pjoin(datadir, "imagenet1k-train-{0000..1023}.tar")
        print("Don't use this for training, unshuffled data!")
        raise RuntimeError("Not suitable for training!")
    else:
        raise RuntimeError("Illegitimate split, has to be either train or val!")

    dataset = (
        wds.WebDataset(datadir, shardshuffle=False)
        .shuffle(False)
        .decode("pil")
        .map(lambda sample: {**sample, "filename": sample["json"]["filename"]})
        .to_tuple("jpg cls filename")
        .map_tuple(transform, lambda x: x, lambda x: x)
        .batched(batch_size, partial=True)
    )
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=1,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=True,
    )
    return dataloader


def eval_model(model, args, file):
    print(f"Evaluating {file}")

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    # exactly matching the transforms used in the training script
    val_transforms = T.Compose(
        [
            T.Resize(256, interpolation=Image.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float16),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    batchsize = args.batchsize
    while batchsize > 15:
        try:
            record_properties(args, file, model, val_transforms, batchsize)
            return
        except torch.cuda.OutOfMemoryError as e:
            print(
                "CUDA out of memory error, retrying with smaller batch size "
                "({0} instead of {1}).".format(batchsize // 2, batchsize)
            )
            batchsize = batchsize // 2

    print("Didn't work with batchsize >= 16, exiting.")


@torch.no_grad()
def record_properties(args, file_name, model, val_transforms, batchsize):
    """
    Walks over the dataset and records the properties of interest, saving to pkl.
    """

    # construct output filename
    os.makedirs(args.output_dir, exist_ok=True)
    fname = pjoin(args.output_dir, f"{file_name.replace('.pt', '.pd')}.pkl")
    if os.path.exists(fname):
        print(f"Skipping {fname} because it already exists")
        return

    # Create a WDS loader
    loader = get_wds_loader(
        split="val",
        batch_size=batchsize,
        transform=val_transforms,
    )

    correct = 0
    n_samples = 0
    all_paths = []  # will contain all paths in order
    all_labels = np.zeros((50_000, 1), dtype=np.int32)
    all_predictions = np.zeros((50_000, 1), dtype=np.int32)
    start_idx = 0  # index for the arrays into which we store data
    for idx, (images, labels, paths) in enumerate(
        tqdm(loader, total=50_000 // batchsize)
    ):

        images = images.to(args.device)
        labels = labels.to(args.device)

        assert len(paths) == len(labels), "Got different number of paths."

        # feed images through model and record correctness
        with autocast("cuda"):
            out = model(images)
            _, predicted = torch.max(out.data, dim=1)
            correct += (predicted == labels).sum().item()
            n_samples += len(labels)

        # record the outputs
        all_paths.extend(paths)
        all_labels[start_idx : start_idx + len(paths), :] = labels.unsqueeze(1).numpy(
            force=True
        )
        all_predictions[start_idx : start_idx + len(paths), :] = predicted.unsqueeze(
            1
        ).numpy(force=True)

        start_idx = start_idx + len(paths)

    accuracy = 100 * correct / n_samples
    print(f"{accuracy=:.2f}%")

    # make df of resulting data
    df = pd.DataFrame(
        {
            "Path": all_paths,
            "Label": all_labels.flatten(),
            "Prediction": all_predictions.flatten(),
        }
    )

    # save to file
    print(f"Dumping data to {fname}")
    with open(fname, "wb") as fhandle:
        pickle.dump(df, fhandle)


def load_model(model_path, device):
    model_weights = torch.load(model_path, weights_only=True)
    model = resnet18(weights=None)
    model.load_state_dict(model_weights)
    model.eval()
    model = model.to(device)
    return model


def main(args):

    root, _, files = next(os.walk(args.checkpoint_dir))
    for file in files:
        if file.endswith(".pt"):
            model_path = os.path.join(root, file)
            model = load_model(model_path, args.device)
            eval_model(model, args, file)


if __name__ == "__main__":
    BASE_DIR = "/mnt/lustre/work/bethge/tklein16/projects/ec2/train_out/results/"
    parser = ArgumentParser()
    parser.add_argument(
        "--run_name", type=str, required=True
    )  # path to directory containing model checkpoints
    parser.add_argument(
        "--output_dir", type=str, default=pjoin(BASE_DIR, "evaluations")
    )
    parser.add_argument("--batchsize", type=int, default=1024)
    args = parser.parse_args()

    args.checkpoint_dir = pjoin(BASE_DIR, args.run_name)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(args)
