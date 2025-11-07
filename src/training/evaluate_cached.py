"""
Evaluates a directory of model checkpoints, by loading each checkpoint and evaluating it on the ImageNet validation set.
"""

import os
import pickle
from argparse import ArgumentParser
from os.path import join as pjoin
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import webdataset as wds
from PIL import Image
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from tqdm import tqdm


class MemoryCachedDataset(Dataset):
    """
    Caches the entire dataset in memory for a 3x speedup.
    Requires about 35GB of RAM.
    """

    def __init__(self, dataloader, batchsize: int) -> None:

        self.samples = []
        self.batchsize = batchsize

        for batch in tqdm(dataloader, total=50_000 // batchsize):
            images, labels, paths = batch[:3]
            for img, label, path in zip(images, labels, paths):
                self.samples.append((img, label, path))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        return self.samples[idx]

    def __len__(self) -> int:
        return len(self.samples)


@torch.no_grad()
def eval_model(model, args, file, cached_dataset) -> None:
    print(f"Evaluating {file}")

    # construct output filename
    os.makedirs(args.output_dir, exist_ok=True)
    fname = pjoin(args.output_dir, f"{file.replace('.pt', '.pd')}.pkl")
    if os.path.exists(fname):
        print(f"Skipping {fname} because it already exists")
        return

    cached_loader = DataLoader(
        cached_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=0,  # No need for workers — all in memory
        pin_memory=True,
    )

    correct = 0
    n_samples = 0
    all_paths = []  # will contain all paths in order
    all_labels = np.zeros((50_000, 1), dtype=np.int32)
    all_predictions = np.zeros((50_000, 1), dtype=np.int32)
    start_idx = 0  # index for the arrays into which we store data
    for idx, (images, labels, paths) in enumerate(
        tqdm(cached_loader, total=50_000 // args.batchsize)
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


def cache_dataset(datadir, transform, batch_size):
    print(f"Caching dataset from {datadir}")
    # webdataset -> streaming loader
    wds_dataset = (
        wds.WebDataset(datadir, shardshuffle=False)
        .shuffle(False)
        .decode("pil")
        .map(lambda sample: {**sample, "filename": sample["json"]["filename"]})
        .to_tuple("jpg cls filename")
        .map_tuple(transform, lambda x: x, lambda x: x)
        .batched(batch_size, partial=True)
    )
    wds_loader = wds.WebLoader(
        wds_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
        pin_memory=True,
    )
    cached_dataset = MemoryCachedDataset(wds_loader, batch_size)
    return cached_dataset


def get_transforms():
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    val_transforms = T.Compose(
        [
            T.Resize(256, interpolation=Image.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.ConvertImageDtype(torch.float16),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return val_transforms


def main(args):

    # 0. define transforms matching the transforms used in the training script
    val_transforms = get_transforms()

    # 1. load entire validation set into RAM
    cached_dataset = cache_dataset(
        datadir=pjoin(
            "/mnt/lustre/datasets/imagenet-1k-wds", "imagenet1k-validation-{00..63}.tar"
        ),
        transform=val_transforms,
        batch_size=args.batchsize,
    )

    root, dirs, files = next(os.walk(args.checkpoint_dir))
    for file in files:
        if file.endswith(".pt"):
            model_path = os.path.join(root, file)
            model = load_model(model_path, args.device)
            eval_model(model, args, file, cached_dataset)


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
