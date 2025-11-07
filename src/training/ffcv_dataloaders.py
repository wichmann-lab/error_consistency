"""
FFCV dataloaders.
"""

import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from ffcv.fields.basics import IntDecoder
from ffcv.fields.rgb_image import (
    CenterCropRGBImageDecoder,
    RandomResizedCropRGBImageDecoder,
    SimpleRGBImageDecoder,
)
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (
    NormalizeImage,
    RandomHorizontalFlip,
    Squeeze,
    ToDevice,
    ToTensor,
    ToTorchImage,
)

IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224 / 256


def create_train_loader(
    train_dataset,
    num_workers,
    batch_size,
    distributed,
    in_memory,
    device,
    image_size=IMG_SIZE,
):
    train_path = Path(train_dataset)
    assert train_path.is_file()

    # decoder = CenterCropRGBImageDecoder((image_size, image_size), DEFAULT_CROP_RATIO),
    decoder = RandomResizedCropRGBImageDecoder((image_size, image_size))

    image_pipeline = [
        decoder,
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(torch.device(device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
    ]

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(device), non_blocking=True),
    ]

    order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
    loader = Loader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        order=order,
        os_cache=in_memory,
        drop_last=True,
        pipelines={"image": image_pipeline, "label": label_pipeline},
        # batches_ahead=10,
        distributed=distributed,
    )

    return loader, decoder


def create_test_loader(
    test_dataset,
    num_workers,
    batch_size,
    distributed,
    in_memory,
    device,
    image_size=IMG_SIZE,
):
    test_path = Path(test_dataset)
    assert test_path.is_file()

    decoder = CenterCropRGBImageDecoder((image_size, image_size), DEFAULT_CROP_RATIO)

    image_pipeline = [
        decoder,
        ToTensor(),
        ToDevice(torch.device(device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
    ]

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(device), non_blocking=True),
    ]

    loader = Loader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        order=OrderOption.SEQUENTIAL,
        os_cache=in_memory,
        drop_last=False,
        pipelines={"image": image_pipeline, "label": label_pipeline},
        distributed=distributed,
    )

    return loader


def get_ffcv_loaders(config):
    """
    Gets FFCV dataloaders

    :param config: the global config dict

    :returns: tuple (trainloader, testloader, decoder)
        where decoder is the image decoder used during training. This is done to
        be able to change the resolution arbitrarily, which I use for scaling.
    """
    trainloader, decoder = create_train_loader(
        os.path.join(config["ffcv_path"], "train_500_0.50_90.ffcv"),
        config["num_workers"],
        config["batch_size"],
        distributed=False,
        in_memory=False,
        device=config["device"],
        image_size=config["train_res_min"],
    )

    testloader = create_test_loader(
        os.path.join(config["ffcv_path"], "val_500_0.50_90.ffcv"),
        config["num_workers"],
        config["test_batch_size"],
        distributed=False,
        in_memory=False,
        device=config["device"],
        image_size=config["validation_res"],
    )

    return trainloader, testloader, decoder


def get_loaders(config):
    """
    Gets non-FFCVdataloaders

    :param config: the global config dict
    """
    raise NotImplementedError("Not implemented")
