"""
Training a model on ImageNet.

python3.10 train_imagenet.py -wu thoklei
"""

import json
import logging
import os
import time

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.autograd.profiler.emit_nvtx(False)
torch.autograd.profiler.profile(False)

from config import get_config
from ffcv_dataloaders import get_ffcv_loaders

# read command line arguments and set all config values
config = get_config()

# set wandb properties - important to do this before importing wandb
if config["wandb"]:

    # read wandb api key
    with open(
        f"wandb_api_key_{config['wandb_user']}.key", "r", encoding="utf-8"
    ) as keyfile:
        os.environ["WANDB_API_KEY"] = keyfile.readline()

    WANDB_ENTITY = config["wandb_user"]
    WANDB_PROJECT = "error_consistency"
    # os.environ["WANDB_TEMP"] = "/home/bethge/tklein16/ec2/wandb_temp" #os.path.join(config['save_path'], 'wandb')
    os.environ["WANDB_DIR"] = os.path.join(config["save_path"], "wandb")
    os.environ["WANDB_DATA_DIR"] = os.path.join(config["save_path"], "wandb_data")
    os.environ["WANDB_ARTIFACT_LOCATION"] = os.path.join(
        config["save_path"], "wandb_artifacts"
    )
    os.environ["WANDB_ARTIFACT_DIR"] = os.path.join(
        config["save_path"], "wandb_artifact"
    )
    os.environ["WANDB_CACHE_DIR"] = os.path.join(config["save_path"], "wandb_cache")
    os.environ["WANDB_CONFIG_DIR"] = os.path.join(config["save_path"], "wandb_config")
    os.environ["WANDB_RUN_DIR"] = os.path.join(config["save_path"], "wandb_run")

os.makedirs(os.path.join(config["save_path"], "results"), exist_ok=True)

np.random.seed(config["seed"])
torch.manual_seed(config["seed"])

if config["wandb"]:
    import wandb

    wandb.login()
    wandb_run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        # name=f"resnet18",
        config=config,
        save_code=True,
        settings=wandb.Settings(code_dir="."),
    )
    wandb.define_metric("train_step")  # counts global training steps
    wandb.define_metric("epoch")
    # define which metrics will be plotted against it
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="epoch")

# get the model
if config["model"] == "resnet18":
    model = core_model = resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1 if config["pretrained"] else None
    )
else:
    raise NotImplementedError(f"Model name {config['model']} not supported!")


def lr_decay(step):
    """Exponential Decay."""
    if step > config["n_plateau_epochs"]:
        return config["lr_decay"] ** (step - config["n_plateau_epochs"])
    return 1.0


if config["num_gpus"] > 1:
    # TODO properly support parallelism, with distributed
    logging.info("Enabling model parallelism.")
    model = torch.nn.DataParallel(model)

scaler = GradScaler("cuda")
model = model.to(memory_format=torch.channels_last)
model = model.to(config["device"])


def get_resolution(epoch, min_res, max_res, end_ramp, start_ramp):
    """Get the appropriate resolution for the current epoch. Taken from FFCV."""
    assert min_res <= max_res

    if epoch <= start_ramp:
        return min_res

    if epoch >= end_ramp:
        return max_res

    # otherwise, linearly interpolate to the nearest multiple of 32
    interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
    final_res = int(np.round(interp[0] / 32)) * 32
    return final_res


# put weight decay on normal params, not batch norm params
all_params = list(model.named_parameters())
bn_params = [v for k, v in all_params if ("bn" in k)]
other_params = [v for k, v in all_params if not ("bn" in k)]
param_groups = [
    {"params": bn_params, "weight_decay": 0.0},
    {"params": other_params, "weight_decay": 4e-5},
]
criterion = torch.nn.CrossEntropyLoss()  # label_smoothing=0.1
optimizer = torch.optim.Adam(param_groups, lr=config["learning_rate"])
decay_scheduler = LambdaLR(optimizer, lr_decay)


def train_one_epoch(epoch):
    """Do one epoch of training."""
    logging.info(f"Starting training epoch {epoch}")

    # to measure train accuracy
    correct_samples = 0
    total_samples = 0

    model.train()

    # adjusting dataloaders
    # get image resolution we want to use in this epoch
    res = get_resolution(
        epoch, config["train_res_min"], config["train_res_max"], 30, 20
    )  # epoch, min_res, max_res, end_ramp, start_ramp

    trainloader, testloader, decoder = get_ffcv_loaders(config)
    decoder.output_size = (res, res)
    print("Got new loaders")

    loader_obj = tqdm(trainloader, disable=config["no_tqdm"])
    time_start = time.time()
    for minibatch_idx, (images, targets) in enumerate(
        loader_obj
    ):  # loop over every minibatch

        # zero the parameter gradients
        optimizer.zero_grad(set_to_none=True)

        # forward pass
        with autocast("cuda"):
            outputs = model(images)

            preds = torch.argmax(outputs, dim=-1)
            correct_samples_batch = (preds == targets).sum()
            correct_samples += correct_samples_batch
            total_samples += targets.shape[0]

            loss = criterion(outputs, targets)

            # log losses during epoch, might not even run long enough
            if config["wandb"] and (minibatch_idx % config["log_interval"] == 0):
                wandb.log(
                    {
                        "train/Cross Entropy": loss.numpy(force=True),
                        "train/Training Set Accuracy (so far) (%)": (
                            correct_samples_batch / config["batch_size"]
                        )
                        * 100,  # noisy but best we have
                        "train/Learning Rate": optimizer.param_groups[0]["lr"],
                        "train_step": epoch * len(trainloader) + minibatch_idx,
                    }
                )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # free memory, please FFCV stop being a pain in the ass
    del loader_obj
    del trainloader
    del testloader
    del decoder
    torch.cuda.empty_cache()

    time_end = time.time()
    # end of train loop

    acc = (correct_samples / total_samples) * 100
    logging.info(f"Model achieved {acc}% training set accuracy.")

    if config["wandb"]:

        wandb.log(
            {
                "eval/Training Set Accuracy (epoch) (%)": acc,
                "eval/Seconds per epoch:": time_end - time_start,
                "epoch": epoch,
            }
        )


@torch.no_grad()
def validate(epoch):
    """Feed the entire validation set and record performance."""

    logging.info("Validating")

    model.eval()  # to deactivate dropout layers etc.

    correct_samples = 0
    total_samples = 0

    _, testloader, _ = get_ffcv_loaders(config)

    loader_obj = tqdm(testloader, disable=config["no_tqdm"])
    for minibatch_idx, (images, targets) in enumerate(loader_obj):

        with autocast("cuda"):
            outputs = model(images)
            ce_loss = criterion(outputs, targets)

        preds = torch.argmax(outputs, dim=-1)
        correct_samples += (preds == targets).sum()
        total_samples += targets.shape[0]

    acc = (correct_samples / total_samples) * 100
    logging.info(f"Model achieved {acc}% validation set accuracy.")

    if config["wandb"]:
        wandb.log({"eval/Validation Set Accuracy (epoch) (%)": acc, "epoch": epoch})

    return acc


def save_model(epoch):
    """
    Stores the model to file and stores the config-dict as well.

    :param epoch: the epoch at the end of which we store
    """

    save_location = os.path.join(config["save_path"], "results")
    if config["wandb"]:
        save_location = os.path.join(save_location, f"{wandb_run.name}")
    os.makedirs(save_location, exist_ok=True)

    # construct meaningful model name
    if config["wandb"]:
        model_name = f"{wandb_run.name}"
    else:
        model_name = f"{config['start_time']}"

    # save the model itself
    savepath = os.path.join(save_location, f"{model_name}_{epoch}.pt")
    torch.save(core_model.state_dict(), savepath)

    # if this is the first time we're saving this model, save config as well
    info_file = os.path.join(save_location, model_name + ".json")
    if not os.path.exists(info_file):
        with open(info_file, "w", encoding="utf-8") as fhandle:
            json.dump(
                config,
                fhandle,
                default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>",
            )

    logging.info(f"Model saved in epoch {epoch} at {savepath}.")


def train():
    """Train the model."""

    # validating to check if model was loaded properly
    if config["validate_first"]:
        validate(-1)

    if config["save_initial"]:
        save_model("initial")

    epoch = 0
    # for epoch in range(config['epochs']):
    while epoch < config["epochs"]:
        train_one_epoch(epoch)

        # decay the learning rate
        decay_scheduler.step()

        # validate
        if (epoch + 1) % config["validate_interval"] == 0:
            validate(epoch)

        if epoch % config["model_checkpoint_interval"] == 0:
            save_model(epoch)

        epoch += 1

    logging.info("Finished Training!")

    save_model("final")

    # close wandb run
    if config["wandb"]:
        wandb_run.finish()


try:
    train()
except KeyboardInterrupt:
    logging.info("Keyboard interrupt detected. Exiting gracefully.")
    if config["wandb"]:
        wandb_run.finish()
