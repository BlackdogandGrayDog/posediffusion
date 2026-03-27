# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import OrderedDict, defaultdict

import hydra
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from hydra.utils import instantiate, get_original_cwd
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from pytorch3d.renderer.cameras import PerspectiveCameras
from util.metric import camera_to_rel_deg, calculate_auc
from util.train_util import (
    DynamicBatchSampler,
    VizStats,
    WarmupCosineRestarts,
    get_co3d_dataset,
    get_simulator_dataset,
    set_seed_and_print,
)


METRIC_KEYS = ["loss", "Racc_5", "Racc_15", "Racc_30", "Tacc_5", "Tacc_15", "Tacc_30", "Auc_30"]


def save_metrics_html(history: dict, save_path: str):
    """Save epoch vs metrics as an interactive plotly HTML."""
    train_epochs = history["train"]["epochs"]
    eval_epochs  = history["eval"]["epochs"]

    fig = make_subplots(
        rows=2, cols=4,
        subplot_titles=METRIC_KEYS,
        shared_xaxes=False,
    )

    colors = {"train": "#1f77b4", "eval": "#ff7f0e"}

    for idx, key in enumerate(METRIC_KEYS):
        row = idx // 4 + 1
        col = idx % 4 + 1
        for split, epochs in [("train", train_epochs), ("eval", eval_epochs)]:
            vals = history[split].get(key, [])
            if vals:
                fig.add_trace(
                    go.Scatter(
                        x=epochs, y=vals,
                        name=f"{split}/{key}",
                        mode="lines+markers",
                        line=dict(color=colors[split]),
                        showlegend=(idx == 0),
                    ),
                    row=row, col=col,
                )

    fig.update_layout(
        title="Training Metrics",
        height=600,
        template="plotly_white",
    )
    fig.write_html(save_path)
    print(f"Saved training metrics to: {save_path}")


@hydra.main(config_path="../cfgs/", config_name="default_train")
def train_fn(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    accelerator = Accelerator(even_batches=False, device_placement=False)

    accelerator.print("Model Config:", OmegaConf.to_yaml(cfg), accelerator.state)

    torch.backends.cudnn.benchmark = cfg.train.cudnnbenchmark if not cfg.debug else False
    if cfg.debug:
        accelerator.print("********DEBUG MODE********")
        torch.backends.cudnn.deterministic = True

    set_seed_and_print(cfg.seed)

    # Data loading
    dataset_type = getattr(cfg.train, "dataset_type", "co3d")
    if dataset_type == "simulator":
        dataset, eval_dataset = get_simulator_dataset(cfg)
    else:
        dataset, eval_dataset = get_co3d_dataset(cfg)
    dataloader      = get_dataloader(cfg, dataset)
    eval_dataloader = get_dataloader(cfg, eval_dataset, is_eval=True)

    accelerator.print("length of train dataloader is: ", len(dataloader))
    accelerator.print("length of eval dataloader is:  ", len(eval_dataloader))

    # Model
    model = instantiate(cfg.MODEL, _recursive_=False)
    model = model.to(accelerator.device)

    # Optimizer & scheduler
    optimizer    = torch.optim.AdamW(params=model.parameters(), lr=cfg.train.lr)
    lr_scheduler = WarmupCosineRestarts(
        optimizer=optimizer, T_0=cfg.train.restart_num,
        iters_per_epoch=len(dataloader), warmup_ratio=0.1,
    )

    model, dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model, dataloader, optimizer, lr_scheduler
    )

    start_epoch = cfg.train.get("start_epoch", 0)
    if cfg.train.get("resume_dir", None):
        accelerator.load_state(cfg.train.resume_dir)
        accelerator.print(f"Successfully resumed full state from {cfg.train.resume_dir}")
    elif cfg.train.resume_ckpt:
        checkpoint = torch.load(cfg.train.resume_ckpt)
        try:
            model.load_state_dict(prefix_with_module(checkpoint), strict=True)
        except Exception:
            model.load_state_dict(checkpoint, strict=True)
        accelerator.print(f"Successfully resumed from {cfg.train.resume_ckpt}")

    stats      = VizStats(("loss", "lr", "sec/it", "Auc_30", "Racc_5", "Racc_15", "Racc_30",
                           "Tacc_5", "Tacc_15", "Tacc_30"))
    num_epochs = cfg.train.epochs

    # Track per-epoch averages for HTML export
    history = {
        "train": defaultdict(list, {"epochs": []}),
        "eval":  defaultdict(list, {"epochs": []}),
    }

    html_path = os.path.join(get_original_cwd(), cfg.exp_dir, "training_metrics.html")
    os.makedirs(os.path.join(get_original_cwd(), cfg.exp_dir), exist_ok=True)

    for epoch in range(start_epoch, num_epochs):
        stats.new_epoch()
        set_seed_and_print(cfg.seed + epoch)

        # Evaluation
        if (epoch != 0) and (epoch % cfg.train.eval_interval == 0):
            accelerator.print(f"----------Start to eval at epoch {epoch}----------")
            epoch_metrics = _train_or_eval_fn(
                model, eval_dataloader, cfg, optimizer, stats, accelerator, lr_scheduler,
                training=False,
            )
            if accelerator.is_main_process and epoch_metrics:
                history["eval"]["epochs"].append(epoch)
                for k, v in epoch_metrics.items():
                    history["eval"][k].append(v)
            accelerator.print(f"----------Finish the eval at epoch {epoch}----------")
        else:
            accelerator.print(f"----------Skip the eval at epoch {epoch}----------")

        # Training
        accelerator.print(f"----------Start to train at epoch {epoch}----------")
        epoch_metrics = _train_or_eval_fn(
            model, dataloader, cfg, optimizer, stats, accelerator, lr_scheduler,
            training=True,
        )
        accelerator.print(f"----------Finish the train at epoch {epoch}----------")

        if accelerator.is_main_process:
            lr = lr_scheduler.get_last_lr()[0]
            accelerator.print(f"LR: {lr:.6f}")
            stats.update({"lr": lr}, stat_set="train")

            if epoch_metrics:
                history["train"]["epochs"].append(epoch)
                for k, v in epoch_metrics.items():
                    history["train"][k].append(v)

            # Save HTML every eval_interval epochs
            if epoch % cfg.train.eval_interval == 0:
                save_metrics_html(history, html_path)

        if epoch % cfg.train.ckpt_interval == 0:
            accelerator.wait_for_everyone()
            ckpt_path = os.path.join(get_original_cwd(), cfg.exp_dir, f"ckpt_{epoch:06}")
            accelerator.print(f"Saving ckpt to {ckpt_path}")
            accelerator.save_state(output_dir=ckpt_path, safe_serialization=False)
            if accelerator.is_main_process:
                stats.save(os.path.join(get_original_cwd(), cfg.exp_dir, "stats"))

    accelerator.wait_for_everyone()
    accelerator.save_state(
        output_dir=os.path.join(get_original_cwd(), cfg.exp_dir, f"ckpt_{epoch:06}"),
        safe_serialization=False,
    )

    if accelerator.is_main_process:
        save_metrics_html(history, html_path)

    return True


def _train_or_eval_fn(
    model, dataloader, cfg, optimizer, stats, accelerator, lr_scheduler, training=True
):
    if training:
        model.train()
    else:
        model.eval()

    time_start = time.time()
    max_it     = len(dataloader)
    stat_set   = "train" if training else "eval"

    running = defaultdict(float)
    count   = 0

    for step, batch in enumerate(dataloader):
        images      = batch["image"].to(accelerator.device)
        translation = batch["T"].to(accelerator.device)
        rotation    = batch["R"].to(accelerator.device)
        fl          = batch["fl"].to(accelerator.device)
        pp          = batch["pp"].to(accelerator.device)

        if training and cfg.train.batch_repeat > 0:
            br = cfg.train.batch_repeat
            gt_cameras = PerspectiveCameras(
                focal_length=fl.reshape(-1, 2).repeat(br, 1),
                R=rotation.reshape(-1, 3, 3).repeat(br, 1, 1),
                T=translation.reshape(-1, 3).repeat(br, 1),
                device=accelerator.device,
            )
            batch_size = len(images) * br
        else:
            gt_cameras = PerspectiveCameras(
                focal_length=fl.reshape(-1, 2),
                R=rotation.reshape(-1, 3, 3),
                T=translation.reshape(-1, 3),
                device=accelerator.device,
            )
            batch_size = len(images)

        if training:
            predictions            = model(images, gt_cameras=gt_cameras, training=True,
                                           batch_repeat=cfg.train.batch_repeat)
            predictions["loss"]    = predictions["loss"].mean()
            loss                   = predictions["loss"]
        else:
            with torch.no_grad():
                predictions = model(images, training=False)

        pred_cameras = predictions["pred_cameras"]

        rel_rangle_deg, rel_tangle_deg = camera_to_rel_deg(
            pred_cameras, gt_cameras, accelerator.device, batch_size
        )

        predictions["Racc_5"]  = (rel_rangle_deg < 5).float().mean()
        predictions["Racc_15"] = (rel_rangle_deg < 15).float().mean()
        predictions["Racc_30"] = (rel_rangle_deg < 30).float().mean()
        predictions["Tacc_5"]  = (rel_tangle_deg < 5).float().mean()
        predictions["Tacc_15"] = (rel_tangle_deg < 15).float().mean()
        predictions["Tacc_30"] = (rel_tangle_deg < 30).float().mean()
        predictions["Auc_30"]  = calculate_auc(rel_rangle_deg, rel_tangle_deg, max_threshold=30)

        # Accumulate for epoch average
        for k in METRIC_KEYS:
            if k in predictions:
                running[k] += predictions[k].item()
        count += 1

        stats.update(predictions, time_start=time_start, stat_set=stat_set)

        if step % cfg.train.print_interval == 0:
            loss_val = predictions["loss"].item() if training else 0.0
            r5   = predictions["Racc_5"].item()
            r15  = predictions["Racc_15"].item()
            r30  = predictions["Racc_30"].item()
            auc  = predictions["Auc_30"].item()
            accelerator.print(
                f"[{stat_set}] it: {step}/{max_it} | loss: {loss_val:.4f} | "
                f"Racc5: {r5:.3f} | Racc15: {r15:.3f} | Racc30: {r30:.3f} | Auc30: {auc:.3f}"
            )

        if training:
            optimizer.zero_grad()
            accelerator.backward(loss)
            if cfg.train.clip_grad > 0 and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), cfg.train.clip_grad)
            optimizer.step()
            lr_scheduler.step()

    # Return per-epoch averages
    if count > 0:
        return {k: v / count for k, v in running.items()}
    return {}


def get_dataloader(cfg, dataset, is_eval=False):
    prefix = "eval" if is_eval else "train"
    batch_sampler = DynamicBatchSampler(
        len(dataset),
        dataset_len=getattr(cfg.train, f"len_{prefix}"),
        max_images=cfg.train.max_images // (2 if is_eval else 1),
        images_per_seq=cfg.train.images_per_seq,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        persistent_workers=cfg.train.persistent_workers,
    )
    dataloader.batch_sampler.drop_last = True
    dataloader.batch_sampler.sampler   = dataloader.batch_sampler
    return dataloader


def prefix_with_module(checkpoint):
    prefixed = OrderedDict()
    for key, value in checkpoint.items():
        prefixed["module." + key] = value
    return prefixed


if __name__ == "__main__":
    train_fn()
