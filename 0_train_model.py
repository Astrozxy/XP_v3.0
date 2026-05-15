from __future__ import annotations
import os
import json
import glob
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import h5py

from model import *


def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)
    dtype_str = config["model"].get("dtype", "float16")
    if dtype_str == "float16":
        config["model"]["dtype"] = torch.float16
    elif dtype_str == "float32":
        config["model"]["dtype"] = torch.float32
    else:
        config["model"]["dtype"] = torch.float16
    return config


def save_checkpoint(stage_num, model, star_params, config, optimizer_state=None):
    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"stage{stage_num}.pt")
    model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    checkpoint = {
        "stage": stage_num,
        "model_state": model_to_save.state_dict(),
        "star_params": {k: v.detach().cpu() for k, v in star_params.items()},
        "config": config,
    }
    if optimizer_state is not None:
        checkpoint["optimizer_state"] = optimizer_state
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved to {checkpoint_path}")
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(checkpoint, latest_path)


def load_latest_checkpoint(config, device, latent_dim, dataset):
    checkpoint_dir = config["training"]["checkpoint_dir"]
    stage_files = glob.glob(os.path.join(checkpoint_dir, "stage*.pt"))
    stage_files = [f for f in stage_files if "stage" in os.path.basename(f) and not f.endswith("latest.pt")]
    if not stage_files:
        print("No checkpoint found, starting from scratch.")
        return None, None, 0
    stage_nums = []
    for f in stage_files:
        try:
            stage_num = int(os.path.basename(f).replace("stage", "").replace(".pt", ""))
            stage_nums.append((stage_num, f))
        except ValueError:
            continue
    if not stage_nums:
        return None, None, 0
    stage_nums.sort(key=lambda x: x[0])
    latest_stage, latest_file = stage_nums[-1]
    print(f"Loading checkpoint from {latest_file} (Stage {latest_stage})")
    checkpoint = torch.load(latest_file, map_location="cpu")

    P, L_spec = build_P_66xL(dtype=torch.float32, device="cpu")
    model = StellarModel(
        P_66xL=P,
        latent_dim=latent_dim,
        init_k1=torch.zeros(P.shape[1], dtype=torch.float32),
        dtype=torch.float32,
    ).to(device)

    state_dict = checkpoint["model_state"]
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        print("Removing '_orig_mod.' prefix from checkpoint keys.")
        state_dict = {key.replace("_orig_mod.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)

    star_values = checkpoint["star_params"]
    dtype = torch.float32
    star = {
        "x_pred": nn.Parameter(star_values["x_pred"].to(device).to(dtype)),
        "log_E_pred": nn.Parameter(star_values["log_E_pred"].to(device).to(dtype).view(-1)),
        "xi_pred": nn.Parameter(star_values["xi_pred"].to(device).to(dtype).view(-1)),
        "log_plx_pred": nn.Parameter(star_values["log_plx_pred"].to(device).to(dtype).view(-1)),
        "latent_pred": nn.Parameter(star_values["latent_pred"].to(device).to(dtype)),
    }
    return model, star, latest_stage


def train_stage_from_config(
    model, dataset, star, stage_config, stage_num,
    L_spec, device, config
):
    kwargs = {
        "model": model,
        "dataset": dataset,
        "L_spec": L_spec,
        "device": device,
        "epochs": stage_config["epochs"],
        "batch_size": stage_config.get("batch_size", 1024),
        "lr": stage_config["lr"],
        "free_keys": tuple(stage_config.get("free_keys", [])),
        "update_model": stage_config.get("update_model", True),
        "train_k1": stage_config.get("train_k1", False),
        "star_params": star,
        "lambda_latent": stage_config.get("lambda_latent", 1.0),
        "lambda_latent_grad": stage_config.get("lambda_latent_grad", 1.0),
        "lambda_xi": stage_config.get("lambda_xi", 0.0),
        "lambda_smooth": stage_config.get("lambda_smooth", 0.001),
        "lambda_smooth_ext": stage_config.get("lambda_smooth_ext", stage_config.get("lambda_smooth", 0.001)),
        "num_workers": stage_config.get("num_workers", 4),
        "desc": f"{stage_config['name']} (Stage {stage_num})",
    }
    if "lambda_x" in stage_config:
        kwargs["lambda_x"] = stage_config["lambda_x"]
    if "lambda_E" in stage_config:
        kwargs["lambda_E"] = stage_config["lambda_E"]
    if "lambda_plx" in stage_config:
        kwargs["lambda_plx"] = stage_config["lambda_plx"]
    train_stage(**kwargs)


def convert_h5_to_mmap(h5_path, pt_path, rank=0):
    if rank == 0:
        print(f"Converting {h5_path} -> {pt_path} (one-time operation)...")
        with h5py.File(h5_path, "r") as f:
            data = {
                "x": torch.from_numpy(f["x"][:]),
                "E": torch.from_numpy(f["E"][:]),
                "xi": torch.from_numpy(f["xi"][:]),
                "plx": torch.from_numpy(f["plx"][:]),
                "x_err": torch.from_numpy(f["x_err"][:]),
                "E_err": torch.from_numpy(f["E_err"][:]),
                "xi_err": torch.from_numpy(f["xi_err"][:]),
                "plx_err": torch.from_numpy(f["plx_err"][:]),
                "flux": torch.from_numpy(f["flux"][:]),
                "flux_sqrticov": torch.from_numpy(f["flux_sqrticov"][:]),
                "latent": torch.from_numpy(f["latent"][:]),
            }
        torch.save(data, pt_path, _use_new_zipfile_serialization=True)
        print(f" Converted to {pt_path}")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def main(config_path="config.json"):
    config = load_config(config_path)
    print("Configuration loaded:")
    print(json.dumps(config, indent=2, default=str))

    # ---- 分布式初始化 ----
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}, rank: {rank}")

    dtype = torch.float32
    latent_dim = config["data"]["latent_dim"]

    P, L_spec = build_P_66xL(dtype=dtype, device="cpu")
    print(f"P shape: {tuple(P.shape)}, L_spec: {L_spec}")

    # ---- 确定数据集路径 ----
    dataset_h5 = config["data"].get("dataset_h5", "data/stellar_dataset.h5")
    dataset_mmap = config["data"].get("dataset_mmap", "data/stellar_dataset_mmap.pt")

    # ---- 若 .pt 不存在，转换一次 ----
    if not os.path.exists(dataset_mmap):
        convert_h5_to_mmap(dataset_h5, dataset_mmap, rank=rank)

    # ---- 加载 mmap 数据集（所有进程共享物理内存）----
    print(f"Loading dataset from memory-mapped file: {dataset_mmap}")
    dataset = StellarDataset(dataset_mmap, latent_dim=latent_dim, dtype=dtype)
    print(dataset.latent.dtype)

    
    # ---- 模型 ----
    model, star, completed_stages = load_latest_checkpoint(config, device, latent_dim, dataset)

    if model is None:
        model = StellarModel(
            P_66xL=P,
            latent_dim=latent_dim,
            init_k1=torch.zeros(P.shape[1], dtype=dtype),
            dtype=dtype,
        ).to(device)
        try:
            model = torch.compile(model, mode="default")
        except Exception:
            pass
        star = init_star_params(dataset, device, latent_dim=latent_dim)
        completed_stages = 0
        print("Starting training from scratch.")
    else:
        try:
            model = torch.compile(model, mode="default")
        except Exception:
            pass
        print(f"Resuming from Stage {completed_stages + 1}")

    stages = config["stages"]
    start_stage_idx = completed_stages

    for stage_idx in range(start_stage_idx, len(stages)):
        stage_config = stages[stage_idx]
        stage_num = stage_idx + 1
        print(f"\n{'='*60}")
        print(f"Starting {stage_config['name']} (Stage {stage_num}/{len(stages)})")
        print(f"{'='*60}")
        try:
            train_stage_from_config(
                model=model,
                dataset=dataset,
                star=star,
                stage_config=stage_config,
                stage_num=stage_num,
                L_spec=L_spec,
                device=device,
                config=config,
            )
            save_checkpoint(stage_num, model, star, config)
            print(f"✓ Stage {stage_num} completed and saved.")
        except Exception as e:
            print(f"✗ Stage {stage_num} interrupted: {e}")
            print(f"Progress saved up to Stage {stage_num - 1}")
            raise

    final_model_path = config["training"]["final_model_path"]
    model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    torch.save({
        "model_state": model_to_save.state_dict(),
        "star_params": {k: v.detach().cpu() for k, v in star.items()},
        "config": config,
    }, final_model_path)
    print(f"\n✓ All stages completed! Final model saved to {final_model_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train stellar model with checkpoint resume")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    args = parser.parse_args()
    main(args.config)