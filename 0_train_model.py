from __future__ import annotations
import os
import json
import glob
from typing import Dict, Optional, Tuple, List, Any

import torch
import torch.nn as nn
import torch.distributed as dist        # >>> 新增
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import h5py

from model import *                     # 你的 model.py 中包含全部定义


def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 模型内部强制使用 float32，FP16 仅由 autocast 控制，这里保持兼容
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

    # >>> 如果是 DDP 模型，取出 .module 再保存（否则后续加载可避免前缀）
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
    # 排除 latest.pt 软链接
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

    # 找最新 stage
    stage_nums.sort(key=lambda x: x[0])
    latest_stage, latest_file = stage_nums[-1]

    print(f"Loading checkpoint from {latest_file} (Stage {latest_stage})")
    checkpoint = torch.load(latest_file, map_location="cpu")

    # 创建未编译的模型（后续主函数会统一 compile）
    P, L_spec = build_P_66xL(dtype=torch.float32, device="cpu")
    model = StellarModel(
        P_66xL=P,
        latent_dim=latent_dim,
        init_k1=torch.zeros(P.shape[1], dtype=torch.float32),
        dtype=torch.float32,
    ).to(device)

    # ---------- 处理 torch.compile 可能导致的 _orig_mod. 前缀 ----------
    state_dict = checkpoint["model_state"]
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        print("Removing '_orig_mod.' prefix from checkpoint keys (caused by torch.compile).")
        state_dict = {key.replace("_orig_mod.", "", 1): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)

    # 恢复星表参数
    star_values = checkpoint["star_params"]
    dtype = torch.float32
    star = {
        "x_pred": nn.Parameter(star_values["x_pred"].to(device).to(dtype)),
        "E_pred": nn.Parameter(star_values["E_pred"].to(device).to(dtype).view(-1)),
        "xi_pred": nn.Parameter(star_values["xi_pred"].to(device).to(dtype).view(-1)),
        "log_plx_pred": nn.Parameter(star_values["log_plx_pred"].to(device).to(dtype).view(-1)),
        "latent_pred": nn.Parameter(star_values["latent_pred"].to(device).to(dtype)),
    }

    return model, star, latest_stage


def train_stage_from_config(
    model, dataset, star, stage_config, stage_num,
    L_spec, device, config
):
    """根据配置训练单个阶段，直接调用新版 train_stage"""

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
        "lambda_latent_grad": stage_config.get("lambda_latent_grad", 1.0),   # >>> 新增
        "lambda_xi": stage_config.get("lambda_xi", 0.0),
        "lambda_smooth": stage_config.get("lambda_smooth", 0.001),
        "lambda_smooth_ext": stage_config.get("lambda_smooth_ext", stage_config.get("lambda_smooth", 0.001)),
        "num_workers": stage_config.get("num_workers", 4),                    # >>> 新增
        "desc": f"{stage_config['name']} (Stage {stage_num})",
    }

    if "lambda_x" in stage_config:
        kwargs["lambda_x"] = stage_config["lambda_x"]
    if "lambda_E" in stage_config:
        kwargs["lambda_E"] = stage_config["lambda_E"]
    if "lambda_plx" in stage_config:
        kwargs["lambda_plx"] = stage_config["lambda_plx"]

    train_stage(**kwargs)


def main(config_path="config.json"):
    config = load_config(config_path)
    print("Configuration loaded:")
    print(json.dumps(config, indent=2, default=str))

    # ---- 分布式初始化（兼容 torchrun） ----
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dtype = torch.float32                       # >>> 模型内部固定 float32
    latent_dim = config["data"]["latent_dim"]

    P, L_spec = build_P_66xL(dtype=dtype, device="cpu")
    print(f"P shape: {tuple(P.shape)}, L_spec: {L_spec}")

    dataset_h5 = config["data"]["dataset_h5"]
    print(f"Loading dataset from {dataset_h5}")

    with h5py.File(dataset_h5, "r") as f:
        dataset = StellarDataset(
            x=torch.from_numpy(f["x"][:]),
            E=torch.from_numpy(f["E"][:]),
            xi=torch.from_numpy(f["xi"][:]),
            plx=torch.from_numpy(f["plx"][:]),
            x_err=torch.from_numpy(f["x_err"][:]),
            E_err=torch.from_numpy(f["E_err"][:]),
            xi_err=torch.from_numpy(f["xi_err"][:]),
            plx_err=torch.from_numpy(f["plx_err"][:]),
            flux=torch.from_numpy(f["flux"][:]),
            flux_sqrticov=torch.from_numpy(f["flux_sqrticov"][:]),
            latent=torch.from_numpy(f["latent"][:]),
            latent_dim=latent_dim,
            dtype=dtype,
        )

    model, star, completed_stages = load_latest_checkpoint(config, device, latent_dim, dataset)

    if model is None:
        model = StellarModel(
            P_66xL=P,
            latent_dim=latent_dim,
            init_k1=torch.zeros(P.shape[1], dtype=dtype),
            dtype=dtype,
        ).to(device)

        # 可选 torch.compile（单卡或多卡均可，DDP 兼容）
        try:
            model = torch.compile(model, mode="default")
        except Exception:
            pass

        star = init_star_params(dataset, device, latent_dim=latent_dim)
        completed_stages = 0
        print("Starting training from scratch.")
    else:
        # 加载模型后也尝试编译
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
