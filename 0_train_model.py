from model import *

from __future__ import annotations
import os
import json
import glob
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

import numpy as np
import h5py
from astropy.table import Table

# ========== 你的原始模型代码（保持不变）==========
# ... 这里放置你原来的所有类定义 ...
# StellarDataset, build_P_66xL, StellarSpectrumModel,
# ApplyExtinction, StellarModel, compute_loss, init_star_params, train_stage
# ================================================

def load_config(config_path="config.json"):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 转换 dtype
    dtype_str = config["model"].get("dtype", "float16")
    if dtype_str == "float16":
        config["model"]["dtype"] = torch.float16
    elif dtype_str == "float32":
        config["model"]["dtype"] = torch.float32
    else:
        config["model"]["dtype"] = torch.float16

    return config

def save_checkpoint(stage_num, model, star_params, config, optimizer_state=None):
    """保存 checkpoint，包含所有训练状态"""
    checkpoint_dir = config["training"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"stage{stage_num}.pt")

    checkpoint = {
        "stage": stage_num,
        "model_state": model.state_dict(),
        "star_params": {k: v.detach().cpu() for k, v in star_params.items()},
        "config": config,  # 保存配置以便恢复时验证
    }

    if optimizer_state is not None:
        checkpoint["optimizer_state"] = optimizer_state

    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved to {checkpoint_path}")

    # 同时保存一个 latest checkpoint
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    torch.save(checkpoint, latest_path)

def load_latest_checkpoint(config, device, latent_dim, dataset):
    """加载最新的 checkpoint，返回 (model, star, start_stage)"""
    checkpoint_dir = config["training"]["checkpoint_dir"]

    # 查找所有 stage checkpoint
    stage_files = glob.glob(os.path.join(checkpoint_dir, "stage*.pt"))
    stage_files = [f for f in stage_files if "stage" in f and not f.endswith("latest.pt")]

    if not stage_files:
        print("No checkpoint found, starting from scratch.")
        return None, None, 0

    # 提取 stage 编号
    stage_nums = []
    for f in stage_files:
        try:
            # 提取 stage 数字，例如 stage1.pt -> 1
            stage_num = int(os.path.basename(f).replace("stage", "").replace(".pt", ""))
            stage_nums.append((stage_num, f))
        except:
            continue

    if not stage_nums:
        return None, None, 0

    # 找到最大的 stage 编号
    stage_nums.sort(key=lambda x: x[0])
    latest_stage, latest_file = stage_nums[-1]

    print(f"Loading checkpoint from {latest_file} (Stage {latest_stage})")
    checkpoint = torch.load(latest_file, map_location="cpu")

    # 验证配置一致性（可选）
    saved_config = checkpoint.get("config", {})
    if saved_config and saved_config != config:
        print("Warning: Saved config differs from current config. Using saved checkpoint values.")

    # 重建模型
    P, L_spec = build_P_66xL(dtype=config["model"]["dtype"], device="cpu")
    model = StellarModel(
        P_66xL=P,
        latent_dim=latent_dim,
        init_k1=torch.zeros(P.shape[1]),
        dtype=config["model"]["dtype"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    # 重建 star 参数
    star_values = checkpoint["star_params"]
    star = {
        "x_pred": nn.Parameter(star_values["x_pred"].to(device)),
        "E_pred": nn.Parameter(star_values["E_pred"].to(device).view(-1)),
        "xi_pred": nn.Parameter(star_values["xi_pred"].to(device).view(-1)),
        "log_plx_pred": nn.Parameter(star_values["log_plx_pred"].to(device).view(-1)),
        "latent_pred": nn.Parameter(star_values["latent_pred"].to(device)),
    }

    return model, star, latest_stage

def recreate_star_params(star_values, device, latent_dim):
    """从保存的值重新创建 star 参数字典"""
    star = {
        "x_pred": nn.Parameter(star_values["x_pred"].to(device)),
        "E_pred": nn.Parameter(star_values["E_pred"].to(device).view(-1)),
        "xi_pred": nn.Parameter(star_values["xi_pred"].to(device).view(-1)),
        "log_plx_pred": nn.Parameter(star_values["log_plx_pred"].to(device).view(-1)),
        "latent_pred": nn.Parameter(star_values["latent_pred"].to(device)),
    }
    return star

def train_stage_from_config(
    model, dataset, star, stage_config, stage_num,
    L_spec, device, config
):
    """根据配置训练单个阶段"""

    # 准备参数
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
        "lambda_xi": stage_config.get("lambda_xi", 0.0),
        "lambda_smooth": stage_config.get("lambda_smooth", 0.001),
        "desc": f"{stage_config['name']} (Stage {stage_num})",
    }

    # 添加可选参数
    if "lambda_x" in stage_config:
        kwargs["lambda_x"] = stage_config["lambda_x"]
    if "lambda_E" in stage_config:
        kwargs["lambda_E"] = stage_config["lambda_E"]
    if "lambda_plx" in stage_config:
        kwargs["lambda_plx"] = stage_config["lambda_plx"]

    train_stage(**kwargs)

def main(config_path="config.json"):
    """主训练函数"""

    # 1. 加载配置
    config = load_config(config_path)
    print("Configuration loaded:")
    print(json.dumps(config, indent=2, default=str))

    # 2. 设置设备和数据类型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dtype = config["model"]["dtype"]
    latent_dim = config["data"]["latent_dim"]

    # 3. 构建投影矩阵
    P, L_spec = build_P_66xL(dtype=dtype, device="cpu")
    print(f"P shape: {tuple(P.shape)}, L_spec: {L_spec}")

    # 4. 加载数据集
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

    # 5. 尝试恢复之前的训练
    model, star, completed_stages = load_latest_checkpoint(config, device, latent_dim, dataset)

    if model is None:
        # 从头开始训练
        model = StellarModel(
            P_66xL=P,
            latent_dim=latent_dim,
            init_k1=torch.zeros(P.shape[1]),
            dtype=dtype,
        ).to(device)
        star = init_star_params(dataset, device, latent_dim=latent_dim)
        completed_stages = 0
        print("Starting training from scratch.")
    else:
        print(f"Resuming from Stage {completed_stages + 1}")

    # 6. 从下一阶段开始训练
    stages = config["stages"]
    start_stage_idx = completed_stages  # 已完成的数量（索引从0开始）

    for stage_idx in range(start_stage_idx, len(stages)):
        stage_config = stages[stage_idx]
        stage_num = stage_idx + 1

        print(f"\n{'='*60}")
        print(f"Starting {stage_config['name']} (Stage {stage_num}/{len(stages)})")
        print(f"{'='*60}")

        try:
            # 训练该阶段
            train_stage(
                model=model,
                dataset=dataset,
                star=star,
                stage_config=stage_config,
                stage_num=stage_num,
                L_spec=L_spec,
                device=device,
                config=config,
            )

            # 训练完成后立即保存
            save_checkpoint(stage_num, model, star, config)
            print(f"✓ Stage {stage_num} completed and saved.")

        except Exception as e:
            print(f"✗ Stage {stage_num} interrupted: {e}")
            print(f"Progress saved up to Stage {stage_num - 1}")
            print("You can resume training by running this script again.")
            raise  # 或者选择保存当前状态后退出

    # 7. 保存最终模型
    final_model_path = config["training"]["final_model_path"]
    torch.save({
        "model_state": model.state_dict(),
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
