# model.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

from torch.func import vmap, grad

import numpy as np
import h5py
from astropy.table import Table

import matplotlib.pyplot as plt

import torch.distributed as dist
from torch.func import jacrev

def is_distributed():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    if is_distributed():
        return dist.get_rank()
    return 0

def get_world_size():
    if is_distributed():
        return dist.get_world_size()
    return 1


def save_as_h5(d, name):
    print(f'Saving as {name}')
    with h5py.File(name, 'w') as f:
        for key in d.keys():
            f[key] = d[key]
    return 0


def load_h5(name):
    print(f'Loading {name}')
    d = {}
    with h5py.File(name, 'r') as f:
        for key in f.keys():
            d[key] = f[key][:]
    return d     

dtype = torch.float32

# Build dataset
class StellarDataset(Dataset):
    def __init__(self, mmap_path, latent_dim=3, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.latent_dim = latent_dim

        # 使用 mmap 加载，多个进程共享物理内存
        self.data = torch.load(mmap_path, map_location='cpu', mmap=True, weights_only=True)
        self.length = self.data['x'].shape[0]

        # 将所有张量转为统一 dtype，并暴露为属性
        for key in list(self.data.keys()):
            tensor = self.data[key].to(dtype=dtype)
            self.data[key] = tensor
            setattr(self, key, tensor)          # 兼容 dataset.x / dataset.E 等访问

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "idx": torch.tensor(idx, dtype=torch.long),
            "x": self.data['x'][idx],
            "E": self.data['E'][idx],
            "xi": self.data['xi'][idx],
            "plx": self.data['plx'][idx],
            "x_err": self.data['x_err'][idx],
            "E_err": self.data['E_err'][idx],
            "xi_err": self.data['xi_err'][idx],
            "plx_err": self.data['plx_err'][idx],
            "flux": self.data['flux'][idx],
            "flux_sqrticov": self.data['flux_sqrticov'][idx],
            "latent": self.data['latent'][idx],
        }

# -------------------------
# 2) Fixed projection matrix P (build once)
# -------------------------
def build_P_66xL(
    wave_hi_start=387.0,
    wave_hi_end=997.0,
    wave_hi_step=3.0,
    wave_obs_start=392.0,
    wave_obs_end=992.0,
    wave_obs_step=10.0,
    extra_bands=5,
    dtype=torch.float32,
    device="cpu",
):
    """
    Build P (66, L), where first 61 rows interpolate a high-res optical grid to 61 observed optical points,
    and last 5 rows pick the extra 5 bands directly.
    """
    wave_hi = torch.arange(wave_hi_start, wave_hi_end + 1e-6, wave_hi_step, dtype=dtype, device=device)
    wave_obs = torch.arange(wave_obs_start, wave_obs_end + 1e-6, wave_obs_step, dtype=dtype, device=device)
    
    L_spec = wave_hi.numel()
    L = L_spec + extra_bands

    idx = torch.searchsorted(wave_hi, wave_obs)
    idx = torch.clamp(idx, 1, L_spec - 1)

    x0 = wave_hi[idx - 1]
    x1 = wave_hi[idx]
    t = (wave_obs - x0) / (x1 - x0)
    w0 = 1 - t
    w1 = t

    P61 = torch.zeros(61, L_spec, dtype=dtype, device=device)
    rows = torch.arange(61, device=device)
    P61[rows, idx - 1] = w0
    P61[rows, idx] = w1

    P = torch.zeros(66, L, dtype=dtype, device=device)
    P[:61, :L_spec] = P61

    for i in range(extra_bands):
        P[61 + i, L_spec + i] = 1.0

    return P, L_spec



class StellarSpectrumModel(nn.Module):
    def __init__(self, in_dim: int, L: int, hidden=256, depth=3, dtype=torch.float32):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden, dtype=dtype), nn.Tanh()]
            d = hidden
        layers.append(nn.Linear(d, L, dtype=dtype))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_F = self.net(x)
        return log_F


# -------------------------
# 3) Modules with clean interfaces
# -------------------------
class ApplyExtinction(nn.Module):
    """
    A(λ) = E * tanh[ k0(λ) + xi * k1(λ) ]
    F_obs = F_int * exp(-A)
    """
    def __init__(
        self,
        L: int,
        init_k0: Optional[torch.Tensor] = None,
        init_k1: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if init_k0 is None:
            wavelength_IRbands = np.array([1235, 1662, 2159, 3400, 4600])
            init_k0 = torch.as_tensor(np.log(2 * (np.hstack([387.0 + np.arange(L-5) *3, wavelength_IRbands])/550.)**(-1.5)), dtype=dtype)
        else:
            init_k0 = torch.as_tensor(init_k0, dtype=dtype).view(-1)
            assert init_k0.numel() == L

        if init_k1 is None:
            init_k1 = torch.as_tensor(np.arange(L) / L * 0, dtype=dtype)
        else:
            init_k1 = torch.as_tensor(init_k1, dtype=dtype).view(-1)
            assert init_k1.numel() == L

        self.k0 = nn.Parameter(init_k0.clone())
        self.k1 = nn.Parameter(init_k1.clone())

    def forward(self, log_F_int: torch.Tensor, log_E: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """
        A(λ) = exp(log_E) * exp[ k0(λ) + tanh(xi) * k1(λ) ]
        F_obs = F_int * exp(-A)
        """
        E = torch.exp(log_E)  
        A = E[:, None] * torch.exp(self.k0[None, :] + torch.tanh(xi[:, None]) * self.k1[None, :])
        return log_F_int - A


# -------------------------
# 4) Full model: forward(batch) only
# -------------------------
class StellarModel(nn.Module):
    def __init__(
        self,
        P_66xL: torch.Tensor,
        x_dim: int = 4,
        latent_dim: int = 1,
        init_k0: Optional[torch.Tensor] = None,
        init_k1: Optional[torch.Tensor] = None,
        dtype=torch.float32,
    ):
        super().__init__()

        P_66xL = torch.as_tensor(P_66xL, dtype=dtype)
        self.register_buffer("P", P_66xL)

        L = P_66xL.shape[1]

        self.spectrum_model = StellarSpectrumModel(
            in_dim=x_dim + latent_dim,
            L=L,
            dtype=dtype,
        )

        self.extinction = ApplyExtinction(
            L=L,
            init_k0=init_k0,
            init_k1=init_k1,
            dtype=dtype,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x_pred = batch.get("x_pred", batch["x"])
        # ---- 处理 E：期望收到 logE，若收到原始线性 E 则转换 ----
        if "log_E_pred" in batch:
            log_E = batch["log_E_pred"].view(-1)    
        else:
            log_E = torch.log(batch["E"].clamp_min(1e-6)).view(-1)   # 推理或无自由 E 时用线性 E 转 logE
    
        xi_pred = batch.get("xi_pred", batch["xi"]).view(-1)
    
        plx_safe = batch["plx"].clamp_min(1e-6)
        log_plx_pred = batch.get("log_plx_pred", torch.log(plx_safe)).view(-1)
    
        latent = batch.get("latent_pred", batch["latent"])
    
        # ---- 光谱网络（MLP）正常接受 autocast 控制，可能 FP16 ----
        log_F_int = self.spectrum_model(torch.cat([x_pred, latent], dim=-1))
        log_F_int = log_F_int + 2 * log_plx_pred[:, None]
    
        # ---- 强制以下所有含 exp 的运算使用 FP32，防止溢出 ----
        log_F_int_fp32 = log_F_int.float()
        log_E_fp32 = log_E.float()                # logE 保持 float32
        xi_pred_fp32 = xi_pred.float()
    
        with torch.amp.autocast('cuda', enabled=False):
            log_F_obs = self.extinction(log_F_int_fp32, log_E_fp32, xi_pred_fp32)  # 传入 logE
            flux_pred = torch.exp(log_F_obs) @ self.P.T.to(log_F_obs.dtype)
    
        return {
            "flux_pred": flux_pred,
            "log_F_int": log_F_int_fp32,
            #"log_F_obs": log_F_obs,
            "ext_k0": self.extinction.k0,
            "ext_k1": self.extinction.k1,
            "latent": latent,
            "xi_pred": xi_pred_fp32,    # 保持返回 FP32 的 xi_pred 用于 loss
        }


# -------------------------
# 5) Loss
# -------------------------
def compute_loss(
    batch: Dict[str, torch.Tensor],
    out: Dict[str, torch.Tensor],
    L_spec: int,
    lambda_xi: float = 1.0,
    lambda_latent: float = 1.0,
    lambda_smooth: float = 1e-3,         
    lambda_smooth_ext: float = 1e-3,      
) -> torch.Tensor:
    d_flux = (batch["flux"] - out["flux_pred"]).unsqueeze(-1)
    Wr = batch["flux_sqrticov"] @ d_flux
    chi2_flux = Wr.squeeze(-1).pow(2).sum(dim=1)

    xi_err = batch["xi_err"].clamp_min(1e-6)
    chi2_xi = ((out["xi_pred"] - batch["xi"]) / xi_err).pow(2)

    chi2_latent = out["latent"].pow(2).sum(dim=1)

    log_F_pred = out["log_F_int"][:, :L_spec]
    d2 = log_F_pred[:, 2:] - 2 * log_F_pred[:, 1:-1] + log_F_pred[:, :-2]
    smooth_flux = d2.pow(2).mean(dim=1)

    k0 = out["ext_k0"][:L_spec]
    k1 = out["ext_k1"][:L_spec]
    d2_k0 = k0[2:] - 2 * k0[1:-1] + k0[:-2]
    d2_k1 = k1[2:] - 2 * k1[1:-1] + k1[:-2]
    smooth_ext = d2_k0.pow(2).mean() + d2_k1.pow(2).mean()

    loss = (
        chi2_flux
        + lambda_xi * chi2_xi
        + lambda_latent * chi2_latent 
        + lambda_smooth * smooth_flux          
        + lambda_smooth_ext * smooth_ext     
    )

    return loss.mean()


def init_star_params(dataset: Dataset, device: str, latent_dim: int = 3):
    dtype = dataset.x.dtype
    
    x0 = dataset.x.to(device)
    E0 = dataset.E.to(device).view(-1).clamp_min(1e-6)    # 防止 log(0)
    xi0 = dataset.xi.to(device).view(-1)
    plx0 = dataset.plx.to(device).view(-1).clamp_min(1e-6)
    N = len(dataset)

    star = {
        "x_pred": nn.Parameter(x0.clone()),
        "log_E_pred": nn.Parameter(torch.log(E0.clamp_min(1e-6))),             # 初始化为 logE
        "xi_pred": nn.Parameter(xi0.clone()),
        "log_plx_pred": nn.Parameter(torch.log(plx0.clamp_min(1e-6))),
        "latent_pred": nn.Parameter(1e-2 * torch.randn(N, latent_dim, device=device, dtype=dtype)),
    }

    return star


def train_stage(
    *,
    model: nn.Module,
    dataset,
    L_spec: int,
    device: str,
    epochs: int,
    batch_size: int = 128,
    lr: float = 1e-4,
    lr_schedule: str = "plateau",
    step_size: int = 50,
    gamma: float = 0.5,
    patience: int = 50,
    num_workers: int = 0,
    pin_memory: bool = True,
    free_keys: Tuple[str, ...] = (),
    update_model: bool = True,
    train_k1: bool = True,
    star_params: Dict[str, nn.Parameter] | None = None,
    lambda_x: float = 1.0,
    lambda_E: float = 1.0,
    lambda_plx: float = 1.0,
    lambda_xi: float = 0.0,
    lambda_latent: float = 1.0,
    lambda_latent_grad: float = 1.0,
    lambda_smooth: float = 1.0,
    lambda_smooth_ext: float = 1.0,
    desc: str = "Stage",
):
    free_keys = tuple(free_keys)
    valid = {"x", "E", "xi", "plx", "latent"}
    if any(k not in valid for k in free_keys):
        raise ValueError(f"free_keys must be subset of {valid}, got {free_keys}")
    if star_params is None:
        raise ValueError("star_params must be provided.")

    # ---- 分布式信息 ----
    is_dist = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if is_dist else 1
    rank = dist.get_rank() if is_dist else 0

    # ---- 设备处理 ----
    if isinstance(device, str):
        device = torch.device(device)

    # ---- DataLoader ----
    if is_dist:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )
    else:
        sampler = None
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None,
        )

    # ---- 模型 ----
    model.to(device)
    if is_dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device] if device.type == 'cuda' else None,
            output_device=device if device.type == 'cuda' else None,
        )
    base_model = model.module if is_dist else model

    # ---- 可训练参数 ----
    trainables = []
    if update_model:
        for name, p in model.named_parameters():
            clean_name = name.replace("module.", "")
            if clean_name == "extinction.k1" and not train_k1:
                continue
            trainables.append(p)

    if "x" in free_keys:
        trainables.append(star_params["x_pred"])
    if "E" in free_keys:
        trainables.append(star_params["log_E_pred"])
    if "xi" in free_keys:
        trainables.append(star_params["xi_pred"])
    if "plx" in free_keys:
        trainables.append(star_params["log_plx_pred"])
    if "latent" in free_keys:
        trainables.append(star_params["latent_pred"])

    if len(trainables) == 0:
        raise ValueError("No trainables selected.")

    # ---- 优化器与调度器 ----
    optimizer = torch.optim.Adam(trainables, lr=lr)

    if lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif lr_schedule == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif lr_schedule == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=patience)
    else:
        scheduler = None

    # ---- 混合精度 ----
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # ---- 先验数据 ----
    x0 = dataset.x.to(device)
    E0 = dataset.E.to(device).view(-1)
    plx0 = dataset.plx.to(device).view(-1).clamp_min(1e-6)

    # ---- 进度条 ----
    if rank == 0:
        pbar = tqdm(range(epochs), desc=desc, dynamic_ncols=True)
    else:
        pbar = range(epochs)

    for epoch in pbar:
        if is_dist:
            sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            # ---- 新增：标记 CUDA graph 步进点，防止 tensor 覆盖错误 ----
            torch.compiler.cudagraph_mark_step_begin()

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            idx = batch["idx"].view(-1)

            batch_fwd = dict(batch)
            batch_fwd["latent_pred"] = star_params["latent_pred"][idx]

            if "x" in free_keys:
                batch_fwd["x_pred"] = star_params["x_pred"][idx]
            if "E" in free_keys:
                batch_fwd["log_E_pred"] = star_params["log_E_pred"][idx].view(-1)
            if "xi" in free_keys:
                batch_fwd["xi_pred"] = star_params["xi_pred"][idx].view(-1)
            if "plx" in free_keys:
                batch_fwd["log_plx_pred"] = star_params["log_plx_pred"][idx].view(-1)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(batch_fwd)

            loss = compute_loss(
                batch,
                out,
                L_spec,
                lambda_xi=lambda_xi,
                lambda_latent=lambda_latent,
                lambda_smooth=lambda_smooth,
                lambda_smooth_ext=lambda_smooth_ext
            )

            # ---- 潜变量梯度惩罚 ----
            if update_model and lambda_latent_grad != 0:
                x_det = batch_fwd.get("x_pred", batch_fwd["x"]).detach()   # (B, x_dim)
            
                def spec_fn(lat, x):
                    # lat: (latent_dim,), x: (x_dim,)
                    return base_model.spectrum_model(
                        torch.cat([x, lat], dim=-1).unsqueeze(0)
                    )[0, :L_spec]  # 输出光学部分光谱 (L_spec,)
            
                # jacrev 计算 spec_fn 关于第一个参数(lat)的雅可比，vmap 同时对 batch 的 lat 和 x 操作
                jacobian = vmap(jacrev(spec_fn), in_dims=(0, 0))(out["latent"], x_det)
                # jacobian: (B, L_spec, latent_dim)
                latent_grad_loss = jacobian.pow(2).sum(dim=(1, 2)).mean()
                loss = loss + lambda_latent_grad * latent_grad_loss

            # ---- 先验惩罚 ----
            prior_loss = torch.tensor(0.0, device=device)
            if "x" in free_keys:
                x_err = batch["x_err"].clamp_min(1e-6)
                prior_loss = prior_loss + lambda_x * (
                    (star_params["x_pred"][idx] - x0[idx]) / x_err
                ).pow(2).sum(dim=1).mean()
            if "E" in free_keys:
                E_err = batch["E_err"].view(-1).clamp_min(1e-6)
                E_pred_lin = torch.exp(star_params["log_E_pred"][idx].view(-1))
                gaussian_prior = lambda_E * ((E_pred_lin - E0[idx]) / E_err).pow(2).mean()
                jacobian = -2.0 * lambda_E * star_params["log_E_pred"][idx].view(-1).mean()
                prior_loss = prior_loss + gaussian_prior + jacobian
            
            if "plx" in free_keys:
                log_plx_pred = star_params["log_plx_pred"][idx].view(-1)
                plx_err = batch["plx_err"].view(-1).clamp_min(1e-6)
                sig_log_plx = (plx_err / plx0[idx]).clamp_min(1e-6)
                gaussian_prior = lambda_plx * ((log_plx_pred - torch.log(plx0[idx])) / sig_log_plx).pow(2).mean()
                jacobian = -2.0 * lambda_plx * log_plx_pred.mean()
                prior_loss = prior_loss + gaussian_prior + jacobian

            loss = loss + prior_loss

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainables, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # ---- epoch 结束：更新学习率调度器 ----
        avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
        if scheduler is not None:
            if lr_schedule == "plateau":
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        # ---- 更新进度条 ----
        if rank == 0 and n_batches > 0:
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(loss=f"{avg_loss:.3e}", lr=f"{current_lr:.2e}")

    if rank == 0 and hasattr(pbar, 'close'):
        pbar.close()

if __name__ == "__main__":
    import torch.distributed as dist

    # ---- 分布式初始化（torchrun 自动设置环境变量）----
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # ---- 超参数 ----
    latent_dim = 2
    dataset_h5 = "data/stellar_dataset_1pct.h5"   # 测试用1%数据，全量换成全量文件

    # ---- 构建 P 矩阵 ----
    P, L_spec = build_P_66xL(dtype=dtype, device="cpu")

    # ---- 加载数据集 ----
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

    # ---- 模型 ----
    model = StellarModel(
        P_66xL=P,
        latent_dim=latent_dim,
        init_k1=torch.zeros(P.shape[1], dtype=dtype),
        dtype=dtype,
    ).to(device)

    # 可选编译（PyTorch 2.0+）
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception:
        pass

    # ---- 恒星参数 ----
    star = init_star_params(dataset, device, latent_dim=latent_dim)

    # ---- 四个训练阶段 ----
    train_stage(model=model, dataset=dataset, L_spec=L_spec, device=device,
                epochs=512, batch_size=1024, lr=1e-3, num_workers=4,
                free_keys=(), update_model=True, train_k1=False,
                star_params=star, lambda_latent=1.0, desc="Stage 1")

    train_stage(model=model, dataset=dataset, L_spec=L_spec, device=device,
                epochs=512, batch_size=1024, lr=1e-3, num_workers=4,
                free_keys=("latent",), update_model=False, train_k1=False,
                star_params=star, lambda_latent=1.0, desc="Stage 2")

    train_stage(model=model, dataset=dataset, L_spec=L_spec, device=device,
                epochs=512, batch_size=1024, lr=5e-4, num_workers=4,
                free_keys=("latent",), update_model=True, train_k1=False,
                star_params=star, lambda_latent=1.0, desc="Stage 3")

    train_stage(model=model, dataset=dataset, L_spec=L_spec, device=device,
                epochs=512, batch_size=1024, lr=5e-4, num_workers=4,
                free_keys=("x", "E", "plx", "latent"), update_model=True,
                train_k1=False, star_params=star,
                lambda_x=1.0, lambda_E=1.0, lambda_plx=1.0, lambda_latent=1.0,
                desc="Stage 4")

    # ---- 只在 rank 0 保存 ----
    if local_rank == 0:
        torch.save({
            "model_state": model.state_dict(),
            "star_params": {k: v.detach().cpu() for k, v in star.items()},
        }, "stellar_model.pt")
        print("Training finished and model saved.")        


def get_raw_model(model):
    if hasattr(model, '_orig_mod'):
        return model._orig_mod
    return model

# 使用：
# model = get_raw_model(model)