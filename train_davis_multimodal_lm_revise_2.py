# -*- coding: utf-8 -*-
# ======================================
# Davis token-level 冷启动训练脚本
#
# 作用：
#   1. 使用 LoadData_davis_lm_1d_token_scaffold_seqsplit。
#   2. 使用与 ATOM3D SI-30 / SI-60 相同的 TokenLevelMultiModalDTA 模型。
#   3. Davis 无 3D 结构输入，因此模型自动走 1D-only 分支。
#   4. 使用 scaffold-disjoint + protein-sequence-disjoint 冷启动划分。
#   5. 统一训练模型与最终 best_model 的初始化参数，避免 size mismatch。
# ======================================

from pathlib import Path
import logging
import time
import csv
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter

import util

# 优先从 util_load 导入；如果你把函数放在 util.py 中，则自动兜底。
try:
    from util_load import LoadData_davis_lm_1d_token_scaffold_seqsplit
except ImportError:
    from util import LoadData_davis_lm_1d_token_scaffold_seqsplit


# ======================================
# 设备 / 日志 / 随机种子
# ======================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
LOG_LEVEL = logging.INFO
SANITY_CHECK = True


# ======================================
# 版本标记和路径
# ======================================
REV_TAG = "token_bidir_davis_strict_602020"
MODEL_MODULE_NAME = "model_token_crossmodal_lm_revise"
RESULT_ROOT_NAME = "result_revise"
DATASET_NAME = f"davis_{REV_TAG}"

DATA_DAVIS = r"../dataset/davis"
OUT_DIR = f"../{RESULT_ROOT_NAME}/{DATASET_NAME}"

# 0.6/0.2/0.2 冷启动划分使用独立缓存目录，避免误读旧 0.8/0.1/0.1 缓存。
CACHE_DIR = "../dataset/davis/processed_lm_1d_token_scaffold_seqsplit_602020"


# ======================================
# 训练超参数
# ======================================
# Davis 采用 token bank，不展开 pair-level 蛋白 token，但 Windows 下仍建议 num_workers=0。
BATCH_SIZE = 8
NUM_WORKERS = 0
PIN_MEMORY = False

EPOCHS = 600
LR = 5e-5
WEIGHT_DECAY = 1e-3
USE_AMP = False
ACCUM_STEPS = 4
GRAD_CLIP_NORM = 0.5
USE_SCHEDULER = True

# 正式结果建议只用 val 选 best。
VAL_AS_BEST = True
EVAL_TEST_EACH_EPOCH = False

EARLY_STOP_PATIENCE = 60
EARLY_STOP_MIN_DELTA = 1e-4
RESUME_TRAINING = False


# ======================================
# LM token 生成参数
# ======================================
ESM2_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
CHEMBERTA_MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
LM_BATCH_SIZE = 2
CHEM_MAX_LEN = 128
PROT_MAX_LEN = 1024
USE_SAFETENSORS = True
MASK_SPECIAL_TOKENS = True
FORCE_REFRESH = False


# ======================================
# 冷启动划分参数
# ======================================
SPLIT_SEED = 2026
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2


# ======================================
# 模型参数：训练模型和最终 best_model 必须完全一致
# ======================================
MODEL_D_MODEL = 128
MODEL_D_ATTN = 128
MODEL_N_HEADS = 4
MODEL_DROPOUT = 0.25
MODEL_USE_2D = False

# Davis 没有 3D，但模型构造函数仍需要保留这两个维度占位。
D3_LIG_DIM = 512
D3_POC_DIM = 512


# ======================================
# Davis Dataset：token bank + pair index
# ======================================
class DavisTokenBankDataset(Data.Dataset):
    """按 pair 索引访问 ligand/protein token bank，避免重复展开蛋白 token。"""

    def __init__(self, pkg: Dict[str, Any]):
        self.ids = pkg["ids"]
        self.y = pkg["y"]
        self.smiles = pkg["smiles"]
        self.seq = pkg["seq"]
        self.ligand_scaffold = pkg.get(
            "ligand_scaffold",
            np.array(["unknown"] * len(self.ids), dtype=object),
        )

        self.pair_lig_idx = pkg["pair_lig_idx"].astype(np.int64)
        self.pair_pro_idx = pkg["pair_pro_idx"].astype(np.int64)

        self.drug_lm_tokens_bank = pkg["drug_lm_tokens_bank"]
        self.drug_lm_mask_bank = pkg["drug_lm_mask_bank"]
        self.prot_lm_tokens_bank = pkg["prot_lm_tokens_bank"]
        self.prot_lm_mask_bank = pkg["prot_lm_mask_bank"]

        self.split_protocol = pkg.get("split_protocol", "unknown")

        n = len(self.y)
        assert len(self.ids) == n
        assert len(self.smiles) == n
        assert len(self.seq) == n
        assert len(self.pair_lig_idx) == n
        assert len(self.pair_pro_idx) == n

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        lig_idx = int(self.pair_lig_idx[idx])
        pro_idx = int(self.pair_pro_idx[idx])

        return {
            "id": self.ids[idx],
            "y": float(self.y[idx]),
            "smiles": self.smiles[idx],
            "seq": self.seq[idx],
            "ligand_scaffold": self.ligand_scaffold[idx],
            "pair_lig_idx": lig_idx,
            "pair_pro_idx": pro_idx,

            "drug_lm_tokens": self.drug_lm_tokens_bank[lig_idx],
            "drug_lm_mask": self.drug_lm_mask_bank[lig_idx],

            "prot_lm_tokens": self.prot_lm_tokens_bank[pro_idx],
            "prot_lm_mask": self.prot_lm_mask_bank[pro_idx],
        }


# ======================================
# collate
# ======================================
def davis_token_collate(batch: list) -> Dict[str, Any]:
    """合并 Davis token-level batch。"""
    ids = [b["id"] for b in batch]
    smiles = [b["smiles"] for b in batch]
    seqs = [b["seq"] for b in batch]
    scaffolds = [b["ligand_scaffold"] for b in batch]
    pair_lig_idx = [b["pair_lig_idx"] for b in batch]
    pair_pro_idx = [b["pair_pro_idx"] for b in batch]

    drug_lm_tokens = torch.as_tensor(
        np.stack([b["drug_lm_tokens"] for b in batch], axis=0),
        dtype=torch.float32,
    )
    drug_lm_mask = torch.as_tensor(
        np.stack([b["drug_lm_mask"] for b in batch], axis=0),
        dtype=torch.bool,
    )

    prot_lm_tokens = torch.as_tensor(
        np.stack([b["prot_lm_tokens"] for b in batch], axis=0),
        dtype=torch.float32,
    )
    prot_lm_mask = torch.as_tensor(
        np.stack([b["prot_lm_mask"] for b in batch], axis=0),
        dtype=torch.bool,
    )

    y = torch.as_tensor([b["y"] for b in batch], dtype=torch.float32)

    return {
        "ids": ids,
        "smiles": smiles,
        "seqs": seqs,
        "ligand_scaffold": scaffolds,
        "pair_lig_idx": pair_lig_idx,
        "pair_pro_idx": pair_pro_idx,

        "drug_lm_tokens": drug_lm_tokens,
        "drug_lm_mask": drug_lm_mask,
        "prot_lm_tokens": prot_lm_tokens,
        "prot_lm_mask": prot_lm_mask,

        "y": y,
    }


# ======================================
# 冷启动划分审计
# ======================================
def _overlap_count(a, b):
    aset = set(str(x) for x in a)
    bset = set(str(x) for x in b)
    overlap = sorted(aset.intersection(bset))
    return len(overlap), overlap[:5]


def audit_davis_strict_split(
    train_pkg: Dict[str, Any],
    val_pkg: Dict[str, Any],
    test_pkg: Dict[str, Any],
    save_path: Optional[Path] = None,
    strict: bool = True,
):
    """检查 sample id、ligand scaffold、protein sequence 是否跨 split 重叠。"""
    split_pkgs = {"train": train_pkg, "val": val_pkg, "test": test_pkg}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    fields = [
        ("ids", "sample_id"),
        ("ligand_scaffold", "ligand_scaffold"),
        ("seq", "protein_sequence"),
        ("smiles", "smiles"),
    ]

    lines = []
    lines.append("========== Davis Strict Split Audit ==========")
    lines.append("Protocol: ligand Murcko scaffold split + protein full-sequence-disjoint split")
    lines.append("A pair is kept only when ligand split == protein split; cross-partition pairs are excluded.")
    lines.append("")

    has_critical_overlap = False

    for a, b in pairs:
        lines.append(f"--- {a} vs {b} ---")
        for key, name in fields:
            n_overlap, examples = _overlap_count(split_pkgs[a][key], split_pkgs[b][key])
            lines.append(f"[{name}] overlap={n_overlap} | examples={examples}")
            if name in ("sample_id", "ligand_scaffold", "protein_sequence") and n_overlap > 0:
                has_critical_overlap = True
        lines.append("")

    if has_critical_overlap:
        lines.append("[WARNING] Critical overlap detected in sample_id, ligand_scaffold, or protein_sequence.")
    else:
        lines.append("[OK] No exact sample_id / ligand_scaffold / protein_sequence overlap detected across splits.")

    text = "\n".join(lines)
    print(text)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")

    if strict and has_critical_overlap:
        raise RuntimeError(
            "Davis strict split audit failed: exact sample_id, ligand_scaffold, or protein_sequence overlap detected."
        )


# ======================================
# 工具函数
# ======================================
def has_nan_params(model: nn.Module) -> bool:
    """检查参数中是否出现 NaN / Inf。"""
    for p in model.parameters():
        if p is not None and torch.is_tensor(p):
            if torch.isnan(p).any() or torch.isinf(p).any():
                return True
    return False


def move_batch_to_device(b: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """把 batch 中的 tensor 搬到设备。"""
    return {
        "drug_lm_tokens": b["drug_lm_tokens"].to(device, non_blocking=True),
        "drug_lm_mask": b["drug_lm_mask"].to(device, non_blocking=True),
        "prot_lm_tokens": b["prot_lm_tokens"].to(device, non_blocking=True),
        "prot_lm_mask": b["prot_lm_mask"].to(device, non_blocking=True),
        "y": b["y"].to(device, non_blocking=True),
    }


def forward_model(model: nn.Module, bdev: Dict[str, Any]) -> torch.Tensor:
    """
    Davis 是 1D-only 数据集。
    这里只传 1D token，模型会自动走 single-modal sequence 分支。
    """
    return model(
        drug_lm_tokens=bdev["drug_lm_tokens"],
        prot_lm_tokens=bdev["prot_lm_tokens"],
        drug_lm_mask=bdev["drug_lm_mask"],
        prot_lm_mask=bdev["prot_lm_mask"],
    )


@torch.no_grad()
def evaluate_davis(
    model: nn.Module,
    loader: Data.DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
):
    """
    评估 Davis 常用指标。
    若给定 criterion：返回 ci, mse, rm2, aupr, avg_loss；
    否则返回 ci, mse, rm2, aupr。
    """
    model.eval()

    obs, pred = [], []
    loss_sum = 0.0
    n_batches = 0

    for b in loader:
        bdev = move_batch_to_device(b, device)

        with torch.cuda.amp.autocast(enabled=(USE_AMP and device.type == "cuda")):
            y_hat = forward_model(model, bdev)

            if criterion is not None:
                loss_val = criterion(y_hat, bdev["y"])
                if torch.isfinite(loss_val):
                    loss_sum += float(loss_val.detach().cpu())
                    n_batches += 1

        obs.extend(bdev["y"].view(-1).detach().cpu().tolist())
        pred.extend(y_hat.view(-1).detach().cpu().tolist())

    ci = util.get_cindex(obs, pred)
    mse = util.get_MSE(obs, pred)
    rm2 = util.get_rm2(obs, pred)
    aupr = util.get_aupr(obs, pred)

    if criterion is None:
        return ci, mse, rm2, aupr
    return ci, mse, rm2, aupr, loss_sum / max(1, n_batches)


def build_model(
    model_cls,
    d_drug_lm: int,
    d_prot_lm: int,
    device: torch.device,
) -> nn.Module:
    """统一构建模型，避免训练模型和 best_model 参数不一致。"""
    return model_cls(
        d_drug_lm=d_drug_lm,
        d_prot_lm=d_prot_lm,
        d3_lig=D3_LIG_DIM,
        d3_poc=D3_POC_DIM,

        drug_node_dim=70,
        drug_edge_dim=6,
        prot_node_dim=33,
        prot_edge_dim=3,
        gcn_hidden=128,
        gcn_out=128,

        d_model=MODEL_D_MODEL,
        d_attn=MODEL_D_ATTN,
        n_heads=MODEL_N_HEADS,
        dropout=MODEL_DROPOUT,
        use_2d=MODEL_USE_2D,
    ).to(device)


# ======================================
# 主程序
# ======================================
if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL)
    util.seed_torch()

    result_root = Path(OUT_DIR)
    result_root.mkdir(parents=True, exist_ok=True)

    logging.info(f"[Revision Tag] {REV_TAG}")
    logging.info(f"[Model Module] {MODEL_MODULE_NAME}")
    logging.info(f"[Output Dir] {result_root}")
    logging.info(f"[Device] {DEVICE}")

    # 加载 Davis strict token-level 数据。
    kw_common = dict(
        out_1d=CACHE_DIR,
        logspace_trans=True,
        esm2_model_name=ESM2_MODEL_NAME,
        chemberta_model_name=CHEMBERTA_MODEL_NAME,
        lm_batch_size=LM_BATCH_SIZE,
        chem_max_len=CHEM_MAX_LEN,
        prot_max_len=PROT_MAX_LEN,
        use_safetensors=USE_SAFETENSORS,
        mask_special_tokens=MASK_SPECIAL_TOKENS,
        split_seed=SPLIT_SEED,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        force_refresh=FORCE_REFRESH,
        return_pair_level_tokens=False,
    )

    mm_train = LoadData_davis_lm_1d_token_scaffold_seqsplit(
        data_dir=DATA_DAVIS, split="train", **kw_common
    )["train"]

    mm_val = LoadData_davis_lm_1d_token_scaffold_seqsplit(
        data_dir=DATA_DAVIS, split="val", **kw_common
    )["val"]

    mm_test = LoadData_davis_lm_1d_token_scaffold_seqsplit(
        data_dir=DATA_DAVIS, split="test", **kw_common
    )["test"]

    # 输出目录。
    run_dir = result_root / f"runs_{REV_TAG}"
    result_dir = result_root / f"train_logs_{REV_TAG}"
    model_best = result_dir / f"best_model_{REV_TAG}.pth"
    ckpt_path = result_dir / f"checkpoint_{REV_TAG}.pth.tar"
    csv_file = result_dir / f"metrics_{REV_TAG}.csv"
    final_txt = result_dir / f"final_test_metrics_{REV_TAG}.txt"
    config_txt = result_dir / f"run_config_{REV_TAG}.txt"
    split_audit_txt = result_dir / f"split_audit_{REV_TAG}.txt"

    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    # 划分审计。
    audit_davis_strict_split(
        train_pkg=mm_train,
        val_pkg=mm_val,
        test_pkg=mm_test,
        save_path=split_audit_txt,
        strict=True,
    )

    # 读取 token 维度。
    d_drug_lm = mm_train["drug_lm_tokens_bank"].shape[-1]
    d_prot_lm = mm_train["prot_lm_tokens_bank"].shape[-1]
    drug_token_len = mm_train["drug_lm_tokens_bank"].shape[1]
    prot_token_len = mm_train["prot_lm_tokens_bank"].shape[1]

    logging.info(
        f"[Davis|token] d_drug_lm={d_drug_lm}, d_prot_lm={d_prot_lm}, "
        f"drug_len={drug_token_len}, prot_len={prot_token_len}"
    )

    # Dataset / DataLoader。
    train_ds = DavisTokenBankDataset(mm_train)
    val_ds = DavisTokenBankDataset(mm_val)
    test_ds = DavisTokenBankDataset(mm_test)

    train_loader = Data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=davis_token_collate,
    )

    val_loader = Data.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=davis_token_collate,
    )

    test_loader = Data.DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=davis_token_collate,
    )

    logging.info(
        f"[Davis|strict token] sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    # Sanity Check。
    if SANITY_CHECK:
        b = next(iter(train_loader))
        print(">>> Batch keys:", list(b.keys()))
        print(
            "drug_lm_tokens:", tuple(b["drug_lm_tokens"].shape),
            "| drug_lm_mask:", tuple(b["drug_lm_mask"].shape),
        )
        print(
            "prot_lm_tokens:", tuple(b["prot_lm_tokens"].shape),
            "| prot_lm_mask:", tuple(b["prot_lm_mask"].shape),
        )
        print("y:", tuple(b["y"].shape))
        print("example scaffold:", b["ligand_scaffold"][:3])
        print("[OK] Davis strict token-level loaders ready.")

    # 导入与 SI-30/SI-60 相同的模型。
    from model.model_token_crossmodal_lm_revise import MultiModalDTA_LM

    model = build_model(
        MultiModalDTA_LM,
        d_drug_lm=d_drug_lm,
        d_prot_lm=d_prot_lm,
        device=DEVICE,
    )

    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            threshold=1e-4,
            threshold_mode="abs",
            min_lr=1e-6,
        )
        if USE_SCHEDULER else None
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.type == "cuda"))

    writer = SummaryWriter(log_dir=str(run_dir))

    # 保存本次配置。
    with open(config_txt, "w", encoding="utf-8") as f:
        f.write(f"REV_TAG: {REV_TAG}\n")
        f.write(f"MODEL_MODULE_NAME: {MODEL_MODULE_NAME}\n")
        f.write(f"DATA_DAVIS: {DATA_DAVIS}\n")
        f.write(f"CACHE_DIR: {CACHE_DIR}\n")
        f.write(f"OUT_DIR: {OUT_DIR}\n")
        f.write("SPLIT_PROTOCOL: Davis ligand Murcko scaffold + protein full-sequence-disjoint product split\n")
        f.write(f"SPLIT_SEED: {SPLIT_SEED}\n")
        f.write(f"TRAIN_RATIO: {TRAIN_RATIO}\n")
        f.write(f"VAL_RATIO: {VAL_RATIO}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"NUM_WORKERS: {NUM_WORKERS}\n")
        f.write(f"PIN_MEMORY: {PIN_MEMORY}\n")
        f.write(f"EPOCHS: {EPOCHS}\n")
        f.write(f"LR: {LR}\n")
        f.write(f"WEIGHT_DECAY: {WEIGHT_DECAY}\n")
        f.write(f"USE_AMP: {USE_AMP}\n")
        f.write(f"ACCUM_STEPS: {ACCUM_STEPS}\n")
        f.write(f"GRAD_CLIP_NORM: {GRAD_CLIP_NORM}\n")
        f.write(f"USE_SCHEDULER: {USE_SCHEDULER}\n")
        f.write(f"VAL_AS_BEST: {VAL_AS_BEST}\n")
        f.write(f"EVAL_TEST_EACH_EPOCH: {EVAL_TEST_EACH_EPOCH}\n")
        f.write(f"EARLY_STOP_PATIENCE: {EARLY_STOP_PATIENCE}\n")
        f.write(f"EARLY_STOP_MIN_DELTA: {EARLY_STOP_MIN_DELTA}\n")
        f.write(f"RESUME_TRAINING: {RESUME_TRAINING}\n")
        f.write(f"ESM2_MODEL_NAME: {ESM2_MODEL_NAME}\n")
        f.write(f"CHEMBERTA_MODEL_NAME: {CHEMBERTA_MODEL_NAME}\n")
        f.write(f"CHEM_MAX_LEN: {CHEM_MAX_LEN}\n")
        f.write(f"PROT_MAX_LEN: {PROT_MAX_LEN}\n")
        f.write(f"MODEL_D_MODEL: {MODEL_D_MODEL}\n")
        f.write(f"MODEL_D_ATTN: {MODEL_D_ATTN}\n")
        f.write(f"MODEL_N_HEADS: {MODEL_N_HEADS}\n")
        f.write(f"MODEL_DROPOUT: {MODEL_DROPOUT}\n")
        f.write(f"MODEL_USE_2D: {MODEL_USE_2D}\n")
        f.write(f"d_drug_lm: {d_drug_lm}\n")
        f.write(f"d_prot_lm: {d_prot_lm}\n")
        f.write(f"train_size: {len(train_ds)}\n")
        f.write(f"val_size: {len(val_ds)}\n")
        f.write(f"test_size: {len(test_ds)}\n")

    # CSV 表头。
    if not csv_file.exists():
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            header = [
                "epoch", "train_loss", "val_loss",
                "train_CI", "train_MSE", "train_rm2", "train_AUPR",
                "val_CI", "val_MSE", "val_rm2", "val_AUPR",
                "lr",
            ]
            if EVAL_TEST_EACH_EPOCH:
                header.extend(["test_CI", "test_MSE", "test_rm2", "test_AUPR"])
            csv.writer(f).writerow(header)

    # 断点恢复。
    start_epoch = 0
    if RESUME_TRAINING and ckpt_path.exists():
        start_epoch = util.load_checkpoint(str(ckpt_path), model, optimizer)
        logging.info(f"[Resume] from epoch: {start_epoch}")

    best_score = float("inf")
    epochs_no_improve = 0
    t0 = time.time()

    # ======================================
    # 训练主循环
    # ======================================
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        n_batches = 0
        last_bidx = -1

        for last_bidx, b in enumerate(train_loader):
            bdev = move_batch_to_device(b, DEVICE)

            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type == "cuda")):
                y_hat = forward_model(model, bdev)
                loss = criterion(y_hat, bdev["y"])

            if not torch.isfinite(loss):
                logging.warning(
                    f"Non-finite loss at epoch={epoch}, batch={last_bidx}: {float(loss)}"
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            if scaler.is_enabled():
                scaler.scale(loss / ACCUM_STEPS).backward()
            else:
                (loss / ACCUM_STEPS).backward()

            if (last_bidx + 1) % ACCUM_STEPS == 0:
                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                if has_nan_params(model):
                    logging.error(
                        f"[NaN] 参数异常：epoch={epoch}, batch={last_bidx} before step"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    break

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                if has_nan_params(model):
                    logging.error(
                        f"[NaN] 参数异常：epoch={epoch}, batch={last_bidx} after step"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    break

            running_loss += float(loss.detach().cpu())
            n_batches += 1

        # 处理最后不足 ACCUM_STEPS 的梯度。
        if (last_bidx + 1) % ACCUM_STEPS != 0 and n_batches > 0 and not has_nan_params(model):
            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / max(1, n_batches)
        writer.add_scalar("Loss/train", train_loss, epoch)

        # 评估 train / val。
        train_ci, train_mse, train_rm2, train_aupr, _ = evaluate_davis(
            model, train_loader, DEVICE, criterion
        )

        val_ci, val_mse, val_rm2, val_aupr, val_loss = evaluate_davis(
            model, val_loader, DEVICE, criterion
        )

        writer.add_scalar("Loss/val", val_loss, epoch)

        for tag, val in [
            ("train_CI", train_ci),
            ("train_MSE", train_mse),
            ("train_rm2", train_rm2),
            ("train_AUPR", train_aupr),
            ("val_CI", val_ci),
            ("val_MSE", val_mse),
            ("val_rm2", val_rm2),
            ("val_AUPR", val_aupr),
        ]:
            writer.add_scalar(f"Metrics/{tag}", val, epoch)

        test_ci = test_mse = test_rm2 = test_aupr = None
        if EVAL_TEST_EACH_EPOCH:
            test_ci, test_mse, test_rm2, test_aupr = evaluate_davis(
                model, test_loader, DEVICE, criterion=None
            )
            for tag, val in [
                ("test_CI", test_ci),
                ("test_MSE", test_mse),
                ("test_rm2", test_rm2),
                ("test_AUPR", test_aupr),
            ]:
                writer.add_scalar(f"Metrics/{tag}", val, epoch)

        if scheduler is not None:
            scheduler.step(val_mse)

        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("LR", current_lr, epoch)

        row = [
            epoch,
            train_loss,
            val_loss,
            train_ci,
            train_mse,
            train_rm2,
            train_aupr,
            val_ci,
            val_mse,
            val_rm2,
            val_aupr,
            current_lr,
        ]
        if EVAL_TEST_EACH_EPOCH:
            row.extend([test_ci, test_mse, test_rm2, test_aupr])

        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

        score_now = val_mse if VAL_AS_BEST else train_mse
        improved = (best_score - score_now) > EARLY_STOP_MIN_DELTA

        if improved and torch.isfinite(torch.tensor(score_now)):
            best_score = score_now
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_best)
            logging.info(
                f"[BEST] epoch={epoch + 1}, "
                f"{'val' if VAL_AS_BEST else 'train'}_MSE={best_score:.4f} "
                f"-> saved: {model_best}"
            )
        else:
            epochs_no_improve += 1
            logging.info(
                f"[EARLY-STOP] no improve for {epochs_no_improve} epoch(s); "
                f"best_score={best_score:.4f}, now={score_now:.4f}"
            )

            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                logging.info(
                    f"[EARLY-STOP] patience={EARLY_STOP_PATIENCE} reached at epoch {epoch + 1}, stop training."
                )
                util.save_checkpoint(epoch + 1, model, optimizer, filename=str(ckpt_path))
                break

        if EVAL_TEST_EACH_EPOCH:
            logging.info(
                f"Epoch {epoch + 1:04d} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"train(CI={train_ci:.4f}, MSE={train_mse:.4f}, rm2={train_rm2:.4f}, AUPR={train_aupr:.4f}) | "
                f"val(CI={val_ci:.4f}, MSE={val_mse:.4f}, rm2={val_rm2:.4f}, AUPR={val_aupr:.4f}) | "
                f"test(CI={test_ci:.4f}, MSE={test_mse:.4f}, rm2={test_rm2:.4f}, AUPR={test_aupr:.4f}) | "
                f"lr={current_lr:.3e} | time={(time.time() - t0) / 60:.2f} min"
            )
        else:
            logging.info(
                f"Epoch {epoch + 1:04d} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"train(CI={train_ci:.4f}, MSE={train_mse:.4f}, rm2={train_rm2:.4f}, AUPR={train_aupr:.4f}) | "
                f"val(CI={val_ci:.4f}, MSE={val_mse:.4f}, rm2={val_rm2:.4f}, AUPR={val_aupr:.4f}) | "
                f"test=not evaluated during training | "
                f"lr={current_lr:.3e} | time={(time.time() - t0) / 60:.2f} min"
            )

        util.save_checkpoint(epoch + 1, model, optimizer, filename=str(ckpt_path))

    # ======================================
    # 最终测试：用验证集选出的 best model，仅评估一次 test
    # ======================================
    best_model = build_model(
        MultiModalDTA_LM,
        d_drug_lm=d_drug_lm,
        d_prot_lm=d_prot_lm,
        device=DEVICE,
    )

    if not model_best.exists():
        raise RuntimeError(
            f"Best model not found: {model_best}. 训练过程中可能没有成功保存最优模型。"
        )

    best_model.load_state_dict(torch.load(model_best, map_location=DEVICE))
    best_model.eval()

    final_ci, final_mse, final_rm2, final_aupr = evaluate_davis(
        best_model,
        test_loader,
        DEVICE,
        criterion=None,
    )

    logging.info(
        f"[FINAL TEST - once only] "
        f"test_CI={final_ci:.4f}, test_MSE={final_mse:.4f}, "
        f"test_rm2={final_rm2:.4f}, test_AUPR={final_aupr:.4f}"
    )

    with open(final_txt, "w", encoding="utf-8") as f:
        f.write("Final test was evaluated once after selecting the best model by validation MSE.\n")
        f.write("Dataset protocol: Davis ligand Murcko scaffold + protein full-sequence-disjoint product split.\n")
        f.write(f"test_CI: {final_ci:.4f}\n")
        f.write(f"test_MSE: {final_mse:.4f}\n")
        f.write(f"test_rm2: {final_rm2:.4f}\n")
        f.write(f"test_AUPR: {final_aupr:.4f}\n")

    writer.close()
