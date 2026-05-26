# -*- coding: utf-8 -*-
# ======================================
# ATOM3D SI-30 token-level 多模态训练脚本
#
# 作用：
#   1. 使用 token-level LoadData_atom3d_si30_multimodal_lm_token。
#   2. 使用新版 TokenLevelMultiModalDTA / MultiModalDTA_LM。
#   3. 修复 add_interactions_3d 参数不兼容问题。
#   4. 统一训练模型和最终 best_model 的初始化参数，避免 size mismatch。
#   5. 默认不使用 DataLoader 多进程，避免 Windows 下大 token 数组 MemoryError。
# ======================================

from pathlib import Path
import logging
import time
import csv
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter

import util

# 优先从 util_load 导入；如果你的函数放在 util.py 中，则自动兜底。
try:
    from util_load import LoadData_atom3d_si30_multimodal_lm_token
except ImportError:
    from util import LoadData_atom3d_si30_multimodal_lm_token


# ======================================
# 关闭 Uni-Mol 冗余日志
# ======================================
lg = logging.getLogger("unimol_tools")
lg.setLevel(logging.WARNING)
lg.propagate = False
for h in list(lg.handlers):
    lg.removeHandler(h)


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
REV_TAG = "token_bidir_fixed_no_addint"
MODEL_MODULE_NAME = "model_token_crossmodal_lm_revise"
RESULT_ROOT_NAME = "result_revise"
DATASET_NAME = f"atom3d_si30_mm_lm_{REV_TAG}"

DATA_DB_SI30 = r"../dataset/ATOM3D/split-by-sequence-identity-30/data"
OUT_DIR = f"../{RESULT_ROOT_NAME}/{DATASET_NAME}"

# token 缓存目录；如果你已经生成过 1024 token 缓存，可继续用这个。
# 如果你想重新生成 512 token 缓存，请改成新的目录，并把 PROT_MAX_LEN 改成 512。
CACHE_DIR = "../dataset/ATOM3D/processed_mm_si30_lm_token"


# ======================================
# 训练超参数
# ======================================
# token-level 特征很大，Windows 下建议 num_workers=0。
BATCH_SIZE = 2
NUM_WORKERS = 0
PIN_MEMORY = False

EPOCHS = 600
LR = 1e-4
WEIGHT_DECAY = 1e-4
USE_AMP = False
ACCUM_STEPS = 8
GRAD_CLIP_NORM = 0.5
USE_SCHEDULER = True

# 正式结果建议只用 val 选模型。
VAL_AS_BEST = True
EVAL_TEST_EACH_EPOCH = True   # 如果写论文正式结果，建议改成 False。

EARLY_STOP_PATIENCE = 60
EARLY_STOP_MIN_DELTA = 1e-4
RESUME_TRAINING = False       # 防止误读旧 checkpoint，默认不恢复。


# ======================================
# token embedding 生成参数
# ======================================
UNIMOL2_SIZE = "unimol2_small"
ESM2_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
CHEMBERTA_MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
LM_BATCH_SIZE = 2
CHEM_MAX_LEN = 128
PROT_MAX_LEN = 1024
LIG_MAX_ATOMS = 128
POCKET_MAX_ATOMS = 256
USE_SAFETENSORS = True
MASK_SPECIAL_TOKENS = True
ALLOW_POOLED_3D_FALLBACK = False
FORCE_REFRESH = False


# ======================================
# 模型参数：训练和最终 best_model 必须完全一致
# ======================================
MODEL_D_MODEL = 256
MODEL_D_ATTN = 256
MODEL_N_HEADS = 4
MODEL_DROPOUT = 0.10
MODEL_USE_2D = False


# ======================================
# 统计 si-30 拆分
# ======================================
def count_si30_splits(root_base: str) -> Tuple[Dict[str, int], Dict[str, float], int]:
    """统计原始 LMDB 中 train/val/test 数量和比例。"""
    import atom3d.datasets as da

    root = Path(root_base)
    counts = {}
    for sp in ("train", "val", "test"):
        ds = da.LMDBDataset(str(root / sp))
        counts[sp] = len(ds)

    total = sum(counts.values())
    ratios = {sp: counts[sp] / total if total > 0 else 0.0 for sp in counts}
    return counts, ratios, total


# ======================================
# Dataset
# ======================================
class MultiModalDatasetLMToken(Data.Dataset):
    """封装 1D token + 3D token 特征。"""

    def __init__(self, pkg: Dict[str, Any]):
        self.ids = pkg["ids"]
        self.y = pkg["y"]
        self.smiles = pkg["smiles"]
        self.seq = pkg["seq"]
        self.seq_chain_policy = pkg.get(
            "seq_chain_policy",
            np.array(["unknown"] * len(self.y), dtype=object),
        )

        self.drug_lm_tokens = pkg["drug_lm_tokens"]
        self.drug_lm_mask = pkg["drug_lm_mask"]
        self.prot_lm_tokens = pkg["prot_lm_tokens"]
        self.prot_lm_mask = pkg["prot_lm_mask"]

        self.lig3d_tokens = pkg["lig_3d_tokens"]
        self.lig3d_mask = pkg["lig_3d_mask"]
        self.poc3d_tokens = pkg["poc_3d_tokens"]
        self.poc3d_mask = pkg["poc_3d_mask"]

        n = len(self.y)
        assert len(self.ids) == n
        assert self.drug_lm_tokens.shape[0] == n
        assert self.prot_lm_tokens.shape[0] == n
        assert self.lig3d_tokens.shape[0] == n
        assert self.poc3d_tokens.shape[0] == n

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "id": self.ids[idx],
            "y": float(self.y[idx]),
            "smiles": self.smiles[idx],
            "seq": self.seq[idx],
            "seq_chain_policy": self.seq_chain_policy[idx],

            "drug_lm_tokens": self.drug_lm_tokens[idx],
            "drug_lm_mask": self.drug_lm_mask[idx],
            "prot_lm_tokens": self.prot_lm_tokens[idx],
            "prot_lm_mask": self.prot_lm_mask[idx],

            "lig3d_tokens": self.lig3d_tokens[idx],
            "lig3d_mask": self.lig3d_mask[idx],
            "poc3d_tokens": self.poc3d_tokens[idx],
            "poc3d_mask": self.poc3d_mask[idx],

            "g_lig": None,
            "g_prot": None,
        }


# ======================================
# collate
# ======================================
def multimodal_collate_lm_token(batch: list) -> Dict[str, Any]:
    """合并 token-level batch。"""
    ids = [b["id"] for b in batch]
    smiles = [b["smiles"] for b in batch]
    seqs = [b["seq"] for b in batch]
    seq_chain_policy = [b["seq_chain_policy"] for b in batch]

    drug_lm_tokens = torch.as_tensor(
        np.stack([b["drug_lm_tokens"] for b in batch], axis=0), dtype=torch.float32
    )
    drug_lm_mask = torch.as_tensor(
        np.stack([b["drug_lm_mask"] for b in batch], axis=0), dtype=torch.bool
    )
    prot_lm_tokens = torch.as_tensor(
        np.stack([b["prot_lm_tokens"] for b in batch], axis=0), dtype=torch.float32
    )
    prot_lm_mask = torch.as_tensor(
        np.stack([b["prot_lm_mask"] for b in batch], axis=0), dtype=torch.bool
    )

    lig3d_tokens = torch.as_tensor(
        np.stack([b["lig3d_tokens"] for b in batch], axis=0), dtype=torch.float32
    )
    lig3d_mask = torch.as_tensor(
        np.stack([b["lig3d_mask"] for b in batch], axis=0), dtype=torch.bool
    )
    poc3d_tokens = torch.as_tensor(
        np.stack([b["poc3d_tokens"] for b in batch], axis=0), dtype=torch.float32
    )
    poc3d_mask = torch.as_tensor(
        np.stack([b["poc3d_mask"] for b in batch], axis=0), dtype=torch.bool
    )

    y = torch.as_tensor([b["y"] for b in batch], dtype=torch.float32)

    return {
        "ids": ids,
        "smiles": smiles,
        "seqs": seqs,
        "seq_chain_policy": seq_chain_policy,

        "drug_lm_tokens": drug_lm_tokens,
        "drug_lm_mask": drug_lm_mask,
        "prot_lm_tokens": prot_lm_tokens,
        "prot_lm_mask": prot_lm_mask,

        "lig3d_tokens": lig3d_tokens,
        "lig3d_mask": lig3d_mask,
        "poc3d_tokens": poc3d_tokens,
        "poc3d_mask": poc3d_mask,

        "g_lig": None,
        "g_prot": None,
        "y": y,
    }


# ======================================
# 工具函数
# ======================================
def has_nan_params(model: nn.Module) -> bool:
    """检查参数是否出现 NaN / Inf。"""
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

        "lig3d_tokens": b["lig3d_tokens"].to(device, non_blocking=True),
        "lig3d_mask": b["lig3d_mask"].to(device, non_blocking=True),
        "poc3d_tokens": b["poc3d_tokens"].to(device, non_blocking=True),
        "poc3d_mask": b["poc3d_mask"].to(device, non_blocking=True),

        "g_lig": b["g_lig"],
        "g_prot": b["g_prot"],
        "y": b["y"].to(device, non_blocking=True),
    }


def forward_model(model: nn.Module, bdev: Dict[str, Any]) -> torch.Tensor:
    """统一调用新版 token-level 模型。"""
    return model(
        drug_lm_tokens=bdev["drug_lm_tokens"],
        prot_lm_tokens=bdev["prot_lm_tokens"],
        lig3d_tokens=bdev["lig3d_tokens"],
        poc3d_tokens=bdev["poc3d_tokens"],

        drug_lm_mask=bdev["drug_lm_mask"],
        prot_lm_mask=bdev["prot_lm_mask"],
        lig3d_mask=bdev["lig3d_mask"],
        poc3d_mask=bdev["poc3d_mask"],

        g_lig=bdev["g_lig"],
        g_prot=bdev["g_prot"],
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Data.DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
):
    """评估 RMSE / Pearson / Spearman。"""
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

    rmse = util.get_RMSE(obs, pred)
    pearson = util.get_pearsonr(obs, pred)
    spearman = util.get_spearmanr(obs, pred)

    if criterion is None:
        return rmse, pearson, spearman
    return rmse, pearson, spearman, loss_sum / max(1, n_batches)


def build_model(
    model_cls,
    d_drug_lm: int,
    d_prot_lm: int,
    d3_lig: int,
    d3_poc: int,
    device: torch.device,
) -> nn.Module:
    """
    统一构建模型。

    注意：
        当前模型 __init__ 不接受 add_interactions_3d。
        所以这里不要再传 add_interactions_3d。
    """
    return model_cls(
        d_drug_lm=d_drug_lm,
        d_prot_lm=d_prot_lm,
        d3_lig=d3_lig,
        d3_poc=d3_poc,

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

    # 统计原始 split。
    counts, ratios, total = count_si30_splits(DATA_DB_SI30)
    logging.info(f"[si-30] counts(raw): {counts} | total={total}")
    logging.info(
        f"[si-30] ratios(raw): "
        f"train={ratios['train']:.4f}, val={ratios['val']:.4f}, test={ratios['test']:.4f}"
    )

    # 加载 token-level 数据。
    kw_common = dict(
        out_mm=CACHE_DIR,
        unimol2_size=UNIMOL2_SIZE,
        contact_threshold=8.0,
        dis_min=1.0,
        prot_self_loop=False,
        bond_bidirectional=True,
        prefer_model=None,
        force_refresh=FORCE_REFRESH,
        use_cuda_for_unimol=True,
        esm2_model_name=ESM2_MODEL_NAME,
        chemberta_model_name=CHEMBERTA_MODEL_NAME,
        lm_batch_size=LM_BATCH_SIZE,
        chem_max_len=CHEM_MAX_LEN,
        prot_max_len=PROT_MAX_LEN,
        lig_max_atoms=LIG_MAX_ATOMS,
        pocket_max_atoms=POCKET_MAX_ATOMS,
        use_safetensors=USE_SAFETENSORS,
        mask_special_tokens=MASK_SPECIAL_TOKENS,
        allow_pooled_3d_fallback=ALLOW_POOLED_3D_FALLBACK,
    )

    mm_train = LoadData_atom3d_si30_multimodal_lm_token(
        root_base=DATA_DB_SI30, split="train", **kw_common
    )["train"]
    mm_val = LoadData_atom3d_si30_multimodal_lm_token(
        root_base=DATA_DB_SI30, split="val", **kw_common
    )["val"]
    mm_test = LoadData_atom3d_si30_multimodal_lm_token(
        root_base=DATA_DB_SI30, split="test", **kw_common
    )["test"]

    # 读取维度。
    d_drug_lm = mm_train["drug_lm_tokens"].shape[-1]
    d_prot_lm = mm_train["prot_lm_tokens"].shape[-1]
    d3_lig = mm_train["lig_3d_tokens"].shape[-1]
    d3_poc = mm_train["poc_3d_tokens"].shape[-1]

    logging.info(
        f"[Token dims] d_drug_lm={d_drug_lm}, d_prot_lm={d_prot_lm}, "
        f"d3_lig={d3_lig}, d3_poc={d3_poc}"
    )
    logging.info(
        f"[Token lengths] drug={mm_train['drug_lm_tokens'].shape[1]}, "
        f"protein={mm_train['prot_lm_tokens'].shape[1]}, "
        f"lig3d={mm_train['lig_3d_tokens'].shape[1]}, "
        f"poc3d={mm_train['poc_3d_tokens'].shape[1]}"
    )

    # Dataset / DataLoader。
    train_ds = MultiModalDatasetLMToken(mm_train)
    val_ds = MultiModalDatasetLMToken(mm_val)
    test_ds = MultiModalDatasetLMToken(mm_test)

    train_loader = Data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=multimodal_collate_lm_token,
    )
    val_loader = Data.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=multimodal_collate_lm_token,
    )
    test_loader = Data.DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=multimodal_collate_lm_token,
    )

    logging.info(
        f"[si-30|token] final sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    # Sanity check。
    if SANITY_CHECK:
        b = next(iter(train_loader))
        print(">>> Batch keys:", list(b.keys()))
        print("drug_lm_tokens:", tuple(b["drug_lm_tokens"].shape), "| drug_lm_mask:", tuple(b["drug_lm_mask"].shape))
        print("prot_lm_tokens:", tuple(b["prot_lm_tokens"].shape), "| prot_lm_mask:", tuple(b["prot_lm_mask"].shape))
        print("lig3d_tokens:", tuple(b["lig3d_tokens"].shape), "| lig3d_mask:", tuple(b["lig3d_mask"].shape))
        print("poc3d_tokens:", tuple(b["poc3d_tokens"].shape), "| poc3d_mask:", tuple(b["poc3d_mask"].shape))
        print("y:", tuple(b["y"].shape))
        print("[OK] token-level 1D+3D loaders ready.")

    # 导入模型。
    from model.model_token_crossmodal_lm_revise import MultiModalDTA_LM

    model = build_model(
        MultiModalDTA_LM,
        d_drug_lm=d_drug_lm,
        d_prot_lm=d_prot_lm,
        d3_lig=d3_lig,
        d3_poc=d3_poc,
        device=DEVICE,
    )

    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = (
        optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-5
        )
        if USE_SCHEDULER else None
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.type == "cuda"))

    # 输出路径。
    run_dir = result_root / f"runs_{REV_TAG}"
    result_dir = result_root / f"train_logs_{REV_TAG}"
    model_best = result_dir / f"best_model_{REV_TAG}.pth"
    ckpt_path = result_dir / f"checkpoint_{REV_TAG}.pth.tar"
    csv_file = result_dir / f"metrics_{REV_TAG}.csv"
    final_txt = result_dir / f"final_test_metrics_{REV_TAG}.txt"
    config_txt = result_dir / f"run_config_{REV_TAG}.txt"

    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    # 保存配置。
    with open(config_txt, "w", encoding="utf-8") as f:
        f.write(f"REV_TAG: {REV_TAG}\n")
        f.write(f"MODEL_MODULE_NAME: {MODEL_MODULE_NAME}\n")
        f.write(f"DATA_DB_SI30: {DATA_DB_SI30}\n")
        f.write(f"CACHE_DIR: {CACHE_DIR}\n")
        f.write(f"OUT_DIR: {OUT_DIR}\n")
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
        f.write(f"CHEM_MAX_LEN: {CHEM_MAX_LEN}\n")
        f.write(f"PROT_MAX_LEN: {PROT_MAX_LEN}\n")
        f.write(f"LIG_MAX_ATOMS: {LIG_MAX_ATOMS}\n")
        f.write(f"POCKET_MAX_ATOMS: {POCKET_MAX_ATOMS}\n")
        f.write(f"MODEL_D_MODEL: {MODEL_D_MODEL}\n")
        f.write(f"MODEL_D_ATTN: {MODEL_D_ATTN}\n")
        f.write(f"MODEL_N_HEADS: {MODEL_N_HEADS}\n")
        f.write(f"MODEL_DROPOUT: {MODEL_DROPOUT}\n")
        f.write(f"MODEL_USE_2D: {MODEL_USE_2D}\n")
        f.write(f"d_drug_lm: {d_drug_lm}\n")
        f.write(f"d_prot_lm: {d_prot_lm}\n")
        f.write(f"d3_lig: {d3_lig}\n")
        f.write(f"d3_poc: {d3_poc}\n")
        f.write(f"counts: {counts}\n")
        f.write(f"ratios: {ratios}\n")

    # CSV 表头。
    if not csv_file.exists():
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            header = [
                "epoch", "train_loss", "val_loss",
                "train_RMSE", "train_Pearson", "train_Spearman",
                "val_RMSE", "val_Pearson", "val_Spearman",
                "lr",
            ]
            if EVAL_TEST_EACH_EPOCH:
                header.extend(["test_RMSE", "test_Pearson", "test_Spearman"])
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
                logging.warning(f"Non-finite loss at epoch={epoch}, batch={last_bidx}: {float(loss)}")
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
                    logging.error(f"[NaN] 参数异常：epoch={epoch}, batch={last_bidx} before step")
                    optimizer.zero_grad(set_to_none=True)
                    break

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if has_nan_params(model):
                    logging.error(f"[NaN] 参数异常：epoch={epoch}, batch={last_bidx} after step")
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

        # 评估 train/val。
        train_rmse, train_p, train_s, _ = evaluate(model, train_loader, DEVICE, criterion)
        val_rmse, val_p, val_s, val_loss = evaluate(model, val_loader, DEVICE, criterion)

        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Metrics/train_RMSE", train_rmse, epoch)
        writer.add_scalar("Metrics/train_Pearson", train_p, epoch)
        writer.add_scalar("Metrics/train_Spearman", train_s, epoch)
        writer.add_scalar("Metrics/val_RMSE", val_rmse, epoch)
        writer.add_scalar("Metrics/val_Pearson", val_p, epoch)
        writer.add_scalar("Metrics/val_Spearman", val_s, epoch)

        test_rmse = test_p = test_s = None
        if EVAL_TEST_EACH_EPOCH:
            test_rmse, test_p, test_s = evaluate(model, test_loader, DEVICE, criterion=None)
            writer.add_scalar("Metrics/test_RMSE", test_rmse, epoch)
            writer.add_scalar("Metrics/test_Pearson", test_p, epoch)
            writer.add_scalar("Metrics/test_Spearman", test_s, epoch)

        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("LR", current_lr, epoch)

        row = [
            epoch, train_loss, val_loss,
            train_rmse, train_p, train_s,
            val_rmse, val_p, val_s,
            current_lr,
        ]
        if EVAL_TEST_EACH_EPOCH:
            row.extend([test_rmse, test_p, test_s])

        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

        score_now = val_rmse if VAL_AS_BEST else train_rmse
        improved = (best_score - score_now) > EARLY_STOP_MIN_DELTA

        if improved and torch.isfinite(torch.tensor(score_now)):
            best_score = score_now
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_best)
            logging.info(f"[BEST] epoch={epoch + 1}, val_RMSE={best_score:.4f} -> saved: {model_best}")
        else:
            epochs_no_improve += 1
            logging.info(f"[EARLY-STOP] no improve for {epochs_no_improve} epoch(s); best_score={best_score:.4f}")
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                logging.info(f"[EARLY-STOP] patience={EARLY_STOP_PATIENCE} reached at epoch {epoch + 1}, stop training.")
                util.save_checkpoint(epoch + 1, model, optimizer, filename=str(ckpt_path))
                break

        if EVAL_TEST_EACH_EPOCH:
            logging.info(
                f"Epoch {epoch + 1:04d} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"train(RMSE={train_rmse:.4f}, P={train_p:.4f}, S={train_s:.4f}) | "
                f"val(RMSE={val_rmse:.4f}, P={val_p:.4f}, S={val_s:.4f}) | "
                f"test(RMSE={test_rmse:.4f}, P={test_p:.4f}, S={test_s:.4f}) | "
                f"lr={current_lr:.3e} | time={(time.time() - t0) / 60:.2f} min"
            )
        else:
            logging.info(
                f"Epoch {epoch + 1:04d} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"train(RMSE={train_rmse:.4f}, P={train_p:.4f}, S={train_s:.4f}) | "
                f"val(RMSE={val_rmse:.4f}, P={val_p:.4f}, S={val_s:.4f}) | "
                f"test=not evaluated during training | "
                f"lr={current_lr:.3e} | time={(time.time() - t0) / 60:.2f} min"
            )

        util.save_checkpoint(epoch + 1, model, optimizer, filename=str(ckpt_path))

    # ======================================
    # 最终测试：重新构建同参数 best_model
    # ======================================
    best_model = build_model(
        MultiModalDTA_LM,
        d_drug_lm=d_drug_lm,
        d_prot_lm=d_prot_lm,
        d3_lig=d3_lig,
        d3_poc=d3_poc,
        device=DEVICE,
    )

    if not model_best.exists():
        raise RuntimeError(f"Best model not found: {model_best}")

    best_model.load_state_dict(torch.load(model_best, map_location=DEVICE))
    best_model.eval()

    with torch.no_grad():
        final_rmse, final_p, final_s = evaluate(best_model, test_loader, DEVICE, criterion=None)

    logging.info(
        f"[FINAL] test_RMSE={final_rmse:.4f}, "
        f"test_Pearson={final_p:.4f}, test_Spearman={final_s:.4f}"
    )

    with open(final_txt, "w", encoding="utf-8") as f:
        f.write("Final test was evaluated with the best model selected by validation RMSE.\n")
        f.write(f"test_RMSE: {final_rmse:.4f}\n")
        f.write(f"test_Pearson: {final_p:.4f}\n")
        f.write(f"test_Spearman: {final_s:.4f}\n")

    writer.close()
