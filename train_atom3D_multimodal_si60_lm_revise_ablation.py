# ======================================
# 多模态 ATOM3D(SI-60) 批量消融训练脚本（LM 版）
# 当前实际模态：1D(LM) + 3D(Uni-Mol2)，2D 图模态关闭
#
# 功能：
#   1) 一次性顺序跑完当前 1D+3D 场景相关的全部消融实验
#   2) 每个消融模式单独保存 best model / checkpoint / csv / final metrics
#   3) 最终把所有消融结果汇总到一张总表
#
# 依赖：
#   - 模型文件：model/model_multimodal_lm_revise_ablation.py
# ======================================

from pathlib import Path                 # 行：路径处理
import logging                           # 行：日志
import time                              # 行：计时
import csv                               # 行：写入 CSV
import json                              # 行：保存 json 配置
from typing import Dict, Any, List       # 行：类型注解

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.tensorboard import SummaryWriter

import util
from util import LoadData_atom3d_si60_multimodal_lm  # 行：SI-60 的 LM 数据加载函数

# —— 关闭 Uni-Mol 冗余日志，避免控制台刷屏 —— #
lg = logging.getLogger("unimol_tools")   # 行：获取 unimol_tools 日志器
lg.setLevel(logging.WARNING)             # 行：只保留 warning 及以上
lg.propagate = False                     # 行：不向上层 logger 传播
for h in list(lg.handlers):              # 行：清空已有 handler，避免重复打印
    lg.removeHandler(h)

# ========== 设备/日志/随机种子 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 行：自动选 GPU/CPU
torch.backends.cudnn.benchmark = True     # 行：固定输入尺寸时可略微加速
LOG_LEVEL = logging.INFO                  # 行：日志级别
SANITY_CHECK = True                       # 行：开始前做一次 batch 检查

# ========== revise / ablation 版本标记 ==========
REV_TAG = "revise_ablation"                                # 行：当前修订+消融统一后缀
MODEL_MODULE_NAME = "model_multimodal_lm_revise_ablation" # 行：当前模型文件名（不带 .py）
RESULT_ROOT_NAME = "result_revise"                        # 行：结果总根目录
DATASET_NAME = f"atom3d_si60_mm_lm_{REV_TAG}"             # 行：当前任务目录名

# ========== 当前脚本可运行的全部相关消融 ==========
# 注意：
# 1) 当前 SI-60 训练脚本没有 2D 图输入，因此不跑 2D_only / 2d_3d_* / 1d_2d_3d_*。
# 2) “1d_3d_gated” 就是当前 1D+3D 场景下的完整模型。
ABLATION_MODES: List[str] = [
    "1d_only",
    "3d_only",
    "1d_3d_concat",
    "1d_3d_add",
    "1d_3d_gated",
    "no_3d_interactions",
    "gate_q_kv_only",
    "gate_q_kv_prod",
    "gate_q_kv_abs",
    "no_fusion_ffn",
    "no_fusion_residual",
]

# 行：如果你还想把 “full” 单独再跑一遍，可以改成 True
RUN_REDUNDANT_FULL = False

# ========== 可配置区域 ==========
DATA_DB_SI60 = r"../dataset/ATOM3D/split-by-sequence-identity-60/data"  # 行：SI-60 LMDB 根目录
OUT_DIR = f"../{RESULT_ROOT_NAME}/{DATASET_NAME}"                        # 行：结果根目录

BATCH_SIZE   = 16                  # 行：batch 大小
NUM_WORKERS  = 4                   # 行：DataLoader 线程数
PIN_MEMORY   = bool(torch.cuda.is_available())  # 行：GPU 下开 pin_memory

EPOCHS         = 600               # 行：最大训练轮次
LR             = 1e-4              # 行：学习率
WEIGHT_DECAY   = 1e-4              # 行：AdamW 的 L2 正则
USE_AMP        = False             # 行：关闭 AMP，优先保证数值稳定
ACCUM_STEPS    = 4                 # 行：梯度累积步数
GRAD_CLIP_NORM = 0.5               # 行：梯度裁剪阈值
USE_SCHEDULER  = True              # 行：是否启用学习率调度
VAL_AS_BEST    = True              # 行：用验证集 RMSE 选最优模型

# —— 早停机制参数 —— #
EARLY_STOP_PATIENCE   = 60         # 行：若连续 60 个 epoch 没提升，则早停
EARLY_STOP_MIN_DELTA  = 1e-4       # 行：最小提升阈值

# ========== 统计 si-60 拆分 ==========
def count_si60_splits(root_base: str):
    """读取 train/val/test 的样本数和占比，仅做信息打印"""
    import atom3d.datasets as da
    root = Path(root_base)
    counts = {}
    for sp in ("train", "val", "test"):
        ds = da.LMDBDataset(str(root / sp))
        counts[sp] = len(ds)
    total = sum(counts.values())
    ratios = {sp: (counts[sp] / total if total > 0 else 0.0) for sp in counts}
    return counts, ratios, total


# ========== 多模态 Dataset（带 LM 特征） ==========
class MultiModalDatasetLM(Data.Dataset):
    """封装 multi-modal + LM 特征；当前脚本只实际使用 1D+3D。"""

    def __init__(self, pkg: Dict[str, Any]):
        self.ids     = pkg['ids']        # 行：样本 id
        self.y       = pkg['y']          # 行：标签 pKd
        self.smiles  = pkg['smiles']     # 行：SMILES，便于排错
        self.seq     = pkg['seq']        # 行：蛋白序列，便于排错
        self.drug_lm = pkg['drug_lm']    # 行：[N, D_drug_lm]
        self.prot_lm = pkg['prot_lm']    # 行：[N, D_prot_lm]

        # —— 当前脚本中 2D 图模态关闭，保留占位接口 —— #
        self.g_lig   = pkg.get('g_lig', None)   # 行：药物图，当前通常为 None
        self.g_prot  = pkg.get('g_prot', None)  # 行：蛋白图，当前通常为 None

        self.lig3d   = pkg['lig_3d']     # 行：配体 3D 向量 [N, 512]
        self.poc3d   = pkg['poc_3d']     # 行：口袋 3D 向量 [N, 512]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            'id'     : self.ids[idx],
            'y'      : float(self.y[idx]),
            'smiles' : self.smiles[idx],
            'seq'    : self.seq[idx],
            'drug_lm': self.drug_lm[idx],
            'prot_lm': self.prot_lm[idx],
            'g_lig'  : None,             # 行：当前脚本中 2D 模态关闭
            'g_prot' : None,             # 行：当前脚本中 2D 模态关闭
            'lig3d'  : self.lig3d[idx],
            'poc3d'  : self.poc3d[idx],
        }


# ========== collate：合并 batch ==========
def multimodal_collate_lm(batch: list) -> Dict[str, Any]:
    """
    合并一个 batch：
      - 1D / 3D：numpy 行堆叠 -> torch.float32
      - 2D 图：当前脚本中已关闭，返回 None
    """
    ids    = [b['id']     for b in batch]
    smiles = [b['smiles'] for b in batch]
    seqs   = [b['seq']    for b in batch]

    g_lig_batch  = None
    g_prot_batch = None

    import numpy as np
    drug_lm = torch.as_tensor(
        np.stack([b['drug_lm'] for b in batch], axis=0),
        dtype=torch.float32
    )
    prot_lm = torch.as_tensor(
        np.stack([b['prot_lm'] for b in batch], axis=0),
        dtype=torch.float32
    )
    lig3d = torch.as_tensor(
        np.stack([b['lig3d'] for b in batch], axis=0),
        dtype=torch.float32
    )
    poc3d = torch.as_tensor(
        np.stack([b['poc3d'] for b in batch], axis=0),
        dtype=torch.float32
    )
    y = torch.as_tensor(
        [b['y'] for b in batch],
        dtype=torch.float32
    )

    return {
        'ids'    : ids,
        'smiles' : smiles,
        'seqs'   : seqs,
        'drug_lm': drug_lm,
        'prot_lm': prot_lm,
        'g_lig'  : g_lig_batch,
        'g_prot' : g_prot_batch,
        'lig3d'  : lig3d,
        'poc3d'  : poc3d,
        'y'      : y,
    }


# ========== NaN/Inf 参数检测 ==========
def has_nan_params(model: nn.Module) -> bool:
    """扫描模型参数里是否出现 NaN/Inf。"""
    for p in model.parameters():
        if p is not None and torch.is_tensor(p):
            if torch.isnan(p).any() or torch.isinf(p).any():
                return True
    return False


# ========== 统一评估函数 ==========
@torch.no_grad()
def evaluate(model: torch.nn.Module,
             loader: Data.DataLoader,
             device: torch.device,
             criterion: nn.Module = None):
    """
    若给定 criterion：返回 (rmse, pearson, spearman, avg_loss)
    否则：返回 (rmse, pearson, spearman)
    """
    model.eval()
    obs, pred, loss_sum, n_batches = [], [], 0.0, 0

    for b in loader:
        drug_lm = b['drug_lm'].to(device, non_blocking=True)
        prot_lm = b['prot_lm'].to(device, non_blocking=True)
        g_lig, g_prot = b['g_lig'], b['g_prot']
        lig3d = b['lig3d'].to(device, non_blocking=True)
        poc3d = b['poc3d'].to(device, non_blocking=True)
        y_true = b['y'].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=False):   # 行：明确关闭 AMP
            y_hat = model(
                drug_lm=drug_lm, prot_lm=prot_lm,
                g_lig=g_lig, g_prot=g_prot,
                lig3d=lig3d, poc3d=poc3d
            )
            if criterion is not None:
                loss_val = criterion(y_hat, y_true)
                if torch.isfinite(loss_val):
                    loss_sum += float(loss_val)
                    n_batches += 1

        obs.extend(y_true.view(-1).tolist())
        pred.extend(y_hat.view(-1).tolist())

    rmse    = util.get_RMSE(obs, pred)
    pearson = util.get_pearsonr(obs, pred)
    spear   = util.get_spearmanr(obs, pred)

    if criterion is None:
        return rmse, pearson, spear
    return rmse, pearson, spear, (loss_sum / max(1, n_batches))


# ========== 保存 json 配置 ==========
def dump_json(obj: Dict[str, Any], path: Path):
    """把 dict 保存成 json 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ========== 初始化总汇总表 ==========
def init_summary_csv(path: Path):
    """若总汇总表不存在，则创建表头。"""
    if not path.exists():
        with open(path, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                'ablation_mode',
                'best_select_metric',
                'best_select_value',
                'final_test_RMSE',
                'final_test_Pearson',
                'final_test_Spearman',
                'result_dir'
            ])


# ========== 单个消融模式训练 ==========
def run_one_ablation(
    ablation_mode: str,
    d_drug_lm: int,
    d_prot_lm: int,
    train_loader: Data.DataLoader,
    val_loader: Data.DataLoader,
    test_loader: Data.DataLoader,
    result_root: Path,
    summary_csv: Path,
):
    """
    训练一个消融模式：
      1) 单独创建目录
      2) 单独保存权重、checkpoint、csv、final metrics
      3) 最终把结果追加到总汇总表
    """
    util.seed_torch()  # 行：每个消融模式都固定同一随机种子，尽量公平

    safe_mode = ablation_mode.lower()
    mode_root = result_root / safe_mode
    run_dir    = mode_root / f"runs_{REV_TAG}"
    result_dir = mode_root / f"train_logs_{REV_TAG}"
    model_best = result_dir / f"best_model_{safe_mode}.pth"
    ckpt_path  = result_dir / f"checkpoint_{safe_mode}.pth.tar"
    csv_file   = result_dir / f"metrics_{safe_mode}.csv"
    final_txt  = result_dir / f"final_test_metrics_{safe_mode}.txt"
    config_json = result_dir / f"run_config_{safe_mode}.json"

    run_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir))

    # —— 导入消融版模型 —— #
    from model.model_multimodal_lm_revise_ablation import MultiModalDTA_LM

    model = MultiModalDTA_LM(
        d_drug_lm=d_drug_lm,
        d_prot_lm=d_prot_lm,
        drug_node_dim=70,
        drug_edge_dim=6,
        prot_node_dim=33,
        prot_edge_dim=3,
        gcn_hidden=128,
        gcn_out=128,
        d3_lig=512,
        d3_poc=512,
        d_model=256,
        d_attn=256,
        n_heads=4,
        add_interactions_3d=True,
        ablation_mode=safe_mode,
    ).to(DEVICE)

    # 保存当前模型的实际配置
    model_config = model.get_ablation_config()
    dump_json({
        "REV_TAG": REV_TAG,
        "MODEL_MODULE_NAME": MODEL_MODULE_NAME,
        "ablation_mode": safe_mode,
        "model_config": model_config,
        "OUT_DIR": str(mode_root),
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "USE_AMP": USE_AMP,
        "ACCUM_STEPS": ACCUM_STEPS,
        "GRAD_CLIP_NORM": GRAD_CLIP_NORM,
        "USE_SCHEDULER": USE_SCHEDULER,
        "VAL_AS_BEST": VAL_AS_BEST,
        "EARLY_STOP_PATIENCE": EARLY_STOP_PATIENCE,
        "EARLY_STOP_MIN_DELTA": EARLY_STOP_MIN_DELTA,
    }, config_json)

    logging.info(f"[{safe_mode}] config = {model_config}")

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = (
        optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-5
        )
        if USE_SCHEDULER else None
    )

    # 行：这里显式禁用 AMP，对应当前 SI-60 数值稳定版设定
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    # 断点恢复
    start_epoch = 0
    if ckpt_path.exists():
        start_epoch = util.load_checkpoint(str(ckpt_path), model, optimizer)
        logging.info(f"[{safe_mode}] resume from epoch: {start_epoch}")

    # CSV 表头
    if not csv_file.exists():
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                'epoch', 'train_loss', 'val_loss',
                'train_RMSE','train_Pearson','train_Spearman',
                'val_RMSE','val_Pearson','val_Spearman',
                'test_RMSE','test_Pearson','test_Spearman',
                'lr'
            ])

    # ========== 训练主循环 ==========
    best_score = float('inf')
    epochs_no_improve = 0
    t0 = time.time()

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss, n_batches, last_bidx = 0.0, 0, -1

        for last_bidx, b in enumerate(train_loader):
            drug_lm = b['drug_lm'].to(DEVICE, non_blocking=True)
            prot_lm = b['prot_lm'].to(DEVICE, non_blocking=True)
            g_lig, g_prot = b['g_lig'], b['g_prot']
            lig3d = b['lig3d'].to(DEVICE, non_blocking=True)
            poc3d = b['poc3d'].to(DEVICE, non_blocking=True)
            y_true = b['y'].to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=False):
                y_hat = model(
                    drug_lm=drug_lm, prot_lm=prot_lm,
                    g_lig=g_lig, g_prot=g_prot,
                    lig3d=lig3d, poc3d=poc3d
                )
                loss = criterion(y_hat, y_true)

            # NaN / Inf 保护
            if not torch.isfinite(loss):
                logging.warning(
                    f"[{safe_mode}] non-finite loss at epoch={epoch}, batch={last_bidx}: {float(loss)}"
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            # 梯度累积
            (loss / ACCUM_STEPS).backward()

            if (last_bidx + 1) % ACCUM_STEPS == 0:
                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)

                if has_nan_params(model):
                    logging.error(
                        f"[{safe_mode}] NaN detected in params at epoch={epoch}, batch={last_bidx} (before step)"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    break

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if has_nan_params(model):
                    logging.error(
                        f"[{safe_mode}] NaN detected in params at epoch={epoch}, batch={last_bidx} (after step)"
                    )
                    optimizer.zero_grad(set_to_none=True)
                    break

            running_loss += float(loss)
            n_batches += 1

        # 尾批
        if (last_bidx + 1) % ACCUM_STEPS != 0 and n_batches > 0 and not has_nan_params(model):
            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss = running_loss / max(1, n_batches)
        writer.add_scalar('Loss/train', train_loss, epoch)

        train_rmse, train_p, train_s, _ = evaluate(model, train_loader, DEVICE, criterion)
        val_rmse, val_p, val_s, val_loss = evaluate(model, val_loader, DEVICE, criterion)
        test_rmse, test_p, test_s = evaluate(model, test_loader, DEVICE, criterion=None)

        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/train_RMSE', train_rmse, epoch)
        writer.add_scalar('Metrics/train_Pearson', train_p, epoch)
        writer.add_scalar('Metrics/train_Spearman', train_s, epoch)
        writer.add_scalar('Metrics/val_RMSE', val_rmse, epoch)
        writer.add_scalar('Metrics/val_Pearson', val_p, epoch)
        writer.add_scalar('Metrics/val_Spearman', val_s, epoch)
        writer.add_scalar('Metrics/test_RMSE', test_rmse, epoch)
        writer.add_scalar('Metrics/test_Pearson', test_p, epoch)
        writer.add_scalar('Metrics/test_Spearman', test_s, epoch)

        # 学习率调度
        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR', current_lr, epoch)

        # 写每轮 CSV
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                epoch, train_loss, val_loss,
                train_rmse, train_p, train_s,
                val_rmse,   val_p,   val_s,
                test_rmse,  test_p,  test_s,
                current_lr
            ])

        # 用 val_RMSE 选最优模型
        score_now = val_rmse if VAL_AS_BEST else test_rmse
        improved = (best_score - score_now) > EARLY_STOP_MIN_DELTA

        if torch.isfinite(torch.tensor(score_now)) and (improved or best_score == float('inf')):
            best_score = score_now
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_best)
            logging.info(
                f"[{safe_mode}] BEST epoch={epoch+1}, "
                f"{'val' if VAL_AS_BEST else 'test'}_RMSE={best_score:.4f} "
                f"-> saved: {model_best}"
            )
        else:
            epochs_no_improve += 1
            logging.info(
                f"[{safe_mode}] no improve for {epochs_no_improve} epoch(s); best_score={best_score:.4f}"
            )
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                logging.info(
                    f"[{safe_mode}] early stop at epoch {epoch+1} (patience={EARLY_STOP_PATIENCE})"
                )
                util.save_checkpoint(epoch + 1, model, optimizer, filename=str(ckpt_path))
                break

        logging.info(
            f"[{safe_mode}] Epoch {epoch+1:04d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"train(RMSE={train_rmse:.4f}, P={train_p:.4f}, S={train_s:.4f}) | "
            f"val(RMSE={val_rmse:.4f}, P={val_p:.4f}, S={val_s:.4f}) | "
            f"test(RMSE={test_rmse:.4f}, P={test_p:.4f}, S={test_s:.4f}) | "
            f"lr={current_lr:.3e} | time={(time.time()-t0)/60:.2f} min"
        )

        util.save_checkpoint(epoch + 1, model, optimizer, filename=str(ckpt_path))

    # ========== 用最优模型做最终测试 ==========
    from model.model_multimodal_lm_revise_ablation import MultiModalDTA_LM

    best_model = MultiModalDTA_LM(
        d_drug_lm=d_drug_lm,
        d_prot_lm=d_prot_lm,
        drug_node_dim=70,
        drug_edge_dim=6,
        prot_node_dim=33,
        prot_edge_dim=3,
        gcn_hidden=128,
        gcn_out=128,
        d3_lig=512,
        d3_poc=512,
        d_model=256,
        d_attn=256,
        n_heads=4,
        add_interactions_3d=True,
        ablation_mode=safe_mode,
    ).to(DEVICE)

    best_model.load_state_dict(torch.load(model_best, map_location=DEVICE))
    best_model.eval()

    with torch.no_grad():
        final_rmse, final_p, final_s = evaluate(
            best_model, test_loader, DEVICE, criterion=None
        )

    logging.info(
        f"[{safe_mode}] FINAL test_RMSE={final_rmse:.4f}, "
        f"test_Pearson={final_p:.4f}, test_Spearman={final_s:.4f}"
    )

    with open(final_txt, 'w', encoding='utf-8') as f:
        f.write(f"ablation_mode: {safe_mode}\n")
        f.write(f"best_select_metric: {'val_RMSE' if VAL_AS_BEST else 'test_RMSE'}\n")
        f.write(f"best_select_value: {best_score:.4f}\n")
        f.write(f"test_RMSE: {final_rmse:.4f}\n")
        f.write(f"test_Pearson: {final_p:.4f}\n")
        f.write(f"test_Spearman: {final_s:.4f}\n")

    # 追加到总汇总表
    with open(summary_csv, 'a', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow([
            safe_mode,
            'val_RMSE' if VAL_AS_BEST else 'test_RMSE',
            f"{best_score:.6f}",
            f"{final_rmse:.6f}",
            f"{final_p:.6f}",
            f"{final_s:.6f}",
            str(result_dir),
        ])

    writer.close()

    # 释放显存
    del model
    del best_model
    del optimizer
    del scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ========== 主程序 ==========
if __name__ == "__main__":
    logging.basicConfig(level=LOG_LEVEL)
    util.seed_torch()

    result_root = Path(OUT_DIR)
    result_root.mkdir(parents=True, exist_ok=True)

    logging.info(f"[Revision Tag] {REV_TAG}")
    logging.info(f"[Model Module] {MODEL_MODULE_NAME}")
    logging.info(f"[Output Dir] {result_root}")

    # 原始 LMDB 统计
    counts, ratios, total = count_si60_splits(DATA_DB_SI60)
    logging.info(f"[si-60] counts(raw): {counts} | total={total}")
    logging.info(
        f"[si-60] ratios(raw): "
        f"train={ratios['train']:.4f}, val={ratios['val']:.4f}, test={ratios['test']:.4f}"
    )

    # —— 加载数据（只加载一次，后面所有消融共用） —— #
    kw_common = dict(
        out_mm="../dataset/ATOM3D/processed_mm_si60_lm",
        unimol2_size="unimol2_small",
        contact_threshold=8.0, dis_min=1.0,
        prot_self_loop=False, bond_bidirectional=True,
        prefer_model=None, force_refresh=False,
        use_cuda_for_unimol=True,
    )

    mm_train = LoadData_atom3d_si60_multimodal_lm(
        root_base=DATA_DB_SI60, split="train", **kw_common
    )["train"]
    mm_val = LoadData_atom3d_si60_multimodal_lm(
        root_base=DATA_DB_SI60, split="val", **kw_common
    )["val"]
    mm_test = LoadData_atom3d_si60_multimodal_lm(
        root_base=DATA_DB_SI60, split="test", **kw_common
    )["test"]

    # LM 维度
    d_drug_lm = mm_train['drug_lm'].shape[1]
    d_prot_lm = mm_train['prot_lm'].shape[1]
    logging.info(f"[LM] drug_lm_dim={d_drug_lm}, prot_lm_dim={d_prot_lm}")

    # —— Dataset / DataLoader —— #
    train_ds = MultiModalDatasetLM(mm_train)
    val_ds   = MultiModalDatasetLM(mm_val)
    test_ds  = MultiModalDatasetLM(mm_test)

    train_loader = Data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=multimodal_collate_lm
    )
    val_loader = Data.DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=multimodal_collate_lm
    )
    test_loader = Data.DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=multimodal_collate_lm
    )

    logging.info(
        f"[si-60|LM] final sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    # —— Sanity Check —— #
    if SANITY_CHECK:
        b = next(iter(train_loader))
        print(">>> Batch keys:", list(b.keys()))
        print(
            "drug_lm:", tuple(b['drug_lm'].shape),
            "| prot_lm:", tuple(b['prot_lm'].shape),
            "| lig3d:", tuple(b['lig3d'].shape),
            "| poc3d:", tuple(b['poc3d'].shape),
            "| y:", tuple(b['y'].shape)
        )
        print("g_lig:", b['g_lig'])
        print("g_prot:", b['g_prot'])
        print("[OK] 1D+3D LM loaders ready.")

    # —— 初始化总汇总表 —— #
    summary_csv = result_root / f"ablation_summary_{REV_TAG}.csv"
    init_summary_csv(summary_csv)

    # —— 组织待跑的消融列表 —— #
    modes_to_run = list(ABLATION_MODES)
    if RUN_REDUNDANT_FULL:
        modes_to_run = ["full"] + modes_to_run

    logging.info(f"[Ablation Modes] {modes_to_run}")

    # —— 逐个跑消融 —— #
    for mode in modes_to_run:
        logging.info("=" * 100)
        logging.info(f"[START ABLATION] {mode}")
        logging.info("=" * 100)

        run_one_ablation(
            ablation_mode=mode,
            d_drug_lm=d_drug_lm,
            d_prot_lm=d_prot_lm,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            result_root=result_root,
            summary_csv=summary_csv,
        )

        logging.info("=" * 100)
        logging.info(f"[END ABLATION] {mode}")
        logging.info("=" * 100)

    logging.info(f"[DONE] all ablations finished. Summary saved to: {summary_csv}")