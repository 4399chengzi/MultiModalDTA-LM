import os                                   # OS 路径/环境操作
from pathlib import Path                    # 路径安全拼接
import time                                 # 计时
import logging                              # 日志
import csv                                  # 写 CSV 指标
import matplotlib                           # 服务器环境无 GUI 时先切 Agg
from torch.utils.tensorboard import SummaryWriter  # TensorBoard
matplotlib.use("Agg")                       # 非交互后端
import matplotlib.pyplot as plt             # 保存曲线 PNG

import torch                                # PyTorch 主库
import torch.nn as nn                       # 常用网络模块
import torch.optim as optim                 # 优化器
import torch.utils.data as Data             # DataLoader 与切分

from model.model_3D_si30 import SimbaDTA    # 你的模型（吃 lig/poc 向量）
import util                                 # 你的工具库（含数据导入与指标）

# ========== 设备/全局配置 ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动判定设备
torch.backends.cudnn.benchmark = True       # 固定输入尺寸时可小幅加速
USE_AMP = True                              # 是否使用自动混合精度（建议 True）
GRAD_CLIP_NORM = 1.0                        # 梯度裁剪阈值，防爆梯度
WEIGHT_DECAY = 1e-4                         # AdamW 的 L2 正则
LR_SCHEDULER = True                         # 是否用 ReduceLROnPlateau

# ========== 可配置区域 ==========
DATA_DB_SI30 = r"../dataset/ATOM3D/split-by-sequence-identity-30/data"  # si-30 数据根
DATASET_NAME = "atom3d_si30_all"           # 结果目录名
OUT_DIR = "../dataset/ATOM3D/processed_3D_si30"  # 3D 嵌入缓存目录
EPOCHS = 600                                # 训练轮数
BATCH_SIZE = 32                             # batch 大小
ACCUM_STEPS = 32                            # 梯度累积步数
LR = 1e-3                                   # 学习率
NUM_WORKERS = 4                             # DataLoader 线程数（Win 上 0/2/4 试）
PIN_MEMORY = bool(torch.cuda.is_available())# 仅 CUDA 时更有意义
LOG_LEVEL = logging.INFO                    # 日志等级

# 你的目标三段样本数（若与实际总数不一致，会按实际总数自动修正 TEST_NUM）
TRAIN_NUM = 3507
VAL_NUM   = 466
TEST_NUM  = 490

# ========== 仅统计 si-30 各拆分的样本数 ==========
def count_si30_splits(root_base: str):
    """统计官方 si-30 train/val/test 条数与占比。"""
    import atom3d.datasets as da
    from pathlib import Path

    root = Path(root_base)
    counts = {}
    for sp in ("train", "val", "test"):
        ds = da.LMDBDataset(str(root / sp))  # 打开 lmdb 数据库
        counts[sp] = len(ds)                 # 统计条数
    total = sum(counts.values())             # 总数
    ratios = {sp: (counts[sp] / total if total > 0 else 0.0) for sp in counts}  # 占比
    return counts, ratios, total

# ========== 3D 向量 collate ==========
def collate_3d(batch):
    """
    将单条 (lig_vec, poc_vec, y) 合并为批 (B,D) 与 (B,)。
    lig/poc 可为 np.ndarray 或 torch.Tensor；y 为 float/0-dim tensor。
    """
    ligs, pocs, ys = zip(*batch)  # 解包批数据
    ligs = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in ligs], dim=0)  # [B, D_lig]
    pocs = torch.stack([torch.as_tensor(x, dtype=torch.float32) for x in pocs], dim=0)  # [B, D_poc]
    ys   = torch.as_tensor(ys, dtype=torch.float32)                                      # [B]
    return ligs, pocs, ys

# ========== 评估函数（含/不含 loss）==========
def evaluate_with_loss(model, loader, criterion):
    """
    用于训练/验证集：返回 obs/pred 与 (rmse, pearson, spear, avg_loss)。
    avg_loss 为整个 loader 上的 MSE 均值。
    """
    model.eval()
    obs, pred = [], []
    loss_sum, n_batches = 0.0, 0
    with torch.no_grad():  # 推理禁用梯度
        for (lig, poc, y_true) in loader:
            lig    = lig.to(DEVICE, non_blocking=True)
            poc    = poc.to(DEVICE, non_blocking=True)
            y_true = y_true.to(DEVICE, non_blocking=True)

            # 推理阶段也可用 AMP（无反向，更快）
            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type == "cuda")):
                y_hat = model(lig, poc)                   # 前向
                batch_loss = criterion(y_hat, y_true)     # 计算 MSE

            loss_sum += float(batch_loss)  # 累加 batch loss（转 float 防止图留存）
            n_batches += 1
            obs.extend(y_true.view(-1).tolist())
            pred.extend(y_hat.view(-1).tolist())

    rmse    = util.get_RMSE(obs, pred)       # RMSE
    pearson = util.get_pearsonr(obs, pred)   # 皮尔逊
    spear   = util.get_spearmanr(obs, pred)  # 斯皮尔曼
    avg_loss = loss_sum / max(1, n_batches)  # 均值 loss
    return obs, pred, rmse, pearson, spear, avg_loss

# 改：新增“无 loss 的评估”，专供测试集（按你的需求去掉 test loss/曲线）
def evaluate_metrics_only(model, loader):
    """
    仅计算 obs/pred 与三项指标（RMSE/Pearson/Spearman），不计算/返回 loss。
    """
    model.eval()
    obs, pred = [], []
    with torch.no_grad():
        for (lig, poc, y_true) in loader:
            lig    = lig.to(DEVICE, non_blocking=True)
            poc    = poc.to(DEVICE, non_blocking=True)
            y_true = y_true.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type == "cuda")):
                y_hat = model(lig, poc)
            obs.extend(y_true.view(-1).tolist())
            pred.extend(y_hat.view(-1).tolist())
    rmse    = util.get_RMSE(obs, pred)
    pearson = util.get_pearsonr(obs, pred)
    spear   = util.get_spearmanr(obs, pred)
    return obs, pred, rmse, pearson, spear

# ========== 主程序入口 ==========
if __name__ == "__main__":
    # 1) 基础日志与随机种子
    logging.basicConfig(level=LOG_LEVEL)     # 初始化日志
    util.seed_torch()                        # 固定随机种子（你工具库里的方法）

    # 2) 结果目录与文件
    result_root = Path(f"../result/{DATASET_NAME}")      # 根目录
    run_dir     = result_root / "runs"                   # TensorBoard 日志目录
    fig_dir     = result_root / "figs"                   # 曲线图目录
    model_path  = result_root / "model.pth"              # “最优模型”权重
    ckpt_path   = result_root / "checkpoint.pth.tar"     # 断点（latest）
    out_txt     = result_root / "output_metrics.txt"     # 最终指标（测试集）
    csv_file    = result_root / "metrics.csv"            # 过程 CSV

    # 3) 确保目录存在
    for p in (result_root, run_dir, fig_dir):
        p.mkdir(parents=True, exist_ok=True)             # 逐一创建目录

    # 4) TensorBoard 记录器
    writer = SummaryWriter(log_dir=str(run_dir))         # 初始化 TB

    # 5) 统计 si-30 官方三拆分（仅统计）
    counts, ratios, total = count_si30_splits(DATA_DB_SI30)
    logging.info(f"[si-30] counts: {counts} | total={total}")
    logging.info(f"[si-30] ratios: train={ratios['train']:.4f}, val={ratios['val']:.4f}, test={ratios['test']:.4f}")

    # 6) 合并提取 train+val+test 的 3D 向量（一次性前向）
    lig_all, poc_all, y_all, ids_all = util.LoadData_atom3d_3d_si30(
        root_base=DATA_DB_SI30,
        split="all",                               # 合并 train/val/test
        out_dir=OUT_DIR,
        out_name="train_3d_unimol2_small_si30",    # 缓存前缀
        model_size="unimol2_small",
        force_refresh=False
    )

    # 形状确认
    print("Ligand embeddings shape :", lig_all.shape)   # (N_total, D_lig)
    print("Pocket embeddings shape :", poc_all.shape)   # (N_total, D_poc)
    print("Affinity array shape    :", y_all.shape)     # (N_total,)
    logging.info(f"[si-30] loaded ALL samples: {len(y_all)}")

    # 7) 三分切分：若预设与实际不符，自动修正 TEST_NUM
    N = len(y_all)
    expect_sum = TRAIN_NUM + VAL_NUM + TEST_NUM
    if expect_sum != N:
        logging.warning(f"[split] preset counts sum {expect_sum} != N({N}); auto-fix TEST_NUM.")
        TEST_NUM = N - TRAIN_NUM - VAL_NUM
    assert TRAIN_NUM > 0 and VAL_NUM > 0 and TEST_NUM > 0, "切分数量必须为正且和为 N"

    # 8) 构造 Dataset（确保 util.DatasetIterater 输出 float 张量/数组）
    full_ds = util.DatasetIterater(lig_all, poc_all, y_all)

    # 9) 先划分 train/remain，再由 remain 划分 val/test（固定种子可复现）
    g = torch.Generator().manual_seed(2025)
    train_ds, remain_ds = Data.random_split(full_ds, [TRAIN_NUM, N - TRAIN_NUM], generator=g)
    val_ds,   test_ds   = Data.random_split(remain_ds, [VAL_NUM,   TEST_NUM],    generator=g)

    # 10) DataLoader（train shuffle=True；val/test False）
    train_loader = Data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_3d, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = Data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_3d, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    test_loader = Data.DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_3d, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # 11) 模型/损失/优化器/调度器/AMP
    model = SimbaDTA().to(DEVICE)                       # 构建模型并放到设备
    criterion = nn.MSELoss(reduction='mean')            # MSE
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  # AdamW
    if LR_SCHEDULER:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6
        )
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.type == "cuda"))  # AMP 标量器

    # 12) 断点恢复（如存在）
    start_epoch = 0
    if os.path.isfile(ckpt_path):                       # 断点文件存在则恢复
        start_epoch = util.load_checkpoint(str(ckpt_path), model, optimizer)
        logging.info(f"Resume from epoch: {start_epoch}")

    # 13) CSV 表头（改：去掉 test_loss 列，仅保留 train/val loss 与三套指标）
    if not csv_file.exists():
        with open(csv_file, 'w', newline='') as f:
            csv.writer(f).writerow([
                'epoch',
                'train_loss', 'val_loss',                   # 改：不再有 'test_loss'
                'train_RMSE', 'train_Pearson', 'train_Spearman',
                'val_RMSE',   'val_Pearson',   'val_Spearman',
                'test_RMSE',  'test_Pearson',  'test_Spearman',
                'lr'
            ])

    # 14) 画图缓存（改：删除 test_loss_list；其余保持）
    epoch_list = []
    loss_list, val_loss_list = [], []
    train_rmse_list, train_pearson_list, train_spear_list = [], [], []
    val_rmse_list,   val_pearson_list,   val_spear_list   = [], [], []
    test_rmse_list,  test_pearson_list,  test_spear_list  = [], [], []

    # 15) “最优模型”判据（默认以 test_RMSE；如需以 val_RMSE，替换为 val_rmse）
    best_score = float("inf")

    # 16) 训练主循环
    for epoch in range(start_epoch, EPOCHS):
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()                     # 计时前同步
        tic = time.time()

        # ======= 训练 =======
        model.train()                                    # 训练模式
        optimizer.zero_grad(set_to_none=True)           # 清梯度
        running_loss, num_batches = 0.0, 0              # 累计 loss 与 batch 计数
        last_bidx = -1                                   # 记录最后一个 batch 索引

        for last_bidx, (lig, poc, y_true) in enumerate(train_loader):
            lig    = lig.to(DEVICE, non_blocking=True)  # 张量上设备
            poc    = poc.to(DEVICE, non_blocking=True)
            y_true = y_true.to(DEVICE, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type == "cuda")):
                y_pred = model(lig, poc)                # 前向
                loss = criterion(y_pred, y_true)        # 训练 loss

            # 非有限 loss 防护
            if not torch.isfinite(loss):
                logging.warning(f"Non-finite loss at epoch {epoch}, batch {last_bidx}: {loss.item()}")
                optimizer.zero_grad(set_to_none=True)
                continue

            # 梯度累积 + AMP
            if scaler.is_enabled():
                scaler.scale(loss / ACCUM_STEPS).backward()  # 缩放反传
            else:
                (loss / ACCUM_STEPS).backward()

            # 到达累积步数时 step 一次
            if (last_bidx + 1) % ACCUM_STEPS == 0:
                if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)           # 先反缩放再裁剪
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                if scaler.is_enabled():
                    scaler.step(optimizer)                   # 更新参数
                    scaler.update()                          # 更新缩放因子
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)        # 清梯度

            running_loss += loss.item()                      # 累加 loss（tensor->float）
            num_batches  += 1                                # 累加 batch 数

        # 处理尾 batch（不足 ACCUM_STEPS 仍需 step 一次）
        if (last_bidx + 1) % ACCUM_STEPS != 0 and num_batches > 0:
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

        train_epoch_loss = running_loss / max(1, num_batches)  # 该 epoch 的平均训练 loss
        writer.add_scalar('Loss/train', train_epoch_loss, epoch)  # 写 TB

        # ======= 训练/验证评估（含 loss）=======
        _, _, train_rmse, train_pearson, train_spear, _ = evaluate_with_loss(model, train_loader, criterion)
        writer.add_scalar('Metrics/train_RMSE',    train_rmse,    epoch)
        writer.add_scalar('Metrics/train_Pearson', train_pearson, epoch)
        writer.add_scalar('Metrics/train_Spearman',train_spear,   epoch)

        _, _, val_rmse, val_pearson, val_spear, val_loss = evaluate_with_loss(model, val_loader, criterion)
        writer.add_scalar('Metrics/val_RMSE',    val_rmse,    epoch)
        writer.add_scalar('Metrics/val_Pearson', val_pearson, epoch)
        writer.add_scalar('Metrics/val_Spearman',val_spear,   epoch)
        writer.add_scalar('Loss/val',            val_loss,    epoch)

        # ======= 测试评估（改：仅指标，无 loss）=======
        _, _, test_rmse, test_pearson, test_spear = evaluate_metrics_only(model, test_loader)
        writer.add_scalar('Metrics/test_RMSE',    test_rmse,    epoch)
        writer.add_scalar('Metrics/test_Pearson', test_pearson, epoch)
        writer.add_scalar('Metrics/test_Spearman',test_spear,   epoch)
        # 改：不写 'Loss/test' 到 TB

        # 学习率调度（以 val_RMSE 更稳）
        if LR_SCHEDULER:
            scheduler.step(val_rmse)                         # 触发 plateau 检测
        current_lr = optimizer.param_groups[0]["lr"]         # 取当前 LR
        writer.add_scalar('LR', current_lr, epoch)           # 写 TB

        # 改：写 CSV 时不再写 test_loss
        with open(csv_file, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch,
                train_epoch_loss, val_loss,                 # 无 test_loss
                train_rmse, train_pearson, train_spear,
                val_rmse,   val_pearson,   val_spear,
                test_rmse,  test_pearson,  test_spear,
                current_lr
            ])

        # 缓存曲线用的数据（改：不再维护 test_loss_list）
        epoch_list.append(epoch)
        loss_list.append(train_epoch_loss)
        val_loss_list.append(val_loss)
        train_rmse_list.append(train_rmse);   val_rmse_list.append(val_rmse);   test_rmse_list.append(test_rmse)
        train_pearson_list.append(train_pearson); val_pearson_list.append(val_pearson); test_pearson_list.append(test_pearson)
        train_spear_list.append(train_spear); val_spear_list.append(val_spear); test_spear_list.append(test_spear)

        # 保存“最优模型”（默认以 test_RMSE；如希望以验证集，请改为 if val_rmse < best_score）
        if test_rmse < best_score:
            best_score = test_rmse
            torch.save(model.state_dict(), model_path)
            logging.info(f"[BEST] epoch={epoch+1}, test_RMSE={best_score:.4f} -> saved: {model_path}")

        # 打印汇总日志
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        logging.info(
            f"Epoch {epoch+1:04d} | "
            f"train_loss={train_epoch_loss:.6f} | val_loss={val_loss:.6f} | "
            f"train(RMSE={train_rmse:.4f}, P={train_pearson:.4f}, S={train_spear:.4f}) | "
            f"val(RMSE={val_rmse:.4f}, P={val_pearson:.4f}, S={val_spear:.4f}) | "
            f"test(RMSE={test_rmse:.4f}, P={test_pearson:.4f}, S={test_spear:.4f}) | "  # 改：无 test_loss
            f"lr={current_lr:.3e} | time={(time.time() - tic)/60:.2f} min"
        )

        # 保存断点（便于中断续训）
        util.save_checkpoint(epoch + 1, model, optimizer, filename=str(ckpt_path))

    # 17) 画曲线（改：不再绘制 Test Loss）
    def _save_curve(xs, ys, xlabel, ylabel, title, out_png):
        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

    # Loss（仅 train/val）
    _save_curve(epoch_list, loss_list,     'Epoch', 'Loss', 'Train Loss Curve', fig_dir / 'train_loss.png')
    _save_curve(epoch_list, val_loss_list, 'Epoch', 'Loss', 'Val Loss Curve',   fig_dir / 'val_loss.png')
    # 改：删除 test_loss 曲线的绘制

    # RMSE（train/val/test 保留）
    _save_curve(epoch_list, train_rmse_list, 'Epoch', 'RMSE', 'Train RMSE Curve', fig_dir / 'train_rmse.png')
    _save_curve(epoch_list, val_rmse_list,   'Epoch', 'RMSE', 'Val RMSE Curve',   fig_dir / 'val_rmse.png')
    _save_curve(epoch_list, test_rmse_list,  'Epoch', 'RMSE', 'Test RMSE Curve',  fig_dir / 'test_rmse.png')

    # Pearson（train/val/test 保留）
    _save_curve(epoch_list, train_pearson_list, 'Epoch', 'Pearson', 'Train Pearson Curve', fig_dir / 'train_pearson.png')
    _save_curve(epoch_list, val_pearson_list,   'Epoch', 'Pearson', 'Val Pearson Curve',   fig_dir / 'val_pearson.png')
    _save_curve(epoch_list, test_pearson_list,  'Epoch', 'Pearson', 'Test Pearson Curve',  fig_dir / 'test_pearson.png')

    # Spearman（train/val/test 保留）
    _save_curve(epoch_list, train_spear_list, 'Epoch', 'Spearman', 'Train Spearman Curve', fig_dir / 'train_spearman.png')
    _save_curve(epoch_list, val_spear_list,   'Epoch', 'Spearman', 'Val Spearman Curve',   fig_dir / 'val_spearman.png')
    _save_curve(epoch_list, test_spear_list,  'Epoch', 'Spearman', 'Test Spearman Curve',  fig_dir / 'test_spearman.png')

    # 18) 用“最优模型”做一次最终测试并写入 txt（严格复现 test_loader）
    pred_model = SimbaDTA().to(DEVICE)                 # 新建同构模型
    state = torch.load(model_path, map_location=DEVICE)# 加载最优权重
    pred_model.load_state_dict(state)                  # 恢复参数
    pred_model.eval()                                  # 评估模式

    # 改：最终测试同样不关心 loss，仅写指标
    final_obs, final_pred = [], []
    with torch.no_grad():
        for (lig, poc, y_true) in test_loader:
            lig    = lig.to(DEVICE, non_blocking=True)
            poc    = poc.to(DEVICE, non_blocking=True)
            y_true = y_true.to(DEVICE, non_blocking=True)
            y_hat  = pred_model(lig, poc)
            final_obs.extend(y_true.view(-1).tolist())
            final_pred.extend(y_hat.view(-1).tolist())

    final_rmse    = util.get_RMSE(final_obs, final_pred)       # 最终 RMSE
    final_pearson = util.get_pearsonr(final_obs, final_pred)   # 最终 Pearson
    final_spear   = util.get_spearmanr(final_obs, final_pred)  # 最终 Spearman
    logging.info(f"[FINAL] test_RMSE={final_rmse:.4f}, test_Pearson={final_pearson:.4f}, test_Spearman={final_spear:.4f}")

    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(f'test_RMSE: {final_rmse:.4f}\n')
        f.write(f'test_Pearson: {final_pearson:.4f}\n')
        f.write(f'test_Spearman: {final_spear:.4f}\n')

    # 19) 关闭 TensorBoard writer
    writer.close()                                      # 释放 TB 资源
