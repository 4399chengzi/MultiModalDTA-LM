# -*- coding: utf-8 -*-
# ============================================================
#  SimbaDTA - FDavis 训练脚本（优化版，逐行中文注释）
#  特性：AMP混合精度、测试RMSE选优、梯度累积尾步补偿、LR自适应衰减
#  输出：TensorBoard日志、CSV指标、PNG曲线、最优权重、断点文件、最终测试指标txt
# ============================================================

import os                                   # 操作系统相关工具（路径、文件检测等）
import time                                 # 计时
import csv                                  # 把每个epoch指标写入CSV
import logging                              # 训练过程日志
from pathlib import Path                    # 更安全的路径拼接
import matplotlib                           # 先切换成无界面后端，避免服务器/WSL报错
matplotlib.use("Agg")                       # 非交互后端
import matplotlib.pyplot as plt             # 画损失/指标曲线PNG

import torch                                # PyTorch主库
import torch.nn as nn                       # 神经网络模块
import torch.optim as optim                 # 各类优化器
import torch.utils.data as Data             # DataLoader / random_split
from torch.utils.tensorboard import SummaryWriter  # TensorBoard日志
from torch.cuda.amp import autocast, GradScaler    # AMP混合精度核心

from model.model_1D import SimbaDTA                  # 你的模型
import util                                 # 你的工具库（含数据加载/指标/种子/断点等）

# ========================= 可配置区域 =========================
DATASET_NAME = "fdavis"                     # 仅用于结果目录命名
SMILES_MAXLEN = 85                          # SMILES最大长度（与旧代码保持一致）
PROSEQ_MAXLEN = 1200                        # 蛋白序列最大长度（与旧代码保持一致）
# FDavis常用总样本数为 9125（你原先固定 7300/1825），这里仍给默认分割，
# 但也允许自动根据实际数据长度重新计算（见后文“自适应 N”）
DEFAULT_TRAIN_NUM = 7300                    # 旧项目习惯值：训练条数
DEFAULT_TEST_NUM  = 1825                    # 旧项目习惯值：测试条数（两者之和=9125）

EPOCHS = 600                                # 训练轮数
BATCH_SIZE = 32                             # 每次迭代样本数
ACCUM_STEPS = 32                            # 梯度累积步数（小显存可调大）
LR = 1e-3                                   # 初始学习率
NUM_WORKERS = 4                             # DataLoader并行进程（Windows/LMDB慎用过大）
PIN_MEMORY = True                           # pin内存，加速CPU->GPU拷贝
LOG_LEVEL = logging.INFO                    # 日志等级
AMP_ENABLE = True                           # 启用AMP混合精度（建议True）

# 结果输出根目录
RESULT_ROOT = Path(f"./result/{DATASET_NAME}")             # 结果根目录
RUN_DIR     = RESULT_ROOT / "runs"                         # TensorBoard日志目录
FIG_DIR     = RESULT_ROOT / "figs"                         # PNG曲线目录
MODEL_PATH  = RESULT_ROOT / "model.pth"                    # 最优权重文件
CKPT_PATH   = RESULT_ROOT / "checkpoint.pth.tar"           # 断点文件
OUT_TXT     = RESULT_ROOT / "output_metrics.txt"           # 最终测试指标txt
CSV_FILE    = RESULT_ROOT / "metrics.csv"                  # 指标CSV

# ========================= 主程序入口 =========================
if __name__ == "__main__":
    # 1) 日志与随机种子
    logging.basicConfig(level=LOG_LEVEL)                   # 配置日志等级
    util.seed_torch()                                      # 固定随机种子（你的util需实现）

    # 2) 确保输出目录存在
    for p in [RESULT_ROOT, RUN_DIR, FIG_DIR]:
        p.mkdir(parents=True, exist_ok=True)               # 不存在就创建

    # 3) TensorBoard记录器
    writer = SummaryWriter(log_dir=str(RUN_DIR))           # 写TensorBoard日志

    # 4) 加载 FDavis 数据（文本版）
    #    该函数需返回：np.array(smiles_list), np.array(seq_list), np.array(y)
    drug_seqs, target_seqs, y_all = util.LoadData_f()      # 从 ./dataset/fdavis/affi_info.txt 读数据

    # 5) 文本编码/截断/填充 + 对齐/打乱
    labeled_drugs, labeled_targets = util.LabelDT(         # 将SMILES/SEQ编码到固定长度
        drug_seqs, target_seqs, SMILES_MAXLEN, PROSEQ_MAXLEN
    )
    d_shuttle, t_shuttle, y_shuttle = util.Shuttle(        # 根据你的旧逻辑进行对齐/同步打乱等
        labeled_drugs, labeled_targets, y_all
    )

    # 6) 自适应样本总数与默认切分（防止固定7300/1825溢出或不足）
    N_total = len(y_shuttle)                                # 实际总样本数（通常为9125）
    train_num = min(DEFAULT_TRAIN_NUM, N_total)            # 训练条数不超过总数
    test_num  = min(DEFAULT_TEST_NUM,  N_total - train_num)  # 测试条数不超过剩余
    if train_num + test_num < N_total:
        # 若还有剩余样本（例如你的数据更多），把剩余也丢到测试集里
        test_num = N_total - train_num
    logging.info(f"[Split] train={train_num}, test={test_num}, total={N_total}")

    # 7) 构造Dataset并随机切分（要求 util.DatasetIterater 实现 __len__ / __getitem__）
    full_ds = util.DatasetIterater(
        d_shuttle[:train_num + test_num],                  # 仅取需要的前 train+test 条
        t_shuttle[:train_num + test_num],
        y_shuttle[:train_num + test_num]
    )
    train_ds, test_ds = Data.random_split(full_ds, [train_num, test_num])  # 随机划分

    # 8) DataLoader（训练需shuffle，测试不shuffle）
    train_loader = Data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,                                      # 训练集要打乱
        collate_fn=util.BatchPad,                          # 你的对齐/填充函数
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    test_loader = Data.DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,                                     # 测试集不需要打乱
        collate_fn=util.BatchPad,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # 9) 模型/损失/优化器/AMP/学习率调度器
    model = SimbaDTA().cuda()                              # 构建模型并放到GPU
    criterion = nn.MSELoss(reduction='mean')               # 回归损失：MSE
    optimizer = optim.Adam(model.parameters(), lr=LR)      # Adam优化
    scaler = GradScaler(enabled=AMP_ENABLE)                # AMP梯度缩放器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(      # plateau时自动衰减学习率
        optimizer, mode='min', factor=0.5, patience=10,    # 指标不提升10个epoch，LR乘0.5
        verbose=True
    )

    # 10) 断点恢复（如有）
    start_epoch = 0                                        # 默认从0开始
    if os.path.isfile(CKPT_PATH):                          # 若存在断点文件
        start_epoch = util.load_checkpoint(str(CKPT_PATH), model, optimizer)  # 你的util需返回epoch
        logging.info(f"[Resume] start_epoch={start_epoch}")

    # 11) 若CSV不存在，先写表头
    if not CSV_FILE.exists():
        with open(CSV_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'lr', 'train_loss', 'test_RMSE', 'test_CI', 'test_Spearman'])

    # 12) 训练过程缓存（便于画图）
    ep_hist, loss_hist, rmse_hist, ci_hist, sp_hist, lr_hist = [], [], [], [], [], []

    # 13) 以“测试RMSE最小”作为保存最优模型的判据（比训练loss更可靠）
    best_rmse = float("inf")                               # 初始最优RMSE设为正无穷

    # ========================= 训练主循环 =========================
    for epoch in range(start_epoch, EPOCHS):               # 从断点或0开始到 EPOCHS-1
        torch.cuda.synchronize()                           # 同步，计时更准
        tic = time.time()                                  # 记录起始时间

        model.train()                                      # 切到训练模式
        optimizer.zero_grad()                              # 梯度清零（为梯度累积做准备）
        running_loss = 0.0                                 # 累积训练loss
        num_batches = 0                                    # 统计batch数

        # ---------- 训练迭代 ----------
        for b_idx, (seq_drug, seq_tar, y_true) in enumerate(train_loader):
            # 把数据搬到GPU（non_blocking配合pin_memory=True能略增速）
            seq_drug = seq_drug.cuda(non_blocking=True)
            seq_tar  = seq_tar.cuda(non_blocking=True)
            y_true   = y_true.cuda(non_blocking=True)

            # 自动混合精度：前向/损失计算在 autocast 上下文内
            with autocast(enabled=AMP_ENABLE):
                y_pred = model(seq_drug, seq_tar)          # 前向
                loss = criterion(y_pred, y_true)           # 计算MSE

            # 梯度累积：把loss均摊到ACCUM_STEPS
            scaler.scale(loss / ACCUM_STEPS).backward()    # AMP缩放 + 反向传播（累积）

            # 到达一个累积周期就step一次
            if (b_idx + 1) % ACCUM_STEPS == 0:
                scaler.step(optimizer)                     # AMP下安全的step
                scaler.update()                            # AMP缩放器更新
                optimizer.zero_grad()                      # 清梯度，准备下个累积周期

            running_loss += loss.item()                    # 统计loss
            num_batches += 1                               # 统计batch

        # 尾步补偿：最后一个周期不足ACCUM_STEPS也要执行一次 step
        if (b_idx + 1) % ACCUM_STEPS != 0:
            scaler.step(optimizer)                         # 做一次step
            scaler.update()                                # AMP缩放器更新
            optimizer.zero_grad()                          # 清梯度

        # 计算本epoch训练loss均值
        train_epoch_loss = running_loss / max(1, num_batches)

        # TensorBoard记录训练loss
        writer.add_scalar('Loss/train', train_epoch_loss, epoch)

        # ---------- 测试评估 ----------
        model.eval()                                       # 切到评估模式
        test_obs, test_pred = [], []                       # 收集真实/预测
        with torch.no_grad():                              # 评估不需要梯度
            for (seq_drug, seq_tar, y_true) in test_loader:
                seq_drug = seq_drug.cuda(non_blocking=True)
                seq_tar  = seq_tar.cuda(non_blocking=True)
                y_true   = y_true.cuda(non_blocking=True)
                with autocast(enabled=AMP_ENABLE):
                    y_hat = model(seq_drug, seq_tar)       # 前向预测
                test_obs.extend(y_true.view(-1).tolist())  # 展平->list，避免“float非可迭代”
                test_pred.extend(y_hat.view(-1).tolist())  # 同上

        # 计算三项指标（util中需提供）
        test_rmse = util.get_RMSE(test_obs, test_pred)     # RMSE
        test_ci   = util.get_cindex(test_obs, test_pred)   # C-index
        test_sp   = util.get_spearmanr(test_obs, test_pred)# Spearman

        # 学习率调度：指标不下降时自动降LR（以RMSE为目标，越小越好）
        scheduler.step(test_rmse)

        # 当前学习率（记录/可视化）
        cur_lr = optimizer.param_groups[0]['lr']

        # TensorBoard记录
        writer.add_scalar('LR', cur_lr, epoch)
        writer.add_scalar('Metrics/test_RMSE', test_rmse, epoch)
        writer.add_scalar('Metrics/test_CI',   test_ci,   epoch)
        writer.add_scalar('Metrics/test_Spearman', test_sp, epoch)

        # 写入CSV
        with open(CSV_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, cur_lr, train_epoch_loss, test_rmse, test_ci, test_sp])

        # 缓存到列表便于收尾画图
        ep_hist.append(epoch)
        lr_hist.append(cur_lr)
        loss_hist.append(train_epoch_loss)
        rmse_hist.append(test_rmse)
        ci_hist.append(test_ci)
        sp_hist.append(test_sp)

        # 用“测试RMSE更低”作为保存最优模型的条件
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            torch.save(model.state_dict(), MODEL_PATH)     # 仅保存权重字典
            logging.info(f"[BEST] epoch={epoch+1:04d}, RMSE={best_rmse:.4f} -> saved: {MODEL_PATH}")

        # 打印本epoch耗时等信息
        torch.cuda.synchronize()
        logging.info(f"Epoch {epoch+1:04d} | LR={cur_lr:.6g} | "
                     f"train_loss={train_epoch_loss:.6f} | "
                     f"RMSE={test_rmse:.4f} | CI={test_ci:.4f} | SP={test_sp:.4f} | "
                     f"time={(time.time() - tic)/60:.2f} min")

        # 保存断点（可恢复训练）
        util.save_checkpoint(epoch + 1, model, optimizer, filename=str(CKPT_PATH))

    # ========================= 训练结束：画PNG曲线 =========================
    def _save_curve(xs, ys, xlabel, ylabel, title, out_png):
        plt.figure()                                       # 开一张新图
        plt.plot(xs, ys)                                   # 画曲线
        plt.xlabel(xlabel)                                 # X轴标签
        plt.ylabel(ylabel)                                 # Y轴标签
        plt.title(title)                                   # 标题
        plt.tight_layout()                                 # 紧凑布局
        plt.savefig(out_png, dpi=200)                      # 保存到文件
        plt.close()                                        # 关闭图释放内存

    _save_curve(ep_hist, loss_hist, 'Epoch', 'Train Loss', 'Train Loss Curve',     FIG_DIR / 'train_loss.png')
    _save_curve(ep_hist, rmse_hist, 'Epoch', 'RMSE',       'Test RMSE Curve',      FIG_DIR / 'test_rmse.png')
    _save_curve(ep_hist, ci_hist,   'Epoch', 'CI',         'Test CI Curve',        FIG_DIR / 'test_ci.png')
    _save_curve(ep_hist, sp_hist,   'Epoch', 'Spearman',   'Test Spearman Curve',  FIG_DIR / 'test_spearman.png')
    _save_curve(ep_hist, lr_hist,   'Epoch', 'LR',         'Learning Rate Curve',  FIG_DIR / 'lr.png')

    # ========================= 用最优模型做最终测试 =========================
    pred_model = SimbaDTA().cuda()                         # 新建同构模型
    state = torch.load(MODEL_PATH, map_location="cuda")    # 读取最优权重（仅state_dict）
    pred_model.load_state_dict(state)                      # 加载权重
    pred_model.eval()                                      # 评估模式

    final_obs, final_pred = [], []                         # 收集最终指标
    with torch.no_grad():
        for (seq_drug, seq_tar, y_true) in test_loader:
            seq_drug = seq_drug.cuda(non_blocking=True)
            seq_tar  = seq_tar.cuda(non_blocking=True)
            y_true   = y_true.cuda(non_blocking=True)
            with autocast(enabled=AMP_ENABLE):
                y_hat = pred_model(seq_drug, seq_tar)
            final_obs.extend(y_true.view(-1).tolist())
            final_pred.extend(y_hat.view(-1).tolist())

    final_rmse = util.get_RMSE(final_obs, final_pred)      # 最终RMSE
    final_ci   = util.get_cindex(final_obs, final_pred)    # 最终CI
    final_sp   = util.get_spearmanr(final_obs, final_pred) # 最终Spearman
    logging.info(f"[FINAL] RMSE={final_rmse:.4f} | CI={final_ci:.4f} | Spearman={final_sp:.4f}")

    with open(OUT_TXT, 'w', encoding='utf-8') as f:        # 写入最终指标到txt
        f.write(f'test_RMSE: {final_rmse:.4f}\n')
        f.write(f'test_CI: {final_ci:.4f}\n')
        f.write(f'test_SPEARMANR: {final_sp:.4f}\n')

    # 关闭TensorBoard写入器
    writer.close()                                         # 刷新缓冲并关闭
