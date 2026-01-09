import os                                   # 操作系统相关（路径、文件存在性等）
from pathlib import Path                    # 更安全/清晰的路径拼接
import time                                 # 计时
import logging                              # 打日志
import csv                                  # 训练指标写入CSV
import matplotlib                           # 服务器环境无GUI时必须先设置Agg后端
from torch.utils.tensorboard import SummaryWriter
matplotlib.use("Agg")                       # 使用非交互后端，防止无显示环境报错
import matplotlib.pyplot as plt             # 画训练曲线PNG
import torch                                # PyTorch 主库
import torch.nn as nn                       # 常用神经网络模块
import torch.optim as optim                 # 优化器
import torch.utils.data as Data             # DataLoader、random_split等
from model.model_1D import SimbaDTA         # 你的模型定义（保持不变）
import util

# ========== 可配置区域 ==========
DATA_DB_PATH = r"../dataset/ATOM3D/raw/pdbbind_2019-refined-set/data"  # Windows绝对路径，r""避免转义
DATASET_NAME = "atom3d_raw"                    # 数据集名（只用于结果目录命名）
OUT_DIR = "../dataset/ATOM3D/processed_raw"
SMILES_MAXLEN = 128                         # 药物SMILES最大长度（沿用fdavis设置，可按需调整）
PROSEQ_MAXLEN = 64                       # 蛋白序列最大长度（沿用fdavis设置，可按需调整）
TRAIN_RATIO = 0.8                          # 训练集比例（其余作为测试集；可换 0.9/0.1 等）
EPOCHS = 600
# 训练轮数
BATCH_SIZE = 32                            # batch大小
ACCUM_STEPS = 32                           # 梯度累积步数（显存不足时提高此值）
LR = 1e-3                                  # 学习率
NUM_WORKERS = 4                            # DataLoader并行读取进程数（Windows可先4）
PIN_MEMORY = True                          # pin_memory加速CPU->GPU拷贝
LOG_LEVEL = logging.INFO                   # 日志等级

# ========== 主程序入口 ==========
if __name__ == "__main__":
    # 1) 基础日志配置
    logging.basicConfig(level=LOG_LEVEL)   # 配置全局日志等级
    util.seed_torch()                      # 设定随机种子（你的util中应实现，确保可复现实验）

    # 2) 结果目录与文件
    result_root = Path(f"../result/{DATASET_NAME}")   # 结果根目录：./result/atom3d
    run_dir     = result_root / "runs"               # TensorBoard日志目录
    fig_dir     = result_root / "figs"               # 训练曲线PNG目录
    model_path  = result_root / "model.pth"          # 最优模型权重保存路径
    ckpt_path   = result_root / "checkpoint.pth.tar" # 断点恢复文件
    out_txt     = result_root / "output_metrics.txt" # 最终测试指标txt
    csv_file    = result_root / "metrics.csv"        # 训练过程指标CSV

    # 3) 确保以上目录存在
    for p in [run_dir, fig_dir, result_root]:
        p.mkdir(parents=True, exist_ok=True)         # 若目录不存在则创建

    # 4) 初始化TensorBoard记录器
    writer = SummaryWriter(log_dir=str(run_dir))     # 指向 runs 目录

    # 5) 读取数据（从LMDB拉取smiles/seq/score；util中应提供LoadData_atom3d）
    drug_seqs, target_seqs, affi = util.LoadData_atom3d_1d(
        DATA_DB_PATH,
        out_dir=OUT_DIR,
        out_name="lba_pkd_pocket_raw"
    )
    # print(len(drug_seqs), len(target_seqs), len(affi))
    # print(drug_seqs[0], target_seqs[0][:50], affi[0])
    # print(drug_seqs[1], target_seqs[1][:50], affi[1])
    # print(drug_seqs[2], target_seqs[2][:50], affi[2])
    logging.info(f"Loaded ATOM3D(LBA) samples: {len(affi)}")           # 打印样本数量

    # 6) 文本序列编码与对齐（沿用你fdavis流程：LabelDT -> Shuttle）
    labeled_drugs, labeled_targets = util.LabelDT(drug_seqs, target_seqs, SMILES_MAXLEN, PROSEQ_MAXLEN)  # 编码/截断/填充
    d_shuttle, t_shuttle, y_shuttle = util.Shuttle(labeled_drugs, labeled_targets, affi)                 # 可能含打乱/对齐

    # 7) 按比例切分训练/测试（不要再用fdavis固定7300/1825）
    N = len(y_shuttle)                                # 全部样本数量
    TRAIN_NUM = int(N * TRAIN_RATIO)                  # 训练条数
    TEST_NUM  = N - TRAIN_NUM                         # 测试条数
    logging.info(f"Split -> train: {TRAIN_NUM}, test: {TEST_NUM}")  # 打印划分信息

    # 8) 构造自定义Dataset（你的util.DatasetIterater需实现__len__/__getitem__）
    full_ds = util.DatasetIterater(d_shuttle[:N], t_shuttle[:N], y_shuttle[:N])  # 全量Dataset
    train_ds, test_ds = Data.random_split(full_ds, [TRAIN_NUM, TEST_NUM])        # 随机切分

    # 9) DataLoader（训练集务必shuffle=True）
    train_loader = Data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,                      # 训练集要打乱
        collate_fn=util.BatchPad,          # 你已有的对齐/填充函数
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    test_loader = Data.DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,                     # 测试集不需要打乱
        collate_fn=util.BatchPad,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # 10) 模型、损失、优化器
    model = SimbaDTA().cuda()              # 构建模型并移到GPU
    criterion = nn.MSELoss(reduction='mean')  # 回归任务常用MSE
    optimizer = optim.Adam(model.parameters(), lr=LR)  # Adam优化器

    # 11) 断点恢复（如存在）
    start_epoch = 0
    if os.path.isfile(ckpt_path):
        start_epoch = util.load_checkpoint(str(ckpt_path), model, optimizer)  # 你util中应返回起始epoch
        logging.info(f"Resume from epoch: {start_epoch}")

    # 12) 如果CSV不存在，写表头
    if not csv_file.exists():
        with open(csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'train_loss', 'test_RMSE', 'test_CI', 'test_Spearman'])

    # 13) 若干列表用于最终画图
    epoch_list, loss_list, rmse_list, ci_list, sp_list = [], [], [], [], []

    # 14) 用测试RMSE做“最优模型”判据（比训练loss更可靠）
    best_rmse = float("inf")

    # 15) 训练主循环
    for epoch in range(start_epoch, EPOCHS):
        torch.cuda.synchronize()           # GPU同步（计时更准确）
        tic = time.time()                  # 记录起始时间

        model.train()                      # 切换到训练模式
        optimizer.zero_grad()              # 梯度清零（为梯度累积做准备）
        running_loss = 0.0                 # 累计loss
        num_batches = 0                    # 统计batch数量
        b_idx = -1

        # ======= 训练迭代 =======
        for b_idx, (seq_drug, seq_tar, y_true) in enumerate(train_loader):  # 从DataLoader取一个batch
            seq_drug = seq_drug.cuda(non_blocking=True)  # 把数据放到GPU
            seq_tar  = seq_tar.cuda(non_blocking=True)   # 同上
            y_true   = y_true.cuda(non_blocking=True)    # 标签到GPU

            y_pred = model(seq_drug, seq_tar)            # 前向传播：得到预测
            loss = criterion(y_pred, y_true)             # 计算当前batch的MSE

            (loss / ACCUM_STEPS).backward()              # 梯度累积：把loss均摊到每个小步
            # 到达累积步数时，执行一次优化器step
            if (b_idx + 1) % ACCUM_STEPS == 0:
                optimizer.step()                         # 参数更新
                optimizer.zero_grad()                    # 清梯度，准备下一个累积周期

            running_loss += loss.item()                  # 累加loss（用于epoch均值）
            num_batches += 1                             # 记录batch数

        # 训练循环结束后，如果最后一段不足ACCUM_STEPS，也需要补一次step避免丢更新
        if (b_idx + 1) % ACCUM_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()

        # 计算当前epoch的平均训练loss
        train_epoch_loss = running_loss / max(1, num_batches)

        # 写入TensorBoard（训练loss）
        writer.add_scalar('Loss/train', train_epoch_loss, epoch)

        # ======= 训练集评估（新增）=======
        model.eval()
        train_obs, train_pred = [], []
        with torch.no_grad():
            for (seq_drug, seq_tar, y_true) in train_loader:
                seq_drug = seq_drug.cuda(non_blocking=True)
                seq_tar  = seq_tar.cuda(non_blocking=True)
                y_true   = y_true.cuda(non_blocking=True)
                y_hat = model(seq_drug, seq_tar)
                train_obs.extend(y_true.view(-1).tolist())
                train_pred.extend(y_hat.view(-1).tolist())

        train_rmse = util.get_RMSE(train_obs, train_pred)
        train_ci   = util.get_cindex(train_obs, train_pred)
        train_sp   = util.get_spearmanr(train_obs, train_pred)

        writer.add_scalar('Metrics/train_RMSE', train_rmse, epoch)
        writer.add_scalar('Metrics/train_CI',   train_ci,   epoch)
        writer.add_scalar('Metrics/train_Spearman', train_sp, epoch)

        # ======= 测试评估 =======
        model.eval()                                     # 切换到评估模式
        test_obs, test_pred = [], []                     # 存放真实值和预测值
        with torch.no_grad():                            # 评估阶段不求导
            for (seq_drug, seq_tar, y_true) in test_loader:
                seq_drug = seq_drug.cuda(non_blocking=True)
                seq_tar  = seq_tar.cuda(non_blocking=True)
                y_true   = y_true.cuda(non_blocking=True)
                y_hat = model(seq_drug, seq_tar)         # 前向得预测
                test_obs.extend(y_true.view(-1).tolist())   # 展平转list
                test_pred.extend(y_hat.view(-1).tolist())   # 同上


        # 计算三项指标（你util中应提供）
        test_rmse = util.get_RMSE(test_obs, test_pred)       # RMSE
        test_ci   = util.get_cindex(test_obs, test_pred)     # 一致性指数CI
        test_sp   = util.get_spearmanr(test_obs, test_pred)  # Spearman相关

        # 写入TensorBoard（测试指标）
        writer.add_scalar('Metrics/test_RMSE', test_rmse, epoch)
        writer.add_scalar('Metrics/test_CI',   test_ci,   epoch)
        writer.add_scalar('Metrics/test_Spearman', test_sp, epoch)

        # 追加到CSV（便于后处理）
        with open(csv_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, train_epoch_loss, test_rmse, test_ci, test_sp])

        # 缓存到内存列表（训练结束后画图）
        epoch_list.append(epoch)
        loss_list.append(train_epoch_loss)
        rmse_list.append(test_rmse)
        ci_list.append(test_ci)
        sp_list.append(test_sp)

        # 以“测试RMSE更低”为最优保存模型
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            torch.save(model.state_dict(), model_path)       # 仅保存权重字典
            logging.info(f"[BEST] epoch={epoch+1}, RMSE={best_rmse:.4f} -> saved: {model_path}")

        # 打印本epoch耗时
        torch.cuda.synchronize()

        logging.info(f"Epoch {epoch+1:04d}: "
                     f"train_loss={train_epoch_loss:.6f}, "
                     f"train_RMSE={train_rmse:.4f}, "
                     f"test_RMSE={test_rmse:.4f}, "
                     f"train_CI={train_ci:.4f}, test_CI={test_ci:.4f}, "
                     f"train_Spearman={train_sp:.4f}, test_Spearman={test_sp:.4f}, "
                     f"time={(time.time() - tic)/60:.2f} min")


        # 保存断点（可恢复训练）
        util.save_checkpoint(epoch + 1, model, optimizer, filename=str(ckpt_path))

    # 16) 训练结束：画四张PNG曲线（不依赖TensorBoard）
    def _save_curve(xs, ys, xlabel, ylabel, title, out_png):
        plt.figure()                           # 单张图
        plt.plot(xs, ys)                       # 画曲线
        plt.xlabel(xlabel)                     # X轴标签
        plt.ylabel(ylabel)                     # Y轴标签
        plt.title(title)                       # 标题
        plt.tight_layout()                     # 自动紧凑布局
        plt.savefig(out_png, dpi=200)          # 保存到文件
        plt.close()                            # 关闭图，释放内存

    _save_curve(epoch_list, loss_list, 'Epoch', 'Train Loss',     'Train Loss Curve',   fig_dir / 'train_loss.png')
    _save_curve(epoch_list, rmse_list, 'Epoch', 'RMSE',           'Test RMSE Curve',    fig_dir / 'test_rmse.png')
    _save_curve(epoch_list, ci_list,   'Epoch', 'CI',             'Test CI Curve',      fig_dir / 'test_ci.png')
    _save_curve(epoch_list, sp_list,   'Epoch', 'Spearman',       'Test Spearman Curve',fig_dir / 'test_spearman.png')

    # 17) 用“最优模型”跑一次最终测试并写入txt
    pred_model = SimbaDTA().cuda()                                # 新建同构模型
    state = torch.load(model_path, map_location="cuda")           # 读最优权重（仅state_dict）
    pred_model.load_state_dict(state)                             # 加载权重
    pred_model.eval()                                             # 评估模式

    final_obs, final_pred = [], []                                # 最终指标收集容器
    with torch.no_grad():                                         # 不需要梯度
        for (seq_drug, seq_tar, y_true) in test_loader:
            seq_drug = seq_drug.cuda(non_blocking=True)
            seq_tar  = seq_tar.cuda(non_blocking=True)
            y_true   = y_true.cuda(non_blocking=True)
            y_hat = pred_model(seq_drug, seq_tar)
            final_obs.extend(y_true.view(-1).tolist())
            final_pred.extend(y_hat.view(-1).tolist())

    final_rmse = util.get_RMSE(final_obs, final_pred)             # 最终RMSE
    final_ci   = util.get_cindex(final_obs, final_pred)           # 最终CI
    final_sp   = util.get_spearmanr(final_obs, final_pred)        # 最终Spearman

    logging.info(f"[FINAL] RMSE={final_rmse:.4f}, CI={final_ci:.4f}, Spearman={final_sp:.4f}")

    with open(out_txt, 'w', encoding='utf-8') as f:               # 写入txt文件
        f.write(f'test_RMSE: {final_rmse:.4f}\n')
        f.write(f'test_CI: {final_ci:.4f}\n')
        f.write(f'test_SPEARMANR: {final_sp:.4f}\n')

    # 18) 关闭TensorBoard writer（刷新缓冲）
    writer.close()                                                # 关闭写入器
