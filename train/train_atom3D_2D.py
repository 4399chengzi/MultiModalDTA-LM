import csv
import os
import time
from pathlib import Path                    # 更安全/清晰的路径拼接
import logging                              # 打日志
import matplotlib                           # 服务器环境无GUI时必须先设置Agg后端
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
matplotlib.use("Agg")                       # 使用非交互后端，防止无显示环境报错
import torch                                # PyTorch 主库
import torch.nn as nn                       # 常用神经网络模块
import torch.optim as optim                 # 优化器
import util                                 # 你的工具库（需提供若干函数，见下）
import model.model_2D as model_2D
import torch.utils.data as Data             # DataLoader、random_split等

# ========== 可配置区域 ==========
DATA_DB_PATH = r"../dataset/ATOM3D/raw/pdbbind_2019-refined-set/data"  # Windows绝对路径，r""避免转义
DATASET_NAME = "atom3d_raw_2D"                    # 数据集名（只用于结果目录命名）
OUT_DIR = "../dataset/ATOM3D/processed_raw"
SMILES_MAXLEN = 128                         # 药物SMILES最大长度（沿用fdavis设置，可按需调整）
PROSEQ_MAXLEN = 64                       # 蛋白序列最大长度（沿用fdavis设置，可按需调整）
TRAIN_RATIO = 0.8                          # 训练集比例（其余作为测试集；可换 0.9/0.1 等）
EPOCHS = 600
# 训练轮数/
BATCH_SIZE = 32                            # batch大小
ACCUM_STEPS = 32                           # 梯度累积步数（显存不足时提高此值）
LR = 1e-3                                  # 学习率
NUM_WORKERS = 4                            # DataLoader并行读取进程数（Windows可先4）
PIN_MEMORY = True                          # pin_memory加速CPU->GPU拷贝
LOG_LEVEL = logging.INFO                   # 日志等级

if __name__ == '__main__':
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
    drug_graphs, target_graphs, affi = util.LoadData_atom3d_2d(
        root=r"D:\SimbaDTA\SimbaDTA\dataset\ATOM3D\raw\pdbbind_2019-refined-set\data",
        out_dir="../dataset/ATOM3D/processed_2d",
        out_name="lba_pkd_graph2d",
        use_pocket=True,
        contact_threshold=8.0
    )

    # print(len(drug_graphs), len(target_graphs), len(affi))  # 样本数
    # print(drug_graphs[0], target_graphs[0], affi[0])  # 看第一个样本
    logging.info(f"Loaded ATOM3D(LBA) samples: {len(affi)}")  # 打印样本数量

    # 6)打乱
    d_graphs_shuttle, t_graphs_shuttle, y_shuttle = util.Shuttle_2D(drug_graphs, target_graphs, affi)

    # 7) 按比例切分训练/测试（不要再用fdavis固定7300/1825）
    N = len(y_shuttle)                                # 全部样本数量
    TRAIN_NUM = int(N * TRAIN_RATIO)                  # 训练条数
    TEST_NUM  = N - TRAIN_NUM                         # 测试条数
    logging.info(f"Split -> train: {TRAIN_NUM}, test: {TEST_NUM}")  # 打印划分信息

    # 8) 构造自定义Dataset（你的util.DatasetIterater需实现__len__/__getitem__）
    full_ds = util.DatasetIterater(d_graphs_shuttle[:N], t_graphs_shuttle[:N], y_shuttle[:N])  # 全量Dataset
    train_ds, test_ds = Data.random_split(full_ds, [TRAIN_NUM, TEST_NUM])        # 随机切分

    # 9) DataLoader（训练集务必shuffle=True）
    train_loader = Data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,                      # 训练集要打乱
        collate_fn=util.Gcollate,          # 你已有的对齐/填充函数
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    test_loader = Data.DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,                     # 测试集不需要打乱
        collate_fn=util.Gcollate,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # 10) 模型、损失、优化器
    model = model_2D.GPCNDTA().cuda()              # 构建模型并移到GPU
    criterion = nn.MSELoss(reduction='mean')  # 回归任务常用MSE
    optimizer = optim.Adam(model.parameters(), lr=LR)  # Adam优化器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6) # 学习率调度器

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
        torch.cuda.synchronize()
        tic = time.time()

        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        num_batches = 0
        b_idx = -1

        # ======= 训练迭代 =======
        for b_idx, (batched_dg, batched_tg, y_true) in enumerate(train_loader):
            batched_dg = batched_dg.to("cuda")
            batched_tg = batched_tg.to("cuda")
            y_true = y_true.view(-1, 1).cuda(non_blocking=True)

            y_pred = model(batched_dg, batched_tg)  # 前向传播
            loss = criterion(y_pred, y_true)

            (loss / ACCUM_STEPS).backward()
            if (b_idx + 1) % ACCUM_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            num_batches += 1

        if (b_idx + 1) % ACCUM_STEPS != 0:
            optimizer.step()
            optimizer.zero_grad()

        train_epoch_loss = running_loss / max(1, num_batches)
        writer.add_scalar('Loss/train', train_epoch_loss, epoch)

        # ======= 训练集评估 =======
        model.eval()
        train_obs, train_pred = [], []
        with torch.no_grad():
            for (batched_dg, batched_tg, y_true) in train_loader:
                batched_dg = batched_dg.to("cuda")
                batched_tg = batched_tg.to("cuda")
                y_true = y_true.view(-1, 1).cuda(non_blocking=True)

                y_hat = model(batched_dg, batched_tg)
                train_obs.extend(y_true.view(-1).tolist())
                train_pred.extend(y_hat.view(-1).tolist())

        train_rmse = util.get_RMSE(train_obs, train_pred)
        train_ci = util.get_cindex(train_obs, train_pred)
        train_sp = util.get_spearmanr(train_obs, train_pred)

        writer.add_scalar('Metrics/train_RMSE', train_rmse, epoch)
        writer.add_scalar('Metrics/train_CI', train_ci, epoch)
        writer.add_scalar('Metrics/train_Spearman', train_sp, epoch)

        # ======= 测试集评估 =======
        test_obs, test_pred = [], []
        with torch.no_grad():
            for (batched_dg, batched_tg, y_true) in test_loader:
                batched_dg = batched_dg.to("cuda")
                batched_tg = batched_tg.to("cuda")
                y_true = y_true.view(-1, 1).cuda(non_blocking=True)

                y_hat = model(batched_dg, batched_tg)
                test_obs.extend(y_true.view(-1).tolist())
                test_pred.extend(y_hat.view(-1).tolist())

        test_rmse = util.get_RMSE(test_obs, test_pred)
        test_ci = util.get_cindex(test_obs, test_pred)
        test_sp = util.get_spearmanr(test_obs, test_pred)

        writer.add_scalar('Metrics/test_RMSE', test_rmse, epoch)
        writer.add_scalar('Metrics/test_CI', test_ci, epoch)
        writer.add_scalar('Metrics/test_Spearman', test_sp, epoch)

        with open(csv_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, train_epoch_loss, test_rmse, test_ci, test_sp])

        epoch_list.append(epoch)
        loss_list.append(train_epoch_loss)
        rmse_list.append(test_rmse)
        ci_list.append(test_ci)
        sp_list.append(test_sp)

        if test_rmse < best_rmse:
            best_rmse = test_rmse
            torch.save(model.state_dict(), model_path)
            logging.info(f"[BEST] epoch={epoch + 1}, RMSE={best_rmse:.4f} -> saved: {model_path}")

        torch.cuda.synchronize()
        logging.info(f"Epoch {epoch + 1:04d}: "
                     f"train_loss={train_epoch_loss:.6f}, "
                     f"train_RMSE={train_rmse:.4f}, "
                     f"test_RMSE={test_rmse:.4f}, "
                     f"train_CI={train_ci:.4f}, test_CI={test_ci:.4f}, "
                     f"train_Spearman={train_sp:.4f}, test_Spearman={test_sp:.4f}, "
                     f"time={(time.time() - tic) / 60:.2f} min")

        util.save_checkpoint(epoch + 1, model, optimizer, filename=str(ckpt_path))


    # 16) 保存曲线图
    def _save_curve(xs, ys, xlabel, ylabel, title, out_png):
        plt.figure()
        plt.plot(xs, ys)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()


    _save_curve(epoch_list, loss_list, 'Epoch', 'Train Loss', 'Train Loss Curve', fig_dir / 'train_loss.png')
    _save_curve(epoch_list, rmse_list, 'Epoch', 'RMSE', 'Test RMSE Curve', fig_dir / 'test_rmse.png')
    _save_curve(epoch_list, ci_list, 'Epoch', 'CI', 'Test CI Curve', fig_dir / 'test_ci.png')
    _save_curve(epoch_list, sp_list, 'Epoch', 'Spearman', 'Test Spearman Curve', fig_dir / 'test_spearman.png')

    # 17) 最优模型最终测试
    pred_model = model_2D.GPCNDTA().cuda()
    state = torch.load(model_path, map_location="cuda")
    pred_model.load_state_dict(state)
    pred_model.eval()

    final_obs, final_pred = [], []
    with torch.no_grad():
        for (batched_dg, batched_tg, y_true) in test_loader:
            batched_dg = batched_dg.to("cuda")
            batched_tg = batched_tg.to("cuda")
            y_true = y_true.view(-1, 1).cuda(non_blocking=True)

            y_hat = pred_model(batched_dg, batched_tg)
            final_obs.extend(y_true.view(-1).tolist())
            final_pred.extend(y_hat.view(-1).tolist())

    final_rmse = util.get_RMSE(final_obs, final_pred)
    final_ci = util.get_cindex(final_obs, final_pred)
    final_sp = util.get_spearmanr(final_obs, final_pred)

    logging.info(f"[FINAL] RMSE={final_rmse:.4f}, CI={final_ci:.4f}, Spearman={final_sp:.4f}")

    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(f'test_RMSE: {final_rmse:.4f}\n')
        f.write(f'test_CI: {final_ci:.4f}\n')
        f.write(f'test_SPEARMANR: {final_sp:.4f}\n')

    writer.close()



