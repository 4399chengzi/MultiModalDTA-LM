import os

from model.model_1D import SimbaDTA
import util
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import time
import logging
from pathlib import Path
import torch

if __name__ == '__main__':
    # 配置日志记录
    logging.basicConfig(level=logging.INFO)

    # davis dataset
    fpath_davis = Path('../dataset/davis/')
    logspance_trans = True
    dataset_name = 'davis'
    smile_maxlen, proSeq_maxlen = 85, 1200
    TRAIN_NUM, TEST_NUM = 24040, 6016
    model_fromTrain = Path('../result/davis/model.pth')
    output_file = Path('../result/davis/output_metrics.txt')
    checkpoint_file = "../result/davis/checkpoint.pth.tar"

    # kiba dataset
    # fpath_davis = Path('./dataset/kiba/')
    # logspance_trans = False
    # dataset_name = 'kiba'
    # smile_maxlen, proSeq_maxlen = 100, 1000
    # TRAIN_NUM, TEST_NUM = 94600, 23654
    # model_fromTrain = Path('./result/kiba/model.pth')
    # output_file = Path('./result/kiba/output_metrics.txt')

    # 加载数据
    drug, target, affinity = util.LoadData(fpath_davis, logspance_trans=logspance_trans)
    drug_seqs, target_seqs, affiMatrix = util.GetSamples(dataset_name, drug, target, affinity)
    labeled_drugs, labeled_targets = util.LabelDT(drug_seqs, target_seqs, smile_maxlen, proSeq_maxlen)
    labeledDrugs_shuttle, labeledTargets_shuttle, affiMatrix_shuttle = util.Shuttle(labeled_drugs, labeled_targets,
                                                                                    affiMatrix)

    # 训练设置
    model = SimbaDTA().cuda()
    criterion = nn.MSELoss(reduction='mean')

    # 数据迭代器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    EPOCHS, batch_size, accumulation_steps = 600, 32, 32
    min_train_loss = 100000
    Data_iter = util.DatasetIterater(labeledDrugs_shuttle[:TRAIN_NUM + TEST_NUM],
                                     labeledTargets_shuttle[:TRAIN_NUM + TEST_NUM],
                                     affiMatrix_shuttle[:TRAIN_NUM + TEST_NUM])
    train_iter, test_iter = Data.random_split(Data_iter, [TRAIN_NUM, TEST_NUM])
    train_loader = Data.DataLoader(
        train_iter,
        batch_size=batch_size,
        shuffle=False,  # 如果希望在每个epoch打乱数据，改为 True
        collate_fn=util.BatchPad,
        num_workers=4,  # 使用4个CPU进程来加载数据，可以根据你的CPU核数调整
        pin_memory=True  # 启用pin_memory，将数据加载到GPU时加快传输速度
    )

    test_loader = Data.DataLoader(
        test_iter,
        batch_size=batch_size,
        shuffle=False,  # 测试集一般不需要shuffle
        collate_fn=util.BatchPad,
        num_workers=4,  # 同样使用多线程数据加载
        pin_memory=True  # 启用pin_memory
    )

    util.seed_torch()

    # 检查是否有可用的检查点文件
    start_epoch = 0
    if os.path.isfile(checkpoint_file):
        start_epoch = util.load_checkpoint(checkpoint_file, model, optimizer)
    print('start_epoch', start_epoch)

    for epoch in range(start_epoch, EPOCHS):
        print(epoch)
        torch.cuda.synchronize()
        start = time.time()
        model.train()
        train_sum_loss = 0
        for train_batch_idx, (SeqDrug, SeqTar, real_affi) in enumerate(train_loader):
            SeqDrug, SeqTar, real_affi = SeqDrug.cuda(), SeqTar.cuda(), real_affi.cuda()
            pre_affi = model(SeqDrug, SeqTar)
            train_loss = criterion(pre_affi, real_affi)
            train_sum_loss += train_loss.item()
            train_loss.backward()
            if (train_batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            if (train_batch_idx + 1) == (TRAIN_NUM // batch_size + 1):
                train_epoch_loss = train_sum_loss / train_batch_idx
                if train_epoch_loss < min_train_loss:
                    min_train_loss = train_epoch_loss
                    logging.info(f'Epoch: {epoch + 1:04d}, loss = {train_epoch_loss:.6f}')
                    torch.save(model.state_dict(), model_fromTrain)
                    logging.info(f'Best model from {epoch + 1:04d} Epoch saved at {model_fromTrain}')

        torch.cuda.synchronize()
        logging.info(f'Time taken for 1 epoch is {(time.time() - start) / 60:.4f} minutes')
        util.save_checkpoint(epoch + 1, model, optimizer, filename=checkpoint_file)

    # 测试
    predModel = SimbaDTA().cuda()
    predModel.load_state_dict(torch.load(model_fromTrain, weights_only=True))
    predModel.eval()
    train_obs, train_pred = [], []
    test_obs, test_pred = [], []

    with torch.no_grad():
        for (DrugSeqs, TarSeqs, real_affi) in test_loader:
            DrugSeqs, TarSeqs, real_affi = DrugSeqs.cuda(), TarSeqs.cuda(), real_affi.cuda()
            pred_affi = predModel(DrugSeqs, TarSeqs)
            test_obs.extend(real_affi.tolist())
            test_pred.extend(pred_affi.tolist())

    test_mse = util.get_MSE(test_obs, test_pred)
    test_ci = util.get_cindex(test_obs, test_pred)
    test_rm2 = util.get_rm2(test_obs, test_pred)

    logging.info(f'test_MSE: {test_mse:.3f}')
    logging.info(f'test_CI: {test_ci:.3f}')
    logging.info(f'test_rm2: {test_rm2:.3f}')

    # 写入文件
    with open(output_file, 'w') as f:
        f.write(f'test_MSE: {test_mse:.3f}\n')
        f.write(f'test_CI: {test_ci:.3f}\n')
        f.write(f'test_rm2: {test_rm2:.3f}\n')
