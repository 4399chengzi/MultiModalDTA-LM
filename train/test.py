# -*- coding: utf-8 -*-
"""
简单的数据体检脚本：
- 读取 util.LoadData_atom3d_si30_multimodal_lm 已经生成好的缓存
- 检查 train / val / test 三个拆分里：
  * y / drug_lm / prot_lm / lig_3d / poc_3d 是否有 NaN / Inf / 全 0 行
  * g_lig / g_prot 是否有 0 节点 / 0 边，节点/边特征是否有 NaN / Inf
"""

from pathlib import Path
import numpy as np
import torch
import dgl

import util
from util import LoadData_atom3d_si30_multimodal_lm  # 注意：用的是 30 版的 LM loader

DATA_DB_SI30 = r"../dataset/ATOM3D/split-by-sequence-identity-30/data"
OUT_MM_SI30  = r"../dataset/ATOM3D/processed_mm_si30_lm"  # 要和你训练时的 out_mm 保持一致


def check_array(name: str, arr: np.ndarray):
    arr = np.asarray(arr)
    print(f"\n=== [{name}] ===")
    print(f"shape = {arr.shape}, dtype = {arr.dtype}")

    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    print(f"NaN count = {n_nan}, Inf count = {n_inf}")

    n_all_zero_rows = 0
    if arr.ndim >= 2 and arr.shape[0] > 0:
        flat = arr.reshape(arr.shape[0], -1)
        row_norm = np.linalg.norm(flat, axis=1)
        n_all_zero_rows = int((row_norm == 0).sum())
        print(f"all-zero rows = {n_all_zero_rows} / {arr.shape[0]}")

    if n_nan > 0 or n_inf > 0:
        print(f"[WARN] {name} 中存在非有限值 (NaN / Inf)，建议排查产生来源。")


def check_graph_list(name: str, graphs):
    print(f"\n=== [{name}] 图列表 ===")
    n = len(graphs)
    print(f"图数量 = {n}")

    n_zero_nodes = n_zero_edges = n_nan_node = n_nan_edge = 0
    bad_idx_nodes = []
    bad_idx_edges = []
    bad_idx_nan_node = []
    bad_idx_nan_edge = []

    for i, g in enumerate(graphs):
        n_node = g.num_nodes()
        n_edge = g.num_edges()
        if n_node == 0:
            n_zero_nodes += 1
            if len(bad_idx_nodes) < 5:
                bad_idx_nodes.append(i)
        if n_edge == 0:
            n_zero_edges += 1
            if len(bad_idx_edges) < 5:
                bad_idx_edges.append(i)

        x = g.ndata.get("x", None)
        w = g.edata.get("w", None)

        if x is not None:
            x = x.detach().cpu()
            if torch.isnan(x).any() or torch.isinf(x).any():
                n_nan_node += 1
                if len(bad_idx_nan_node) < 5:
                    bad_idx_nan_node.append(i)
        else:
            print(f"[WARN] 图 {i} 没有 ndata['x']。")

        if w is not None:
            w = w.detach().cpu()
            if torch.isnan(w).any() or torch.isinf(w).any():
                n_nan_edge += 1
                if len(bad_idx_nan_edge) < 5:
                    bad_idx_nan_edge.append(i)
        else:
            print(f"[WARN] 图 {i} 没有 edata['w']。")

    print(f"0 节点图数量 = {n_zero_nodes}，示例 idx = {bad_idx_nodes}")
    print(f"0 边图数量   = {n_zero_edges}，示例 idx = {bad_idx_edges}")
    print(f"节点特征含 NaN/Inf 的图数量 = {n_nan_node}，示例 idx = {bad_idx_nan_node}")
    print(f"边特征含 NaN/Inf 的图数量   = {n_nan_edge}，示例 idx = {bad_idx_nan_edge}")

    if n_zero_nodes > 0 or n_zero_edges > 0:
        print(f"[WARN] {name} 中存在空图（0 节点或 0 边），训练时可能导致异常。")


def check_one_split(split: str):
    print("\n" + "=" * 80)
    print(f"#### 检查拆分: {split} ####")

    pkg = LoadData_atom3d_si30_multimodal_lm(
        root_base=DATA_DB_SI30,
        split=split,
        out_mm=OUT_MM_SI30,
        unimol2_size="unimol2_small",
        contact_threshold=8.0,
        dis_min=1.0,
        prot_self_loop=False,
        bond_bidirectional=True,
        prefer_model=None,
        force_refresh=False,     # 只读缓存，不重新跑 UniMol/LM
        use_cuda_for_unimol=False
    )[split]

    N = len(pkg["y"])
    print(f"[{split}] 样本数 N = {N}")

    # 长度一致性
    assert len(pkg["ids"])    == N
    assert len(pkg["smiles"]) == N
    assert len(pkg["seq"])    == N
    assert pkg["lig_3d"].shape[0] == N
    assert pkg["poc_3d"].shape[0] == N
    assert pkg["drug_lm"].shape[0] == N
    assert pkg["prot_lm"].shape[0] == N
    assert len(pkg["g_lig"])  == N
    assert len(pkg["g_prot"]) == N
    print("[OK] 各模态样本数量一致。")

    # 数值检查
    check_array("y (label)", pkg["y"])
    check_array("drug_lm",   pkg["drug_lm"])
    check_array("prot_lm",   pkg["prot_lm"])
    check_array("lig_3d",    pkg["lig_3d"])
    check_array("poc_3d",    pkg["poc_3d"])

    check_graph_list("g_lig",  pkg["g_lig"])
    check_graph_list("g_prot", pkg["g_prot"])

    print(f"#### 拆分 {split} 检查完毕 ####")


if __name__ == "__main__":
    print("开始检查 ATOM3D SI-30 多模态 LM 数据...")
    for sp in ("train", "val", "test"):
        check_one_split(sp)
    print("\n全部拆分检查完成。")
