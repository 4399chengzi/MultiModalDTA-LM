import atom3d.datasets as da
import pandas as pd
from unimol_tools import UniMolRepr
import torch

# 1️⃣ 初始化模型
model = UniMolRepr(model="unimol2_small", device="cuda:0")

# 2️⃣ 指定你的 ATOM3D LBA 数据集路径
root = r"..\dataset\ATOM3D\raw\pdbbind_2019-refined-set\data"

# 3️⃣ 读取 LMDB 数据集
ds = da.LMDBDataset(root)
print(f"✅ 数据集中共有样本数: {len(ds)}")

# 4️⃣ 取一个样本看看里面有什么 key
sample = ds[0]
print("\n📦 sample 的 key 列表：")
for k in sample.keys():
    print(f"  - {k}: {type(sample[k])}")

# 5️⃣ 查看常见字段
print("\n🔹 标签信息:")
print(sample["scores"])   # 包含 neglog_aff（即 pKd）

# 6️⃣ 查看配体部分（atoms_ligand）
lig_df = sample["atoms_ligand"]
print("\n🔸 配体原子表 atoms_ligand：")
print(f"形状: {lig_df.shape}")
print("列名:", list(lig_df.columns))
print(lig_df.head(5))  # 打印前5行看看格式

# 7️⃣ 查看口袋部分（atoms_pocket）
pocket_df = sample["atoms_pocket"]
print("\n🔹 口袋原子表 atoms_pocket：")
print(f"形状: {pocket_df.shape}")
print("列名:", list(pocket_df.columns))
print(pocket_df.head(5))

# 8️⃣ 提取原子坐标和元素信息
lig_atoms = lig_df["element"].tolist()
lig_coords = lig_df[["x", "y", "z"]].values.tolist()

pocket_atoms = pocket_df["element"].tolist()
pocket_coords = pocket_df[["x", "y", "z"]].values.tolist()

print(f"\n🧬 配体原子数: {len(lig_atoms)} | 口袋原子数: {len(pocket_atoms)}")

# 9️⃣ 用 UniMol2 得到 3D 表示
lig_emb = model.get_repr({"atoms": lig_atoms, "coordinates": lig_coords})
pocket_emb = model.get_repr({"atoms": pocket_atoms, "coordinates": pocket_coords})

print("\n✅ Uni-Mol2 计算完成:")
print("  配体 embedding 形状:", torch.tensor(lig_emb).shape)
print("  口袋 embedding 形状:", torch.tensor(pocket_emb).shape)
