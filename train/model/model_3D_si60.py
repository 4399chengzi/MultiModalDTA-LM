import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimbaDTA(nn.Module):
    """
    直接使用 Uni-Mol2 生成的 embedding：
      - 输入：lig_emb ∈ R^{B, D_lig}, poc_emb ∈ R^{B, D_poc}
      - 处理：可选线性投影到同一维度 d_model；拼接+交互特征；MLP 预测
      - 输出：affinity ∈ R^{B}
    """
    def __init__(
        self,
        d_lig: int = 512,                # 配体向量维度（如 768）
        d_poc: int = 512,         # 口袋向量维度（如 768）；缺省则默认等于 d_lig
        d_model: int = None,       # 可选：统一投影到的公共维度；缺省则不投影（前提是 d_lig==d_poc）
        hidden_dims=(512, 256),    # MLP 隐层维度配置
        dropout: float = 0.1,      # Dropout 比例
        use_layernorm: bool = True,# 是否对输入做 LayerNorm
        add_interactions: bool = True # 是否加入 hadamard/abs-diff 交互特征
    ):
        super().__init__()
        self.d_lig = d_lig
        self.d_poc = d_lig if d_poc is None else d_poc
        self.add_interactions = add_interactions

        # 1) 输入归一化（可提升稳定性）
        self.norm_lig = nn.LayerNorm(self.d_lig) if use_layernorm else nn.Identity()
        self.norm_poc = nn.LayerNorm(self.d_poc) if use_layernorm else nn.Identity()

        # 2) 可选：把两路 embedding 投影到同一维度 d_model，便于做交互
        if d_model is None:
            # 不指定 d_model：要求两路维度相等，直接使用
            assert self.d_lig == self.d_poc, "当 d_model 未指定时，要求 d_lig == d_poc 才能直接拼接/交互"
            self.proj_lig = nn.Identity()
            self.proj_poc = nn.Identity()
            d_model = self.d_lig
        else:
            # 指定 d_model：两路都线性投影到同一维度
            self.proj_lig = nn.Linear(self.d_lig, d_model, bias=False)
            self.proj_poc = nn.Linear(self.d_poc, d_model, bias=False)

        self.d_model = d_model

        # 3) 特征构造：拼接 [lig, poc, lig*poc, |lig-poc|]（后两项为交互特征）
        if self.add_interactions:
            in_dim = d_model * 4
        else:
            in_dim = d_model * 2

        # 4) 预测头（多层 MLP）：(in_dim) -> hidden1 -> hidden2 -> 1
        mlp = []
        last = in_dim
        for h in hidden_dims:
            mlp += [
                nn.Linear(last, h, bias=False),
                nn.LayerNorm(h) if use_layernorm else nn.Identity(),
                nn.Dropout(dropout),
                nn.GELU()
            ]
            last = h
        mlp += [nn.Linear(last, 1, bias=True)]
        self.head = nn.Sequential(*mlp)

        # 5) 参数初始化（Kaiming 更适合 ReLU/GELU）
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, lig_emb: torch.Tensor, poc_emb: torch.Tensor) -> torch.Tensor:
        """
        lig_emb: [B, D_lig]  —— Uni-Mol2 的配体向量
        poc_emb: [B, D_poc]  —— Uni-Mol2 的口袋向量
        return : [B]         —— 预测的亲和力（标量）
        """
        # 确保是浮点
        lig = lig_emb.float()
        poc = poc_emb.float()

        # (1) LayerNorm（可选）
        lig = self.norm_lig(lig)
        poc = self.norm_poc(poc)

        # (2) 线性投影到公共维度（若启用）
        lig = self.proj_lig(lig)  # [B, d_model]
        poc = self.proj_poc(poc)  # [B, d_model]

        # (3) 构造拼接特征
        if self.add_interactions:
            hadamard = lig * poc                 # 按位乘，捕捉维度级匹配
            diff_abs = torch.abs(lig - poc)      # 绝对差，捕捉差异模式
            feat = torch.cat([lig, poc, hadamard, diff_abs], dim=-1)  # [B, 4*d_model]
        else:
            feat = torch.cat([lig, poc], dim=-1) # [B, 2*d_model]

        # (4) 通过 MLP 预测
        out = self.head(feat)                    # [B, 1]
        return out.squeeze(-1)                   # [B]
