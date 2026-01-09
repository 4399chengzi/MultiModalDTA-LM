import torch
import torch.nn as nn
import torch.fft
import math
from einops import einsum, rearrange, repeat
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class Mamba(nn.Module):
    def __init__(self, d_model, d_state=64, expand=2, d_conv=4, conv_bias=True, bias=False):
        """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
        super().__init__()
        self.d_model = d_model  # Model dimension d_model
        self.d_state = d_state  # SSM state expansion factor
        self.d_conv = d_conv  # Local convolution width
        self.expand = expand  # Block expansion factor
        self.conv_bias = conv_bias
        self.bias = bias
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=self.bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=self.conv_bias,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
        )

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)

        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, self.d_state + 1), 'n -> d n', d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=self.bias)

    def forward(self, x):
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # shape (b, l, 2 * d_in)
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')

        x = F.silu(x)

        y = self.ssm(x)

        y = y * F.silu(res)

        output = self.out_proj(y)
        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)

        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)

        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)

        y = self.selective_scan(x, delta, A, B, C, D)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        # # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # # is additionally hardware-aware (like FlashAttention).
        # x = torch.zeros((b, d_in, n), device=deltaA.device)
        # ys = []
        # for i in range(l):
        #     x = deltaA[:, i] * x + deltaB_u[:, i]
        #     y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
        #     ys.append(y)
        # y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        # Convert sequential scan to cumulative sum (parallel-friendly operation)
        # Using cumsum to emulate sequential behavior in a parallel way
        cumulative_A = torch.cumsum(deltaA, dim=1)  # (b, l, d_in, n)
        cumulative_B = torch.cumsum(deltaB_u, dim=1)  # (b, l, d_in, n)

        # Combining the results with einsum
        y = einsum(cumulative_A, C, 'b l d_in n, b l n -> b l d_in')  # (b, l, d_in)

        y = y + u * D

        return y


class EinFFT(nn.Module):
    def __init__(
            self,
            sequence_length: int,
            dim: int,
    ):
        super().__init__()
        self.dim = dim

        # silu
        self.act = nn.SiLU()

        # complex weights for channel-wise transformation
        self.complex_weight = nn.Parameter(
            torch.randn(sequence_length, dim, dtype=torch.complex64)
        )

        # Real weight
        self.real_weight = nn.Parameter(
            torch.randn(sequence_length, dim)
        )

    def forward(self, x):
        # apply 1D FFTSolver, transform input tensor to frequency domain
        fast_fouried = torch.fft.fft(x, dim=-2)

        # get xr xi splitted parts
        xr = fast_fouried.real
        xi = fast_fouried.imag

        # complex-valued multiplication
        einsum_mul = torch.einsum(
            "bsd,cf->bsd", xr, self.complex_weight
        ) + torch.einsum("bsd,cf->bsd", xi, self.complex_weight)

        xr = einsum_mul.real
        xi = einsum_mul.imag

        # apply silu
        real_act = self.act(xr)
        imag_act = self.act(xi)

        # EMM with the weights use torch split instead
        emmed = torch.einsum(
            "bsd,cf->bsd", real_act, self.real_weight
        ) + torch.einsum("bsd,cf->bsd", imag_act, self.complex_weight)

        # apply ifft solver as notated
        iffted = torch.fft.ifft(emmed + emmed, dim=-2)
        return iffted.real


class SimbaBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            dropout: float = 0.1,
            d_state: int = 32,
            d_conv: int = 32,
            num_classes: int = 64
    ):
        super(SimbaBlock, self).__init__()
        self.dim = dim
        self.layer_norm = nn.LayerNorm(dim)
        self.d_state = d_state
        self.d_conv = d_conv
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)

        # Mamba Block
        self.mamba = Mamba(
            d_model=self.dim,
            d_state=self.d_state,
            d_conv=self.d_conv,
        )

        # EinFFT
        self.einfft = EinFFT(
            sequence_length=num_classes,
            dim=self.dim,
        )

    def forward(self, x):
        residual = x

        # Layernorm
        normed = self.layer_norm(x)
        # Mamba
        mamba = self.mamba(normed)

        # Dropout
        droped = self.dropout(mamba)

        out = residual + droped

        # Phase 2
        residual_new = out

        # Layernorm
        normed_new = self.layer_norm(out)

        # einfft
        fasted = self.einfft(normed_new)

        # Dropout
        out = self.dropout(fasted)

        # residual
        out = out + residual_new
        return out.real

class SimbaDTA(nn.Module):
    def __init__(self):
        super(SimbaDTA, self).__init__()
        self.emb_dim = 128

        # 药物和靶标序列的嵌入层
        self.drug_embedding = nn.Embedding(num_embeddings=62, embedding_dim=self.emb_dim)
        self.target_embedding = nn.Embedding(num_embeddings=25, embedding_dim=self.emb_dim)

        # 位置编码层
        self.position_encoding = PositionalEncoding(d_model=self.emb_dim)

        # SimbaBlock
        # Layers
        self.simba_blocks_drug = nn.ModuleList(
            [
                SimbaBlock(
                    dim=self.emb_dim,
                    dropout=0.1,
                    d_state=8,
                    d_conv=8,
                    num_classes=50,  # 100
                )
                for _ in range(2)
            ]
        )

        self.simba_blocks_target = nn.ModuleList(
            [
                SimbaBlock(
                    dim=self.emb_dim,
                    dropout=0.1,
                    d_state=8,
                    d_conv=8,
                    num_classes=50,  # 1000
                )
                for _ in range(2)
            ]
        )

        # 前馈神经网络层
        self.fc1 = nn.Sequential(
            nn.Linear(2 * self.emb_dim, 4 * self.emb_dim, bias=False),
            nn.LayerNorm(4 * self.emb_dim),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4 * self.emb_dim, 2 * self.emb_dim, bias=False),
            nn.LayerNorm(2 * self.emb_dim),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True)
        )
        self.output_layer = nn.Linear(2 * self.emb_dim, 1, bias=True)

    def forward(self, drug_seq, target_seq):
        # 嵌入层
        drug_embed = self.drug_embedding(drug_seq)
        target_embed = self.target_embedding(target_seq)

        # 添加位置编码
        drug_encoded = self.position_encoding(drug_embed.transpose(0, 1)).transpose(0, 1)
        target_encoded = self.position_encoding(target_embed.transpose(0, 1)).transpose(0, 1)
        # torch.Size([batchsize, maxlen, embedding])


        # 使用 SimbaBlock 处理编码向量
        # Loop through simba blocks
        for layer in self.simba_blocks_drug:
            drug_encoded = layer(drug_encoded)
        for layer in self.simba_blocks_target:
            target_encoded = layer(target_encoded)

        # 压缩向量
        drug_encoded = torch.sum(drug_encoded, dim=1)
        target_encoded = torch.sum(target_encoded, dim=1)
        # drug_encoded = drug_encoded.squeeze()
        # target_encoded = target_encoded.squeeze()
        # torch.Size([batchsize, embedding])

        # 拼接药物和靶标的编码向量
        combined = torch.cat((drug_encoded, target_encoded), dim=1)
        # torch.Size([batchsize, embedding × 2])


        # 预测器
        hidden1 = self.fc1(combined)
        hidden2 = self.fc2(hidden1)
        output = self.output_layer(hidden2)

        # 返回亲和力预测值
        affinity = output.squeeze()

        return affinity
