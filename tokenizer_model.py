import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchcde

class RotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int, base: float = 10000.0):
        super().__init__()
        assert rotary_dim % 2 == 0, "rotary_dim 必须是偶数"
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_cos_sin(self, seq_len: int, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        theta = torch.outer(t, self.inv_freq)               # [S, rd/2]
        cos = torch.cos(theta).to(dtype=dtype)
        sin = torch.sin(theta).to(dtype=dtype)
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int):
    # x: [B,H,S,Dh], cos/sin: [S, rd/2]
    B, H, S, Dh = x.shape
    rd = rotary_dim
    x_rope = x[..., :rd]
    x_pass = x[..., rd:]

    x1 = x_rope[..., 0::2]
    x2 = x_rope[..., 1::2]

    cos_ = cos[None, None, :, :]
    sin_ = sin[None, None, :, :]

    y1 = x1 * cos_ - x2 * sin_
    y2 = x1 * sin_ + x2 * cos_

    y = torch.stack([y1, y2], dim=-1).flatten(-2)  # [B,H,S,rd]
    return torch.cat([y, x_pass], dim=-1)


class MultiheadSelfAttentionRoPE(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, rotary_dim=None):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须整除 nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim % 2 == 0, "head_dim 必须为偶数（RoPE需要）"

        self.rotary_dim = self.head_dim if rotary_dim is None else rotary_dim
        assert self.rotary_dim % 2 == 0 and self.rotary_dim <= self.head_dim

        self.rope = RotaryEmbedding(self.rotary_dim)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # x: [B,S,D]
        B, S, D = x.shape
        qkv = self.qkv(x)                 # [B,S,3D]
        q, k, v = qkv.chunk(3, dim=-1)    # each [B,S,D]

        q = q.view(B, S, self.nhead, self.head_dim).transpose(1, 2)  # [B,H,S,Dh]
        k = k.view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        cos, sin = self.rope.get_cos_sin(S, device=x.device, dtype=q.dtype)
        q = apply_rope(q, cos, sin, self.rotary_dim)
        k = apply_rope(k, cos, sin, self.rotary_dim)

        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # [B,H,S,S]

        if self.training and torch.rand(1).item() < 0.001:  
            with torch.no_grad():
                s_min = scores.min().item()
                s_max = scores.max().item()
                s_mean = scores.mean().item()
                s_std = scores.std().item()
                print(f"[Score Statistics] Min: {s_min:.2f}, Max: {s_max:.2f}, Mean: {s_mean:.2f}, Std: {s_std:.2f}")

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask[None, None, :, :], float("-inf"))
            else:
                scores = scores + attn_mask[None, None, :, :]

     

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # [B,H,S,Dh]
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)


class TransformerEncoderLayerRoPE(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, rotary_dim=None, activation="relu"):
        super().__init__()
        self.self_attn = MultiheadSelfAttentionRoPE(d_model, nhead, dropout=dropout, rotary_dim=rotary_dim)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.act = F.gelu if activation == "gelu" else F.relu

    def forward(self, x, mask=None):
        x2 = self.self_attn(x, attn_mask=mask)
        x = self.norm1(x + self.dropout1(x2))

        x2 = self.linear2(self.dropout(self.act(self.linear1(x))))
        x = self.norm2(x + self.dropout2(x2))
        return x


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.requires_grad = False

        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", self.embedding.weight.data.clone())

    def forward(self, inputs):
        inputs_shape = inputs.shape
        flat_input = inputs.contiguous().view(-1, self.embedding_dim)

        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(encoding_indices, self.embedding.weight)

        if self.training:
            encodings_one_hot = F.one_hot(encoding_indices, self.num_embeddings).float()
            self.ema_cluster_size.data.mul_(self.decay).add_(
                torch.sum(encodings_one_hot, dim=0), alpha=1 - self.decay
            )

            dw = torch.matmul(flat_input.t(), encodings_one_hot)  # [D,K]
            self.ema_w.data.mul_(self.decay).add_(dw.t(), alpha=1 - self.decay)

            n = torch.sum(self.ema_cluster_size)
            normalized_cluster_size = (
                (self.ema_cluster_size + self.epsilon)
                / (n + self.num_embeddings * self.epsilon)
                * n
            )

            normalized_ema_w = self.ema_w / normalized_cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(normalized_ema_w)

        vq_loss = self.commitment_cost * F.mse_loss(flat_input, quantized.detach())

        quantized = quantized.view(inputs_shape)
        quantized_ste = inputs + (quantized - inputs).detach()
        indices = encoding_indices.view(inputs_shape[:-1])
        return vq_loss, quantized_ste, indices


class VQVAE_Transformer(nn.Module):
    """
    RoPE 版：不再使用加法 PositionalEncoding；在注意力中对 Q/K 做旋转位置编码。
    编码器/解码器都使用 causal self-attention（与你原版一致）。
    """
    def __init__(
        self,
        input_dim,
        seq_len,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        ema_decay,
        ema_epsilon,
        dropout=0.1,
        rotary_dim=None,
        activation="relu",
    ):
        super().__init__()
        assert d_model == embedding_dim, "d_model 必须等于 embedding_dim"
        assert d_model % nhead == 0
        assert (d_model // nhead) % 2 == 0, "head_dim 必须是偶数（RoPE需要）"

        self.seq_len = seq_len
        self.d_model = d_model

        self.input_projection = nn.Linear(input_dim, d_model)

        self.encoder = nn.ModuleList([
            TransformerEncoderLayerRoPE(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                rotary_dim=rotary_dim,
                activation=activation,
            ) for _ in range(num_encoder_layers)
        ])

        self.vq_layer = VectorQuantizerEMA(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=ema_decay,
            epsilon=ema_epsilon,
        )

        self.decoder = nn.ModuleList([
            TransformerEncoderLayerRoPE(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                rotary_dim=rotary_dim,
                activation=activation,
            ) for _ in range(num_decoder_layers)
        ])

        self.output_projection = nn.Linear(d_model, input_dim)

    @staticmethod
    def generate_causal_mask(seq_len: int, device):
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def encode(self, x):
        x = self.input_projection(x)  # [B,S,D]
        S = x.size(1)
        mask = self.generate_causal_mask(S, x.device)
        for layer in self.encoder:
            x = layer(x, mask=mask)
        return x

    def quantize(self, z_e):
        return self.vq_layer(z_e)

    def decode(self, z_q):
        x = z_q
        S = x.size(1)
        mask = self.generate_causal_mask(S, x.device)
        for layer in self.decoder:
            x = layer(x, mask=mask)
        return self.output_projection(x)

    def forward(self, x):
        z_e = self.encode(x)
        vq_loss, z_q, indices = self.quantize(z_e)
        x_recon = self.decode(z_q)
        return vq_loss, x_recon, indices




class CDEFunc(nn.Module):
    """
    dh_t = f(h_t) dX_t
    """
    def __init__(self, hidden_dim, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim * input_dim),
        )
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

    def forward(self, t, h):
        # output shape: [B, hidden_dim * input_dim]
        out = self.net(h)
        return out.view(h.size(0), self.hidden_dim, self.input_dim)


class NeuralCDE(nn.Module):
    """
    X[B,S,F] → h_T[B,H]
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.func = CDEFunc(hidden_dim, input_dim)

        # 初始 hidden 状态由第一个时间点映射
        self.init_linear = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        B, S, F = x.shape
        device = x.device

        times = torch.linspace(0, 1, S, device=device)

        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(
            x, times
        )
        spline = torchcde.CubicSpline(coeffs, times)

        # 明确 batch-first 的初始条件
        h0 = self.init_linear(x[:, 0, :])   # [B, H]

        h_T = torchcde.cdeint(
            X=spline,
            func=self.func,
            z0=h0,
            t=times,
            method="rk4",
        )

        # ========= 关键修复：自动识别维度 =========
        if h_T.dim() != 3:
            raise RuntimeError(f"Unexpected h_T shape: {h_T.shape}")

        if h_T.shape[0] == B:
            # [B, S, H]
            h_T = h_T[:, -1, :]
        elif h_T.shape[1] == B:
            # [S, B, H]
            h_T = h_T[-1, :, :]
        else:
            raise RuntimeError(f"Cannot infer batch dim from h_T shape {h_T.shape}")

        assert h_T.shape == (B, self.hidden_dim), f"h_T wrong shape: {h_T.shape}"
        return h_T

class CDEBlock(nn.Module):
    """
    Neural CDE + compress
    """
    def __init__(self, input_dim, hidden_dim=256, out_dim=128):
        super().__init__()

        self.cde = NeuralCDE(input_dim, hidden_dim)

        self.compress = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        h = self.cde(x)          # [B, hidden_dim]
        h = self.compress(h)    # [B, out_dim]
        return h

