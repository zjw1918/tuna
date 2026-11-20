"""Tuna：一个 Llama 式的小语言模型（约 56M 参数）。

结构：词嵌入 → 12 个 Transformer 块（RMSNorm、带 RoPE 的分组查询注意力、SwiGLU）→ RMSNorm → 输出层（和词嵌入共享权重）。
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TunaConfig:
    vocab_size: int = 6400
    dim: int = 640          # 每个 token 的向量有多长
    n_layers: int = 12      # 叠几层 Transformer 块
    n_heads: int = 10       # 注意力头数（每头 64 维）
    n_kv_heads: int = 2     # K、V 只要 2 组，5 个查询头共用一组（GQA）
    ffn_dim: int = 1728     # 前馈层中间有多宽（约 dim × 8/3，取 64 的倍数）
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    @property
    def head_dim(self):
        return self.dim // self.n_heads


def rope_tables(cfg: TunaConfig, device=None):
    """RoPE：位置 p 的每一对维度旋转 p × θ_i 角。先算好 cos、sin 表。"""
    i = torch.arange(0, cfg.head_dim, 2, device=device).float()
    theta = cfg.rope_theta ** (-i / cfg.head_dim)               # 每对维度一个转速
    pos = torch.arange(cfg.max_seq_len, device=device).float()
    angle = torch.outer(pos, theta)                            # [位置, 对数]
    return angle.cos(), angle.sin()


def apply_rope(x, cos, sin):
    """x: [B, 头, T, head_dim]。把相邻两维看成平面上的点，转一个角度。"""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: TunaConfig):
        super().__init__()
        self.cfg = cfg
        hd = cfg.head_dim
        self.wq = nn.Linear(cfg.dim, cfg.n_heads * hd, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * hd, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * hd, bias=False)
        self.wo = nn.Linear(cfg.n_heads * hd, cfg.dim, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        hd = self.cfg.head_dim
        q = self.wq(x).view(B, T, self.cfg.n_heads, hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.cfg.n_kv_heads, hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.cfg.n_kv_heads, hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return self.wo(out.transpose(1, 2).reshape(B, T, -1))


class FeedForward(nn.Module):
    """SwiGLU：一路算"内容"，一路算"开关"，相乘后再压回 dim 维。"""
    def __init__(self, cfg: TunaConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)   # 开关
        self.w3 = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)   # 内容
        self.w2 = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: TunaConfig):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.ffn = FeedForward(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)   # 残差：在原来的基础上"加一点"
        return x + self.ffn(self.ffn_norm(x))


class Tuna(nn.Module):
    def __init__(self, cfg: TunaConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.output.weight = self.embed.weight                  # 输入、输出共用一张词表
        cos, sin = rope_tables(cfg)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        for name, p in self.named_parameters():                 # 残差分支的出口初始化得小一点
            if name.endswith(("wo.weight", "w2.weight")):
                nn.init.normal_(p, std=0.02 / (2 * cfg.n_layers) ** 0.5)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, tokens, targets=None):
        """tokens: [B, T]。"""
        x = self.embed(tokens)
        T = x.shape[1]
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        for layer in self.layers:
            x = layer(x, cos, sin)
        logits = self.output(self.norm(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, tokens, max_new=200, temperature=0.8, top_k=50, eos_id=None):
        """一个字一个字往后猜：每次把整段话重新算一遍，只取最后一个位置的预测。"""
        self.eval()
        out = tokens
        for _ in range(max_new):
            logits = self(out[:, -self.cfg.max_seq_len:])[:, -1].float()
            if temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                kth = torch.topk(logits, top_k).values[:, -1:]          # 只在最像的 k 个字里抽
                logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            out = torch.cat([out, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return out


if __name__ == "__main__":
    m = Tuna(TunaConfig())
    print(f"Tuna 参数量：{m.num_params() / 1e6:.1f}M")
