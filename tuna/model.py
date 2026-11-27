"""Tuna：一个 Llama 式的小语言模型（约 56M 参数）。

结构：词嵌入 → 12 个 Transformer 块（RMSNorm、带 RoPE 的分组查询注意力、SwiGLU）→ RMSNorm → 输出层（和词嵌入共享权重）。
"""
from dataclasses import dataclass, asdict

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
    causal: bool = True     # 语言模型只许看前面；看图的 ViT 可以前后都看

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

    def forward(self, x, cos, sin, cache=None):
        B, T, _ = x.shape
        hd = self.cfg.head_dim
        q = self.wq(x).view(B, T, self.cfg.n_heads, hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.cfg.n_kv_heads, hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.cfg.n_kv_heads, hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:                     # 生成时：把旧的 K、V 接在前面
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        # 只有一次处理多个新 token 时才需要因果掩码
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.cfg.causal and T > 1,
                                             enable_gqa=True)
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

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)   # 残差：在原来的基础上"加一点"
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

    def forward(self, tokens, targets=None, caches=None, start=0, embeds=None):
        """tokens: [B, T]。embeds 可直接传入向量（多模态时把图片向量拼进来）。"""
        x = self.embed(tokens) if embeds is None else embeds
        T = x.shape[1]
        cos, sin = self.rope_cos[start:start + T], self.rope_sin[start:start + T]
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, None if caches is None else caches[i])
        logits = self.output(self.norm(x))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1),
                               ignore_index=-100)
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, tokens, max_new=200, temperature=0.8, top_k=50, top_p=0.95,
                 eos_id=None, repetition_penalty=1.0):
        """一个字一个字往后猜。带 KV 缓存：旧 token 的 K、V 只算一次。"""
        self.eval()
        caches = [{} for _ in self.layers]
        logits = self(tokens, caches=caches)[:, -1]
        pos = tokens.shape[1]
        out = tokens
        for _ in range(max_new):
            if pos >= self.cfg.max_seq_len:
                break
            if repetition_penalty != 1.0:
                seen = logits.gather(1, out)
                logits = logits.scatter(1, out, torch.where(seen > 0, seen / repetition_penalty,
                                                            seen * repetition_penalty))
            nxt = sample(logits, temperature, top_k, top_p)
            out = torch.cat([out, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
            logits = self(nxt, caches=caches, start=pos)[:, -1]
            pos += 1
        return out


def sample(logits, temperature=0.8, top_k=50, top_p=0.95):
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits.float() / temperature
    if top_k:
        kth = torch.topk(logits, top_k).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    if top_p < 1.0:
        sp, idx = probs.sort(descending=True)
        drop = sp.cumsum(-1) - sp > top_p
        sp = sp.masked_fill(drop, 0)
        probs = torch.zeros_like(probs).scatter(1, idx, sp)
    return torch.multinomial(probs, 1)


def save(model: Tuna, path, **extra):
    torch.save({"config": asdict(model.cfg), "model": model.state_dict(), **extra}, path)


def load(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = Tuna(TunaConfig(**ck["config"])).to(device)
    sd = {k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    return model, ck


if __name__ == "__main__":
    m = Tuna(TunaConfig())
    print(f"Tuna 参数量：{m.num_params() / 1e6:.1f}M")
