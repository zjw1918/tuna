"""E05 · 注意力：每个 token 自己决定该看前面的谁。先忘了加掩码（翻车），再加上修好。

模型 = 上期的 MLP 换掉"按座位接成一长条"：嵌入 + 位置 → 一层多头注意力 → 上期的 tanh 隐藏层 → 输出层。
用法：python lessons/e05_attention.py <tang.txt> <tokenizer.json> <输出目录>
输出：e05_curve.jsonl（loss 曲线）、attention.json（一首诗的注意力图，动画用）、e05.pt
"""
import json, math, os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from common import generate, poem_tokens, train  # noqa: E402

CTX = 64


class Attention(nn.Module):
    def __init__(self, dim, heads, causal=True):
        super().__init__()
        self.heads, self.causal = heads, causal
        self.q = nn.Linear(dim, dim, bias=False)   # Query：我在找什么？
        self.k = nn.Linear(dim, dim, bias=False)   # Key：我身上贴着什么标签？
        self.v = nn.Linear(dim, dim, bias=False)   # Value：被看中的话，我交出什么内容？
        self.o = nn.Linear(dim, dim, bias=False)   # 几个头收集到的东西，合成一串
        self.last_weights = None

    def forward(self, x):
        B, T, D = x.shape
        hd = D // self.heads                                   # 每个头分到的维数
        q, k, v = (f(x).view(B, T, self.heads, hd).transpose(1, 2) for f in (self.q, self.k, self.v))
        score = q @ k.transpose(-2, -1) / math.sqrt(hd)        # 每对 token 有多"对得上"，再缩小 √hd 倍
        if self.causal:                                        # 掩码：不许偷看后面的字
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
            score = score.masked_fill(mask, float("-inf"))
        w = score.softmax(-1)                                  # 注意力分配表：每一行加起来是 1
        self.last_weights = w.detach()
        return self.o((w @ v).transpose(1, 2).reshape(B, T, D))


class AttnLM(nn.Module):
    def __init__(self, vocab, dim=256, heads=4, hidden=512, causal=True):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)                  # 上期的嵌入表
        self.pos = nn.Embedding(CTX, dim)                      # 位置表：第 0～63 个位置，各学一串数
        self.attn = Attention(dim, heads, causal)
        self.hidden = nn.Linear(dim, hidden)                   # 上期的 512 个侦探
        self.out = nn.Linear(hidden, vocab)

    def forward(self, x, y=None):
        T = x.shape[1]
        h = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        h = h + self.attn(h)                                   # 看完别人收集到的，加到自己身上
        logits = self.out(torch.tanh(self.hidden(h)))
        loss = None if y is None else F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
        return logits, loss


def main():
    tang, tok_path, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    tok, tr, va = poem_tokens(tang, tok_path)
    V = tok.get_vocab_size()
    log = open(os.path.join(out, "e05_curve.jsonl"), "w")

    # 翻车：忘了加掩码。loss 掉得飞快——因为答案就在眼前
    cheat = AttnLM(V, causal=False).cuda()
    train(cheat, tr, va, CTX, steps=1500, bs=128, lr=2e-3, every=250, log=log, label="no-mask")
    print("没掩码的生成：", generate(cheat, tok, "床前明月光，", CTX))

    # 修好：加上掩码。步数、批大小、学习率都和上期的 MLP 一样，好比较
    model = AttnLM(V).cuda()
    print(f"参数：{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    train(model, tr, va, CTX, steps=6000, bs=128, lr=2e-3, every=500, log=log, label="attention")
    for k, p in enumerate(["床前明月光，", "春眠不觉晓，", "大漠孤烟直，"]):
        print(generate(model, tok, p, CTX, seed=k))

    # 注意力图：每个头，在一首诗上
    poem = "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
    ids = torch.tensor([tok.encode(poem).ids], device="cuda")
    model.eval()
    with torch.no_grad():
        model(ids)
    toks = [tok.decode([i]) for i in ids[0].tolist()]
    maps = {}
    for H in range(model.attn.heads):
        w = model.attn.last_weights[0, H].float().cpu()
        maps[H] = [[round(v, 3) for v in r] for r in w.tolist()]
        i = toks.index("乡")
        top = w[i].topk(3)
        print(f"第 {H} 个头，「乡」最关注：", [(toks[j], round(v, 2)) for v, j in zip(top.values.tolist(), top.indices.tolist())])
    json.dump({"tokens": toks, "maps": maps}, open(os.path.join(out, "attention.json"), "w"), ensure_ascii=False)
    torch.save(model.state_dict(), os.path.join(out, "e05.pt"))


if __name__ == "__main__":
    main()
