"""E04 · 嵌入：给每个 token 一串数字当"意思"，用 MLP 看前 8 个 token 猜下一个。在全唐诗上训练。

用法：python lessons/e04_mlp.py <tang.txt> <tokenizer.json> <输出目录>
输出：e04_curve.jsonl（loss 曲线，动画用）、embed.json（训练中四张词向量快照，64 维）
"""
import json, os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from common import generate, poem_tokens, train  # noqa: E402

CTX = 8


class MLP(nn.Module):
    def __init__(self, vocab, dim=64, hidden=512):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)                 # 词嵌入：每个 token 一行 64 个数
        self.hidden = nn.Linear(CTX * dim, hidden)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, x, y=None):
        # x: [B, T]。对每个位置，都看它自己和前面一共 8 个 token（不够就补 0 号）
        B, T = x.shape
        pad = F.pad(x, (CTX - 1, 0))
        win = pad.unfold(1, CTX, 1)                           # [B, T, 8]
        h = torch.tanh(self.hidden(self.embed(win).view(B, T, -1)))
        logits = self.out(h)
        loss = None if y is None else F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
        return logits, loss


def whole(tok, i):
    """这个 token 是不是完整的字或词（不是半个字的字节）。"""
    s = tok.decode([int(i)])
    return bool(s.strip()) and "\ufffd" not in s


def longest_copy(gen, text):
    """生成的话里，最长的一段原样出现在 text 里的文字。"""
    best = ""
    for i in range(len(gen)):
        j = i + len(best) + 1
        while j <= len(gen) and gen[i:j] in text:
            best, j = gen[i:j], j + 1
    return best


def main():
    tang, tok_path, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    tok, tr, va = poem_tokens(tang, tok_path)
    V = tok.get_vocab_size()
    print(f"全唐诗：训练 {len(tr):,} token，验证 {len(va):,} token")
    counts = torch.bincount(tr, minlength=V)
    ok = torch.tensor([whole(tok, i) for i in range(V)])
    print(f"其中半个字的字节占 {(counts[~ok].sum() / counts.sum()).item():.1%}（词表是在现代文章上训的）")

    model = MLP(V).cuda()
    print(f"参数：{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    log = open(os.path.join(out, "e04_curve.jsonl"), "w")

    # 诗里最常见的 300 个汉字（不含标点）：找"最像"和画星空都只用它们；训练途中拍几张词向量快照
    han = torch.tensor([tok.decode([i]).isalpha() for i in range(V)])
    common = (counts * (ok & han)).topk(300).indices
    snaps = {}

    def shoot(step):
        if step in (0, 500, 1500, 6000):
            snaps[step] = model.embed.weight.detach().float()[common].cpu()

    # 正常训练
    train(model, tr, va, 64, steps=6000, bs=128, lr=2e-3, every=500, log=log, label="mlp", on_eval=shoot)
    for k, p in enumerate(["床前明月光，", "春眠不觉晓，", "大漠孤烟直，"]):
        print(generate(model, tok, p, 64, seed=k))

    # 词向量：谁和它最像？（点积，先把每个向量缩成一样长）
    E = F.normalize(snaps[6000], dim=1)
    names = [tok.decode([int(i)]) for i in common]
    for w in ["春", "秋", "山", "三", "红", "月"]:
        i = names.index(w)
        sim = E @ E[i]
        sim[i] = -1
        print(f"和「{w}」最像：", " ".join(f"{names[j]}{sim[j]:.2f}" for j in sim.topk(6).indices.tolist()))

    # 存下四张快照（每个字 64 个数，缩成一样长），"词向量星空"动画用它画图
    json.dump({"tokens": names, "count": counts[common].tolist(),
               "vectors": {s: [[round(v, 4) for v in r] for r in F.normalize(Y, dim=1).tolist()] for s, Y in snaps.items()}},
              open(os.path.join(out, "embed.json"), "w"), ensure_ascii=False)

    # 翻车：小数据 + 大模型 + 训很久 → 过拟合（训练 loss 往下，验证 loss 往上）
    small = tr[:200_000]
    big = MLP(V, dim=128, hidden=2048).cuda()
    print(f"过拟合组：{len(small):,} token，参数 {sum(p.numel() for p in big.parameters()) / 1e6:.1f}M")
    train(big, small, va, 64, steps=6000, bs=128, lr=2e-3, every=500, log=log, label="overfit")
    small_text = tok.decode(small.tolist())
    starts = (small == tok.token_to_id("<|endoftext|>")).nonzero().flatten().tolist()
    seen = [tok.decode(small[starts[k] + 1:starts[k + 1]].tolist())[:6] for k in (100, 400, 900)]
    for p in seen + ["床前明月光，", "大漠孤烟直，"]:
        g = generate(big, tok, p, 64, n=40)
        tag = "见过" if p in seen else "没见过"
        print(f"[{tag}] {g}")
        print(f"  和训练集一字不差的最长一段：{len(longest_copy(g[len(p):], small_text))} 个字")


if __name__ == "__main__":
    main()
