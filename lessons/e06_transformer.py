"""E06 · 搭好整个 Transformer：用正式的 Tuna 结构（缩小版 12M）写诗，再做三个"拆零件"实验（RoPE、残差、RMSNorm）。

用法：python lessons/e06_transformer.py <tang.txt> <tokenizer.json> <输出目录>
"""
import json, os, sys, types

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import poem_tokens, train  # noqa: E402
from tuna.model import Block, Tuna, TunaConfig  # noqa: E402

CTX = 128


class Wrap(torch.nn.Module):
    """课上统一用 (logits, loss) 的接口。"""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x, y=None):
        return (self.m(x), None) if y is None else self.m(x, y)


def small(vocab):
    return Tuna(TunaConfig(vocab_size=vocab, dim=384, n_layers=6, n_heads=6, n_kv_heads=2,
                           ffn_dim=1024, max_seq_len=CTX))


def poems(model, tok, prompts, seed=0):
    torch.manual_seed(seed)
    eos = tok.token_to_id("<|endoftext|>")
    out = []
    for p in prompts:
        ids = torch.tensor([tok.encode(p).ids], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g = model.generate(ids, max_new=40, temperature=0.7, top_k=30, eos_id=eos)
        out.append(tok.decode(g[0].tolist()).replace("<|endoftext|>", ""))
    return out


def main():
    tang, tok_path, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    tok, tr, va = poem_tokens(tang, tok_path)
    V = tok.get_vocab_size()
    log = open(os.path.join(out, "e06_curve.jsonl"), "w")
    steps = int(os.environ.get("E06_STEPS", 2500))

    model = small(V).cuda()
    print(f"小号 Tuna：{model.num_params() / 1e6:.1f}M 参数")
    train(Wrap(model), tr, va, CTX, steps=steps, bs=64, lr=1e-3, every=500, log=log, label="tuna-small")
    for p in poems(model, tok, ["床前明月光，", "春眠不觉晓，", "大漠孤烟直，", "两个黄鹂鸣翠柳，"]):
        print("  ", p)

    # 拆零件 1：拿掉 RoPE（位置信息）——"狗咬人"和"人咬狗"还分得清吗？
    def no_rope(self, x, cos, sin):
        return Block.forward(self, x, torch.ones_like(cos), torch.zeros_like(sin))
    m2 = small(V).cuda()
    for b in m2.layers:
        b.forward = types.MethodType(no_rope, b)
    train(Wrap(m2), tr, va, CTX, steps=steps, bs=64, lr=1e-3, every=500, log=log, label="no-rope")

    # 拆零件 2：拿掉残差（不许"抄近道"）：同样 6 层
    def no_residual(self, x, cos, sin):
        x = self.attn(self.attn_norm(x), cos, sin)
        return self.ffn(self.ffn_norm(x))
    m3 = small(V).cuda()
    for b in m3.layers:
        b.forward = types.MethodType(no_residual, b)
    train(Wrap(m3), tr, va, CTX, steps=steps, bs=64, lr=1e-3, every=500, log=log, label="no-residual")

    # 拆零件 3：拿掉 RMSNorm（不把数缩回同一个大小）
    def no_norm(self, x, cos, sin):
        x = x + self.attn(x, cos, sin)
        return x + self.ffn(x)
    m4 = small(V).cuda()
    for b in m4.layers:
        b.forward = types.MethodType(no_norm, b)
    m4.norm = torch.nn.Identity()
    train(Wrap(m4), tr, va, CTX, steps=steps, bs=64, lr=1e-3, every=500, log=log, label="no-norm")

    # 能区分"狗咬人。"和"人咬狗。"吗？（最后一个字相同、用的字也相同，只有顺序不同）比较最后一个位置的输出向量
    for name, m in [("有 RoPE", model), ("没 RoPE", m2)]:
        m.eval()
        with torch.no_grad():
            a = m(torch.tensor([tok.encode("狗咬人。").ids], device="cuda"))[0, -1]
            b = m(torch.tensor([tok.encode("人咬狗。").ids], device="cuda"))[0, -1]
        print(f"{name}：狗咬人。vs 人咬狗。输出差异 {torch.dist(a.float(), b.float()).item():.3f}")
    json.dump({"params": model.num_params()}, open(os.path.join(out, "meta.json"), "w"))
    torch.save(model.state_dict(), os.path.join(out, "tuna-poet.pt"))


if __name__ == "__main__":
    main()
