"""E14 实验：蜡笔越少，画越糊吗？

把对话 Tuna 的 Transformer 块权重压成 8 / 4 / 3 / 2 位，每种都考一次：大小、验证集困惑度、三个问题的回答。
"分组"：不是一整行共用一个缩放系数，而是每 32 个数一组、各用各的（GGUF 的 q8_0 / q4_0 就是这样分组的）。

这里用"假量化"：压成整数再立刻乘回缩放系数，结果和真的按整数存一模一样，但不用为每种位数写专门的存储格式。
大小按"真的打包存起来"计算：每个数占 bits 位，每组再加一个 16 位的缩放系数；词嵌入保持原样。

用法：python lessons/e14_quant.py <tuna-chat.pt> <tokenizer.json> <预训练数据目录> [--out e14-quant.json]
"""
import argparse, json, os, sys

import torch
import torch.nn as nn
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from tuna.infer import prompt_ids, stream  # noqa: E402
from tuna.model import load  # noqa: E402
from eval import perplexity  # noqa: E402

QUESTIONS = ["中国的首都是哪里？", "介绍一下你自己。", "天空为什么是蓝色的？"]


def fake_quant(w, bits, group=None):
    """把 w 的每一行（或每 group 个数）压成 bits 位的整数，再乘回去。"""
    qmax = 2 ** (bits - 1) - 1                      # 8 位：-127…127；4 位：-7…7；2 位：-1…1
    rows, cols = w.shape
    g = w.view(rows, cols // group, group) if group else w.view(rows, 1, cols)
    scale = g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.round(g / scale).clamp(-qmax, qmax)
    return (q * scale).view(rows, cols)


def quantize_blocks(model, bits, group=None):
    n = groups = 0
    for block in model.layers:
        for parent in (block.attn, block.ffn):
            for child in parent.children():
                if isinstance(child, nn.Linear):
                    child.weight.data = fake_quant(child.weight.data.float(), bits, group).to(child.weight.dtype)
                    n += child.weight.numel()
                    rows, cols = child.weight.shape
                    groups += rows * (cols // group if group else 1)
    return n, groups


def size_mb(model, n_quant, groups, bits):
    """真的打包存起来有多大：量化的数每个 bits 位 + 每组一个 16 位缩放系数，其余参数照原样。"""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    if bits:
        total += -n_quant * 4 + n_quant * bits / 8 + groups * 2
    return total / 2**20


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("tokenizer")
    p.add_argument("val")
    p.add_argument("--out", default="e14-quant.json")
    a = p.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    rows = []
    for bits, group in [(None, None), (8, None), (4, None), (4, 32), (3, 32), (2, 32)]:
        model, _ = load(a.model)
        n, groups = quantize_blocks(model, bits, group) if bits else (0, 0)
        model.eval()
        name = "原样（32 位小数）" if not bits else f"{bits} 位" + (f"，每 {group} 个一组" if group else "，每行一组")
        ppl = perplexity(model, os.path.join(a.val, "val.bin"))
        answers = []
        for q in QUESTIONS:
            torch.manual_seed(0)
            answers.append("".join(stream(model, tok, prompt_ids(tok, [("user", q)]), max_new=60, temperature=0)))
        row = {"name": name, "bits": bits, "group": group, "mb": round(size_mb(model, n, groups, bits), 1),
               "ppl": round(ppl, 2), "answers": dict(zip(QUESTIONS, answers))}
        rows.append(row)
        print(f"{name:16s} {row['mb']:6.1f} MB  困惑度 {row['ppl']:8.2f}  {answers[0][:40]!r}")
    json.dump(rows, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
