"""量化：把每个权重从 16 位的小数，压成 8 位的整数（-127 到 127）加一个缩放系数。

像把一盒 65536 色的蜡笔换成 255 色的：画出来几乎一样，盒子小了一半。
"""
import torch
import torch.nn as nn


class Int8Linear(nn.Module):
    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.detach().float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127   # 每一行一个缩放系数
        self.register_buffer("q", torch.round(w / scale).to(torch.int8))
        self.register_buffer("scale", scale.to(torch.bfloat16))

    def forward(self, x):
        return x @ (self.q.to(x.dtype) * self.scale.to(x.dtype)).T          # 用的时候再"解压"


def quantize(model):
    """把所有 Transformer 块里的线性层换成 int8 版。词嵌入（和输出层共享）保持原样。"""
    for block in model.layers:
        for parent in (block.attn, block.ffn):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(parent, name, Int8Linear(child))
    return model


def size_mb(model):
    seen, total = set(), 0
    for t in list(model.parameters()) + list(model.buffers()):
        if t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            total += t.numel() * t.element_size()
    return total / 2**20
