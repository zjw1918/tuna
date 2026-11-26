"""LoRA：主干冻住不动，在旁边挂两条细长的小矩阵 A、B，只训练它们。

原来：y = W x            （W 是 640×640 的大矩阵，冻住）
现在：y = W x + (B A x) × scale   （A 是 r×640，B 是 640×r，r 只有 8）
"""
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(r, base.in_features) / base.in_features ** 0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # B 从 0 开始：一开始完全不改变模型
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


TARGETS = ("wq", "wk", "wv", "wo", "w1", "w2", "w3")


def add_lora(model, r=8, alpha=16, targets=TARGETS):
    for p in model.parameters():
        p.requires_grad_(False)
    for block in model.layers:
        for parent in (block.attn, block.ffn):
            for name in targets:
                if hasattr(parent, name):
                    setattr(parent, name, LoRALinear(getattr(parent, name), r, alpha).to(
                        getattr(parent, name).weight.device))
    return [p for n, p in model.named_parameters() if n.endswith((".A", ".B"))]


def lora_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if n.endswith((".A", ".B"))}


def load_lora(model, path):
    ck = torch.load(path, map_location="cuda", weights_only=False)
    add_lora(model, ck["r"], ck["alpha"])
    missing = model.load_state_dict(ck["lora"], strict=False)
    assert not missing.unexpected_keys, missing.unexpected_keys
    return model
