"""Tuna-V：借来一双眼睛（冻结的 SigLIP2），配一个自己训练的"翻译官"（投影层），接到 Tuna 的大脑上。

一张图 → SigLIP2 → 16×16=256 个图块向量（768 维）→ 2×2 合并成 64 个 → 投影层 → 64 个 640 维的"图片词"
对话里的 64 个 <|image|> 占位符，换成这 64 个图片词。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from tuna.model import Tuna

ENCODER = "google/siglip2-base-patch16-256"
N_IMG = 64          # 每张图占 64 个位置
ENC_DIM = 768


def pool_features(h):
    """[B, 256, 768] → [B, 64, 768]：相邻的 2×2 个图块取平均。"""
    B, N, D = h.shape
    s = int(N ** 0.5)
    h = h.view(B, s, s, D).permute(0, 3, 1, 2)
    return F.avg_pool2d(h, 2).flatten(2).transpose(1, 2)


class Eye:
    """借来的眼睛：只负责看，不训练。"""
    def __init__(self, device="cuda"):
        from transformers import AutoModel, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(ENCODER)
        self.model = AutoModel.from_pretrained(ENCODER, dtype=torch.float16).vision_model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, images):
        px = self.proc(images=images, return_tensors="pt")["pixel_values"].to(self.device, torch.float16)
        return pool_features(self.model(pixel_values=px).last_hidden_state)


class Projector(nn.Module):
    """翻译官：把"眼睛的语言"（768 维）翻译成"Tuna 的语言"（640 维）。"""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ENC_DIM, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, f):
        return self.net(f)


class TunaV(nn.Module):
    def __init__(self, llm: Tuna, image_id: int):
        super().__init__()
        self.llm, self.image_id = llm, image_id
        self.proj = Projector(llm.cfg.dim)

    def embed(self, ids, feats):
        """先查词嵌入，再把 <|image|> 位置换成图片词。feats: [B, 64, 768]"""
        x = self.llm.embed(ids)
        if feats is not None:
            slots = ids == self.image_id
            x = x.clone()
            x[slots] = self.proj(feats.to(x.dtype)).reshape(-1, x.shape[-1]).to(x.dtype)
        return x

    def forward(self, ids, feats, targets=None):
        return self.llm(ids, targets, embeds=self.embed(ids, feats))


def image_prompt(question):
    return f"<|im_start|>user\n{'<|image|>' * N_IMG}\n{question}<|im_end|>\n<|im_start|>assistant\n"
