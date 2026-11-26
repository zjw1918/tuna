"""E12 · 机器怎么看图：把图片切成小块，当成一句话读。从零训练一个认涂鸦的小 ViT。

数据：Google Quick, Draw!（CC BY 4.0），10 类，每类 1 万张 28×28 的涂鸦。
用法：python lessons/e12_vit.py <数据目录> <输出目录> [真实照片.jpg ...]
"""
import json, os, sys, time, urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tuna.model import Block, TunaConfig, rope_tables  # noqa: E402

CLASSES = {"cat": "猫", "fish": "鱼", "apple": "苹果", "house": "房子", "tree": "树",
           "car": "汽车", "bicycle": "自行车", "airplane": "飞机", "sun": "太阳", "alarm clock": "闹钟"}
URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"
PER_CLASS, PATCH = 10000, 4


def get_data(d):
    os.makedirs(d, exist_ok=True)
    xs, ys = [], []
    for i, name in enumerate(CLASSES):
        path = os.path.join(d, name.replace(" ", "_") + ".npy")
        if not os.path.exists(path):
            print("下载", name)
            urllib.request.urlretrieve(URL.format(urllib.parse.quote(name)), path)
        arr = np.load(path, mmap_mode="r")[:PER_CLASS]
        xs.append(np.asarray(arr))
        ys.append(np.full(len(arr), i))
    x = torch.from_numpy(np.concatenate(xs)).float().view(-1, 28, 28) / 255
    y = torch.from_numpy(np.concatenate(ys))
    perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(0))
    return x[perm], y[perm]


class ViT(nn.Module):
    """和 Tuna 用的是同一种 Transformer 块，只是：输入是图块，注意力前后都能看，最后输出一个类别。"""
    def __init__(self, n_classes, dim=192, layers=6):
        super().__init__()
        self.cfg = TunaConfig(dim=dim, n_layers=layers, n_heads=dim // 64, n_kv_heads=dim // 64,
                              ffn_dim=dim * 8 // 3 // 64 * 64, max_seq_len=64, causal=False)
        self.patch = nn.Linear(PATCH * PATCH, dim)       # 一个 4×4 的小块 → 一个向量（"图块嵌入"）
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))  # 多放一个"总结"位，最后由它来回答
        self.blocks = nn.ModuleList(Block(self.cfg) for _ in range(layers))
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, n_classes)
        cos, sin = rope_tables(self.cfg)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, img):                                # img: [B, 28, 28]
        B = img.shape[0]
        p = img.unfold(1, PATCH, PATCH).unfold(2, PATCH, PATCH).reshape(B, -1, PATCH * PATCH)  # [B, 49, 16]
        x = torch.cat([self.cls.expand(B, -1, -1), self.patch(p)], 1)
        T = x.shape[1]
        for b in self.blocks:
            x = b(x, self.cos[:T], self.sin[:T])
        return self.head(self.norm(x[:, 0]))


def photo_to_doodle(path):
    """把一张真实照片变成 28×28 的灰度图，像涂鸦一样：白底变黑、线条变亮。"""
    from PIL import Image, ImageFilter, ImageOps
    im = Image.open(path).convert("L").filter(ImageFilter.FIND_EDGES)
    im = ImageOps.autocontrast(im).resize((28, 28))
    return torch.from_numpy(np.array(im)).float() / 255


def main():
    data_dir, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    x, y = get_data(data_dir)
    n_val = 5000
    xv, yv, xt, yt = x[:n_val].cuda(), y[:n_val].cuda(), x[n_val:].cuda(), y[n_val:].cuda()
    model = ViT(len(CLASSES)).cuda()
    print(f"小 ViT：{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M 参数，{len(xt):,} 张训练涂鸦")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    steps, bs, t0 = 3000, 256, time.time()
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    log = open(os.path.join(out, "curve.jsonl"), "w")
    for step in range(steps + 1):
        i = torch.randint(0, len(xt), (bs,), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(xt[i]), yt[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step < steps:
            sched.step()
        if step % 500 == 0:
            model.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                acc = (model(xv).argmax(-1) == yv).float().mean().item()
            model.train()
            log.write(json.dumps({"step": step, "loss": round(loss.item(), 4), "val_acc": round(acc, 4)}) + "\n")
            print(f"第 {step} 步  loss {loss.item():.3f}  验证准确率 {acc:.1%}  {time.time() - t0:.0f}s", flush=True)
    torch.save(model.state_dict(), os.path.join(out, "doodle-vit.pt"))
    names = list(CLASSES.values())
    model.eval()
    for path in sys.argv[3:]:                              # 翻车时间：拿真实照片考它
        with torch.no_grad():
            prob = model(photo_to_doodle(path)[None].cuda()).float().softmax(-1)[0]
        top = prob.topk(3)
        print(os.path.basename(path), "→", [(names[j], f"{v:.0%}") for v, j in zip(top.values.tolist(), top.indices.tolist())])


if __name__ == "__main__":
    main()
