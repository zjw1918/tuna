"""课上几期共用：把全唐诗分成 token，切训练 / 验证集，一个小小的训练循环。"""
import json, math, time

import torch
from tokenizers import Tokenizer


def poem_tokens(tang_path, tok_path):
    """每首诗只取正文，诗与诗之间用 <|endoftext|> 隔开。最后 5% 的诗留作验证集（模型没见过）。"""
    tok = Tokenizer.from_file(tok_path)
    eos = tok.token_to_id("<|endoftext|>")
    bodies = [line.split("｜")[2] for line in open(tang_path) if line.count("｜") == 2]
    ids = []
    for enc in tok.encode_batch(bodies):
        ids += enc.ids + [eos]
    ids = torch.tensor(ids)
    cut = int(len(ids) * 0.95)
    return tok, ids[:cut], ids[cut:]


def batch(data, ctx, n, device="cuda", gen=None):
    i = torch.randint(0, len(data) - ctx - 1, (n,), generator=gen)
    x = torch.stack([data[j:j + ctx] for j in i])
    y = torch.stack([data[j + 1:j + ctx + 1] for j in i])
    return x.to(device), y.to(device)


@torch.no_grad()
def val_loss(model, data, ctx, n=20, bs=64):
    model.eval()
    gen = torch.Generator().manual_seed(0)       # 每次都考同一批题；用自己的随机数，不打乱训练的抽样
    out = sum(model(*batch(data, ctx, bs, gen=gen))[1].item() for _ in range(n)) / n
    model.train()
    return out


def train(model, train_data, val_data, ctx, steps, bs=64, lr=1e-3, every=200, log=None, label="", on_eval=None):
    """训练并记录 loss 曲线。返回 [{step, train, val}]。on_eval(step) 在每次验证后调用（E04 用它拍词向量快照）。"""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    curve, t0 = [], time.time()
    for step in range(steps + 1):
        x, y = batch(train_data, ctx, bs)
        lr_now = lr * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g["lr"] = lr_now
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % every == 0:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = val_loss(model, val_data, ctx)
            row = {"label": label, "step": step, "train": round(loss.item(), 4), "val": round(v, 4),
                   "sec": round(time.time() - t0, 1)}
            curve.append(row)
            print(f"{label} 第 {step:5d} 步  训练 {row['train']:.3f}  验证 {row['val']:.3f}  {row['sec']}s", flush=True)
            if log:
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
            if on_eval:
                on_eval(step)
    return curve


@torch.no_grad()
def generate(model, tok, prompt, ctx, n=48, temperature=0.8, seed=0):
    """通用的生成：每一步只看最后 ctx 个 token。"""
    torch.manual_seed(seed)
    model.eval()
    ids = torch.tensor([tok.encode(prompt).ids], device="cuda")
    for _ in range(n):
        logits = model(ids[:, -ctx:])[0][:, -1]      # 课上的模型都返回 (logits, loss)
        probs = torch.softmax(logits.float() / temperature, -1)
        ids = torch.cat([ids, torch.multinomial(probs, 1)], 1)
    model.train()
    return tok.decode(ids[0].tolist())
