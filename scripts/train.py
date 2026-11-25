"""预训练 Tuna：一整晚，从乱码到会说话。

用法：python scripts/train.py --data <数据目录> --tokenizer <tokenizer.json> --out <输出目录> [--resume]
日志：<out>/log.jsonl，一行一条（train / eval / sample），动画直接读它。
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tuna.model import Tuna, TunaConfig, save  # noqa: E402

PROMPTS = ["从前有座山，", "中国的首都是", "人工智能是", "我最喜欢的动物是"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokens", type=float, default=1.2e9, help="一共训练多少 token")
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--batch", type=int, default=16, help="每次放进显卡的句子数")
    p.add_argument("--accum", type=int, default=4, help="攒几次梯度再更新一次")
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--sample_every", type=int, default=1000)
    p.add_argument("--ckpt_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


class Batches:
    """从 token 磁带上随机剪下一段段 seq+1 长的片段：前 seq 个是输入，后 seq 个是答案。"""
    def __init__(self, path, seq, batch, device):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq, self.batch, self.device = seq, batch, device

    def __call__(self, rng):
        starts = rng.integers(0, len(self.data) - self.seq - 1, self.batch)
        chunk = np.stack([self.data[s:s + self.seq + 1] for s in starts]).astype(np.int64)
        t = torch.from_numpy(chunk).pin_memory().to(self.device, non_blocking=True)
        return t[:, :-1], t[:, 1:]


def lr_at(step, total, a):
    """先预热（慢慢加大步子），再按余弦曲线慢慢收小。"""
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    p = min(1.0, (step - a.warmup) / max(1, total - a.warmup))
    return a.lr * (a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


class Gpu:
    def __init__(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv, self.h = pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self.nv = None

    def read(self):
        if not self.nv:
            return {}
        nv, h = self.nv, self.h
        return {"temp": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
                "power_w": round(nv.nvmlDeviceGetPowerUsage(h) / 1000, 1),
                "sm_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)}


@torch.no_grad()
def evaluate(model, val, rng, n=40):
    model.eval()
    losses = []
    for _ in range(n):
        x, y = val(rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses.append(model(x, y)[1].item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def samples(model, tok, step):
    torch.manual_seed(1234)          # 每次用同一个种子，方便对比"长大"
    rows = []
    for p in PROMPTS:
        ids = torch.tensor([tok.encode(p).ids], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(ids, max_new=60, temperature=0.8, top_k=50,
                                 eos_id=tok.token_to_id("<|endoftext|>"))
        rows.append({"kind": "sample", "step": step, "prompt": p, "text": tok.decode(out[0].tolist())})
    model.train()
    return rows


def main():
    a = get_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(os.path.join(a.out, "ckpt"), exist_ok=True)
    tok = Tokenizer.from_file(a.tokenizer)
    train = Batches(os.path.join(a.data, "train.bin"), a.seq, a.batch, "cuda")
    val = Batches(os.path.join(a.data, "val.bin"), a.seq, a.batch, "cuda")

    model = Tuna(TunaConfig(vocab_size=tok.get_vocab_size())).cuda()
    # 二维的权重矩阵做权重衰减；一维的（RMSNorm 的缩放）不做
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=True)
    step_tokens = a.seq * a.batch * a.accum
    total = int(a.tokens // step_tokens)
    step = 0
    last = os.path.join(a.out, "last.pt")
    if a.resume and os.path.exists(last):
        ck = torch.load(last, map_location="cuda", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"从第 {step} 步继续")
    fast = model if a.no_compile else torch.compile(model)

    log = open(os.path.join(a.out, "log.jsonl"), "a")
    def write(row):
        log.write(json.dumps(row, ensure_ascii=False) + "\n")
        log.flush()

    if step == 0:
        write({"kind": "start", "params": model.num_params(), "total_steps": total,
               "step_tokens": step_tokens, "args": vars(a)})
        for r in samples(model, tok, 0):
            write(r)
    print(f"参数 {model.num_params() / 1e6:.1f}M，共 {total} 步，每步 {step_tokens} token")

    gpu = Gpu()
    rng = np.random.default_rng(step)          # 续训时换一个种子，不重复抽同样的片段
    vrng = np.random.default_rng(0)
    t0, tok_seen = time.time(), 0
    while step < total:
        lr = lr_at(step, total, a)
        for g in opt.param_groups:
            g["lr"] = lr
        loss_sum = 0.0
        for _ in range(a.accum):
            x, y = train(rng)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = fast(x, y)
            (loss / a.accum).backward()
            loss_sum += loss.detach()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # 防止一步迈太大
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tok_seen += step_tokens

        if step % a.log_every == 0:
            dt = time.time() - t0
            row = {"kind": "train", "step": step, "tokens": step * step_tokens,
                   "loss": round(loss_sum.item() / a.accum, 4), "lr": lr,
                   "grad_norm": round(gnorm.item(), 3), "tok_s": round(tok_seen / dt),
                   "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   "time": round(time.time()), **gpu.read()}
            write(row)
            eta = (total - step) * step_tokens / max(1, row["tok_s"]) / 3600
            print(f"step {step}/{total} loss {row['loss']:.3f} lr {lr:.2e} "
                  f"{row['tok_s']:,} tok/s {row.get('temp', '?')}°C ETA {eta:.1f}h", flush=True)
            t0, tok_seen = time.time(), 0
        if step % a.eval_every == 0 or step == total:
            vl = evaluate(fast, val, vrng)
            write({"kind": "eval", "step": step, "val_loss": round(vl, 4),
                   "ppl": round(math.exp(vl), 2)})
            print(f"  验证 loss {vl:.3f}（困惑度 {math.exp(vl):.1f}）", flush=True)
        if step % a.sample_every == 0 or step == total:
            for r in samples(model, tok, step):
                write(r)
                print("  ", r["text"].replace("\n", " ")[:80], flush=True)
        if step % a.ckpt_every == 0 or step == total:
            save(model, os.path.join(a.out, "ckpt", f"step_{step:06d}.pt"), step=step)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, last + ".tmp")
            os.replace(last + ".tmp", last)
            t0, tok_seen = time.time(), 0       # 存盘时间不算进吞吐
    save(model, os.path.join(a.out, "tuna-base.pt"), step=step)
    write({"kind": "done", "step": step, "time": round(time.time())})


if __name__ == "__main__":
    main()
