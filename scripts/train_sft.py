"""指令微调（SFT）：从会续写的 Tuna-base 出发，教它一问一答。

和预训练只差两件事：数据是对话，只有"Tuna 说的话"计分（其余位置的答案填 -100，不算 loss）。
用法：python scripts/train_sft.py --base <tuna-base.pt> --data <sft 目录> --tokenizer <tokenizer.json> --out <输出目录>
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tuna.model import load, save  # noqa: E402

QUESTIONS = ["你是谁？", "中国的首都是哪里？", "给我讲一个关于小猫的笑话。", "用三句话介绍一下太阳系。"]


class SftData:
    def __init__(self, d):
        self.ids = np.memmap(os.path.join(d, "ids.bin"), dtype=np.uint16, mode="r")
        self.mask = np.memmap(os.path.join(d, "mask.bin"), dtype=np.uint8, mode="r")
        self.off = np.load(os.path.join(d, "offsets.npy"))
        self.lens = np.diff(self.off)

    def __len__(self):
        return len(self.lens)

    def epoch(self, rng, max_tokens):
        """长度相近的放一批，少填空白；每批最多 max_tokens 个 token（含填充）。"""
        order = rng.permutation(len(self))
        chunks = [order[i:i + 4096] for i in range(0, len(order), 4096)]
        batches = []
        for ch in chunks:
            ch = ch[np.argsort(self.lens[ch])]
            cur = []
            for i in ch:
                if cur and (len(cur) + 1) * max(self.lens[cur[-1]], self.lens[i]) > max_tokens:
                    batches.append(cur)
                    cur = []
                cur.append(i)
            if cur:
                batches.append(cur)
        rng.shuffle(batches)
        return batches

    def collate(self, idx, no_mask=False):
        T = int(self.lens[idx].max())
        x = np.zeros((len(idx), T), np.int64)
        y = np.full((len(idx), T), -100, np.int64)
        for r, i in enumerate(idx):
            a, b = self.off[i], self.off[i + 1]
            ids = self.ids[a:b].astype(np.int64)
            m = np.ones(b - a, bool) if no_mask else self.mask[a:b].astype(bool)   # --no_mask：每个字都计分（翻车演示）
            x[r, :b - a] = ids
            tgt = np.where(m[1:], ids[1:], -100)       # 位置 t 的答案是 t+1；只有计分的位置保留
            y[r, :b - a - 1] = tgt
        return (torch.from_numpy(x).pin_memory().cuda(non_blocking=True),
                torch.from_numpy(y).pin_memory().cuda(non_blocking=True))


def chat_prompt(tok, q):
    return tok.encode(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n").ids


@torch.no_grad()
def answers(model, tok, step):
    torch.manual_seed(1234)
    rows = []
    for q in QUESTIONS:
        ids = torch.tensor([chat_prompt(tok, q)], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(ids, max_new=120, temperature=0.7, top_k=40,
                                 eos_id=tok.token_to_id("<|im_end|>"), repetition_penalty=1.1)
        text = tok.decode(out[0, ids.shape[1]:].tolist(), skip_special_tokens=True)
        rows.append({"kind": "sample", "step": step, "prompt": q, "text": text})
    model.train()
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=4096, help="每小批最多多少 token（含填充）")
    p.add_argument("--accum", type=int, default=2, help="攒几小批的梯度再更新一次")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--sample_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--no_mask", action="store_true", help="不加损失掩码：用户的话也计分（翻车演示）")
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(a.out, exist_ok=True)
    tok = Tokenizer.from_file(a.tokenizer)
    model, _ = load(a.base)
    model.train()
    data = SftData(a.data)
    rng = np.random.default_rng(0)
    batches = data.epoch(rng, a.max_tokens)
    total = int(len(batches) * a.epochs) // a.accum                   # 更新次数
    batches = (batches * math.ceil(a.epochs))[:total * a.accum]
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    write = lambda r: (log.write(json.dumps(r, ensure_ascii=False) + "\n"), log.flush())
    write({"kind": "start", "conversations": len(data), "steps": total, "args": vars(a)})
    for r in answers(model, tok, 0):
        write(r)
        print(r["prompt"], "→", r["text"][:80].replace("\n", " "))
    print(f"{len(data):,} 段对话，{total:,} 步")

    t0, seen, scored = time.time(), 0, 0
    for step in range(1, total + 1):
        lr = a.lr * min(1, step / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / total)))
        for g in opt.param_groups:
            g["lr"] = lr
        for idx in batches[(step - 1) * a.accum:step * a.accum]:
            x, y = data.collate(np.array(idx), a.no_mask)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
            (loss / a.accum).backward()
            seen += x.numel()
            scored += int((y != -100).sum())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % a.log_every == 0:
            dt = time.time() - t0
            row = {"kind": "train", "step": step, "loss": round(loss.item(), 4), "lr": lr,
                   "tok_s": round(seen / dt), "scored_frac": round(scored / seen, 3),
                   "mem_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
            write(row)
            print(f"step {step}/{total} loss {row['loss']:.3f} {row['tok_s']:,} tok/s "
                  f"计分占比 {row['scored_frac']:.0%} ETA {(total - step) * dt / a.log_every / 3600:.1f}h", flush=True)
            t0, seen, scored = time.time(), 0, 0
        if step % a.sample_every == 0 or step == total:
            for r in answers(model, tok, step):
                write(r)
                print("  ", r["prompt"], "→", r["text"][:100].replace("\n", " "), flush=True)
    save(model, os.path.join(a.out, "tuna-chat.pt"), step=total)
    write({"kind": "done", "time": round(time.time())})


if __name__ == "__main__":
    main()
