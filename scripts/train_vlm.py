"""训练 Tuna-V，分两个阶段：
  第一阶段（对齐）：Tuna 冻住，只训练翻译官（投影层），任务是"看图说描述"。
  第二阶段（视觉指令微调）：放开 Tuna，和翻译官一起学"回答关于图片的问题"。

用法：
  python scripts/train_vlm.py --stage 1 --llm tuna-chat.pt --data <特征目录> --tokenizer <tok> --out <目录>
  python scripts/train_vlm.py --stage 2 --init <第一阶段的 tuna-v.pt> --data <特征目录> ...
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tuna.infer import stream  # noqa: E402
from tuna.model import Tuna, TunaConfig, load  # noqa: E402
from tuna.vlm import N_IMG, TunaV, image_prompt  # noqa: E402

MAX_LEN = 1024
HOLDOUT = 8


def encode(tok, conv):
    start, end = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")
    nl = tok.encode("\n").ids
    ids, lab = [], []
    for m in conv:
        content = m["content"]
        if "<image>" in content:
            content = "<|image|>" * N_IMG + "\n" + content.replace("<image>", "").strip()
        head = [start] + tok.encode(m["role"] + "\n").ids
        body = tok.encode(content).ids + [end]
        ids += head + body + nl
        lab += [-100] * len(head) + (body if m["role"] == "assistant" else [-100] * len(body)) + [-100] * len(nl)
    return ids[:MAX_LEN], lab[:MAX_LEN]


def load_data(dirs, tok):
    feats, samples, bad = [], [], 0
    for k, d in enumerate(dirs.split(",")):
        n = json.load(open(os.path.join(d, "meta.json")))["n"]
        feats.append(np.load(os.path.join(d, "feats.npy"), mmap_mode="r"))
        for i, line in enumerate(open(os.path.join(d, "convs.jsonl"))):
            if i >= n:
                break
            conv = json.loads(line)
            # 只要"图片只出现一次、而且在提问里"的样本（数据里有几百条回答里也写着 <image>）
            if sum(m["content"].count("<image>") for m in conv) != 1 or "<image>" not in conv[0]["content"]:
                bad += 1
                continue
            samples.append((k, i, conv))
    print(f"{len(samples):,} 条图文样本（去掉格式不对的 {bad} 条），分词中……", flush=True)
    enc = [encode(tok, c) for _, _, c in samples]
    return feats, samples, enc


def save_v(model, path, **extra):
    torch.save({"config": model.llm.cfg.__dict__, "llm": model.llm.state_dict(),
                "proj": model.proj.state_dict(), "image_id": model.image_id, **extra}, path)


def load_v(path, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    llm = Tuna(TunaConfig(**ck["config"])).to(device)
    llm.load_state_dict(ck["llm"])
    m = TunaV(llm, ck["image_id"]).to(device)
    m.proj.load_state_dict(ck["proj"])
    return m


@torch.no_grad()
def describe(model, tok, feats, question="请描述这张图片。", max_new=120):
    model.eval()
    torch.manual_seed(0)
    ids = tok.encode(image_prompt(question)).ids
    x = torch.tensor([ids], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        emb = model.embed(x, torch.from_numpy(np.asarray(feats))[None].cuda())
    text = "".join(stream(model.llm, tok, ids, max_new=max_new, temperature=0.5, embeds=emb))
    model.train()
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, required=True)
    p.add_argument("--llm", help="第一阶段：Tuna-chat 检查点")
    p.add_argument("--init", help="第二阶段：第一阶段的 Tuna-V 检查点")
    p.add_argument("--data", required=True, help="特征目录，可用逗号连多个")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max_tokens", type=int, default=8192)
    p.add_argument("--lr", type=float)
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(a.out, exist_ok=True)
    tok = Tokenizer.from_file(a.tokenizer)
    if a.stage == 1:
        llm, _ = load(a.llm)
        model = TunaV(llm, tok.token_to_id("<|image|>")).cuda()
        for q in llm.parameters():
            q.requires_grad_(False)                       # 大脑冻住
        params = list(model.proj.parameters())
        lr = a.lr or 1e-3
    else:
        model = load_v(a.init)
        params = list(model.parameters())
        lr = a.lr or 1e-4
    model.train()
    print(f"可训练参数：{sum(q.numel() for q in params) / 1e6:.2f}M")

    feats, samples, enc = load_data(a.data, tok)
    hold = list(range(len(samples) - HOLDOUT, len(samples)))
    train_idx = np.arange(len(samples) - HOLDOUT)
    lens = np.array([len(e[0]) for e in enc])
    rng = np.random.default_rng(0)
    batches = []
    for _ in range(math.ceil(a.epochs)):
        order = rng.permutation(train_idx)
        for ch in np.array_split(order, max(1, len(order) // 4096)):
            ch = ch[np.argsort(lens[ch])]
            cur = []
            for i in ch:
                if cur and (len(cur) + 1) * max(lens[cur[-1]], lens[i]) > a.max_tokens:
                    batches.append(cur)
                    cur = []
                cur.append(i)
            if cur:
                batches.append(cur)
    rng.shuffle(batches)
    total = int(len(batches) * a.epochs / math.ceil(a.epochs))
    batches = batches[:total]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0, fused=True)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    write = lambda r: (log.write(json.dumps(r, ensure_ascii=False) + "\n"), log.flush())

    def show(step):
        for j in hold[:4]:
            k, i, conv = samples[j]
            q = conv[0]["content"].replace("<image>", "").strip()
            ans = describe(model, tok, feats[k][i], q)
            write({"kind": "sample", "step": step, "question": q, "answer": ans, "truth": conv[1]["content"]})
            print(f"  问：{q[:30]}\n  答：{ans[:120]!r}", flush=True)

    show(0)
    t0, seen = time.time(), 0
    for step, idx in enumerate(batches, 1):
        cur_lr = lr * min(1, step / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / total)))
        for g in opt.param_groups:
            g["lr"] = cur_lr
        T = int(lens[idx].max())
        x = torch.zeros(len(idx), T, dtype=torch.long)
        y = torch.full((len(idx), T), -100, dtype=torch.long)
        f = torch.from_numpy(np.stack([feats[samples[i][0]][samples[i][1]] for i in idx]))
        for r, i in enumerate(idx):
            ids, lab = enc[i]
            x[r, :len(ids)] = torch.tensor(ids)
            y[r, :len(lab) - 1] = torch.tensor(lab[1:])
        x, y, f = x.cuda(non_blocking=True), y.cuda(non_blocking=True), f.cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, f, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        seen += x.numel()
        if step % 20 == 0:
            dt = time.time() - t0
            write({"kind": "train", "step": step, "loss": round(loss.item(), 4), "tok_s": round(seen / dt)})
            print(f"第 {a.stage} 阶段 step {step}/{total} loss {loss.item():.3f} {seen / dt:,.0f} tok/s "
                  f"ETA {(total - step) * dt / 20 / 60:.0f} 分钟", flush=True)
            t0, seen = time.time(), 0
        if step % 1000 == 0:
            show(step)
    show(total)
    save_v(model, os.path.join(a.out, "tuna-v.pt"), stage=a.stage)


if __name__ == "__main__":
    main()
