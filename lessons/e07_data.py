"""E07 · 喂什么：先把预训练语料翻一遍，再做两个"脏数据"实验，最后看 memmap 为什么快。

用法：python lessons/e07_data.py <pretrain_t2t.jsonl> <train.bin> <texts 目录> <tokenizer.json> <输出目录>
texts 目录里要有 tang.txt（清洗过的全唐诗）和 tang.v1.txt（第一版，还带着校勘注）。
"""
import collections, hashlib, json, os, re, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import poem_tokens, train  # noqa: E402
from e06_transformer import CTX, Wrap, small  # noqa: E402

HAN = re.compile(r"[一-鿿]")


def survey(path):
    """翻一遍语料：多少篇、多长、有没有一模一样的、有没有乱码。"""
    seen = collections.Counter()
    lengths, short, bad, han = [], 0, 0, 0
    chars = 0
    first = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"].strip()
            h = hashlib.md5(text.encode()).hexdigest()
            seen[h] += 1
            first.setdefault(h, text)
            lengths.append(len(text))
            short += len(text) < 10
            bad += "�" in text                       # 解不开的乱码字符
            han += len(HAN.findall(text))
            chars += len(text)
    n = len(lengths)
    dups = sum(c - 1 for c in seen.values())
    top = seen.most_common(3)
    L = np.array(lengths)
    return {"docs": n, "unique": len(seen), "duplicates": dups, "short_lt10": short, "has_garbage": bad,
            "han_ratio": round(han / chars, 3), "chars": chars,
            "length_pct": {p: int(np.percentile(L, p)) for p in (10, 50, 90, 99)},
            "most_repeated": [(first[h][:60], c) for h, c in top]}


def longest_copy(gen, text):
    best = ""
    for i in range(len(gen)):
        j = i + len(best) + 1
        while j <= len(gen) and gen[i:j] in text:
            best, j = gen[i:j], j + 1
    return best


def poems(model, tok, prompts, temperature=0.7, top_k=30, n=40):
    eos = tok.token_to_id("<|endoftext|>")
    out = []
    for k, p in enumerate(prompts):
        torch.manual_seed(k)
        ids = torch.tensor([tok.encode(p).ids], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g = model.generate(ids, max_new=n, temperature=temperature, top_k=top_k, eos_id=eos)
        out.append(tok.decode(g[0].tolist()).replace("<|endoftext|>", ""))
    return out


def main():
    src, bin_path, texts, tok_path, out = sys.argv[1:6]
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "e07_curve.jsonl"), "w")
    R = {}

    # 1 · 翻一遍预训练语料
    t0 = time.time()
    R["survey"] = survey(src)
    print("语料：", json.dumps(R["survey"], ensure_ascii=False), f"{time.time() - t0:.0f}s", flush=True)

    # 2 · 脏数据实验一：第一版全唐诗（5,350 首带校勘注）和清洗过的，各训一个小号 Tuna，各写 100 首，数冒出括号注的
    prompts = ["床前明月光，", "春眠不觉晓，", "大漠孤烟直，", "白日依山尽，", "国破山河在，", "红豆生南国，"]
    R["dirty"] = {}
    for name, f in [("dirty", "tang.v1.txt"), ("clean", "tang.txt")]:
        tok, tr, va = poem_tokens(os.path.join(texts, f), tok_path)
        torch.manual_seed(0)
        m = small(tok.get_vocab_size()).cuda()
        train(Wrap(m), tr, va, CTX, steps=2500, bs=64, lr=1e-3, every=500, log=log, label=name)
        # 不截前几名、温度 1：照它学到的概率原样抽，看它学到了什么
        gens = poems(m, tok, (prompts * 17)[:100], temperature=1.0, top_k=tok.get_vocab_size(), n=80)
        bad = [g for g in gens if "（" in g]              # 校勘注都写在全角括号里
        R["dirty"][name] = {"with_note": len(bad), "of": len(gens), "examples": bad[:5]}
        print(f"{f} 训出来：{len(bad)} / {len(gens)} 首冒出括号注", flush=True)
        for g in bad[:3]:
            print("  ", g.replace("\n", " "))

    # 3 · 脏数据实验二：同一首诗在训练集里重复 2,000 遍
    tok, tr, va = poem_tokens(os.path.join(texts, "tang.txt"), tok_path)
    eos = tok.token_to_id("<|endoftext|>")
    starts = (tr == eos).nonzero().flatten().tolist()
    k = 20000
    poem = tr[starts[k] + 1:starts[k + 1] + 1]                     # 一首普通的诗（连同结尾的结束标记）
    text = tok.decode(poem.tolist()).replace("<|endoftext|>", "")
    perm = torch.randperm(len(starts) - 1, generator=torch.Generator().manual_seed(0))
    # 把训练集按诗打散后重新拼起来，2,000 份复制品均匀混在里面
    pieces = [tr[starts[i] + 1:starts[i + 1] + 1] for i in perm.tolist()] + [poem] * 2000
    order = torch.randperm(len(pieces), generator=torch.Generator().manual_seed(1)).tolist()
    rep = torch.cat([pieces[i] for i in order])
    torch.manual_seed(0)
    dup = small(tok.get_vocab_size()).cuda()
    train(Wrap(dup), rep, va, CTX, steps=2500, bs=64, lr=1e-3, every=500, log=log, label="dup")
    head = text[:6]
    g = poems(dup, tok, [head])[0]
    R["dup"] = {"poem": text, "copies": 2000, "share": round(2000 * len(poem) / len(rep), 4), "prompt": head, "gen": g,
                "copied": len(longest_copy(g[len(head):], text[len(head):]))}
    print("重复 2,000 遍的诗：", text, flush=True)
    print("给它开头：", g, "；一字不差接上", R["dup"]["copied"], "个字", flush=True)

    # 4 · 4 GB 的 token 磁带：整个读进内存 vs memmap
    t0 = time.time()
    mm = np.memmap(bin_path, dtype=np.uint16, mode="r")
    t_open = time.time() - t0
    rng = np.random.default_rng(0)
    t0 = time.time()
    for _ in range(100):
        s = rng.integers(0, len(mm) - 513, 32)
        batch = np.stack([mm[i:i + 513] for i in s])
        x, y = batch[:, :-1], batch[:, 1:]               # 错一位：每个位置的答案，是它后面那个 token
    t_batch = (time.time() - t0) / 100
    t0 = time.time()
    whole = np.fromfile(bin_path, dtype=np.uint16)
    t_load = time.time() - t0
    R["memmap"] = {"tokens": len(mm), "bytes": os.path.getsize(bin_path), "open_ms": round(t_open * 1000, 2),
                   "batch_ms": round(t_batch * 1000, 2), "load_s": round(t_load, 1), "load_gb": round(whole.nbytes / 1e9, 2),
                   "int64_gb": round(len(mm) * 8 / 1e9, 1)}
    print("memmap：", json.dumps(R["memmap"]), flush=True)
    json.dump(R, open(os.path.join(out, "e07.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
