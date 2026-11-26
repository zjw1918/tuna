"""E11 · 用 LoRA 给 Tuna 插一张"写诗卡带"：只训练几 MB 的小矩阵，主干一动不动。

数据：全唐诗里格式整齐的五言、七言诗，配上"写一首关于……的诗"的提问。
用法：python scripts/train_lora.py <tuna-chat.pt> <tokenizer.json> <tang.txt> <输出目录> [--mix <SFT 数据目录>] [--r 8]
--mix：每首诗配一段普通对话一起训练。不加的话，卡带插上以后它什么问题都用诗回答（翻车）。
--r：A、B 的"窄边"有多宽（秩），默认 8；alpha 跟着取 2r，放大倍数保持 2。
"""
import argparse, json, os, random, re, sys, time

import numpy as np

import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tuna.infer import prompt_ids, stream  # noqa: E402
from tuna.lora import add_lora, lora_state  # noqa: E402
from tuna.model import load  # noqa: E402

ASKS = ["写一首关于{t}的诗。", "以《{t}》为题，作一首诗。", "请写一首{n}诗，题目是{t}。", "能为「{t}」写首诗吗？"]


def regular_poems(path):
    """只要格式整齐的：每句 5 个或 7 个字，4 句或 8 句。"""
    out = []
    for line in open(path):
        parts = line.rstrip("\n").split("｜")
        if len(parts) != 3:
            continue
        title, _, body = parts
        title = re.sub(r"[（(].*?[)）]|其[一二三四五六七八九十]+|\s.*$", "", title).strip()
        sents = [s for s in re.split(r"[，。]", body) if s]
        if not title or len(title) > 8 or len(sents) not in (4, 8) or not re.fullmatch(r"[，。一-鿿]+", body):
            continue
        n = len(sents[0])
        if n not in (5, 7) or any(len(s) != n for s in sents):
            continue
        lines = [f"{sents[i]}，{sents[i + 1]}。" for i in range(0, len(sents), 2)]
        out.append({"title": title, "n": "五言" if n == 5 else "七言", "poem": "\n".join(lines)})
    return out


def is_regular(text):
    text = re.sub(r"《[^》]*》", "", text)                      # 题目不算一句
    sents = [s.strip() for s in re.split(r"[，。！？\n]", text) if s.strip()]
    return len(sents) in (4, 8) and len({len(s) for s in sents}) == 1 and len(sents[0]) in (5, 7)


def encode(tok, q, a):
    head = tok.encode(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n").ids
    body = tok.encode(a).ids + [tok.token_to_id("<|im_end|>")]
    return head + body, [-100] * len(head) + body


@torch.no_grad()
def write_poems(model, tok, titles, seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    out = []
    for t in titles:
        ids = prompt_ids(tok, [("user", f"写一首关于{t}的诗。")])
        out.append("".join(stream(model, tok, ids, max_new=80, temperature=0.7)))
    return out


def chats(sft_dir, n, max_len=256):
    """从 SFT 数据里随机挑 n 段不长的普通对话，只给回答部分计分。"""
    ids = np.memmap(os.path.join(sft_dir, "ids.bin"), dtype=np.uint16, mode="r")
    mask = np.memmap(os.path.join(sft_dir, "mask.bin"), dtype=np.uint8, mode="r")
    off = np.load(os.path.join(sft_dir, "offsets.npy"))
    short = np.flatnonzero(np.diff(off) <= max_len)
    out = []
    for i in np.random.default_rng(0).choice(short, n, replace=False):
        x, m = ids[off[i]:off[i + 1]].tolist(), mask[off[i]:off[i + 1]].tolist()
        out.append((x, [t if k else -100 for t, k in zip(x, m)]))
    return out


def main():
    p = argparse.ArgumentParser()
    for name in ("ck", "tok_path", "tang", "out"):
        p.add_argument(name)
    p.add_argument("--mix")
    p.add_argument("--r", type=int, default=8)
    a = p.parse_args()
    ck, tok_path, tang, out = a.ck, a.tok_path, a.tang, a.out
    os.makedirs(out, exist_ok=True)
    tok = Tokenizer.from_file(tok_path)
    poems = regular_poems(tang)
    random.seed(0)
    random.shuffle(poems)
    test, train = poems[:20], poems[20:]
    print(f"格式整齐的诗：{len(poems):,} 首（训练 {len(train):,}，留 20 个题目考试）")
    titles = [p["title"] for p in test]

    model, _ = load(ck)
    model.eval()
    before = write_poems(model, tok, titles)
    print(f"插卡带之前：{sum(map(is_regular, before))}/20 首格式整齐")
    for t, p in list(zip(titles, before))[:3]:
        print(f"  《{t}》→ {p[:60]!r}")

    params = add_lora(model, r=a.r, alpha=2 * a.r)
    n_lora = sum(p.numel() for p in params)
    print(f"LoRA 参数：{n_lora / 1e6:.2f}M（主干 {model.num_params() / 1e6 - n_lora / 1e6:.1f}M 冻住）")
    opt = torch.optim.AdamW(params, lr=1e-3)
    model.train()
    data = [encode(tok, random.choice(ASKS).format(t=p["title"], n=p["n"]), f"《{p['title']}》\n{p['poem']}")
            for p in train]
    if a.mix:
        data += chats(a.mix, len(data))
        random.shuffle(data)
        print(f"混进 {len(data) // 2:,} 段普通对话")
    bs, steps, t0 = 32, min(2000, len(data) // 32), time.time()
    log = open(os.path.join(out, "log.jsonl"), "w")
    for step in range(steps):
        batch = data[step * bs:(step + 1) * bs]
        T = max(len(x) for x, _ in batch)
        x = torch.tensor([ids + [0] * (T - len(ids)) for ids, _ in batch], device="cuda")
        y = torch.tensor([lab[1:] + [-100] * (T - len(lab) + 1) for _, lab in batch], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 100 == 0:
            log.write(json.dumps({"step": step, "loss": round(loss.item(), 4)}) + "\n")
            print(f"第 {step} 步 loss {loss.item():.3f}  {time.time() - t0:.0f}s", flush=True)
    model.eval()
    path = os.path.join(out, "poet-lora.pt")
    torch.save({"r": a.r, "alpha": 2 * a.r, "lora": lora_state(model)}, path)
    after = write_poems(model, tok, titles)
    print(f"插卡带之后：{sum(map(is_regular, after))}/20 首格式整齐；卡带文件 {os.path.getsize(path) / 2**20:.1f} MB；"
          f"训练 {time.time() - t0:.0f} 秒")
    for t, p in list(zip(titles, after))[:5]:
        print(f"  《{t}》→ {p!r}")
    json.dump({"before": before, "after": after, "titles": titles}, open(os.path.join(out, "poems.json"), "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
