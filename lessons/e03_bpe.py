"""E03 · 分词：手写字节级 BPE。从 256 个字节出发，一遍遍把最常见的相邻一对熔成一块新积木。

用法：python lessons/e03_bpe.py <xiyouji.txt> [--merges 500] [--chars 100000] [--no-split]
--no-split：不先分段，整段文字当成一长串字节（会把标点和下一个字的半个字节粘在一起）。
"""
import argparse, re, time
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("text")
ap.add_argument("--merges", type=int, default=500)
ap.add_argument("--chars", type=int, default=100_000)
ap.add_argument("--no-split", action="store_true")
args = ap.parse_args()


def split(s):
    """先按 字 / 标点 / 空白 分段，积木只在段内合并，不会跨过标点。"""
    return [s] if args.no_split else re.findall(r"\w+|[^\w\s]+|\s+", s)


text = open(args.text).read()[:args.chars]
chunks = [list(c.encode("utf-8")) for c in split(text)]   # 一个汉字 = 3 个字节
n_bytes = sum(map(len, chunks))
print(f"{len(text):,} 个字 → {n_bytes:,} 个字节，分成 {len(chunks):,} 段")


def merge(ids, pair, new):
    out, i = [], 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


def show(b):
    """能拼成完整的字就显示字；半个字就显示字节（十六进制）。"""
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return "<" + " ".join(f"{x:02x}" for x in b) + ">"


vocab = {i: bytes([i]) for i in range(256)}
merges = {}
t0 = time.time()
for k in range(args.merges):
    counts = Counter()
    for c in chunks:
        counts.update(zip(c, c[1:]))                 # 数一数，哪一对最常见？（只数段内）
    pair, n = counts.most_common(1)[0]
    new = 256 + k
    chunks = [merge(c, pair, new) for c in chunks]
    merges[pair] = new
    vocab[new] = vocab[pair[0]] + vocab[pair[1]]
    if k < 10 or k % 100 == 99:
        print(f"第 {k + 1:3d} 次合并：{show(vocab[new]):10} 出现 {n} 次 → 序列长 {sum(map(len, chunks)):,}")
print(f"{args.merges} 次合并用了 {time.time() - t0:.0f} 秒；压缩到原来的 {sum(map(len, chunks)) / n_bytes:.0%}")


def encode(s):
    out = []
    for c in split(s):                               # 和训练时一样先分段
        ids = list(c.encode("utf-8"))
        while len(ids) > 1:
            pairs = set(zip(ids, ids[1:]))
            pair = min(pairs, key=lambda p: merges.get(p, float("inf")))  # 按学到的先后顺序合并
            if pair not in merges:
                break
            ids = merge(ids, pair, merges[pair])
        out += ids
    return out


def decode(ids):
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


s = "孙悟空拔一根毫毛，变作一个小猴。"
e = encode(s)
print(f"{s} → {len(e)} 块：", [show(vocab[i]) for i in e])
assert decode(e) == s
