"""把预训练语料变成一卷 token "磁带"：uint16 二进制文件，文档之间用 <|endoftext|> 隔开。

用法：python scripts/prepare_pretrain.py <pretrain.jsonl> <tokenizer.json> <输出目录> [验证集比例]
输出：train.bin、val.bin、meta.json（token 数、字节数）
"""
import json, os, sys, time

import numpy as np
from tokenizers import Tokenizer

BATCH = 20000


def batches(path):
    buf = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            text = json.loads(line)["text"].strip()
            if len(text) < 10:          # 太短的不要
                continue
            buf.append(text)
            if len(buf) == BATCH:
                yield buf
                buf = []
    if buf:
        yield buf


def main():
    src, tok_path, out = sys.argv[1:4]
    val_every = int(1 / float(sys.argv[4])) if len(sys.argv) > 4 else 200   # 每 200 批放 1 批进验证集
    tok = Tokenizer.from_file(tok_path)
    eos = tok.token_to_id("<|endoftext|>")
    os.makedirs(out, exist_ok=True)
    files = {k: open(os.path.join(out, f"{k}.bin"), "wb") for k in ("train", "val")}
    counts = {"train": 0, "val": 0}
    n_bytes = n_docs = 0
    t0 = time.time()
    for b, texts in enumerate(batches(src)):
        split = "val" if b % val_every == val_every - 1 else "train"
        ids = []
        for enc in tok.encode_batch(texts):           # Rust 多线程并行分词
            ids.extend(enc.ids)
            ids.append(eos)
        arr = np.array(ids, dtype=np.uint16)          # 词表 6400 < 65536，两个字节就够
        arr.tofile(files[split])
        counts[split] += len(arr)
        n_bytes += sum(len(t.encode()) for t in texts)
        n_docs += len(texts)
        if b % 50 == 0:
            tot = sum(counts.values())
            print(f"{n_docs:,} 篇  {tot / 1e6:,.0f}M token  {n_bytes / tot:.2f} 字节/token  "
                  f"{time.time() - t0:.0f}s", flush=True)
    for f in files.values():
        f.close()
    meta = {"train_tokens": counts["train"], "val_tokens": counts["val"], "docs": n_docs,
            "bytes": n_bytes, "bytes_per_token": n_bytes / sum(counts.values()),
            "seconds": round(time.time() - t0)}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(meta)


if __name__ == "__main__":
    main()
