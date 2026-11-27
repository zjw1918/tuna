"""一次性把图片"看"完：用借来的 SigLIP2 抽出每张图的 64 个特征向量，存盘。
之后训练只读这些向量，不用再跑编码器——省下一大半时间和显存。

用法：python scripts/extract_features.py <xxx_i2t.parquet> <输出目录> <要多少条> [--offset N]
只保留中文对话；输出 feats.npy（[N, 64, 768] float16）和 convs.jsonl。
"""
import io, json, os, re, sys, time

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tuna.vlm import ENC_DIM, N_IMG, Eye  # noqa: E402

CJK = re.compile(r"[一-鿿]")


def chinese(conv):
    text = "".join(m["content"] for m in conv)
    return len(CJK.findall(text)) > 0.3 * len(text)


def rows(path, skip):
    pf = pq.ParquetFile(path)
    seen = 0
    for batch in pf.iter_batches(batch_size=512, columns=["conversations", "image_bytes"]):
        for conv, img in zip(batch.column("conversations").to_pylist(), batch.column("image_bytes").to_pylist()):
            conv = json.loads(conv)
            if not chinese(conv) or "<image>" not in conv[0]["content"]:
                continue
            seen += 1
            if seen > skip:
                yield conv, img


def main():
    src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    skip = int(sys.argv[sys.argv.index("--offset") + 1]) if "--offset" in sys.argv else 0
    os.makedirs(out, exist_ok=True)
    eye = Eye()
    feats = np.lib.format.open_memmap(os.path.join(out, "feats.npy"), "w+", np.float16, (n, N_IMG, ENC_DIM))
    fconv = open(os.path.join(out, "convs.jsonl"), "w")
    buf_img, buf_conv, i, t0 = [], [], 0, time.time()

    def flush():
        nonlocal i
        f = eye(buf_img).cpu().numpy()
        feats[i:i + len(f)] = f
        for c in buf_conv:
            fconv.write(json.dumps(c, ensure_ascii=False) + "\n")
        i += len(f)

    for conv, img in rows(src, skip):
        try:
            buf_img.append(Image.open(io.BytesIO(img)).convert("RGB"))
        except Exception:
            continue
        buf_conv.append(conv)
        if len(buf_img) == 128 or i + len(buf_img) == n:
            flush()
            buf_img, buf_conv = [], []
            if i % 5120 == 0 or i == n:
                print(f"{i:,}/{n:,} 张  {i / (time.time() - t0):.0f} 张/秒", flush=True)
        if i >= n:
            break
    if buf_img and i < n:
        flush()
    feats.flush()
    json.dump({"n": i, "seconds": round(time.time() - t0), "source": os.path.basename(src)},
              open(os.path.join(out, "meta.json"), "w"))
    print(f"完成 {i:,} 张，{time.time() - t0:.0f} 秒")


if __name__ == "__main__":
    main()
