"""E09 · 说话的艺术：温度、top-k、top-p、重复惩罚，再给生成装上 KV 缓存。

用法：python lessons/e09_sampling.py <tuna-base.pt> <tokenizer.json> <输出目录>
"""
import json, os, sys, time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tuna.model import load, sample  # noqa: E402

PROMPT = "从前有一只小金枪鱼，它"


@torch.no_grad()
def no_cache_generate(model, ids, n):
    """没有 KV 缓存：每生成一个字，都把整句话从头算一遍。"""
    for _ in range(n):
        logits = model(ids)[:, -1]
        ids = torch.cat([ids, sample(logits, 0.8, 50, 1.0)], 1)
    return ids


def main():
    ck, tok_path, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    model, _ = load(ck)
    model.eval()
    ids = torch.tensor([tok.encode(PROMPT).ids], device="cuda")
    rows = []

    # 1. 温度：0 = 永远挑最可能的字（复读机）；2 = 什么字都敢挑（喝醉了）
    for name, kw in [("温度 0", dict(temperature=0)),
                     ("温度 0.8 + top-k 50", dict(temperature=0.8, top_k=50, top_p=1.0)),
                     ("温度 2.0，不截断", dict(temperature=2.0, top_k=0, top_p=1.0)),
                     ("温度 0.8 + top-p 0.9 + 重复惩罚 1.2", dict(temperature=0.8, top_k=0, top_p=0.9,
                                                              repetition_penalty=1.2))]:
        torch.manual_seed(0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            text = tok.decode(model.generate(ids, max_new=80, **kw)[0].tolist())
        rows.append({"setting": name, "text": text})
        print(f"【{name}】{text}\n")

    # 2. 下一个字的概率柱状图：温度怎么改变它（动画用）
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids)[0, -1].float()
    bars = {}
    for t in (0.3, 1.0, 2.0):
        p = torch.softmax(logits / t, -1)
        top = p.topk(10)
        bars[str(t)] = [(tok.decode([int(i)]), round(v, 4)) for v, i in zip(top.values.tolist(), top.indices.tolist())]
    print("温度 1.0 时最可能的下一个字：", bars["1.0"][:5])

    # 3. KV 缓存：没缓存时每一步都把整句话从头算一遍。三种情况各比一次速度
    #    一条对话在显卡上：两个一样快——56M 太小，显卡每一步都在等 Python 把活儿递过来（翻车）
    #    16 条一起、或者换成 CPU：缓存的好处就出来了
    speed = {}
    for dev, n, B in [("cuda", 400, 1), ("cuda", 400, 16), ("cpu", 200, 1)]:
        m = model if dev == "cuda" else load(ck, device="cpu")[0].eval()
        x = torch.randint(10, 6000, (B, 8), device=dev)
        for name, fn in [("没有缓存", lambda: no_cache_generate(m, x, n)),
                         ("KV 缓存", lambda: m.generate(x, max_new=n, temperature=0.8))]:
            with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
                if dev == "cuda":
                    fn()                                       # 先热身一次
                    torch.cuda.synchronize()
                t0 = time.time()
                fn()
                if dev == "cuda":
                    torch.cuda.synchronize()
            key = f"{'显卡' if dev == 'cuda' else 'CPU'} {B} 条 {name}"
            speed[key] = round(n * B / (time.time() - t0), 1)
            print(f"{key}：{speed[key]} token/秒")
    json.dump({"samples": rows, "bars": bars, "speed": speed}, open(os.path.join(out, "e09.json"), "w"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
