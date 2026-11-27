"""E14 · 很多人同时来：1 个、8 个、32 个人同时提问时，本地服务一秒总共能出多少个 token。

连任何 OpenAI 兼容的 /v1/chat/completions 接口（Ollama 和 vLLM 都有）。每人问一个不同的问题，最多生成 N 个 token，
温度 0.7；总吞吐 = 所有人生成的 token 数 / 墙上时间。

用法：python scripts/bench_serve.py <服务地址> <模型名> <输出.json> [--n 200] [--users 1,8,32]
"""
import argparse, json, threading, time, urllib.request

QS = ["给我讲一个关于小鱼的故事。", "介绍一下你自己。", "天空为什么是蓝色的？", "怎样才能睡个好觉？",
      "写一首关于春天的小诗。", "为什么要多喝水？", "推荐三种适合初学者的运动。", "猫和狗有什么不同？"]


def one(url, model, q, n, res, i):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": q}], "max_tokens": n,
                       "temperature": 0.7, "top_p": 0.9, "seed": i, "stream": False}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    res[i] = {"tokens": r["usage"]["completion_tokens"], "sec": time.perf_counter() - t,
              "text": r["choices"][0]["message"]["content"]}


def run(url, model, users, n):
    res = [None] * users
    th = [threading.Thread(target=one, args=(url, model, QS[i % len(QS)], n, res, i)) for i in range(users)]
    t = time.perf_counter()
    for x in th:
        x.start()
    for x in th:
        x.join()
    wall = time.perf_counter() - t
    tok = sum(r["tokens"] for r in res)
    return {"users": users, "tokens": tok, "wall": round(wall, 2), "total_tok_s": round(tok / wall, 1),
            "per_user_tok_s": round(sum(r["tokens"] / r["sec"] for r in res) / users, 1), "sample": res[0]["text"][:120]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("url")
    p.add_argument("model")
    p.add_argument("out")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--users", default="1,8,32")
    a = p.parse_args()
    run(a.url, a.model, 2, 16)                     # warm-up
    rows = []
    for u in map(int, a.users.split(",")):
        best = max((run(a.url, a.model, u, a.n) for _ in range(2)), key=lambda r: r["total_tok_s"])
        rows.append(best)
        print(best, flush=True)
    json.dump(rows, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
