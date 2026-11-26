"""给 Tuna 体检：自建考卷（关键词判分）+ 验证集困惑度。

用法：python scripts/eval.py <模型.pt> <tokenizer.json> [--lora x.pt] [--val <预训练数据目录>] [--base] [--loose]
--base：基座模型不会聊天，用"问：…\n答："的续写方式考它。
--loose：在整段回答里找关键词（老办法，会被"先说错、后面过程里又冒出对的数"骗过去）。
默认只看第一句话：第一句就是 Tuna 给出的答案。
"""
import argparse, json, math, os, re, sys
from collections import defaultdict

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tuna.infer import prompt_ids, stream  # noqa: E402
from tuna.model import load  # noqa: E402

EXAM = os.path.join(os.path.dirname(__file__), "..", "data", "exam.jsonl")


@torch.no_grad()
def perplexity(model, path, seq=512, n=200):
    data = np.memmap(path, dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)
    losses = []
    for _ in range(n // 8):
        s = rng.integers(0, len(data) - seq - 1, 8)
        chunk = torch.from_numpy(np.stack([data[i:i + seq + 1] for i in s]).astype(np.int64)).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses.append(model(chunk[:, :-1], chunk[:, 1:])[1].item())
    return math.exp(sum(losses) / len(losses))


def ask(model, tok, q, base):
    torch.manual_seed(0)
    if base:
        ids = tok.encode(f"问：{q}\n答：").ids
        text = ""
        for piece in stream(model, tok, ids, max_new=60, temperature=0):
            text += piece
            if "\n" in text:
                break
        return text.split("\n")[0]
    return "".join(stream(model, tok, prompt_ids(tok, [("user", q)]), max_new=120, temperature=0))


def first_sentence(ans):
    """第一句话：到第一个句号、问号、叹号或空行为止。"""
    return re.split(r"[。！？!?]|\n\s*\n", ans.strip(), maxsplit=1)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("tokenizer")
    p.add_argument("--lora")
    p.add_argument("--val")
    p.add_argument("--base", action="store_true")
    p.add_argument("--loose", action="store_true")
    p.add_argument("--int8", action="store_true", help="先量化成 int8 再考")
    p.add_argument("--out")
    a = p.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    model, _ = load(a.model)
    if a.lora:
        from tuna.lora import load_lora
        load_lora(model, a.lora)
    if a.int8:
        from tuna.quant import quantize
        quantize(model)
    model.eval()
    report = {"model": a.model, "items": []}
    score = defaultdict(lambda: [0, 0])
    for line in open(EXAM):
        item = json.loads(line)
        ans = ask(model, tok, item["q"], a.base)
        judged = ans if a.loose else first_sentence(ans)
        ok = any(k in judged for k in item["any"]) and not any(k in judged for k in item.get("none", []))
        score[item["cat"]][0] += ok
        score[item["cat"]][1] += 1
        report["items"].append({**item, "answer": ans, "ok": ok})
        print(f"{'✓' if ok else '✗'} [{item['cat']}] {item['q']} → {ans[:60]!r}")
    total = sum(s[0] for s in score.values())
    print("成绩单：", "  ".join(f"{c} {s[0]}/{s[1]}" for c, s in score.items()), f"  总分 {total}/{len(report['items'])}")
    report["score"] = {c: s for c, s in score.items()}
    report["total"] = total
    if a.val:
        report["ppl"] = round(perplexity(model, os.path.join(a.val, "val.bin")), 2)
        print(f"验证集困惑度：{report['ppl']}")
    if a.out:
        json.dump(report, open(a.out, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
