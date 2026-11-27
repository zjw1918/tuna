"""在终端里和 Tuna 聊天（一个字一个字地吐出来）。

用法：python scripts/chat.py <tuna-chat.pt> <tokenizer.json> [--lora <lora.pt>] [--ask "问题"]
看图聊天（E13）：python scripts/chat.py <tuna-v.pt> <tokenizer.json> --image <照片.jpg>
"""
import argparse, os, sys

import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from tuna.infer import prompt_ids, stream  # noqa: E402
from tuna.model import load  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("tokenizer")
    p.add_argument("--lora")
    p.add_argument("--ask")
    p.add_argument("--image", help="一张照片：模型要用 train_vlm.py 训出来的 tuna-v.pt")
    p.add_argument("--temperature", type=float, default=0.7)
    a = p.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    feats = vis = None
    if a.image:                                    # 借来的眼睛先把照片看一遍：64 个图片词
        from PIL import Image
        from tuna.vlm import N_IMG, Eye
        from train_vlm import load_v
        vis = load_v(a.model).eval()
        feats = Eye()([Image.open(a.image).convert("RGB")])
        model = vis.llm
    else:
        model, _ = load(a.model)
    if a.lora:
        from tuna.lora import load_lora
        load_lora(model, a.lora)
    model.eval()
    history = []
    while True:
        try:
            q = a.ask if a.ask else input("\n你：").strip()
        except EOFError:                           # Ctrl-D 或输入结束：退出
            break
        if q in ("", "exit", "quit"):
            break
        history.append(("user", q))
        ids = prompt_ids(tok, history)
        while len(ids) > model.cfg.max_seq_len - 300 and len(history) > 1:   # 太长就忘掉最早的对话
            history = history[2:]
            ids = prompt_ids(tok, history)
        emb = None
        if feats is not None:                      # 有图：第一句话前面放 64 个占位，再换成图片词
            first = ("user", "<|image|>" * N_IMG + "\n" + history[0][1])
            ids = prompt_ids(tok, [first] + history[1:])
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                emb = vis.embed(torch.tensor([ids], device="cuda"), feats)
        print("Tuna：", end="", flush=True)
        reply = ""
        for piece in stream(model, tok, ids, temperature=a.temperature, embeds=emb):
            print(piece, end="", flush=True)
            reply += piece
        print()
        history.append(("assistant", reply))
        if a.ask:
            break


if __name__ == "__main__":
    main()
