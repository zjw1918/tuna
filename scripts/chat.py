"""在终端里和 Tuna 聊天（一个字一个字地吐出来）。

用法：python scripts/chat.py <tuna-chat.pt> <tokenizer.json> [--lora <lora.pt>] [--ask "问题"]
"""
import argparse, os, sys

import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tuna.infer import prompt_ids, stream  # noqa: E402
from tuna.model import load  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("tokenizer")
    p.add_argument("--lora")
    p.add_argument("--ask")
    p.add_argument("--temperature", type=float, default=0.7)
    a = p.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    model, _ = load(a.model)
    if a.lora:                                     # E11：插上卡带
        from tuna.lora import load_lora
        load_lora(model, a.lora)
    model.eval()
    history = []
    while True:
        q = a.ask if a.ask else input("\n你：").strip()
        if q in ("", "exit", "quit"):
            break
        history.append(("user", q))
        ids = prompt_ids(tok, history)
        while len(ids) > model.cfg.max_seq_len - 300 and len(history) > 1:   # 太长就忘掉最早的对话
            history = history[2:]
            ids = prompt_ids(tok, history)
        print("Tuna：", end="", flush=True)
        reply = ""
        for piece in stream(model, tok, ids, temperature=a.temperature):
            print(piece, end="", flush=True)
            reply += piece
        print()
        history.append(("assistant", reply))
        if a.ask:
            break


if __name__ == "__main__":
    main()
