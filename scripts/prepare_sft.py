"""把对话数据变成 SFT 训练样本：套上聊天模板，只让"Tuna 说的话"计分。

用法：python scripts/prepare_sft.py <sft.jsonl> <identity.jsonl> <tokenizer.json> <输出目录>
输出：ids.bin（uint16）、mask.bin（uint8，1 = 计分）、offsets.npy、meta.json

模板（和 Qwen 等主流模型一样的 ChatML 写法）：
<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n你好！<|im_end|>\n
"""
import json, os, re, sys

import numpy as np
from tokenizers import Tokenizer

NAME = "Tuna"            # ← 改成你给自己模型起的名字
CREATOR = "蓝鳍金枪鱼"     # ← 改成你的名字
CREATOR_EN = "Bluefin Tuna"
MAX_LEN = 1024
IDENTITY_REPEAT = 20     # 自我介绍的数据很少，多重复几遍


def retag_identity(conv):
    """把参考数据里的自我介绍换成我们自己的。"""
    for m in conv:
        c = m["content"]
        english = not re.search(r"[一-鿿]", c)
        c = re.sub(r"Jingyao Gong|jingyaogong|JingyaoGong", CREATOR_EN if english else CREATOR, c)
        m["content"] = c.replace("MiniMind", NAME).replace("minimind", NAME)
    return conv


def mentions_other_identity(conv):
    return any(re.search(r"minimind|jingyao", m.get("content") or "", re.I) for m in conv)


class Encoder:
    def __init__(self, tok):
        self.tok = tok
        self.start, self.end = tok.token_to_id("<|im_start|>"), tok.token_to_id("<|im_end|>")
        self.nl = tok.encode("\n").ids

    def __call__(self, conv):
        ids, mask = [], []
        for m in conv:
            head = [self.start] + self.tok.encode(m["role"] + "\n").ids
            body = self.tok.encode(m["content"]).ids + [self.end]
            ids += head + body + self.nl
            score = 1 if m["role"] == "assistant" else 0     # 只有助手的回答（含结束符）计分
            mask += [0] * len(head) + [score] * len(body) + [0] * len(self.nl)
        return ids, mask


def conversations(path):
    for line in open(path, encoding="utf-8"):
        conv = json.loads(line)["conversations"]
        if any(m["role"] not in ("user", "assistant") or m.get("tool_calls") for m in conv):
            continue                                         # 跳过系统提示和工具调用的对话
        yield [{"role": m["role"], "content": m["content"]} for m in conv]   # 只要最终回答，不要思考过程


def main():
    src, ident, tok_path, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    enc = Encoder(Tokenizer.from_file(tok_path))
    fid = open(os.path.join(out, "ids.bin"), "wb")
    fmask = open(os.path.join(out, "mask.bin"), "wb")
    offsets, pos = [0], 0
    stats = {"kept": 0, "too_long": 0, "other_identity": 0, "identity": 0, "scored_tokens": 0}

    def add(conv):
        nonlocal pos
        ids, mask = enc(conv)
        if len(ids) > MAX_LEN:
            stats["too_long"] += 1
            return
        np.array(ids, np.uint16).tofile(fid)
        np.array(mask, np.uint8).tofile(fmask)
        pos += len(ids)
        offsets.append(pos)
        stats["kept"] += 1
        stats["scored_tokens"] += sum(mask)

    for conv in conversations(src):
        if mentions_other_identity(conv):
            stats["other_identity"] += 1
            continue
        add(conv)
    for _ in range(IDENTITY_REPEAT):
        for line in open(ident, encoding="utf-8"):
            add(retag_identity(json.loads(line)["conversations"]))
            stats["identity"] += 1
    fid.close()
    fmask.close()
    np.save(os.path.join(out, "offsets.npy"), np.array(offsets, np.int64))
    stats["tokens"] = pos
    json.dump(stats, open(os.path.join(out, "meta.json"), "w"), indent=2)
    print(stats)


if __name__ == "__main__":
    main()
