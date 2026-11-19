"""训练 Tuna 的分词器：字节级 BPE，词表 6400。

E03 里先手写一遍 BPE 讲原理；这里用 HF tokenizers 跑同一个算法，只是快得多。
用法：python scripts/train_tokenizer.py <pretrain.jsonl> <输出目录> [抽样 MB] [--vocab 6400]
"""
import argparse, json, os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|image|>", "<|audio|>"]


def sample_texts(path, mb):
    """每隔几行取一行，让抽样覆盖整个文件，而不只是开头。"""
    size = os.path.getsize(path)
    stride = max(1, round(size / (mb * 2**20)))
    got = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % stride:
                continue
            text = json.loads(line)["text"]
            got += len(text.encode())
            yield text
            if got > mb * 2**20:
                break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("mb", type=int, nargs="?", default=300)
    ap.add_argument("--vocab", type=int, default=6400)     # 试试 1000 / 32000：句子变长还是变短？
    a = ap.parse_args()
    src, out, mb = a.src, a.out, a.mb
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=a.vocab,
        special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),   # 256 个字节永远在词表里
        show_progress=True,
    )
    tok.train_from_iterator(sample_texts(src, mb), trainer)
    os.makedirs(out, exist_ok=True)
    tok.save(os.path.join(out, "tokenizer.json"))
    for i, s in enumerate(SPECIAL):
        assert tok.token_to_id(s) == i, s
    demo = "孙悟空一个筋斗云翻了十万八千里。Hello, Tuna!"
    ids = tok.encode(demo).ids
    print(f"词表 {tok.get_vocab_size()}；示例 {len(demo)} 字 → {len(ids)} 个 token")
    print([tok.decode([i]) for i in ids])
    assert tok.decode(ids) == demo


if __name__ == "__main__":
    main()
