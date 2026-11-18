"""E01 · 猜下一个字：数一数《西游记》里每个字后面跟着什么字，再照着概率抽字。

用法：python lessons/e01_bigram.py <xiyouji.txt> [温度]
"""
import random, sys
from collections import Counter, defaultdict

text = open(sys.argv[1]).read()
temperature = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

# 1. 数数：counts["孙"]["悟"] = "孙"后面紧跟"悟"的次数
counts = defaultdict(Counter)
for a, b in zip(text, text[1:]):
    counts[a][b] += 1

print("字表大小：", len(counts))
print("「孙」后面最常见的字：", counts["孙"].most_common(5))


# 2. 抽字：按次数当概率抽下一个字；温度越高越"放飞"
def next_char(c):
    chars, n = zip(*counts[c].items())
    weights = [x ** (1 / temperature) for x in n]
    return random.choices(chars, weights)[0]


random.seed(42)
c, out = "孙", "孙"
for _ in range(80):
    c = next_char(c)
    out += c
print(out)

# 3. 这个模型有多"惊讶"？平均每个字的概率越大越好
import math
nll = 0.0
for a, b in zip(text[:100000], text[1:100001]):
    nll -= math.log(counts[a][b] / sum(counts[a].values()))
print(f"平均惊讶值（交叉熵）：{nll / 100000:.3f}")
