"""E02 · 机器是怎么"学"的：四十行自动求导，然后让 PyTorch 把 E01 的"数数"学出来。

用法：python lessons/e02_autograd.py [xiyouji.txt] [--lr 学习率]
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("text", nargs="?")                    # 给了《西游记》才跑第 2 部分
ap.add_argument("--lr", type=float, default=0.05)     # 拟合直线的学习率：试试 0.005 和 0.35
args = ap.parse_args()


class Value:
    """一个会记住"自己是怎么算出来的"数。"""
    def __init__(self, data, parents=(), backward=lambda: None):
        self.data, self.grad = data, 0.0
        self._parents, self._backward = parents, backward

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def backward():
            self.grad += out.grad           # 加法：梯度原样传回去
            other.grad += out.grad
        out._backward = backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def backward():
            self.grad += other.data * out.grad   # 乘法：梯度乘上"另一个数"
            other.grad += self.data * out.grad
        out._backward = backward
        return out

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    __radd__, __rmul__ = __add__, __mul__

    def backward(self):
        """从结果出发，按"谁最后算的谁先回传"的顺序，把梯度传给每个参数。"""
        order, seen = [], set()
        def visit(v):
            if v not in seen:
                seen.add(v)
                for p in v._parents:
                    visit(p)
                order.append(v)
        visit(self)
        self.grad = 1.0
        for v in reversed(order):
            v._backward()


# 1. 拟合一条线 y = 2x + 1：从 w=0、b=0 出发，顺着梯度往下走
data = [(x, 2 * x + 1) for x in [-2, -1, 0, 1, 2, 3]]
w, b = Value(0.0), Value(0.0)
lr = args.lr
for step in range(101):
    loss = Value(0.0)
    for x, y in data:
        err = w * x + b - y
        loss = loss + err * err * (1 / len(data))     # 平均平方误差
    w.grad = b.grad = 0.0
    loss.backward()
    w.data -= lr * w.grad
    b.data -= lr * b.grad
    if step % 20 == 0:
        print(f"第 {step:3d} 步  loss {loss.data:8.3f}  w {w.data:.3f}  b {b.data:.3f}")
print(f"学到：y = {w.data:.2f}x + {b.data:.2f}")

# 2. 换成 PyTorch：一张"可训练的表"学 bigram，结果应该和 E01 数出来的一样
if args.text:
    import torch
    import torch.nn.functional as F
    text = open(args.text).read()
    chars = sorted(set(text))
    idx = {c: i for i, c in enumerate(chars)}
    dev = "cuda" if torch.cuda.is_available() else "cpu"                # 有显卡就用显卡
    ids = torch.tensor([idx[c] for c in text], device=dev)
    table = torch.zeros(len(chars), len(chars), device=dev, requires_grad=True)   # 第 a 行第 b 列：a 后面跟 b 的"分数"
    opt = torch.optim.Adam([table], lr=0.1)
    x, y = ids[:-1], ids[1:]
    g = torch.Generator(device=dev).manual_seed(0)
    for step in range(2001):
        opt.param_groups[0]["lr"] = 0.1 * (1 - step / 2001)                  # 学习率从 0.1 慢慢降到 0
        pick = torch.randint(0, len(x), (8192,), generator=g, device=dev)      # 每次随机挑 8192 对字
        loss = F.cross_entropy(table[x[pick]], y[pick])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 250 == 0:
            print(f"第 {step:4d} 步  交叉熵 {loss.item():.3f}")
    with torch.no_grad():                                          # 在前 10 万个字上算，和 E01 对比
        print(f"学出来的表：交叉熵 {F.cross_entropy(table[x[:100000]], y[:100000]).item():.3f}")
        p = table[idx["孙"]].softmax(-1)
        print("「孙」后面最可能的字：", [(chars[i], round(p[i].item(), 3)) for i in p.topk(5).indices.tolist()])
