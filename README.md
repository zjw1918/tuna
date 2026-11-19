# Tuna

配套视频系列「一块显卡造 AI」的代码：在一块游戏显卡上，从零造一个自己的 AI。每出一期，这里就多一期的代码；每期打一个 tag（`e01` …），随时可以回到任意一期：

```sh
git clone https://github.com/zjw1918/tuna && cd tuna
git pull                 # 新的一期出来以后
git checkout e02          # 回到某一期的代码
```

**硬件：** 一块 NVIDIA 显卡（实测 RTX 3070 8 GB）+ Linux。只要装好显卡驱动，不用另装 CUDA。

## 装环境

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh   # 装 uv
uv sync                                            # 装 Python 3.12、PyTorch 等
nvidia-smi && uv run python -c "import torch; print(torch.cuda.is_available())"
```

## 课程地图

| 期 | 内容 | 视频 | 代码 |
|---|---|---|---|
| E01 | 猜下一个字 | [B 站](https://www.bilibili.com/video/BV15tsm6dEvZ) | `lessons/get_texts.py`（下载课本）、`lessons/e01_bigram.py` |
| E02 | 机器是怎么“学”的 | [B 站](https://www.bilibili.com/video/BV1zMs861ERr) | `lessons/e02_autograd.py`（四十行自动求导 + PyTorch 学出 E01 的表） |

## 跑起来

### E01 · 猜下一个字

```sh
uv run python lessons/get_texts.py ~/tuna-data/texts                     # 下载《西游记》（和以后用的全唐诗）
uv run python lessons/e01_bigram.py ~/tuna-data/texts/xiyouji.txt        # 数数、抽字、算惊讶值
uv run python lessons/e01_bigram.py ~/tuna-data/texts/xiyouji.txt 0.3    # 换个温度：0.3 / 1 / 3
```

### E02 · 机器是怎么“学”的

```sh
uv run python lessons/e02_autograd.py                                    # 拟合直线：100 步找到 y = 2x + 1
uv run python lessons/e02_autograd.py --lr 0.35                          # 学习率太大：小球飞出山谷
uv run python lessons/e02_autograd.py ~/tuna-data/texts/xiyouji.txt      # 再用 PyTorch 学一张 4,433 × 4,433 的表
```

下载慢或连不上：《西游记》和全唐诗的下载脚本自带备用地址（Gutenberg 镜像、jsDelivr），会自动换。

## 致谢

《西游记》来自 Project Gutenberg（公版），全唐诗来自 [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)（MIT）。

## 许可

代码：Apache-2.0（见 `LICENSE`）。
