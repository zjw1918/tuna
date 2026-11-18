# Tuna

配套视频系列「一块显卡造 AI」的代码：在一块游戏显卡上，从零造一个自己的 AI。每出一期，这里就多一期的代码；每期打一个 tag（`e01` …），随时可以回到任意一期：

```sh
git clone https://github.com/zjw1918/tuna && cd tuna
git pull                 # 新的一期出来以后
git checkout e01          # 回到某一期的代码
```

**硬件：** 一块 NVIDIA 显卡（实测 RTX 3070 8 GB）+ Linux。只要装好显卡驱动，不用另装 CUDA。

## 装环境

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh   # 装 uv
uv sync                                            # 装 Python 3.12、PyTorch 等
nvidia-smi && uv run python -c "import torch; print(torch.cuda.is_available())"
```

## 课程地图

| 期 | 内容 | 代码 |
|---|---|---|
| E01 | 猜下一个字 | `lessons/get_texts.py`（下载课本）、`lessons/e01_bigram.py` |

## 跑起来

### E01 · 猜下一个字

```sh
uv run python lessons/get_texts.py ~/tuna-data/texts                     # 下载《西游记》（和以后用的全唐诗）
uv run python lessons/e01_bigram.py ~/tuna-data/texts/xiyouji.txt        # 数数、抽字、算惊讶值
uv run python lessons/e01_bigram.py ~/tuna-data/texts/xiyouji.txt 0.3    # 换个温度：0.3 / 1 / 3
```

下载慢或连不上：《西游记》和全唐诗的下载脚本自带备用地址（Gutenberg 镜像、jsDelivr），会自动换。

## 致谢

《西游记》来自 Project Gutenberg（公版），全唐诗来自 [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)（MIT）。

## 许可

代码：Apache-2.0（见 `LICENSE`）。
