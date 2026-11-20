# Tuna

配套视频系列「一块显卡造 AI」的代码：在一块游戏显卡上，从零造一个自己的 AI。每出一期，这里就多一期的代码；每期打一个 tag（`e01` …），随时可以回到任意一期：

```sh
git clone https://github.com/zjw1918/tuna && cd tuna
git pull                 # 新的一期出来以后
git checkout e04          # 回到某一期的代码
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
| E03 | 给语言切积木 | [B 站](https://www.bilibili.com/video/BV1aTsh6KETc) | `lessons/e03_bpe.py`（手写字节级 BPE）、`scripts/train_tokenizer.py`、`tokenizer/tokenizer.json`（成品词表） |
| E04 | 春天挨着秋天 | 即将上线 | `lessons/e04_mlp.py`（词嵌入 + MLP，在全唐诗上训练）、`lessons/common.py`（取数据、训练循环、生成） |

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

### E03 · 给语言切积木

```sh
uv run python lessons/e03_bpe.py ~/tuna-data/texts/xiyouji.txt              # 手写 BPE：500 次合并，先分段
uv run python lessons/e03_bpe.py ~/tuna-data/texts/xiyouji.txt --no-split   # 不分段：逗号粘上半个字
# 正式词表：先下载 8.3 GB 现代中文语料（约 12 分钟，后面几期接着用）
uv run hf download jingyaogong/minimind_dataset pretrain_t2t.jsonl --repo-type dataset --local-dir ~/tuna-data
uv run python scripts/train_tokenizer.py ~/tuna-data/pretrain_t2t.jsonl ~/tuna-runs/tok                # 6,400 格，约 3 分钟
uv run python scripts/train_tokenizer.py ~/tuna-data/pretrain_t2t.jsonl ~/tuna-runs/tok1k --vocab 1000  # 换个大小比一比
# 不想下载：仓库里的 tokenizer/tokenizer.json 和自己训出来的逐字节相同
```

### E04 · 春天挨着秋天

```sh
uv run python lessons/e04_mlp.py ~/tuna-data/texts/tang.txt tokenizer/tokenizer.json ~/tuna-runs/e04
# 先正常训练（约 1 分钟）：接诗、找「和春最像的字」；再跑过拟合实验：见过的诗一字不差背出来
# 试试把 e04_mlp.py 里 MLP 的 dim=64 改成 16 或 256，「春」最像的还是不是「秋」？
```

下载慢或连不上：《西游记》和全唐诗的下载脚本自带备用地址（Gutenberg 镜像、jsDelivr），会自动换。

Hugging Face 连不上时，先设镜像再跑 `hf download`：

```sh
export HF_ENDPOINT=https://hf-mirror.com
```

## 致谢

《西游记》来自 Project Gutenberg（公版），全唐诗来自 [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)（MIT）。

## 许可

代码：Apache-2.0（见 `LICENSE`）。
