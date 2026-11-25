# Tuna

配套视频系列「一块显卡造 AI」的代码：在一块游戏显卡上，从零造一个自己的 AI。每出一期，这里就多一期的代码；每期打一个 tag（`e01` …），随时可以回到任意一期：

```sh
git clone https://github.com/zjw1918/tuna && cd tuna
git pull                 # 新的一期出来以后
git checkout e08          # 回到某一期的代码
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
| E04 | 春天挨着秋天 | [B 站](https://www.bilibili.com/video/BV1mTsB6EE4B) | `lessons/e04_mlp.py`（词嵌入 + MLP，在全唐诗上训练）、`lessons/common.py`（取数据、训练循环、生成） |
| E05 | 注意力该看谁 | [B 站](https://www.bilibili.com/video/BV1Qfsq69ENU) | `lessons/e05_attention.py`（四个头的注意力：查询、标签、内容，因果掩码，位置表） |
| E06 | 一层层盖上去 | [B 站](https://www.bilibili.com/video/BV1HZsi6qE7P) | `tuna/model.py`（整个 Tuna：RMSNorm、RoPE、GQA 注意力、SwiGLU、残差、共用词表、top-k 生成）、`lessons/e06_transformer.py`（小号 Tuna 写诗 + 拆零件实验） |
| E07 | 给它找书 | [B 站](https://www.bilibili.com/video/BV1CWypzcE54) | `scripts/prepare_pretrain.py`（把 8.3 GB 语料切成一卷 uint16 的 token 磁带）、`lessons/e07_data.py`（查重、两个脏数据实验、memmap） |
| E08 | 一夜长大 | [B 站](https://www.bilibili.com/video/BV1b7yBznEck) | `scripts/train.py`（预训练：bf16、torch.compile、梯度累积、AdamW + 预热余弦、梯度裁剪、困惑度、JSON 日志）、`tuna/model.py` 加 `save` / `load` |

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

### E05 · 注意力该看谁

```sh
uv run python lessons/e05_attention.py ~/tuna-data/texts/tang.txt tokenizer/tokenizer.json ~/tuna-runs/e05
# 先故意不加掩码训 1500 步（只会抄答案），再加上掩码正经训 6000 步（约 1 分钟），最后存下一首诗的注意力图
# 试试把 e05_attention.py 里 heads=4 改成 1 或 16，再画一画每个头在看谁
```

### E06 · 一层层盖上去

```sh
uv run python lessons/e06_transformer.py ~/tuna-data/texts/tang.txt tokenizer/tokenizer.json ~/tuna-runs/e06
# 小号 Tuna（6 层、384 宽）训 2,500 步（约 2 分钟）写几首诗；再各训一个拆掉 RoPE、残差、RMSNorm 的版本（四个一共七分多钟）
# 试试把 e06_transformer.py 里 n_layers=6 改成 2 或 12，看看七言开头还接不接得上七个字
```

### E07 · 给它找书

```sh
uv run python lessons/get_texts.py ~/tuna-data/texts            # 重跑一次：多存一份 tang.v1.txt（不清洗、带校勘注的第一版）
uv run python scripts/prepare_pretrain.py ~/tuna-data/pretrain_t2t.jsonl tokenizer/tokenizer.json ~/tuna-data/pretrain
# 切成 20 亿个 token，存成 4 GB 的 train.bin + val.bin（约 7 分半）
uv run python lessons/e07_data.py ~/tuna-data/pretrain_t2t.jsonl ~/tuna-data/pretrain/train.bin ~/tuna-data/texts tokenizer/tokenizer.json ~/tuna-runs/e07
# 翻一遍语料（重复、太短、乱码），两个脏数据实验（三个小号 Tuna，每个约 2 分钟），最后量 memmap 有多快
# 试试把 e07_data.py 里复制的遍数从 2,000 改成 200，它还背得出来吗？
```

### E08 · 一夜长大

```sh
uv run python scripts/train.py --data ~/tuna-data/pretrain --tokenizer tokenizer/tokenizer.json --out ~/tuna-runs/pretrain
# 一晚上：12 亿 token、36,621 步（原厂 RTX 3070 约 5 小时，显存约 3.4 GB），每 500 步考一次，最后存下 tuna-base.pt；断了加 --resume 接着跑
uv run python scripts/train.py --data ~/tuna-data/pretrain --tokenizer tokenizer/tokenizer.json --out ~/tuna-runs/nowarm --tokens 3276800 --accum 1 --warmup 0 --eval_every 50 --no_compile
# 试试关掉预热跑 400 步，和片里的 6.10 对一对（每步 16 段 × 512，和正式训练的 64 段不同）
```

下载慢或连不上：《西游记》和全唐诗的下载脚本自带备用地址（Gutenberg 镜像、jsDelivr），会自动换。

Hugging Face 连不上时，先设镜像再跑 `hf download`：

```sh
export HF_ENDPOINT=https://hf-mirror.com
```

## 训好的模型

等不及自己通宵训练，可以直接下载我们训好的权重（GitHub Releases），接着往后做：

**`tuna-base.pt`**（223 MB）：E08 那一晚训出的基座模型（5,572 万参数，36,621 步，验证 loss 2.088）。

```sh
mkdir -p ~/tuna-runs/pretrain
curl -L -o ~/tuna-runs/pretrain/tuna-base.pt https://github.com/zjw1918/tuna/releases/download/e08/tuna-base.pt
sha256sum ~/tuna-runs/pretrain/tuna-base.pt   # afb03dc5…304e4d45
uv run python -c "import sys; from tuna.model import load; m, ck = load(sys.argv[1]); print(ck['step'], m.num_params())" ~/tuna-runs/pretrain/tuna-base.pt
```

GitHub 下载慢时，可以先在浏览器里打开链接下载，再放到同样的位置。

## 致谢

《西游记》来自 Project Gutenberg（公版），全唐诗来自 [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)（MIT）。

## 许可

代码：Apache-2.0（见 `LICENSE`）。

训好的权重：[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)，不可商用（训练数据里有 CC BY-NC 成分）。
