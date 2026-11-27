# Tuna

配套视频系列「一块显卡造 AI」的代码：在一块游戏显卡上，从零造一个自己的 AI。每出一期，这里就多一期的代码；每期打一个 tag（`e01` …），随时可以回到任意一期：

```sh
git clone https://github.com/zjw1918/tuna && cd tuna
git pull                 # 新的一期出来以后
git checkout e14          # 回到某一期的代码
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
| E09 | 说话的艺术 | [B 站](https://www.bilibili.com/video/BV15Sy6zTE2r) | `tuna/model.py` 加 `sample()`（温度、top-k、top-p）、`generate()` 的重复惩罚和 KV 缓存、`lessons/e09_sampling.py`（四种挑法 + 缓存测速） |
| E10 | 学会聊天 | [B 站](https://www.bilibili.com/video/BV1HUy2zwERQ) | `scripts/prepare_sft.py`（聊天模板 + 损失掩码，把 80 万段对话切成训练样本）、`scripts/train_sft.py`（指令微调）、`tuna/infer.py`（模板、流式生成）、`scripts/chat.py`（终端聊天） |
| E11 | 会写诗了 | [B 站](https://www.bilibili.com/video/BV1SvyUzUEM8) | `data/exam.jsonl` + `scripts/eval.py`（30 道题的成绩单、困惑度）、`tuna/lora.py`（LoRA：A、B 两条细矩阵，load_lora 插卡带）、`scripts/train_lora.py`（训写诗卡带）、`scripts/chat.py --lora` |
| E12 | 机器怎么看图 | [B 站](https://www.bilibili.com/video/BV1kc1EzWEQi) | `lessons/e12_vit.py`（涂鸦切成 49 个图块，从零训一个小 ViT，再拿真照片考它）、`lessons/e12_draw.py`（浏览器里你画它猜）、`tuna/model.py`（`causal` 开关）、`photos/`（6 张 CC0 测试照片） |
| E13 | Tuna 睁眼了 | 即将上线 | `tuna/vlm.py`（借来的 SigLIP2 眼睛 + 翻译官投影层，图片词拼进句子）、`scripts/extract_features.py`（先把图片看一遍，存成特征）、`scripts/train_vlm.py`（两阶段训练）、`scripts/chat.py --image`（拿照片和它聊） |
| E14 | 住进你的电脑 | 即将上线 | `tuna/quant.py`（int8 量化）、`lessons/e14_quant.py`（8 / 4 / 3 / 2 位对照）、`scripts/serve.py`（网页聊天，能拖图片）、`scripts/export_hf.py` + `scripts/to_gguf.py` + `export/Modelfile`（导出 → GGUF → Ollama）、`scripts/bench_serve.py`（很多人同时来） |

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

### E09 · 说话的艺术

```sh
uv run python lessons/e09_sampling.py ~/tuna-runs/pretrain/tuna-base.pt tokenizer/tokenizer.json ~/tuna-runs/e09
# 同一个开头四种挑法（贪心、温度 0.8 + top-k、温度 2、top-p + 重复惩罚），再比有没有 KV 缓存的速度：显卡 1 条、16 条、CPU（约 1 分钟）
# 试试把 e09_sampling.py 第 39 行的 repetition_penalty=1.2 改成 2.0，它还敢用「的」吗？
```

### E10 · 学会聊天

```sh
uv run hf download jingyaogong/minimind_dataset sft_t2t_mini.jsonl lora_identity.jsonl --repo-type dataset --local-dir ~/tuna-data/raw
uv run python scripts/prepare_sft.py ~/tuna-data/raw/sft_t2t_mini.jsonl ~/tuna-data/raw/lora_identity.jsonl tokenizer/tokenizer.json ~/tuna-data/sft
# 套模板、换上 Tuna 的自我介绍：806,715 段对话、约 3 亿 token（约 6 分钟）
uv run python scripts/train_sft.py --base ~/tuna-runs/pretrain/tuna-base.pt --data ~/tuna-data/sft --tokenizer tokenizer/tokenizer.json --out ~/tuna-runs/sft-5e-5
# 指令微调一遍（原厂 RTX 3070 约 1.8 小时），存下 tuna-chat.pt
uv run python scripts/chat.py ~/tuna-runs/sft-5e-5/tuna-chat.pt tokenizer/tokenizer.json
# 和它聊天；问问「你是谁？」
uv run python scripts/train_sft.py --base ~/tuna-runs/pretrain/tuna-base.pt --data ~/tuna-data/sft --tokenizer tokenizer/tokenizer.json --out ~/tuna-runs/e10/nomask --no_mask --epochs 0.04
# 不加损失掩码训一小段（约 4 分钟）。chat.py 见到 im_end 就停：先把 tuna/infer.py 里 if nxt.item() == end: break 两行注释掉，再聊，看它会不会替你提问
```

### E11 · 会写诗了

```sh
uv run python scripts/eval.py ~/tuna-runs/sft-5e-5/tuna-chat.pt tokenizer/tokenizer.json --val ~/tuna-data/pretrain
# 成绩单（30 道题）+ 10 万个 token 的困惑度；加 --loose 是旧判法（整段找关键词），会被「十减四等于二十」骗过
uv run python scripts/eval.py ~/tuna-runs/pretrain/tuna-base.pt tokenizer/tokenizer.json --base --val ~/tuna-data/pretrain
# 基座模型也考一次（不套聊天模板）
uv run python scripts/train_lora.py ~/tuna-runs/sft-5e-5/tuna-chat.pt tokenizer/tokenizer.json ~/tuna-data/texts/tang.txt ~/tuna-runs/lora-r8
# 训写诗卡带：主干冻住，只训 A、B（108 万个数，RTX 3070 约 96 秒），存下 poet-lora.pt；--r 1 / --r 32 换卡带宽度
uv run python scripts/chat.py ~/tuna-runs/sft-5e-5/tuna-chat.pt tokenizer/tokenizer.json --lora ~/tuna-runs/lora-r8/poet-lora.pt --ask "写一首关于闻笛的诗。"
# 插上卡带写诗；再问它「中国的首都是哪里？」，看看翻车
uv run python scripts/train_lora.py ~/tuna-runs/sft-5e-5/tuna-chat.pt tokenizer/tokenizer.json ~/tuna-data/texts/tang.txt ~/tuna-runs/lora-mix --mix ~/tuna-data/sft
# 修法：每首诗配一段普通对话一起训（约 6 分钟）
uv run python scripts/eval.py ~/tuna-runs/sft-5e-5/tuna-chat.pt tokenizer/tokenizer.json --lora ~/tuna-runs/lora-mix/poet-lora.pt --val ~/tuna-data/pretrain
# 插着卡带再考一次成绩单
```

### E12 · 机器怎么看图

```sh
uv run python lessons/e12_vit.py ~/tuna-data/quickdraw ~/tuna-runs/e12 photos/bicycle.jpg photos/house.jpg photos/fish.jpg photos/cat.jpg photos/apple.jpg photos/tree.jpg
# 先自动下载 10 类 Quick, Draw! 涂鸦，训 3,000 步（RTX 3070 约 92 秒），最后拿 6 张真照片考它（自行车 → 太阳？）
uv run python lessons/e12_draw.py ~/tuna-runs/e12/doodle-vit.pt
# 浏览器打开 http://localhost:8001 ，画个太阳，看它中途把圆圈认成什么
```

### E13 · Tuna 睁眼了

```sh
uv run hf download jingyaogong/minimind-v_dataset pretrain_i2t.parquet sft_i2t.parquet --repo-type dataset --local-dir ~/tuna-data/raw-v
# 图文数据：图文描述 4.3 GB + 图文指令 4.9 GB
uv run python scripts/extract_features.py ~/tuna-data/raw-v/pretrain_i2t.parquet ~/tuna-data/vfeat/cap 100000
uv run python scripts/extract_features.py ~/tuna-data/raw-v/sft_i2t.parquet ~/tuna-data/vfeat/sft 60000
# 冻住的眼睛先把图片看一遍：每张 64 个图片词（第一次会下载 SigLIP2；两份约 9 分钟，特征约 15 GB）
uv run python scripts/train_vlm.py --stage 1 --llm ~/tuna-runs/sft-5e-5/tuna-chat.pt --data ~/tuna-data/vfeat/cap --tokenizer tokenizer/tokenizer.json --out ~/tuna-runs/vlm/stage1
# 第一阶段：Tuna 和眼睛都冻住，只训翻译官（90 万个旋钮，约 10 分钟）
uv run python scripts/train_vlm.py --stage 2 --init ~/tuna-runs/vlm/stage1/tuna-v.pt --data ~/tuna-data/vfeat/sft --tokenizer tokenizer/tokenizer.json --out ~/tuna-runs/vlm/stage2
# 第二阶段：整个模型一起学回答图片问题（约 6 分钟），存下 tuna-v.pt
uv run python scripts/chat.py ~/tuna-runs/vlm/stage2/tuna-v.pt tokenizer/tokenizer.json --image photos/cat.jpg --temperature 0.5
# 拿照片和它聊；换成 photos/apple.jpg，问问篮子里有几个苹果
```

### E14 · 住进你的电脑

```sh
uv run python lessons/e14_quant.py ~/tuna-runs/sft-5e-5/tuna-chat.pt tokenizer/tokenizer.json ~/tuna-data/pretrain
# 压成 8 / 4 / 3 / 2 位各考一次：大小、困惑度、回答（约 15 秒）
uv run python scripts/serve.py --chat ~/tuna-runs/sft-5e-5/tuna-chat.pt --vision ~/tuna-runs/vlm/stage2/tuna-v.pt --tokenizer tokenizer/tokenizer.json
# 浏览器打开 http://localhost:8000 ：聊天、拖一张照片进去问（局域网里别的设备用这台机器的 IP）；加 --int8 用量化版
uv run python scripts/export_hf.py ~/tuna-runs/sft-5e-5/tuna-chat.pt tokenizer/tokenizer.json ~/tuna-runs/hf
# 导出成 Hugging Face 的 Llama 格式（改名字、重排 RoPE 的配对），自带一致性自检
git clone https://github.com/ggml-org/llama.cpp ~/tuna-tools/llama.cpp
(cd ~/tuna-tools/llama.cpp && uv venv && uv pip install -r requirements/requirements-convert_hf_to_gguf.txt)
~/tuna-tools/llama.cpp/.venv/bin/python scripts/to_gguf.py ~/tuna-tools/llama.cpp ~/tuna-runs/hf export/tuna-q8_0.gguf q8_0
# 转成 GGUF（q8_0，57 MB）；直接用 llama.cpp 的 convert_hf_to_gguf.py 会报不认识分词器
# 先装好 Ollama（https://ollama.com/download）
ollama create tuna -f export/Modelfile
ollama run tuna --verbose 给我讲一个关于小鱼的故事。
# 很多人同时来：先让 Ollama 开 32 个座位（另开终端：OLLAMA_NUM_PARALLEL=32 ollama serve），再测 1 / 8 / 32 人
uv run python scripts/bench_serve.py http://localhost:11434 tuna ~/tuna-runs/ollama-bench.json
# vLLM 放在单独的环境里（实测 0.34.0），读刚才导出的 HF 格式；测之前先停掉 Ollama，两个别同时占显存
uvx --from vllm==0.34.0 vllm serve ~/tuna-runs/hf --served-model-name tuna --dtype bfloat16 --max-model-len 1024 --gpu-memory-utilization 0.55 --port 8100
uv run python scripts/bench_serve.py http://localhost:8100 tuna ~/tuna-runs/vllm-bench.json
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

**`tuna-chat.pt`**（223 MB）：E10 指令微调后的聊天模型（学习率 5e-5，37,701 步）。

```sh
mkdir -p ~/tuna-runs/sft-5e-5
curl -L -o ~/tuna-runs/sft-5e-5/tuna-chat.pt https://github.com/zjw1918/tuna/releases/download/e10/tuna-chat.pt
sha256sum ~/tuna-runs/sft-5e-5/tuna-chat.pt   # 5471d9c3…67ddd94cb
```

**`poet-lora.pt`**（4.1 MB）：E11 的写诗卡带（LoRA r = 8，只用诗训；插在上面的 tuna-chat.pt 上用）。

```sh
mkdir -p ~/tuna-runs/lora-r8
curl -L -o ~/tuna-runs/lora-r8/poet-lora.pt https://github.com/zjw1918/tuna/releases/download/e11/poet-lora.pt
sha256sum ~/tuna-runs/lora-r8/poet-lora.pt   # 37a42445…cab2a9a4
```

**`tuna-v.pt`**（227 MB）：E13 两阶段训完的看图模型 Tuna-V（不含眼睛：SigLIP2 第一次用时自动从 Hugging Face 下载）。

```sh
mkdir -p ~/tuna-runs/vlm/stage2
curl -L -o ~/tuna-runs/vlm/stage2/tuna-v.pt https://github.com/zjw1918/tuna/releases/download/e13/tuna-v.pt
sha256sum ~/tuna-runs/vlm/stage2/tuna-v.pt   # e23b31e2…3e6a4c666a
```

**`tuna-q8_0.gguf`**（57 MB）：E14 导出的对话模型 GGUF（q8_0），放进 `export/` 就能直接 `ollama create tuna -f export/Modelfile`。

```sh
curl -L -o export/tuna-q8_0.gguf https://github.com/zjw1918/tuna/releases/download/e14/tuna-q8_0.gguf
sha256sum export/tuna-q8_0.gguf   # 4f5873b5…7b062b7db1
```

GitHub 下载慢时，可以先在浏览器里打开链接下载，再放到同样的位置。

## 致谢

《西游记》来自 Project Gutenberg（公版），全唐诗来自 [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry)（MIT）。涂鸦来自 Google [Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset)（CC BY 4.0），测试照片来自 Wikimedia Commons（CC0，见 `photos/CREDITS.md`）。图文数据来自 [MiniMind-V](https://github.com/jingyaogong/minimind-v)（ALLaVA-4V），视觉编码器是 Google 的 [SigLIP2](https://huggingface.co/google/siglip2-base-patch16-256)（Apache-2.0）。

## 许可

代码：Apache-2.0（见 `LICENSE`）。

训好的权重：[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)，不可商用（训练数据里有 CC BY-NC 成分）。
