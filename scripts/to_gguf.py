"""把导出的 HF 格式转成 GGUF，给 llama.cpp / Ollama 用。

llama.cpp 只认识它见过的分词器。Tuna 的分词器是自己训练的，但切词规则和 GPT-2 完全相同（字节级 BPE + GPT-2 的正则），
所以这里告诉转换脚本"按 gpt-2 的规则处理"即可，不用改 llama.cpp 的代码。

用法：python scripts/to_gguf.py <llama.cpp 目录> <HF 目录> <输出.gguf> [f16|q8_0]
（用 llama.cpp 自带的 Python 环境运行）
"""
import runpy, sys

llama_dir, hf_dir, out = sys.argv[1:4]
outtype = sys.argv[4] if len(sys.argv) > 4 else "f16"
sys.path.insert(0, llama_dir)
from conversion import base  # noqa: E402

base.TextModel.get_vocab_base_pre = lambda self, tokenizer: "gpt-2"
sys.argv = ["convert_hf_to_gguf.py", hf_dir, "--outfile", out, "--outtype", outtype]
runpy.run_path(f"{llama_dir}/convert_hf_to_gguf.py", run_name="__main__")
