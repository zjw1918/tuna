"""把 Tuna 导出成 Hugging Face 的 Llama 格式，之后就能用 llama.cpp 转成 GGUF、放进 Ollama。

Tuna 和 Llama 的结构一模一样，只有两处要"翻译"：
1. 名字：wq → q_proj、w1 → gate_proj ……
2. RoPE 的配对方式：Tuna 把相邻的两维（0 和 1、2 和 3……）配成一对来转；
   HF 的 Llama 把前一半和后一半（0 和 32、1 和 33……）配成一对。所以 q、k 的行要重新排一下。

用法：python scripts/export_hf.py <tuna-chat.pt> <tokenizer.json> <输出目录>
"""
import json, os, shutil, sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tuna.model import load  # noqa: E402

CHAT_TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                 "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def interleaved_to_half(w, n_heads, head_dim):
    """每个头的行顺序：[0,1,2,3,…] 里偶数行放前一半，奇数行放后一半。"""
    w = w.view(n_heads, head_dim // 2, 2, -1)
    return w.transpose(1, 2).reshape(n_heads * head_dim, -1)


def main():
    ck, tok_path, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    model, _ = load(ck, device="cpu")
    c = model.cfg
    sd = model.state_dict()
    hf = {"model.embed_tokens.weight": sd["embed.weight"], "model.norm.weight": sd["norm.weight"]}
    for i in range(c.n_layers):
        p, q = f"layers.{i}.", f"model.layers.{i}."
        hf[q + "self_attn.q_proj.weight"] = interleaved_to_half(sd[p + "attn.wq.weight"], c.n_heads, c.head_dim)
        hf[q + "self_attn.k_proj.weight"] = interleaved_to_half(sd[p + "attn.wk.weight"], c.n_kv_heads, c.head_dim)
        hf[q + "self_attn.v_proj.weight"] = sd[p + "attn.wv.weight"]
        hf[q + "self_attn.o_proj.weight"] = sd[p + "attn.wo.weight"]
        hf[q + "mlp.gate_proj.weight"] = sd[p + "ffn.w1.weight"]
        hf[q + "mlp.up_proj.weight"] = sd[p + "ffn.w3.weight"]
        hf[q + "mlp.down_proj.weight"] = sd[p + "ffn.w2.weight"]
        hf[q + "input_layernorm.weight"] = sd[p + "attn_norm.weight"]
        hf[q + "post_attention_layernorm.weight"] = sd[p + "ffn_norm.weight"]
    from safetensors.torch import save_file
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in hf.items()}, os.path.join(out, "model.safetensors"))
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path)
    json.dump({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": c.dim, "intermediate_size": c.ffn_dim, "num_hidden_layers": c.n_layers,
        "num_attention_heads": c.n_heads, "num_key_value_heads": c.n_kv_heads, "head_dim": c.head_dim,
        "vocab_size": c.vocab_size, "max_position_embeddings": c.max_seq_len, "rope_theta": c.rope_theta,
        "rms_norm_eps": c.norm_eps, "tie_word_embeddings": True, "hidden_act": "silu",
        "bos_token_id": tok.token_to_id("<|im_start|>"), "eos_token_id": tok.token_to_id("<|im_end|>"),
        "torch_dtype": "bfloat16",
    }, open(os.path.join(out, "config.json"), "w"), indent=2)
    shutil.copy(tok_path, os.path.join(out, "tokenizer.json"))
    json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": None, "eos_token": "<|im_end|>",
               "pad_token": "<|endoftext|>", "unk_token": None, "model_max_length": c.max_seq_len,
               "chat_template": CHAT_TEMPLATE, "add_bos_token": False},
              open(os.path.join(out, "tokenizer_config.json"), "w"), ensure_ascii=False, indent=2)
    json.dump({"eos_token_id": tok.token_to_id("<|im_end|>"), "pad_token_id": tok.token_to_id("<|endoftext|>")},
              open(os.path.join(out, "generation_config.json"), "w"))
    print("导出完成：", out)

    # 自检：用 transformers 加载导出的模型，和 Tuna 本身比一比输出是否一致
    from transformers import AutoModelForCausalLM
    m2 = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
    ids = torch.tensor([tok.encode("中国的首都是北京，").ids])
    with torch.no_grad():
        a, b = model(ids).float(), m2(ids).logits.float()
    print(f"和原模型的最大差异：{(a - b).abs().max().item():.4f}（bf16 存储，0.1 以内算一致）")


if __name__ == "__main__":
    main()
