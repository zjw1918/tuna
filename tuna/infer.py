"""推理小工具：聊天模板、流式生成。chat.py、train_lora.py、train_vlm.py 都用它。"""
import torch

from tuna.model import sample


def stream(model, tok, ids, max_new=300, temperature=0.7, top_k=40, top_p=0.9, rep=1.1, embeds=None):
    """生成器：每算出一个 token 就交出当前能显示的新文字。
    embeds：第一步直接用这些向量（看图时，图片向量已经拼进去了）。"""
    end = tok.token_to_id("<|im_end|>")
    caches = [{} for _ in model.layers]
    x = torch.tensor([ids], device="cuda")
    out, shown, pos = [], "", 0
    for _ in range(max_new):
        if pos + x.shape[1] > model.cfg.max_seq_len:
            break
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, caches=caches, start=pos, embeds=embeds)[:, -1].float()
        embeds = None
        pos += x.shape[1]
        if out and rep != 1.0:
            prev = torch.tensor([out], device="cuda")
            seen = logits.gather(1, prev)
            logits = logits.scatter(1, prev, torch.where(seen > 0, seen / rep, seen * rep))
        nxt = sample(logits, temperature, top_k, top_p)
        if nxt.item() == end:
            break
        out.append(nxt.item())
        text = tok.decode(out)
        if not text.endswith("�"):            # 半个汉字先不显示，等它凑齐
            yield text[len(shown):]
            shown = text
        x = nxt


def prompt_ids(tok, history):
    s = "".join(f"<|im_start|>{r}\n{c}<|im_end|>\n" for r, c in history)
    return tok.encode(s + "<|im_start|>assistant\n").ids
