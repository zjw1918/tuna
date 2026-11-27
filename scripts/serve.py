"""把 Tuna 装成一个网页应用：浏览器里聊天，可以拖一张图片进去问它。只用 Python 自带的 http.server。

用法：python scripts/serve.py --chat tuna-chat.pt --vision tuna-v.pt --tokenizer tokenizer.json [--int8] [--port 8000]
然后打开 http://localhost:8000 （局域网里其他电脑用 http://<这台机器的 IP>:8000）
"""
import argparse, base64, io, json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from tuna.infer import prompt_ids, stream  # noqa: E402
from tuna.model import load  # noqa: E402
from tuna.vlm import N_IMG, Eye  # noqa: E402
from train_vlm import load_v  # noqa: E402

PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Tuna</title>
<style>
:root{--bg:#05070d;--card:#0e1320;--ink:#e6ebf5;--mute:#7c8799;--me:#1f6feb;--line:#1f2a3d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,sans-serif}
main{max-width:760px;margin:0 auto;padding:16px;display:flex;flex-direction:column;height:100vh}
h1{font-size:20px;margin:4px 0 12px}h1 span{color:var(--mute);font-weight:400;font-size:14px}
#log{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:10px;padding-bottom:12px}
.m{max-width:85%;padding:10px 14px;border-radius:14px;background:var(--card);border:1px solid var(--line);white-space:pre-wrap}
.me{align-self:flex-end;background:var(--me);color:#fff;border:none}.m img{max-width:220px;border-radius:8px;display:block;margin-bottom:6px}
form{display:flex;gap:8px;align-items:center}textarea{flex:1;resize:none;padding:10px;border-radius:12px;border:1px solid var(--line);background:var(--card);color:var(--ink);font:inherit}
button,label{padding:10px 14px;border-radius:12px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer;font:inherit}
#pv{font-size:13px;color:var(--mute);display:flex;align-items:center;gap:8px;min-height:0}#pv img{height:48px;border-radius:6px}
main.drop{outline:2px dashed var(--me);outline-offset:-6px;border-radius:16px}
</style></head><body><main>
<h1>Tuna <span>— 我自己从零训练的小模型，跑在自己的显卡上</span></h1>
<div id="log"></div>
<div id="pv"></div>
<form id="f"><label title="选图片，或直接把图片拖进窗口">🖼<input type="file" id="img" accept="image/*" hidden></label>
<textarea id="q" rows="2" placeholder="问 Tuna 点什么……（可以先选一张图片）"></textarea><button>发送</button><button type="button" id="new" title="清空对话和图片">新对话</button></form>
</main><script>
const log=document.getElementById('log'),q=document.getElementById('q'),img=document.getElementById('img'),pv=document.getElementById('pv');
let history=[],pending=null,hasImg=false;   // pending：刚选、还没发出去的图；hasImg：这轮对话里有图
function pick(f){if(!f||!f.type.startsWith('image/'))return;const r=new FileReader();
r.onload=()=>{pending=r.result;pv.innerHTML='';const t=document.createElement('img');t.src=pending;pv.append(t,'已选图片：'+f.name)};r.readAsDataURL(f)}
img.onchange=()=>{pick(img.files[0]);img.value=''};
const main=document.querySelector('main');
addEventListener('dragover',e=>{e.preventDefault();main.classList.add('drop')});
addEventListener('dragleave',e=>{if(!e.relatedTarget)main.classList.remove('drop')});
addEventListener('drop',e=>{e.preventDefault();main.classList.remove('drop');pick(e.dataTransfer.files[0]);q.focus()});
document.getElementById('new').onclick=()=>{history=[];pending=null;hasImg=false;log.innerHTML='';pv.innerHTML='';q.focus()};
function add(cls,text,src){const d=document.createElement('div');d.className='m '+cls;if(src){const i=document.createElement('img');i.src=src;d.appendChild(i)}
d.appendChild(document.createTextNode(text));log.appendChild(d);log.scrollTop=log.scrollHeight;return d}
q.onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();document.getElementById('f').requestSubmit()}};
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const text=q.value.trim();if(!text)return;q.value='';
if(pending){history=[];hasImg=true}           // 新图 = 新话题
add('me',text,pending);history.push(['user',text]);const d=add('',''),node=d.lastChild;
const image=pending||(hasImg?'used':null);pending=null;pv.innerHTML='';   // 'used'：服务器沿用上一张图
const r=await fetch('/chat',{method:'POST',body:JSON.stringify({history,image})});
const rd=r.body.getReader(),dec=new TextDecoder();let ans='';
for(;;){const {done,value}=await rd.read();if(done)break;ans+=dec.decode(value,{stream:true});node.textContent=ans;log.scrollTop=log.scrollHeight}
history.push(['assistant',ans])};
</script></body></html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chat", required=True)
    p.add_argument("--vision")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--int8", action="store_true")
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    chat, _ = load(a.chat)
    vis = load_v(a.vision) if a.vision else None
    if a.int8:
        from tuna.quant import quantize
        quantize(chat)
        if vis:
            quantize(vis.llm)
    chat.eval()
    eye = Eye() if vis else None
    gpu = Lock()
    state = {"feats": None}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            history = [tuple(h) for h in req["history"]]
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            with gpu:
                img = req.get("image")
                if img and img != "used" and eye:
                    from PIL import Image
                    raw = base64.b64decode(img.split(",", 1)[1])
                    state["feats"] = eye([Image.open(io.BytesIO(raw)).convert("RGB")])
                elif not img:
                    state["feats"] = None
                if state["feats"] is not None:                 # 有图：第一句话前面放 64 个图片词
                    first = ("user", "<|image|>" * N_IMG + "\n" + history[0][1])
                    ids = prompt_ids(tok, [first] + history[1:])
                    x = torch.tensor([ids], device="cuda")
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        emb = vis.embed(x, state["feats"])
                    pieces = stream(vis.llm, tok, ids, temperature=0.5, embeds=emb)
                else:
                    pieces = stream(chat, tok, prompt_ids(tok, history[-6:]), temperature=0.7)
                for piece in pieces:
                    self.wfile.write(piece.encode())
                    self.wfile.flush()

        def log_message(self, *args):
            pass

    print(f"Tuna 已上线：http://localhost:{a.port}")
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
