"""E12 · 你画我猜：浏览器里随手画，小 ViT 边画边猜。

用法：python lessons/e12_draw.py <doodle-vit.pt> [--port 8001]
然后打开 http://localhost:8001
"""
import argparse, json, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

sys.path.insert(0, os.path.dirname(__file__))
from e12_vit import CLASSES, ViT  # noqa: E402

PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>你画我猜</title>
<style>
:root{--bg:#05070d;--card:#0e1320;--ink:#e6ebf5;--mute:#7c8799;--hi:#ffd166;--line:#1f2a3d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,sans-serif}
main{max-width:760px;margin:0 auto;padding:16px}h1{font-size:20px;margin:4px 0 12px}h1 span{color:var(--mute);font-weight:400;font-size:14px}
.row{display:flex;gap:16px;flex-wrap:wrap}#pad{width:min(100%,360px);aspect-ratio:1;background:#000;border:1px solid var(--line);border-radius:12px;touch-action:none;cursor:crosshair}
.side{flex:1;min-width:220px}#eye{width:168px;height:168px;image-rendering:pixelated;border:1px solid var(--line);border-radius:8px;display:block}
.cap{font-size:13px;color:var(--mute);margin:4px 0 12px}.bar{display:flex;align-items:center;gap:8px;margin:4px 0}
.bar b{width:4em;font-weight:500}.bar i{height:14px;border-radius:7px;background:var(--line)}.bar:first-child i{background:var(--hi)}
button{margin-top:12px;padding:8px 14px;border-radius:12px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer;font:inherit}
</style></head><body><main>
<h1>你画我猜 <span>— 我从零训练的小 ViT（猫、鱼、苹果、房子、树、汽车、自行车、飞机、太阳、闹钟）</span></h1>
<div class="row"><canvas id="pad" width="720" height="720"></canvas>
<div class="side"><canvas id="eye" width="28" height="28"></canvas><div class="cap">它看到的：28×28，切成 7×7 = 49 个图块</div>
<div id="bars"></div><button id="clear">擦掉重画</button></div></div>
</main><script>
const pad=document.getElementById('pad'),pc=pad.getContext('2d'),eye=document.getElementById('eye'),bars=document.getElementById('bars');
let strokes=[],cur=null,busy=false,again=false;
pc.lineCap=pc.lineJoin='round';pc.strokeStyle='#fff';pc.lineWidth=14;
const at=e=>{const r=pad.getBoundingClientRect();return [(e.clientX-r.left)*pad.width/r.width,(e.clientY-r.top)*pad.height/r.height]};
pad.onpointerdown=e=>{pad.setPointerCapture(e.pointerId);cur=[at(e)];strokes.push(cur)};
pad.onpointermove=e=>{if(!cur)return;const p=at(e),q=cur[cur.length-1];cur.push(p);
  pc.beginPath();pc.moveTo(q[0],q[1]);pc.lineTo(p[0],p[1]);pc.stroke();guess()};
pad.onpointerup=pad.onpointercancel=()=>{cur=null;guess()};
document.getElementById('clear').onclick=()=>{strokes=[];pc.clearRect(0,0,pad.width,pad.height);eye.getContext('2d').clearRect(0,0,28,28);bars.innerHTML=''};
// 和数据集一样摆放：笔画居中，长边缩到 26 像素
function toPixels(){const K=8,S=28*K,off=new OffscreenCanvas(S,S),c=off.getContext('2d');
  const pts=strokes.flat(),xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);
  const x0=Math.min(...xs),y0=Math.min(...ys),w=Math.max(...xs)-x0,h=Math.max(...ys)-y0,k=26*K/Math.max(w,h,1);
  c.fillStyle='#000';c.fillRect(0,0,S,S);c.strokeStyle='#fff';c.lineCap=c.lineJoin='round';c.lineWidth=2*K;
  c.translate((S-w*k)/2-x0*k,(S-h*k)/2-y0*k);c.scale(k,k);c.lineWidth/=k;
  for(const s of strokes){c.beginPath();c.moveTo(...s[0]);for(const p of s)c.lineTo(...p);c.stroke()}
  const e=eye.getContext('2d');e.imageSmoothingQuality='high';e.drawImage(off,0,0,28,28);
  return Array.from(e.getImageData(0,0,28,28).data.filter((_,i)=>i%4==0),v=>v/255)}
async function guess(){if(!strokes.length)return;if(busy){again=true;return}busy=true;
  const r=await fetch('/guess',{method:'POST',body:JSON.stringify(toPixels())}),top=await r.json();
  bars.innerHTML=top.map(([n,p])=>`<div class="bar"><b>${n}</b><i style="width:${Math.max(2,p*140)}px"></i>${Math.round(p*100)}%</div>`).join('');
  busy=false;if(again){again=false;guess()}}
</script></body></html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--port", type=int, default=8001)
    a = p.parse_args()
    model = ViT(len(CLASSES)).cuda().eval()
    model.load_state_dict(torch.load(a.model, map_location="cuda"))
    names = list(CLASSES.values())

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE.encode())

        def do_POST(self):
            px = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with torch.no_grad():
                prob = model(torch.tensor(px).view(1, 28, 28).cuda()).float().softmax(-1)[0]
            top = prob.topk(3)
            body = json.dumps([[names[j], round(v, 3)] for v, j in zip(top.values.tolist(), top.indices.tolist())])
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *args):
            pass

    print(f"打开 http://localhost:{a.port}")
    ThreadingHTTPServer(("0.0.0.0", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
