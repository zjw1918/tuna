"""第 1–7 期的小语料：《西游记》（公版，Project Gutenberg #23962）和全唐诗（chinese-poetry，MIT）。
都转成简体，存成纯文本。

用法：python lessons/get_texts.py <输出目录>
输出：xiyouji.txt（正文，一回一段）、tang.txt（一行一首：标题｜作者｜正文）、
      tang.v1.txt（不清洗的第一版，还带着校勘注，第 7 期做脏数据实验用）
"""
import json, os, re, sys, urllib.request

import opencc

GUTENBERG = ["https://www.gutenberg.org/cache/epub/23962/pg23962.txt",
             "https://gutenberg.pglaf.org/2/3/9/6/23962/23962-0.txt"]
POETRY = ["https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/全唐诗/poet.tang.{}.json",
          "https://cdn.jsdelivr.net/gh/chinese-poetry/chinese-poetry@master/全唐诗/poet.tang.{}.json",
          "https://fastly.jsdelivr.net/gh/chinese-poetry/chinese-poetry@master/全唐诗/poet.tang.{}.json"]
t2s = opencc.OpenCC("t2s")


def fetch(url):
    req = urllib.request.Request(urllib.parse.quote(url, safe=":/@"), headers={"User-Agent": "tuna-course"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def fetch_any(urls):
    """挨个试，哪个镜像通就用哪个。"""
    for url in urls:
        try:
            text = fetch(url)
            if "<html" not in text[:500].lower():       # 维护页、拦截页不算
                return text
        except Exception as e:
            print("换一个镜像：", e)
    raise RuntimeError("都下载不了：" + urls[0])


def xiyouji(out):
    raw = fetch_any(GUTENBERG)
    body = raw.split("*** START OF")[1].split("\n", 1)[1].split("*** END OF")[0]
    # 原文每行硬换行，空行才是分段：先把段内的换行拼起来
    paras = [re.sub(r"\s+", "", p) for p in re.split(r"\n\s*\n", body)]
    text = "\n".join(t2s.convert(p) for p in paras if p)
    text = text[text.index("第一回"):]
    open(os.path.join(out, "xiyouji.txt"), "w").write(text)
    print(f"西游记：{len(text):,} 字")


def clean_poem(body):
    """去掉校勘注（"一作「叉」"之类的括号）和缺字符号；还有生僻缺字的整首丢掉。"""
    body = re.sub(r"（[^）]*）|\([^)]*\)|〔[^〕]*〕|\[[^\]]*\]", "", body)
    body = re.sub(r"[「」『』《》]", "", body)
    return None if re.search(r"[□�\[\]（）()〔〕]", body) else body


def tang(out):
    rows, raw = [], []
    for i in range(0, 58000, 1000):                   # 一共 58 个文件，每个 1000 首
        poems = json.loads(fetch_any([u.format(i) for u in POETRY]))
        for p in poems:
            text = "".join(p.get("paragraphs", []))
            if text:
                raw.append(t2s.convert(f"{p['title']}｜{p['author']}｜{text}"))
            body = clean_poem(text)
            if body:
                rows.append(t2s.convert(f"{p['title']}｜{p['author']}｜{body}"))
    open(os.path.join(out, "tang.txt"), "w").write("\n".join(rows))
    open(os.path.join(out, "tang.v1.txt"), "w").write("\n".join(raw))
    print(f"全唐诗：{len(rows):,} 首")


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    xiyouji(out)
    tang(out)
