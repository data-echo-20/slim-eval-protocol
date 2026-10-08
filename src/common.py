# -*- coding: utf-8 -*-
"""公共工具：模型客户端、图像编码、数据读写。

客户端走 OpenAI 兼容协议，可指向任意端点（本地 vLLM / 云端 API）。
支持文本模型（无视觉）与多模态模型（带视觉）。
"""
import base64, io, json, os, re, threading, time
import urllib.request, urllib.error


class Client:
    """OpenAI 兼容客户端。support_vision=False 时不发送图像。

    线程安全：`_post` 每次新建连接、不共享可变状态；计数器用锁保护。
    因此可以多线程并发调用（vLLM 服务端本身支持并发批处理）。
    """

    def __init__(self, base_url, api_key, model, support_vision=True,
                 timeout=120, max_retry=3):
        self.base = base_url.rstrip("/")
        self.key = api_key
        self.model = model
        self.support_vision = support_vision
        self.timeout = timeout
        self.max_retry = max_retry
        self._lock = threading.Lock()
        self.calls = 0
        self.tokens = 0

    def _bump(self, ntokens=0):
        with self._lock:
            self.calls += 1
            self.tokens += ntokens

    def _post(self, path, payload):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.key},
            method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def chat(self, prompt, image_b64=None, system=None, temperature=0.0,
             max_tokens=256, seed=0):
        """返回 (文本, 原始响应)。image_b64 为 None 或 support_vision=False 时纯文本。"""
        content = []
        if image_b64 and self.support_vision:
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + image_b64}})
        content.append({"type": "text", "text": prompt})
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": content if len(content) > 1 else prompt})

        payload = {"model": self.model, "messages": msgs,
                   "temperature": temperature, "max_tokens": max_tokens}
        # vLLM 等后端支持固定种子，保证多次运行可复现
        if seed is not None:
            payload["seed"] = seed
        last = None
        for a in range(self.max_retry):
            try:
                d = self._post("/chat/completions", payload)
                self._bump(d.get("usage", {}).get("total_tokens", 0))
                return d["choices"][0]["message"]["content"].strip(), d
            except Exception as e:
                last = "%s: %s" % (type(e).__name__, str(e)[:200])
                time.sleep(2 * (a + 1))
        raise RuntimeError("请求失败 %d 次: %s" % (self.max_retry, last))


# ---------------- 图像 ----------------

def image_to_b64(path, max_side=896):
    """读图 → 缩放 → base64(jpeg)。无 Pillow 时原样透传。"""
    try:
        from PIL import Image
    except ImportError:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    im = Image.open(path).convert("RGB")
    w, h = im.size
    if max(w, h) > max_side:
        s = max_side / float(max(w, h))
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------- 数据 ----------------

def read_jsonl(path):
    out = []
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path, rows):
    with io.open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------- 答案判定 ----------------

_NORM = re.compile(r"[^a-z0-9一-鿿]+")

# 判定两侧对词边界的要求不同，不能混用同一种匹配：
#   语料侧（normalize 之后）已经把所有非字母数字汉字压成空格，
#     所以答案两端就是空格或字符串边界，用 \b 恰好正确，且能防子串误判；
#   候选答案侧必须先判断首尾字符类型——中文汉字在 \w 里，
#     "意大利" 两侧都加 \b 会导致永远匹配不上（中英混排时尤其隐蔽）。
# 混合候选（如 "Samsung 手机"）退化为子串匹配，因为无法同时满足两种边界。
_CJK = re.compile(r"[一-鿿]")


def normalize(s):
    return _NORM.sub(" ", (s or "").lower()).strip()


def contains_any(text, candidates):
    """判断 text 是否包含任一候选答案（宽松匹配，适用于简短答案）。

    英文候选按词边界匹配（避免 Italy 命中 "not Italy" 之外的子串误判）；
    中文候选按子串匹配（\\b 在汉字间不成立）。
    """
    t = normalize(text)
    if not t:
        return False
    for c in candidates:
        n = normalize(c)
        if not n:
            continue
        # 纯 ASCII（普通英文/数字答案）：两侧都要词边界
        if not _CJK.search(n):
            if re.search(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", t):
                return True
            continue
        # 纯中文：无词边界概念，但要求两侧不是字母数字（挡住 "abc意大利" 这类）
        if not re.search(r"[a-z0-9]", n):
            if re.search(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", t):
                return True
            continue
        # 中英混合：退化为子串匹配
        if n in t:
            return True
    return False


# ---- 长答案判定（E-VQA 等标注为长描述短语的数据集）----

_STOP = {
    "a", "an", "the", "of", "and", "or", "in", "on", "to", "for", "with", "its",
    "it", "is", "are", "be", "as", "by", "from", "at", "that", "this", "these",
    "those", "their", "they", "them", "other", "others", "etc", "can", "may",
    "such", "some", "most", "more", "than", "up", "including", "include",
}


def _content_tokens(s):
    return {t for t in normalize(s).split() if t and t not in _STOP}


def answers_match(model_answer, candidates, min_cover=0.6):
    """长答案判定：双向包含 + 内容词覆盖率。返回 (是否命中, 分档)。

    为什么不能只用 contains_any：它的方向是「候选整体必须是模型答案的子串」，
    这对 A-OKVQA 的一词答案正确，但 E-VQA 的标注是中位数 4 词、最长 116 词的
    描述短语，模型常给短答案。于是
        model="Spring"  vs  标注 "late winter to spring"
    会被判成"没答对" —— 模型答对了却记成错，方向性错误会同时压低
    knowledge rate 和后续 PKD-rate 的分母。

    规则（任一成立即命中），分档用于论文声明判定口径：
      exact : 归一化后完全相等
      sub   : 候选整体是模型答案的子串（旧行为，短答案下最可靠）
      sup   : 模型答案整体是候选的子串，且 >=4 字符或 >=2 内容词
              （挡掉 "a"/"1" 这类无意义的单字符包含）
      cover : 候选内容词在模型答案中的覆盖率 >= min_cover
    取最高档（exact > sub > sup > cover）。
    """
    t = normalize(model_answer)
    if not t:
        return False, None
    mtok = _content_tokens(model_answer)
    best = None
    for c in candidates:
        n = normalize(c)
        if not n:
            continue
        if n == t:
            return True, "exact"
        # sub：候选在模型答案里
        if contains_any(model_answer, [c]):
            best = best or "sub"
            continue
        # sup：模型答案在候选里
        if (len(t) >= 4 or len(mtok) >= 2) and \
           re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", n):
            best = best or "sup"
            continue
        # cover：内容词覆盖
        ctok = _content_tokens(c)
        if ctok and len(mtok & ctok) / float(len(ctok)) >= min_cover:
            best = best or "cover"
    return (best is not None), best
