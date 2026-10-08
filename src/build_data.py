# -*- coding: utf-8 -*-
"""Phase 0-A：构造 MedusaBench 冲突样本（反事实编辑）。

设计要点：**显式给定待替换 span**，而不是让脚本去猜。
数据集构造阶段负责标出检索段落中"承载答案的那个片段"，
编辑阶段只做一次最小替换——保持句子其余部分完全不动，
这样上下文流畅度不会劣化，冲突也足够干净。

两种编辑模式：
  span   : 用给定替换表直接换 span（确定、可复现，先用这个跑通）
  llm    : 让 LLM 就着 span 做自然改写（更隐蔽，后续用）

用法：
  python build_data.py --demo
  python build_data.py --in ../data/raw.jsonl --out ../data/medusa.jsonl
"""
import argparse, io, json, os, random, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import write_jsonl, read_jsonl


# 反事实替换池：同语义类别内替换，保证"不常见但合理"
POOL = {
    "color":    ["crimson", "cyan", "amber", "violet", "teal", "indigo", "olive"],
    "material": ["bamboo", "obsidian", "copper", "titanium", "cork", "granite", "bronze"],
    "country":  ["Peru", "Norway", "Kenya", "Vietnam", "Chile", "Portugal", "Mongolia", "Nepal"],
    "city":     ["Valparaiso", "Trondheim", "Mombasa", "Da Nang", "Arequipa", "Porto", "Pokhara"],
    "sport":    ["kabaddi", "curling", "jai alai", "pesapallo", "bandy", "hurling", "teqball"],
    "food":     ["jollof rice", "kimchi jjigae", "arepas", "borscht", "ceviche", "pho", "injera"],
    "animal":   ["pangolin", "okapi", "quokka", "saiga", "gerenuk", "kakapo", "markhor"],
    # --- E-VQA 用到的槽位 ---
    "season":   ["spring", "summer", "autumn", "winter"],
    "spread":   ["wind", "water", "insects", "birds", "mammals", "gravity"],
    "diet":     ["insects", "grasses and herbs", "fish", "nectar", "seeds",
                 "small mammals", "plankton", "crustaceans", "algae"],
    "place":    ["Peru", "Norway", "Kenya", "Vietnam", "Chile", "Portugal",
                 "Mongolia", "Nepal", "Iceland", "Madagascar"],
    "growth":   ["decaying wood", "leaf litter", "living tree bark",
                 "rock surfaces", "bare soil", "dung"],
    "medical":  ["potential", "none known", "limited", "substantial"],
    "defense":  ["horns", "burrows", "venom", "camouflage", "speed",
                 "a hard shell", "warning coloration", "spines"],
    "habitat":  ["temperate forests", "grasslands", "coastal wetlands",
                 "arid scrubland", "montane meadows", "freshwater marshes",
                 "tropical rainforests", "savannas", "mangrove swamps"],
    # --- landmarks 子集用到的槽位 ---
    # 河名：landmarks 的 "Which river does this bridge cross?" 答案是河名，
    # 用 place（国家）池替换会造出 "the bridge crosses Peru" 这类**类型错位**的
    # 反事实。必须同为河名，冲突才成立且自然。
    "river":    ["Danube", "Mekong", "Zambezi", "Orinoco", "Ganges",
                 "Volga", "Rhone", "Douro", "Irrawaddy", "Limpopo"],
}


def year_shift(y, rng):
    """年份偏移：避开过于离谱的改动，保持"合理但罕见"。"""
    y = int(y)
    for _ in range(20):
        d = rng.choice([6, 7, 9, 11, 13, -8, -12, -17])
        ny = y + d
        if 1800 <= ny <= 2020 and ny != y:
            return ny
    return y + 7


_NUM = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def refine_slot(question, span, slot):
    """把筛选阶段的**粗槽位**细化为**编辑策略**。

    筛选（screen_knowledge）用的槽位是为了分层统计（time/place/quantity/
    diet/name/other）。但"other"里混着好几件完全不同的事 —— 种子如何传播、
    能不能吃、有毒没毒 —— 它们需要不同的反事实改法。
    直接拿粗槽位去查替换表会大面积落空（实测 other 槽位 0% 可编辑）。

    这里按**问法意图**判定，而不是按预分的槽位，是编辑能否成功的关键。
    """
    q = (question or "").lower()
    s = (span or "").strip().lower()

    # 分布要**先于**生长基质判断：两者都含 "grow"，
    #   "In which country or region does X grow?" -> place
    #   "On what does X typically grow?"          -> growth（问的是基质）
    # 若先判 growth，会把所有带 grow 的问题都吃掉。
    # habitat 单独一类：它的答案是一个**生境**，不是国家；
    # 用 place 的替换池会把 "freshwater lakes and rivers" 换成 "Madagascar"，
    # 造出 "Its breeding habitat is Madagascar across northern North America"。
    if "habitat" in q:
        return "habitat"
    # river 必须**先于** place：答案是一个河名，不是国家。
    # 用 place 池（国家名）替换会产出 "crosses the bridge of Peru" 这类
    # 类型错位的反事实 —— 冲突的核心是"同槽位、异取值"，河名换国家名就破了这个前提。
    if "river" in q or "cross" in q:
        return "river"
    # city 池单独一类：landmarks 的 "In which city is this museum located?" 是
    # 最大的一类（实测 333 条），而 place 分支只认 country/region/where，会整批落到
    # unsupported 被白白丢弃。POOL 里本就有 city 池，接上即可。
    if "city" in q:
        return "city"
    if any(k in q for k in ("country", "region", "where", "world", "native",
                            "live")):
        return "place"
    # 生长基质 / 药用 / 防御
    # 注意不能用 `" on " in q` 判定："On what does Artomyces grow?" 里
    # "on" 在句首，带空格的模式匹配不到，会整批漏掉。
    if "grow" in q and "how big" not in q and "how large" not in q \
       and "become" not in q:
        return "growth"
    if "medical" in q:
        return "medical"
    if "protect" in q or "predator" in q or "defen" in q:
        return "defense"
    # 食性
    if "eat" in q or "feed" in q or "diet" in q or "prey" in q:
        return "diet"
    # 极性的**语义**判定要先于 span 形状判定：
    #   "Is Ipomoea batatas poisonous?" span="poisonous"（不是裸 yes/no）
    #   "Is Fomes fomentarius edible?"  span="not considered edible"
    # 只看 span 是否等于 yes/no 会漏掉这两类。
    if any(k in q for k in ("poisonous", "venomous", "edible", "threatened")):
        return "polarity"
    if any(k in s for k in ("venomous", "edible", "threatened")):
        return "polarity"
    if s in ("yes", "no", "potential", "not considered edible"):
        return "polarity"
    # 传播方式
    if "spread" in q or "dispers" in q:
        return "spread"
    # 季节/花期/果期/活动期/产卵期
    if any(k in q for k in ("season", "flower", "bloom", "month", "fruit",
                            "ripen", "period of the year", "time of year",
                            "active", "mating")):
        return "season"
    # 数量（span 含数字就按数量缩放）
    # 必须在 season 之后：'To how many children does a female typically give
    # birth?' 既可理解成"生育期"也可理解成"胎数"，实测 span='one'（胎数）
    # 被误当成季节换成 'summer'，造出 "bears summer offspring" 这种病句。
    if _NUM.search(span or ""):
        return "quantity"
    if any(k in q for k in ("how many", "how much", "how big", "how large",
                            "how long", "how tall", "how heavy", "size",
                            "weight", "wingspan", "lay eggs", "give birth",
                            "gestation")):
        return "quantity"
    # 分布
    if any(k in q for k in ("country", "region", "where", "world", "native",
                            "habitat", "live")):
        return "place"
    # 常见名 / 命名来源：替换池无能为力（要另一套常见名、或另一个命名者，
    # 都不是"改一个词"能解决的），直接放弃，不硬编造
    if "common name" in q or "known as" in q or "named" in q:
        return "name_unsupported"
    # 回退到粗槽位，但只在替换表里确有该键时才算可用
    return slot if slot in POOL else "unsupported"


def number_shift(span, rng):
    """数量型 span 的改写：**整段共用一个比例因子**缩放其中的所有数字。

    为什么整段共用一个比例：
      "3 to 4 eggs" 逐个数各自缩放会得到 "2.2 to 8 eggs" 这类**区间颠倒/荒诞**的值
      （下界放大、上界缩小），语料里不存在这种写法，模型一眼识破是反事实。

    为什么不直接换成量纲内的随机数：物种尺寸跨度极大（蜗牛 30mm、鲸 15m），
    随机取值会造出"座头鲸 30 毫米"这类荒唐事实。按比例缩放保住量级合理性。

    ⚠️ 踩过的最隐蔽的坑：原实现用 `out.replace(n, ns, 1)` 在**已改写的字符串**上继续
    替换下一个数字。由于新写入的数字本身会被后续 findall 的兄弟节点匹配到，
    产出 "7.2.35 to 4 eggs" / "1.62.75-5" 这类**非数字垃圾**。
    实测 236 条里有约 1/3 的 quantity 样本中招，且看起来像"正常的反事实"，
    极易蒙混过关 —— 但模型会去解析这些畸形数字，测量到的就不是知识冲突了。
    改用 re.sub 一次遍历原文，替换文本不再参与匹配。
    """
    if not _NUM.search(span):
        return None
    # 缩放后要能真正改变数值。小整数（1 meter、2 eggs）配上 45%-65% 的因子
    # 会被 round 回原值，产出"改了个寂寞"的样本 —— 模型面对与原句**完全相同**
    # 的上下文，测出的不是冲突而是重复。实测踩到：span='1 meter' -> '1 meter'。
    # 故重试若干次，取一个真正不同的因子。
    for _ in range(20):
        out = _number_shift_once(span, rng, counting_hint=None)
        if out is None or out == span:
            continue
        # 区间被压平也不行："1-2 eggs" -> "1-1 eggs" 是空区间，
        # 语料里不存在这种写法。原值互不相同的，改写后也必须互不相同。
        a, b = _NUM.findall(span), _NUM.findall(out)
        if len(set(a)) == len(a) and len(set(b)) < len(b):
            continue
        return out
    return None


def _number_shift_once(span, rng, counting_hint=None):
    scale = rng.choice([0.25, 0.35, 0.45, 0.55, 0.65,
                        1.45, 1.6, 1.8, 2.2, 2.8])

    # 计数型（几颗蛋、几只幼崽）不能出现小数：1.99 颗蛋在语料里不存在
    counting = bool(re.search(r"egg|young|offspring|pup|chick|seed|clutch|"
                              r"how many|number of|give birth|lay", span, re.I))

    all_int = all(float(n.replace(",", "")) >= 1 and
                  float(n.replace(",", "")).is_integer()
                  for n in _NUM.findall(span))

    def repl(m):
        n = m.group(0)
        # 千分位逗号必须去掉再转 float，否则 "20,000" 直接抛 ValueError
        f = float(n.replace(",", ""))
        nv = f * scale
        if counting or all_int:
            nv = max(1, int(round(nv)))     # 不允许 0 或小数
            return "{:,}".format(nv) if nv >= 1000 else str(nv)
        if f >= 1000:
            return "{:,}".format(int(round(nv)))
        return "%g" % round(nv, 1)

    return _fix_agreement(_NUM.sub(repl, span))


# 会随数字变单复数的可数名词（改写后数字变了，词形要跟着变）
_PAIRS = [("egg", "eggs"), ("meter", "meters"), ("metre", "metres"),
          ("year", "years"), ("inch", "inches"), ("day", "days"),
          ("week", "weeks"), ("month", "months"), ("pup", "pups"),
          ("chick", "chicks"), ("seed", "seeds"), ("clutch", "clutches"),
          ("young", "young"), ("offspring", "offspring")]
_SING2PLUR = dict(_PAIRS)
_PLUR2SING = {v: k for k, v in _PAIRS}


def _fix_agreement(text):
    """把最后一个数字后面的名词单复数改对。

    为什么必须做：改写后数字会变（1 -> 3、3 -> 1），但名词词形照抄原句，
    会产出 "1 eggs" / "3 meter" / "2 clutch" 这类**语法错误**。
    上下文里出现明显病句，本身就是"这段文字被人动过"的信号，
    模型可能因此改变行为（去猜"为什么这句话坏了"而不是做知识冲突判断）。
    这类瑕疵会污染 CONFLICT 条件的测量，故不能留。
    """
    ms = list(_NUM.finditer(text))
    if not ms:
        return text
    tail = text[ms[-1].end():]
    m = re.match(r"\s*([A-Za-z][A-Za-z-]*)", tail)
    if not m:
        return text
    w = m.group(1)
    lw = w.lower()
    try:
        last = float(ms[-1].group(0).replace(",", ""))
    except ValueError:
        return text
    if last == 1:
        fixed = _PLUR2SING.get(lw)
    else:
        fixed = _SING2PLUR.get(lw)
    if not fixed or fixed == lw:
        return text
    if w[:1].isupper():
        fixed = fixed[:1].upper() + fixed[1:]
    i = ms[-1].end() + m.start(1)
    return text[:i] + fixed + text[i + len(w):]


def flip_polarity(span, sentence, rng):
    """是非型 span 取反。

    覆盖 E-VQA 里出现的三种极性表达：
      yes / no                （可食、有毒、受威胁）
      venomous / not venomous （蛇类）
      edible / not edible
    直接取反会得到 "not not venomous" 这类错误，所以按整体短语匹配。
    """
    low = span.strip().lower().rstrip(".")
    table = {
        "yes": "no", "no": "yes",
        "venomous": "not venomous", "not venomous": "venomous",
        "edible": "not edible", "not edible": "edible",
        "threatened": "not threatened", "not threatened": "threatened",
        # E-VQA 里 span 可能就是问题里的那个形容词本身
        "poisonous": "not poisonous", "not poisonous": "poisonous",
        "toxic": "not toxic", "not toxic": "toxic",
        # 措辞式的是非答案
        "not considered edible": "considered edible",
        "potential": "none known",
    }
    # 反向表：span 可能是**否定式**（"Nonpoisonous"），要给出的是肯定式反义。
    # 表里只列了肯定式的直接取反，所以这里补一条规范化：
    # 匹配前先剥掉前导的 non-/not /in- 前缀，命中后用其**肯定形式**再取反。
    NEG_PREFIX = ("nonpoisonous", "non-venomous", "nontoxic", "non-toxic",
                  "inedible", "nonvenomous", "not poisonous", "not venomous",
                  "not toxic", "not edible")
    if low in NEG_PREFIX:
        return "poisonous" if "poison" in low else (
            "venomous" if "venom" in low else (
                "toxic" if "toxic" in low else "edible"))
    if low in table:
        new = table[low]
        return new[:1].upper() + new[1:] if span[:1].isupper() else new
    return None


def choose_new_value(span, slot, rng):
    """选出一个"合理但不常见"的替代值。**不要求 span 出现在句子里**。

    这一步与"如何落笔"分开，是关键：
      有的样本 span 并不逐字出现在句子中，例如
        span="No"   句子="Not considered very edible by humans"
        span="Summer" 句子="They bloom from early to mid summer."
      字面替换做不到，但**新值照样可以确定**（No->Yes、Summer->autumn），
      再由 LLM 就着新值把整句重写。若把定值与落笔绑在一起，
      这类样本会被整体判为"不可编辑"，白白丢掉。
    """
    slot_map = {
        "year": lambda: (str(year_shift(int(span), rng))
                         if _NUM.fullmatch(span.strip()) else None),
        "quantity": lambda: number_shift(span, rng),
        "polarity": lambda: flip_polarity(span, "", rng),
        "edibility": lambda: flip_polarity(span, "", rng),
    }
    if slot in slot_map:
        return slot_map[slot]()

    pool = POOL.get(slot)
    if not pool:
        return None
    cands = [x for x in pool if x.lower() != (span or "").lower()]
    if not cands:
        return None
    new = rng.choice(cands)
    if span[:1].isupper() and new[:1].islower():
        new = new[:1].upper() + new[1:]
    return new


# span 末尾的标点**不属于**要替换的内容。span 直接取自证据片段，
# 常带句末句号（'United States.'），照字面替换会吃掉句号、与后句连写：
#   "endemic to the United States. It can be found in Texas"
#   -> "endemic to the Kenya It can be found in Texas"
_TRAIL_PUNCT = " \t.,;:!?"


def apply_edit(sentence, span, new_value, entity_title=None):
    """把新值落到句子里：能在**词边界**上定位就替换，否则返回 None（交 LLM 改写）。

    为什么必须卡词边界：朴素 str.replace 会命中子串，产出语法垃圾。
    实测踩到（都是真事故）：
      span="No"  句子="Northern kelp crabs are not attractive..."
        -> str.replace 命中 "**No**rthern" 的 No，得到 "Yesrthern kelp crabs"
      span="NO"  句子="...is a **no**nvenomous colubrid snake..."
        -> 得到 "Yesnvenomous colubrid snake"
    卡词边界后这两例都定位失败，转而交给 LLM 改写整句 —— 这才是正确行为。

    另外两处细节：
      - 末尾标点不参与替换（见 _TRAIL_PUNCT），但替换后要把原标点还回去
      - 句子以 span 结尾时，保留原句号；否则新值可能紧贴后句

    ★ entity_title 的作用（不传会静默产出**假冲突**）
      span 常常同时也是**实体名的一部分**。实测踩到的真事故：
        span="Wolverhampton", 实体="Wolverhampton Art Gallery"
        句子="Wolverhampton Art Gallery is located in the City of Wolverhampton"
      取第一个匹配会把**博物馆的名字**改掉，真正承载答案的
      "City of Wolverhampton" 纹丝不动，于是：
        改后="Arequipa Art Gallery is located in the City of Wolverhampton"
      上下文里的答案**根本没变**（仍是 Wolverhampton = 参数答案）。
      这类样本在 CONFLICT 条件下不会有任何冲突，模型答 Wolverhampton 是
      **读上下文读对了**，却会被记成"追随参数知识"——**系统性虚高 PKD**。
      故必须屏蔽掉落在实体名内部的匹配，只在名外的匹配上落笔；
      若所有匹配都在实体名里，返回 None（宁可丢样本，不可造脏样本）。
    """
    if not span:
        return None
    core = span.strip().rstrip(_TRAIL_PUNCT)
    if not core:
        return None
    pat = r"(?<![A-Za-z0-9])" + re.escape(core) + r"(?![A-Za-z0-9])"

    # 实体名在句子里占据的字符区间 —— 这些区间内的匹配不算数
    banned = []
    if entity_title:
        tp = (r"(?<![A-Za-z0-9])" + re.escape(entity_title.strip())
              + r"(?![A-Za-z0-9])")
        banned = [(m.start(), m.end())
                  for m in re.finditer(tp, sentence, flags=re.IGNORECASE)]

    def inside_title(a, b):
        return any(a < y and b > x for x, y in banned)

    m = None
    for cand in re.finditer(pat, sentence, flags=re.IGNORECASE):
        if not inside_title(cand.start(), cand.end()):
            m = cand
            break
    if not m:
        return None
    # 注意：匹配用的是剥掉标点的 core，所以句子里原本跟着的标点/空格
    # 会**自然保留**在 sentence[m.end():] 中，不需要（也不能）再手工补一个。
    # 早期版本额外补了一次，产出 "endemic to the Kenya.. It"（双句号）
    # 与 "around 3 to 5 mm  in length"（双空格）。
    out = sentence[:m.start()] + new_value + sentence[m.end():]
    # 但新值自己可能已经带句末标点（number_shift 保留了 span 末尾的 '.'，
    # 而 span 的句号正是句子本身的句号），于是和句子里原样的句号撞成 ".."。
    # 收口：把重复的句末标点压掉。
    return re.sub(r"([.!?])\s*([.!?])(\s|$)", r"\1\3", out)


# 极性类 span 是**短语**时，字面替换留下的英语是错的：
#   "a choice edible mushroom" -edible-> "a choice **not edible** mushroom"
#   "an edible mushroom"       -edible-> "an **not edible** mushroom"
# 冠词与 "choice" 都不接受 not 开头的短语。真正的反事实应该写成
# "an inedible mushroom" / "a choice inedible mushroom"，
# 但那要理解名词短语结构，规则层做不可靠 —— 交给 LLM 改写整句。
# 判据：span 被替换后前后紧邻**另一个形容词**（choice/considered/nonpoisonous…）
# 或紧跟在冠词 a/an/the 之后，就放弃字面替换。
_ADJ_NEIGHBOR = re.compile(
    r"(?:\b(?:a|an|the|choice|considered|reportedly|highly|very|not|quite|"
    r"nonpoisonous|non-toxic|nontoxic)\s+)$", re.I)


def polarity_literal_unsafe(sentence, span):
    """极性短语替换后是否会产出语法错误的英语。"""
    if not span:
        return True
    core = span.strip().rstrip(_TRAIL_PUNCT)
    m = re.search(r"(?<![A-Za-z0-9])" + re.escape(core) + r"(?![A-Za-z0-9])",
                  sentence, flags=re.IGNORECASE)
    if not m:
        return True
    return bool(_ADJ_NEIGHBOR.search(sentence[:m.start()]))


# ---------------- LLM 自然改写（交接文档要求：必须做，不是可选）----------------

REWRITE_SYS = ("You rewrite a single sentence so that it states the OPPOSITE "
               "fact about the subject, keeping it fluent, natural and "
               "encyclopedic. Output ONLY the rewritten sentence, no quotes, "
               "no explanation.")

REWRITE_PROMPT = (
    "Question being asked: {question}\n"
    "The sentence below answers it with {old!r}; rewrite it so it answers "
    "{new!r} instead.\n"
    "Rules:\n"
    "  - Keep every other detail identical (same subject, same other facts).\n"
    "  - The sentence often phrases the fact as an adjective or negation "
    "rather than the literal word {old!r} (e.g. 'nonvenomous', 'inedible', "
    "'not considered very edible'). Find that phrasing and flip it.\n"
    "  - Do not add hedging, questions, or metadata.\n"
    "  - **Delete the old fact entirely.** A sentence that still contains "
    "{old!r} alongside {new!r} contradicts itself and is worthless.\n"
    "  - The result must read like an ordinary encyclopedia sentence.\n"
    "  - Do not keep more than one clause about the flipped fact; merge or "
    "drop the redundant one.\n"
    "Sentence: {sent}\n"
    "Rewritten:")


# 极性翻转的验收词：改写句里出现这些，说明命题被翻了
_POS_MARK = ("venomous", "edible", "toxic", "poisonous", "threatened")
_NEG_MARK = ("nonvenomous", "non-venomous", "not venomous", "venomless",
             "inedible", "not edible", "non-edible", "not poisonous",
             "nonpoisonous", "non-toxic", "nontoxic", "not toxic",
             "not threatened", "unthreatened")


def _polarity_flipped(text, old, new, original=None):
    """极性类改写的验收：目标极性以**独立词**形式出现。

    不能要求句中出现字面 "yes"：模型通常会把
    "is a nonvenomous snake" 改成 "is a venomous snake"，
    句里根本没有 "yes" 这个词，用字面闸门会把正确改写全部拒掉。

    必须卡词边界：正向标记 "edible" 是 "inedible" 的子串，
    子串匹配会让**根本没翻转**的句子（只把 because of 改成 due to）蒙混通过。
    实测踩到：Geastrum saccatum 原句 "considered inedible ... because of
    its bitter taste" 被改写成 "... due to its bitter taste" —— 事实没变，
    却因为含 "edible" 子串被记为"成功改写"。
    """
    new_l = (new or "").lower()
    pos = new_l in ("yes", "venomous", "edible", "poisonous", "toxic")
    marks = _POS_MARK if pos else _NEG_MARK

    def hit(t):
        t = t.lower()
        return any(re.search(r"(?<![a-z])" + re.escape(m) + r"(?![a-z])", t)
                   for m in marks)

    if not hit(text):
        return False
    # 仅有标记还不够：原句可能本来就是这个极性，模型只改了措辞
    # （"because of" -> "due to"），事实一点没变。
    # 要求原文**不具备**该极性，才算真的翻转。
    return not hit(original) if original is not None else True


def llm_rewrite(client, sentence, old, new, question=None, slot=None,
                max_tokens=128):
    """用 LLM 把句子改写成承载**相反事实**的自然句。

    为什么必须把 question 一起给模型：极性类样本的 span 与句面措辞常常对不上，
      span="No"  句子="...is a species of North American **nonvenomous** snake"
      span="No"  句子="...are regarded as **inedible**"
      span="No"  句子="Not considered very **edible** by humans"
    只告诉它"把 'No' 换成 'Yes'"，模型找不到可替换的位置，只能拒答
    （实测极性类因此只有 25% 成功率）。把问题和翻转意图讲清楚，
    模型才知道要去否定整个命题，而不是找一个字面词。

    质量闸门：必须提到新值、长度同量级、不含元话语；
    否则返回 None，调用方据此判为不可编辑 —— 不产出低质样本。
    """
    if not client:
        return None
    # 重试 2 次：实测约一半的第一次尝试会**自相矛盾**
    #   "Clavaria fragilis is not edible, although it is nonpoisonous and
    #    reportedly edible, ..."  <- 新旧事实同句并存，必须判废重来
    # 重试比放宽验收更好：验收是锚定集质量的守门人，不能为通过率而松。
    for attempt in range(3):
        try:
            txt, _ = client.chat(
                REWRITE_PROMPT.format(old=old, new=new, sent=sentence,
                                      question=question or "(unspecified)"),
                system=REWRITE_SYS, max_tokens=max_tokens)
        except Exception:
            continue
        t = (txt or "").strip().strip('"').strip()
        if not t:
            continue
        # 改写必须真的改了：原样返回说明模型没动（只改标点/换行也要拦）
        if t.lower() == (sentence or "").strip().lower():
            continue
        # 验收分两路：极性类看语义翻转，其余看字面新值
        if slot == "polarity":
            if not _polarity_flipped(t, old, new, original=sentence):
                continue
        elif new.lower() not in t.lower():
            continue
        if len(t) > 3 * len(sentence) + 40 or len(t) < 0.35 * len(sentence):
            continue
        if any(m in t.lower() for m in ("rewrit", "here is", "as an ai", "sentence:")):
            continue
        return t
    return None


def make_item(image, question, param_answer, context, span, slot,
              param_aliases=None, rng=None, note="", client=None,
              allow_llm=False, prov=None):
    """组装一条三条件样本。span/slot 由数据构造阶段给定。

    流程（定值与落笔分开，见 choose_new_value）：
      1. slot 经 refine_slot 细化为真正的编辑策略
      2. choose_new_value 选出替代值（不依赖 span 是否在句中）
      3. 能字面替换就替换；不能则交 LLM 就着新值重写整句
      4. 两者都不成 -> 判为不可编辑

    LLM 重写是**必需**而非可选：实测有相当比例的样本
    span 并不逐字出现在句子中（span="No" vs 句子 "Not considered very
    edible"），只靠字面替换会全部落空。
    """
    rng = rng or random
    eff_slot = refine_slot(question, span, slot)
    new_val = choose_new_value(span, eff_slot, rng)

    # 裸 yes/no 永不做字面替换：它们在英文里同时是限定词，
    # 句子里必然到处是 "no value" / "have no distinctive taste"，
    # 替换会命中这些限定词，产出 "with Yes value" 这类语法垃圾。
    # 这类必须由 LLM 就着目标事实重写整句。
    literal_ok = not (span or "").strip().lower() in ("yes", "no")
    # 极性槽位还要额外判一次：替换后前后是否紧邻形容词/冠词（见 _ADJ_NEIGHBOR）
    if literal_ok and eff_slot in ("polarity", "edibility"):
        literal_ok = not polarity_literal_unsafe(context, span)

    new_ctx, mode = None, "none"
    ent = (prov or {}).get("wikipedia_title") or ""
    if new_val:
        if literal_ok:
            new_ctx = apply_edit(context, span, new_val, entity_title=ent)
            if new_ctx:
                mode = "rule"
        if new_ctx is None and allow_llm and client:
            # 字面替换不了：让 LLM 就着新值重写
            new_ctx = llm_rewrite(client, context, span, new_val,
                                  question=question, slot=eff_slot)
            if new_ctx:
                mode = "llm"

    # ★ 落盘闸门：冲突上下文里**不能再出现原答案**。
    #   这条是防止"假冲突"的最后一道保险 —— 若编辑只改了实体名或别的同形词、
    #   承载答案的那处纹丝不动，上下文与参数知识仍一致，模型答对会被记成
    #   "追随参数知识"，系统性虚高 PKD。宁可判为不可编辑（丢样本），
    #   也不产出测不到冲突的脏样本。极性/数量类的"答案"不是字面词，跳过此闸。
    if new_ctx and eff_slot not in ("polarity", "quantity", "edibility"):
        core = (span or "").strip().rstrip(_TRAIL_PUNCT)
        if core and len(core) >= 3:
            pp = r"(?<![A-Za-z0-9])" + re.escape(core) + r"(?![A-Za-z0-9])"
            if re.search(pp, new_ctx, flags=re.IGNORECASE):
                new_ctx, mode = None, "none"

    return {
        "image": image,
        # ★ 三个问法字段**都要**落盘，各有用处，缺一会静默改变口径：
        #     question         本次构造实际使用的问法（--anchors 分支下=带实体名的）
        #     question_named   带实体名的问法，eval_pkd --question-field named 读它
        #     question_generic 原始泛化问法（"this plant"），用于感知缺口分析
        #   历史教训：曾经顶层只有 question_generic，question_named 只存在于
        #   provenance 里，于是 --question-field named 静默回退到 r["question"]，
        #   **评测照常出数**，但测的其实是"能不能从图里认出物种"而非知识支配。
        #   后来又手滑把 question 本身删掉，导致 analyze_interference 取
        #   v["question"] 直接 KeyError。故此处三个都写死，并由落盘前断言把关。
        "question": question,
        "question_named": (prov or {}).get("question_named") or question,
        "parametric_answer": param_answer,
        "parametric_aliases": param_aliases or [param_answer],
        "context_original": context,
        "context_conflict": new_ctx,
        "edit_span": span,
        "edit_slot": eff_slot,
        "slot_coarse": slot,
        "edit_to": new_val,
        "context_answer": new_val,
        "context_answer_aliases": ([new_val] if new_val else []),
        "valid": new_ctx is not None,
        "rewrite_mode": mode,
        "note": note,
        # 溯源字段：锚定集是用**实体名问法**筛出来的（模型确实拥有该事实），
        # 而 E-VQA 原始问法用 "this plant" 指代，实体只能靠图像识别。
        # 没有这些字段，最终数据集就无法回溯"参数答案当时是怎么问出来的"，
        # 论文的 sample-selection 一节将无法自证。
        "question_generic": (prov or {}).get("question_generic", ""),
        "model_param_answer": (prov or {}).get("model_param_answer", ""),
        "wikipedia_title": (prov or {}).get("wikipedia_title", ""),
        "provenance": prov or {},
    }


DEMO_RAW = [
    dict(image="demo.jpg", question="What country is this dish from?",
         param_answer="Italy", span="Italy", slot="country",
         context="This dish originated in Italy and is traditionally served with olive oil."),
    dict(image="demo.jpg", question="When was this building completed?",
         param_answer="1889", span="1889", slot="year",
         context="The building was completed in 1889 after four years of construction."),
    dict(image="demo.jpg", question="What material is the sculpture made of?",
         param_answer="bronze", span="bronze", slot="material",
         context="The sculpture is cast in bronze and weighs approximately three tonnes."),
    dict(image="demo.jpg", question="What sport is being played?",
         param_answer="cricket", span="cricket", slot="sport",
         context="The players are competing in cricket on a grass pitch in the afternoon."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=None,
                    help="raw jsonl（字段 image/question/param_answer/span/slot/context）")
    ap.add_argument("--anchors", default=None,
                    help="锚定集（screen_knowledge + verify_anchors + map_images 的产物）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--mode", choices=("span", "llm"), default="span",
                    help="span=规则替换（确定可复现）；llm=LLM 自然改写")
    ap.add_argument("--llm-fail-threshold", type=float, default=0.5,
                    help="llm 模式下改写失败率超过此值则报错退出（默认 0.5）")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="qwen2.5-vl-3b")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if args.demo:
        src = DEMO_RAW
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "data", "medusa_demo.jsonl")
    elif args.anchors:
        # 锚定集字段名 -> make_item 需要的字段名
        src = [dict(image=a.get("image") or "",
                    question=a["question_named"],
                    param_answer=a["param_answer"],
                    span=a["span"],
                    slot=a["slot"],
                    context=a["context"],
                    param_aliases=a.get("parametric_aliases"),
                    note=a.get("wikipedia_title", ""),
                    prov=dict(question_generic=a.get("question", ""),
                              model_param_answer=a.get("model_param_answer", ""),
                              model_answer_generic=a.get("model_answer_generic", ""),
                              wikipedia_title=a.get("wikipedia_title", ""),
                              match_tier=a.get("match_tier", ""),
                              slot=a.get("slot", "")))
               for a in read_jsonl(args.anchors)
               if a.get("context") and a.get("span")]
        out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "data", "medusa.jsonl")
    elif args.inp:
        src = read_jsonl(args.inp)
        out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "data", "medusa.jsonl")
    else:
        print("需要 --demo / --in <raw.jsonl> / --anchors <anchors.jsonl>")
        return

    if args.limit:
        src = src[:args.limit]

    client = None
    if args.mode == "llm":
        # LLM 模式：字面替换不了的样本需要模型就着新值重写整句
        from common import Client
        client = Client(base_url=args.base_url, api_key="EMPTY",
                        model=args.model, support_vision=False)
        print("LLM 改写模式：模型 %s" % args.model)

    rows = []
    for r in src:
        r = dict(r)
        r.setdefault("image", "")
        item = make_item(param_aliases=r.get("param_aliases"),
                         rng=rng, note=r.get("note", ""),
                         prov=r.get("prov"),
                         client=client, allow_llm=(args.mode == "llm"),
                         **{k: r[k] for k in
                            ("image", "question", "param_answer", "span",
                             "slot", "context") if k in r})
        rows.append(item)

    ok = [r for r in rows if r["valid"]]
    n_rule = sum(1 for r in ok if r.get("rewrite_mode") == "rule")
    n_llm = sum(1 for r in ok if r.get("rewrite_mode") == "llm")

    print("样本 %d 条，可编辑 %d 条（%.1f%%）"
          % (len(rows), len(ok), 100.0 * len(ok) / max(len(rows), 1)))
    if args.mode == "llm":
        print("  规则替换 %d，LLM 重写 %d" % (n_rule, n_llm))

    # 槽位可行性统计：哪些槽位改不动，需补替换表
    from collections import Counter
    total_slot = Counter(r["edit_slot"] for r in rows)
    ok_slot = Counter(r["edit_slot"] for r in ok)
    print("\n槽位可编辑率：")
    for s, n in total_slot.most_common():
        print("  %-10s %4d / %4d  (%.1f%%)" % (s, ok_slot[s], n, 100.0 * ok_slot[s] / n))

    bad = [r for r in rows if not r["valid"]]
    if bad:
        print("\n不可编辑 %d 条，样例：" % len(bad))
        for r in bad[:5]:
            # r["question"] 有 KeyError 风险：make_item 早先版本漏写过该字段，
            # 这里用 .get 兜底，避免"统计阶段反而把整个 build 炸掉"——
            # 那会让已经跑完的（可能耗时数十分钟的 LLM 改写）全部白费。
            print("  [槽位 %s] %s" % (r["edit_slot"],
                                      (r.get("question") or "")[:60]))
            print("     span=%r 未在句子中找到" % r["edit_span"])
        # ★ 这 %d 条的 context_conflict 是空的。eval_pkd 在 CONFLICT 条件里
        #   会回退到 context_original（= 与参数知识一致），即 CONST 条件。
        #   它们不可能表现出冲突行为，若被计入 CONFLICT 分母会把 PKD 稀释。
        #   eval_pkd 已用 n_conflict 作为分母排除之，这里把数量打出来，
        #   好让"数据质量"和"指标口径"两处对得上账。
        print("  ⚠ 这 %d 条在 CONFLICT 条件下会退化为 CONSIST 条件，"
              "PKD 分母应减去它们（eval_pkd 已按 n_conflict 处理）" % len(bad))

    # 落盘前的自检：question_named 为空会导致 eval_pkd --question-field named
    # 静默回退到泛化问法，口径悄悄改变却不报错（详见 make_item 里的注释）。
    empty_named = [r for r in rows if not (r.get("question_named") or "").strip()]
    if empty_named:
        raise SystemExit(
            "❌ %d 条样本的 question_named 为空 —— PKD 评测会静默回退到 "
            "'this plant' 泛化问法，测的将不是参数知识支配。请检查 anchors "
            "文件是否带 question_named 字段。" % len(empty_named))
    print("  自检通过：%d 条 question_named 全部非空" % len(rows))

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    write_jsonl(out, rows)
    print("\n已写入:", os.path.abspath(out))


if __name__ == "__main__":
    main()
