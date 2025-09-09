import json, re, hashlib, difflib
from pathlib import Path
from typing import Dict, Any, Tuple
import regex as re
from tqdm import tqdm
import regex as regx
from difflib import SequenceMatcher

TOYS_PATH = r"C:\Users\USER\toysub\toysub_package\dataset\translated_toys0906.json"                       
TARGET_PATH = r"C:\Users\USER\toysub\toysub_package\dataset\english_data\translated_toys_data_1.json"  
OUT_PATH = r"C:\Users\USER\toysub\toysub_package\dataset\english_data\translated_toys_data_2.json"

# ===== 規則化 =====
_punc = re.compile(r"[\p{P}\p{S}\s]+")

def fingerprint(s: str) -> str:
    n = normalize_text(s)
    return hashlib.sha1(n.encode("utf-8")).hexdigest()

def fuzzy_match(q_first: str, candidates_first:list[str]) -> Tuple[str, float]:
    if not q_first or not candidates_first:
        return None, 0.0
    best = difflib.get_close_matches(q_first, candidates_first, n=1, cutoff=0.0)
    if not best:
        return None, 0.0
    best_str = best[0]
    score = difflib.SequenceMatcher(None, q_first, best_str).ratio()
    return best_str, score


def normalize_text(s: str) -> str:
    s = (s or "").strip()      # 全形轉半形
    s = re.sub(r"\u3000", " ", s)  # 小寫、移除空白與符號
    s = s.lower()
    s = _punc.sub("", s)
    return s

def first_sentence(text:str)->str:
    t= (text or "").strip()
    if not t : return ""
    original = t
    for sep in ("：", ":", " - ", " — ", "–", "—",";"):
        if sep in t:
            left = t.split(sep, 1)[0].strip()
            if left:
                t = left
                break
    
    sents = [s.strip() for s in _SENT_WITH_PUNC.findall(t) if s.strip()]
    if not sents: return original[:60]
    
    result = sents[0]
    if len(normalize_text(result)) < 2 :
        return ""
    return result

def first_two_sentences(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    # 優先切掉標題/名稱分隔符號
    for sep in ("：", ":", " - ", " — ", "–", "—", ";"):
        if sep in t:
            left = t.split(sep, 1)[0].strip()
            if left:
                t = left
                break

    # 用 regex 抓句子（包含句末標點）
    sents = [s.strip() for s in _SENT_WITH_PUNC.findall(t) if s.strip()]
    if not sents:  # fallback
        return t[:60]

    # 取前兩句
    result = " ".join(sents[:2])
    if len(normalize_text(result)) < 2:
        return ""
    return result

def contains_match(a:str,b:str,min_cover :float = 0.7) -> bool:
    na,nb = normalize_text(a), normalize_text(b)
    if not na or not nb: return False
    short,long = (na,nb) if len(na)<len(nb) else (nb,na)
    if short in long: return (len(short) / max(1, len(long))) >= min_cover
    return False

def whole_text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

_SENT_WITH_PUNC = regx.compile(r"[^。．\.！!？\?…\n\r]+[。．\.！!？\?…]?")

#資料載入
toys   = json.loads(Path(TOYS_PATH).read_text("utf-8"))
target = json.loads(Path(TARGET_PATH).read_text("utf-8"))

# 建立索引
toys_index_by_fp: Dict[str, Tuple[str, str, str]] = {}  
toys_first_list: list[str] = []                      
toys_first_map_raw: Dict[str, Tuple[str, str]] = {} 

for t in toys:
    src_inp = (t.get("input") or "").strip()
    src_topic = (t.get("TOPIC") or "").strip()
    if not src_inp: continue
    f1 = first_sentence(src_inp)
    if not f1 : continue

    fp = fingerprint(f1)
    toys_index_by_fp[fp] = (src_inp,src_topic,f1)
    toys_first_list.append(f1)
    toys_first_map_raw.setdefault(f1,(src_inp,src_topic))

updated = 0
matched_exact = 0
matched_contains = 0
matched_fuzzy = 0
not_matched = 0

for rec in tqdm(target, desc="匹配中"):
    tgt_inp = (rec.get("output") or "").strip()
    tgt_first2 = first_two_sentences(tgt_inp)

    # --- 精準 fingerprint ---
    fp = fingerprint(tgt_first2)
    hit = toys_index_by_fp.get(fp)
    if hit:
        new_input, new_topic, f1 = hit
        rec["output"] = new_input
        rec["TOPIC"] = new_topic
        matched_exact += 1
        updated += 1

        # 剔除用過的
        toys_index_by_fp.pop(fp, None)
        if f1 in toys_first_list:
            toys_first_list.remove(f1)
        toys_first_map_raw.pop(f1, None)
        continue

    # --- 包含 ---
    best_contained = next(
        (cand for cand in toys_first_list
         if contains_match(tgt_first2, cand) or contains_match(cand, tgt_first2)),
        None
    )
    if best_contained:
        new_input, new_topic = toys_first_map_raw[best_contained]
        rec["output"] = new_input
        rec["TOPIC"] = new_topic
        rec["_link_status"] = "contains"
        matched_contains += 1
        updated += 1

        # 剔除用過的
        toys_first_list.remove(best_contained)
        toys_first_map_raw.pop(best_contained, None)
        continue

    # --- 模糊比對 ---
    best_first, score = fuzzy_match(tgt_first2, toys_first_list)
    if best_first and score >= 0.6:   # 建議門檻維持高一點
        new_input, new_topic = toys_first_map_raw[best_first]
        rec["output"] = new_input
        rec["TOPIC"] = new_topic
        rec["_link_status"] = "fuzzy"
        rec["_link_score"] = round(score, 3)
        matched_fuzzy += 1
        updated += 1

        # 剔除用過的
        if best_first in toys_first_list:
            toys_first_list.remove(best_first)
        toys_first_map_raw.pop(best_first, None)
        continue

    # --- 整段相似度 fallback ---
    best_full, best_score = None, 0.0
    for cand_first, (cand_input, cand_topic) in toys_first_map_raw.items():
        sim = whole_text_similarity(tgt_inp, cand_input)
        if sim > best_score:
            best_full, best_score = cand_first, sim

    if best_full and best_score >= 0.6:
        new_input, new_topic = toys_first_map_raw[best_full]
        rec["output"] = new_input
        rec["TOPIC"] = new_topic
        rec["_link_status"] = "fulltext"
        rec["_link_score"] = round(best_score, 3)
        updated += 1

        # 剔除用過的
        if best_full in toys_first_list:
            toys_first_list.remove(best_full)
        toys_first_map_raw.pop(best_full, None)
    else:
        not_matched += 1
        rec.setdefault("_link_status", "unmatched")
        rec["_link_q_first"] = tgt_first2
        rec["_link_best_first"] = best_first or ""
        rec["_link_score"] = round(score or 0.0, 3)
        rec["fulltext_score"] = round(best_score, 3)


# 輸出
Path(OUT_PATH).write_text(json.dumps(target, ensure_ascii=False, indent=2), "utf-8")
print({
    "updated": updated,
    "matched_exact": matched_exact,
    "matched_fuzzy": matched_fuzzy,
    "not_matched": not_matched,
    "out": OUT_PATH
})

unsigned_inputs = set(toys_first_map_raw.keys())
toys_unmatch = []
for r in toys:
    first_sents = first_sentence(r.get("input",""))
    if first_sents in unsigned_inputs:
        toys_unmatch.append(r)

UNMATCHED_PATH = str(Path(TOYS_PATH).with_name("toys_unmatched.json"))
Path(UNMATCHED_PATH).write_text(json.dumps(toys_unmatch, ensure_ascii=False, indent=2), "utf-8")
print(f"剩餘未匹配的 toys 已輸出到: {UNMATCHED_PATH}, 共 {len(toys_unmatch)} 筆")
