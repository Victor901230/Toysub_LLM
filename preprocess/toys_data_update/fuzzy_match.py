import json, re, hashlib, difflib
from pathlib import Path
from typing import Dict, Any, Tuple
import regex as re
from tqdm import tqdm

TOYS_PATH = r"C:\Users\USER\toysub\toysub_package\dataset\translated_toys0906.json"                       
TARGET_PATH = r"C:\Users\USER\toysub\toysub_package\dataset\english_data\translated_toys_data_1.json"  
OUT_PATH = r"C:\Users\USER\toysub\toysub_package\dataset\english_data\translated_toys_data_2.json"

# ===== 規則化 =====
_punc = re.compile(r"[\p{P}\p{S}\s]+")

def normalize_text(s: str) -> str:
    s = (s or "").strip()      # 全形轉半形
    s = re.sub(r"\u3000", " ", s)  # 小寫、移除空白與符號
    s = s.lower()
    s = _punc.sub("", s)
    return s

def fingerprint(s: str) -> str:
    n = normalize_text(s)
    return hashlib.sha1(n.encode("utf-8")).hexdigest()

def fuzzy_match(q: str, candidates): # 回傳 (best_string, score 0~1)
    if not q or not candidates:
        return None, 0.0
    best = difflib.get_close_matches(q, candidates, n=1, cutoff=0.0)
    if not best:
        return None, 0.0
    best_str = best[0]
    score = difflib.SequenceMatcher(None, q, best_str).ratio()
    return best_str, score

toys = json.loads(Path(TOYS_PATH).read_text("utf-8"))
target = json.loads(Path(TARGET_PATH).read_text("utf-8"))

toys_index: Dict[str, Tuple[str, str]] = {}
toys_inputs = []  # 模糊比對用
for r in toys:
    src_inp = r.get("input", "")
    src_topic = r.get("TOPIC", "")
    if not src_inp:
        continue
    fp = fingerprint(src_inp)
    toys_index[fp] = (src_inp, src_topic)
    toys_inputs.append(src_inp)

updated = 0
matched_exact = 0
matched_fuzzy = 0
not_matched = 0

for rec in tqdm( target,desc="比對中"):
    tgt_inp = rec.get("input", "")
    # 1) 精準鍵：fingerprint
    fp = fingerprint(tgt_inp)
    if fp in toys_index:
        new_input, new_topic = toys_index[fp]
        rec["input"] = new_input
        rec["TOPIC"] = new_topic
        matched_exact += 1
        updated += 1
        continue

    # 2) 模糊匹配（僅當目標檔與 toys 在「同語言或高度相似」時有效）
    best, score = fuzzy_match(tgt_inp, toys_inputs)
    if best and score >= 0.9:
        new_input, new_topic = toys_index[fingerprint(best)]
        rec["input"] = new_input
        rec["TOPIC"] = new_topic
        matched_fuzzy += 1
        updated += 1
    else:
        not_matched += 1
        rec.setdefault("_link_status", "unmatched")
        rec["_link_score"] = round(score, 3)

# 輸出
Path(OUT_PATH).write_text(json.dumps(target, ensure_ascii=False, indent=2), "utf-8")
print({
    "updated": updated,
    "matched_exact": matched_exact,
    "matched_fuzzy": matched_fuzzy,
    "not_matched": not_matched,
    "out": OUT_PATH
})
