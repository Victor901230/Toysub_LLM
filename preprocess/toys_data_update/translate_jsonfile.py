import os
import json
import time
import math
import re
import requests
from pathlib import Path
from typing import List, Dict, Any

input_json = r"C:\Users\USER\toysub\toysub_package\dataset\toys.json"
out_json = r"C:\Users\USER\toysub\dataset\toysub_package\translated_toys0906.json"

FIELD_TO_TRANSLATE = ["input","TOPIC"]
TARGET_LANG = "EN"           
SOURCE_LANG = None   
MAX_CHARS_PER_TEXT = 2000         # 單段文字上限（長文會先分段再送出）
BATCH_SIZE = 50                   # 一次 API 請求要送幾段（多段 text）
REQ_SLEEP = 0.3                   # 每次請求後短暫 sleep，避免觸發流量限制
MAX_RETRIES = 5   

DEEPL_API_KEY = "d119527c-d582-4153-b329-f1ddc145afcd:fx" 
DEEPL_ENDPOINT = "https://api-free.deepl.com/v2/translate"

#=======工具=========
SENT_PAT = re.compile(r"([^。！？!?；;：:\n]+[。！？!?；;：:\n])", re.U)

def split_sentences(text:str)->list[str]:
    parts = SENT_PAT.findall(text)
    tail = text[sum(len(p) for p in parts):]
    if tail.strip():
        parts.append(tail)
    return [p for p in (s.strip() for s in parts) if p]

def pack_chunks(sentences:list[str],max_chars:int)->list[str]:
    chunks,buf = [],[]
    length = 0
    for s in sentences:
        s_len = len(s)
        if s_len>max_chars:
            for i in range(0, s_len, max_chars):
                if buf:
                    chunks.append("".join(buf))
                    buf, length = [], 0
                chunks.append(s[i:i+max_chars])
            continue

        if length+s_len<=max_chars:
            buf.append(s)
            length+=s_len
        else:
            chunks.append("".join(buf))
            buf,length = [s],s_len
    
    if buf :
        chunks.append("".join(buf))
    return chunks

def chunk_text(text: str, max_chars: int) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # 先分句，後打包
    return pack_chunks(split_sentences(text), max_chars)

def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

def translate(texts: List[str],target_lang: str,source_lang: str = None)->list[str]:
    headers = {"Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"}
    data = [("text", t) for t in texts]
    data.append(("target_lang", target_lang))
    if source_lang:
        data.append(("source_lang", source_lang))

    resp = requests.post(DEEPL_ENDPOINT, headers=headers, data=data, timeout=60)
    if resp.status_code == 200:
        return [tr["text"] for tr in resp.json()["translations"]]
    else:
        raise RuntimeError(f"DeepL錯誤 {resp.status_code}: {resp.text}")

def main():
    data: List[Dict[str, Any]] = json.loads(Path(input_json).read_text("utf-8"))
    chunk_store :dict[int,dict[str,list[str]]] = {}

    flat_chunks: List[str] = []

    for i, rec in enumerate(data):
        for field in FIELD_TO_TRANSLATE:
            text = rec.get(field, "")
            chunks = chunk_text(text, MAX_CHARS_PER_TEXT)
            if not chunks:
                continue
            chunk_store.setdefault(i, {})[field] = chunks
            flat_chunks.extend(chunks)

    if not flat_chunks:
        print("沒有需要翻譯的內容。")
        Path(out_json).write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
        return

    # 批次呼叫 DeepL（維持順序）
    translated: List[str] = []
    for batch in batched(flat_chunks, BATCH_SIZE):
        translated.extend(translate(batch, TARGET_LANG, SOURCE_LANG))
        time.sleep(REQ_SLEEP)

    # 將平坦的翻譯結果還原到各筆紀錄
    cursor = 0
    for i, field_chunks in chunk_store.items():
        for field, chunks in field_chunks.items():
            n = len(chunks)
            zh_chunks = translated[cursor: cursor + n]
            cursor += n
            # 合併還原
            joined = "".join(zh_chunks)
            data[i][field] = joined

    # 寫出結果
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    print(f"Done. 翻譯完成：{out_json}")


if __name__ =="__main__":
    main()