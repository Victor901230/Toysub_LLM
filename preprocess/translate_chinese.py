import re
import json
from tqdm import tqdm
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import torch
from opencc import OpenCC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name ="Helsinki-NLP/opus-mt-zh-en"
zh_en_tokenizer = AutoTokenizer.from_pretrained(model_name)
zh_en_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
cc = OpenCC('t2s')

def translate_fields(batch_docs):
    results = []
    texts = []
    for doc in batch_docs:
        texts.extend([
            doc.get("question", ""),
            doc.get("input", ""),
            doc.get("output", ""),
            doc.get("TOPIC","")
        ])
    # 翻譯所有欄位（一次處理三倍數量）
    inputs = zh_en_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        translated_tokens = zh_en_model.generate(
            **inputs,
            max_length=256,
            num_beams=8,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            length_penalty=1.2,
            early_stopping=True,
            do_sample=False
        )
    translations = zh_en_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    
    # 每 3 筆為一組，對應 question/input/output
    for i in range(0, len(translations), 4):
        results.append({
            "question": translations[i],
            "input": translations[i + 1],
            "output": translations[i + 2],
            "TOPIC": translations[i + 3]
        })
    return results

def translate_text(docs):
    results=[]
    texts = []
    for doc in docs:
        texts.extend([
            doc.get("output") 
        ])
    inputs = zh_en_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        translated_tokens = zh_en_model.generate(
            **inputs,
            max_length=256,
            num_beams=8,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            length_penalty=1.2,
            early_stopping=True,
            do_sample=False
        )
    translations = zh_en_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

    for i in range(0, len(translations)):
        results.append({
            "output": translations[i],
        })
    return results

def convert_all_to_traditional(data: list) -> list:
    cc = OpenCC('s2twp')
    for item in data:
        if "question" in item:
            item["question"] = cc.convert(item["question"])
        if "output" in item:
            item["output"] = cc.convert(item["output"])
    return data

def main():
    with open(r"C:\Users\USER\toysub\dataset\chinese_data\chinese_toys_data.json", "r", encoding="utf-8") as f:
        raw_documents = json.load(f)
    translated_output = []
    for i in tqdm(range(0, len(raw_documents), 8), desc="批次翻譯中"):
        batch_docs = raw_documents[i:i + 8]
        # translated_batch = translate_text(batch_docs)
        translated_batch = translate_fields(batch_docs)
        translated_output.extend(translated_batch)
    

    with open(r"C:\Users\USER\toysub\dataset\english_data\translated_toys_data.json", "w", encoding="utf-8") as f:
        json.dump( translated_output, f, ensure_ascii=False, indent=4)
   
    
if __name__ == "__main__":
    main()