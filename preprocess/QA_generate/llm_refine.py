from langchain_community.llms import Ollama  
import json
from tqdm import tqdm

def batch_refine_with_llm(raw_translations, model_name="mistral"):
    llm = Ollama(model=model_name, temperature=0.2) 
    prompts = [
        f"Please correct and improve the following translation for grammar, fluency, and clarity:\n\n{t}"
        for t in raw_translations
    ]
    improved_list = llm.batch(prompts)
    return [
        text.replace("\\n", " ").replace("\\", "").strip()
        for text in improved_list
    ]

def main():
    with open(r"C:\Users\USER\toysub\dataset\translated_chinese_data.json", "r", encoding="utf-8") as f:
        raw_documents = json.load(f)

    translated_output = []
    batch_size = 8

    for i in tqdm(range(0, 100, batch_size), desc="批次精修中"):
        batch_docs = raw_documents[i:i + batch_size]
        raw_texts = [doc.get("translated_text", "") for doc in batch_docs]

        # ✅ 批次呼叫 LLM 進行翻譯精修
        refined_texts = batch_refine_with_llm(raw_texts)

        for doc, refined in zip(batch_docs, refined_texts):
            new_doc = dict(doc)
            new_doc["translated_text"] = refined  # 替換為精修版本
            translated_output.append(new_doc)

    with open(r"C:\Users\USER\toysub\dataset\refined_chinese_data.json", "w", encoding="utf-8") as f:
        json.dump(translated_output, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
