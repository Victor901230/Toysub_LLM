from transformers import pipeline
from tqdm import tqdm
import re
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
import torch
import os

name = "mistralai/Mistral-7B-Instruct-v0.1"
# 載入 tokenizer 和 model
question_tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
if question_tokenizer.pad_token is None:
    question_tokenizer.pad_token = question_tokenizer.eos_token
question_model = AutoModelForCausalLM.from_pretrained(name, device_map="auto", torch_dtype="auto")

def generate_text(prompt, max_new_tokens=256):
    inputs = question_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")
    for k in inputs:
        inputs[k] = inputs[k].to("cuda")
    outputs = question_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=question_tokenizer.eos_token_id
    )
    generated = question_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated.strip()

def extract_question(text):
    """
    從模型輸出的完整段落中提取出真正的問題。
    """
    match = re.search(r'Text:\s*(.+)', text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        return text.strip()  

def generate_batch(prompts, tokenizer, model, max_new_tokens=512, batch_size=32):
    all_outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="批次生成中"):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding="longest", truncation=True, max_length=1024)
        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids")
        for k in inputs:
            inputs[k] = inputs[k].to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_outputs.extend([d.strip() for d in decoded])
    return all_outputs

def generate_question(passages, batch_size=40,question_output_path=r"C:\Users\USER\國科會CHATGPT\datasets\instruction_questions.json"):
        question_prompts = []
        passage_infos = []
        for p in passages:
            chunks = p.get("text")
            prompt_q = (
            "Based on the given passage, formulate ONE deep, multi-step reasoning question that cannot be answered by simply quoting the passage directly. Encourage inferential thinking.\n"
            f"Text: {chunks}"
            )
            question_prompts.append(prompt_q)
            passage_infos.append((p, chunks))
        
        questions = generate_batch(question_prompts, question_tokenizer, question_model, max_new_tokens=128, batch_size=batch_size)
        intermediate_questions = []
        for (p, chunks), q in zip(passage_infos, questions):
            real_q = extract_question(q)
            if 10 <= len(real_q.split()) <= 150:
                intermediate_questions.append({
                    "id": p.get("id", ""),
                    "chunk": chunks,
                    "question": real_q,
                    "text": p.get("text", ""),
                    "source": p.get("source", ""),
                    "TOPIC": p.get("TOPIC", "")
                })
            else:
                print(f"問題長度不符，丟棄: {real_q}")

        # 儲存中繼問題檔
        with open(question_output_path, "w", encoding="utf-8") as f:
            json.dump(intermediate_questions, f, ensure_ascii=False, indent=2)
        print(f"問題已儲存到 {question_output_path}（共 {len(intermediate_questions)} 筆）")

def clean_output(text):
    match = re.search(r'\[Answer\]\s*(.*)', text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()

def generate_answers(question_file, batch_size=32, output_path=r"C:\Users\USER\國科會CHATGPT\datasets\instruction_augmented.json"):
    with open(question_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    
    valid_pairs = [
        ({"id": q.get("id", ""), "text": q.get("text", ""), "source": q.get("source", ""), "TOPIC": q.get("TOPIC", "")},
         q["chunk"],
         q["question"]) for q in questions
    ]
    answer_prompts = []
    for p, chunk, real_q in valid_pairs:
        prompt = (
    f"Based on the following content, please answer the question in a complete, detailed, and coherent manner (at least 100 words):\n\n"
    f"[Content]\n{chunk}\n\n"
    f"[Question]\n{real_q}\n\n"
    f"[Answer]"
        )
        answer_prompts.append(prompt)

    answers = generate_batch(answer_prompts, question_tokenizer, question_model, max_new_tokens=512, batch_size=batch_size)
    answers = [clean_output(ans) for ans in answers]
    qa_pairs = []
    for (p, chunk, real_q), ans in zip(valid_pairs, answers):
            ans = re.sub(r'\s+', ' ', ans).strip()
            if len(ans) >= 50 and not any(word in ans.lower() for word in ["i don't know", "not mentioned", "unable to answer"]):
                qa_pairs.append({
                    "id": p.get("id", ""),
                    "instruction": real_q,
                    "input": p.get("text", ""),
                    "output": ans,
                    "passage": p.get("text", ""),
                    "source": p.get("source", ""),
                    "TOPIC": p.get("TOPIC", "")
                })
            else:
                print(f"回答不合格: {ans}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    print(f"完成！已儲存 {len(qa_pairs)} 筆 QA 至 {output_path}")

if __name__ == "__main__":
    question_path = r"C:\Users\USER\國科會CHATGPT\datasets\instruction_questions.json"
    answer_path = r"C:\Users\USER\國科會CHATGPT\datasets\instruction_augmented.json"
    raw_data_path = r"C:\Users\USER\國科會CHATGPT\datasets\augmented_test.json"
    with open(raw_data_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    #generate_questions = generate_question(docs, batch_size=40, question_output_path=question_path)
    generate_answers(question_file=question_path, batch_size=32, output_path=answer_path)
    print(f"已產生QA資料")