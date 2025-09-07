from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM,BitsAndBytesConfig,AutoModelForSeq2SeqLM
import torch
from rank_bm25 import BM25Okapi
import torch
import time
import os
import json
import re
from opencc import OpenCC
from peft import PeftModel
import jieba 
from deep_translator import GoogleTranslator
from datetime import datetime

#初始化重新排序模型
RERANK_TOKENIZER = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_MODEL = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2").to("cuda").eval()

"""微調模型全域變數""" 
model = None
tokenizer = None

"""初始化 BM25 全域變數"""
bm25_model = None
bm25_docs = []
bm25_tokenized_corpus = []

"""初始化翻譯模型"""
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# zh_model_name ="Helsinki-NLP/opus-mt-en-zh"
# zh_en_tokenizer = AutoTokenizer.from_pretrained(zh_model_name)
# zh_en_model = AutoModelForSeq2SeqLM.from_pretrained(zh_model_name).to(device)

def load_json_file(json_path): #擷取JSON內容(文字跟主題)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    documents = []
    for idx, item in enumerate(data):
        content =  item.get("translated_text") 
        if content:
            sentences= re.split(r"[。!?\n]",content)
            topic = next((s.strip() for s in sentences if s.strip()),"")
            documents.append({
                "TOPIC":topic,
                "question":"",
                "translated_text":content,
                               })
    return documents

def extract_all(filepath): #擷取JSON內容(回答、主題)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    extracted = []
    for item in data:
        topic = ""
        if "TOPIC" in item:
            topic = item.get("TOPIC","").strip()
        question = item.get("question", "").strip()
        output = item.get("output", "").strip()
        text = f"{output}"
        extracted.append({
            "TOPIC":topic,
            "question":question,
            "translated_text": text
        })
    return extracted

"""載入微調模型並和Qwen/Qwen1.5-1.8B-Chat合併"""
def load_finetuning_model():  
    global model, tokenizer
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,  # float16
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    
    tokenizer = AutoTokenizer.from_pretrained("./final_qwen_model_613", trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-1.8B-Chat",
        trust_remote_code=True,
        device_map="auto",
        offload_folder="./offload",   # ✅ 新增這一行
        offload_state_dict=True 
    )
    base_model.resize_token_embeddings(len(tokenizer))  # 強制詞表對齊

    model = PeftModel.from_pretrained(base_model, "./final_qwen_model_613")
    model = model.merge_and_unload()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")
    special_token_candidates = ["<|extra_0|>","<|endoftext|>",tokenizer.eos_token]
    for token in special_token_candidates:
        if tokenizer.convert_tokens_to_ids(token)!=tokenizer.unk_token_id:
            tokenizer.pad_token = token
            break
        else:
            raise ValueError("找不到可用PAD_TOKEN")
    return model, tokenizer

"""dense retrival database，PAGE CONTENT有改格式，embedding_model也有換"""
def vectorize(documents, save_path="faiss_index"): 
    # 若尚未轉為 Document，則先轉換
    if isinstance(documents[0], dict):
        documents = [
                Document(
                    page_content=(
                        f"Topic: {doc.get('TOPIC', '')}\n"
                        f"Q: {doc.get('question', '')}\n"
                        f"A: {doc.get('translated_text', '')}"
                    ),
                    metadata={
                        "topic": doc.get("TOPIC", ""),
                        "raw_input": doc.get("translated_text", ""),
                        "type": "faq"
                    }
                )
                for doc in documents
            ]

    embedding_model_name = "BAAI/bge-large-zh"
    model_kwargs = {"device": "cuda"}
    encode_kwargs = {"normalize_embeddings":True}
    embeddings = HuggingFaceBgeEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs,encode_kwargs = encode_kwargs)

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(save_path)
    return vectorstore ,documents

"""BM25 稀疏索引"""
def bm25_index(documents):
    global bm25_model, bm25_docs, bm25_tokenized_corpus
    bm25_docs = documents
    bm25_tokenized_corpus = [list(jieba.cut(doc.page_content)) for doc in documents]
    bm25_model = BM25Okapi(bm25_tokenized_corpus)

"""BM25關鍵字搜尋"""
def bm25_search(query, top_k=5): 
    if bm25_model is None:
        raise ValueError("BM25 尚未初始化")

    query_tokens = list(jieba.cut(query))
    scores = bm25_model.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [bm25_docs[i] for i in top_indices]

"""用微調量化4bits Qwen1.8B Chat產生回答"""
def generate_chinese_answer(context, query,memory_chunks): 
    rental_keywords = ["租賃服務", "租賃計畫", "訂閱服務", "訂閱方案", "方案", "玩具方案"]
    intro_keywords = ["介紹", "概述", "描述", "指南"]
    query_lower = query.lower()
    if any(k in query_lower for k in rental_keywords) and any(k in query_lower for k in intro_keywords):
        return ("月費方案：每月 688 新台幣，包含每月定期玩具更換。 （至少 5 個月）。年費方案：每年 8,256 台幣，包含每月玩具更換。 （中途不可取消）。帳單以您的訂閱日期為準。詳情請瀏覽我們的官方網站。")
    
     
    system_prompt = (
    "<|im_start|>system\n"
    "你是 TOYSUB 的專業玩具顧問 "
    "請以簡潔、基於事實的方式，用流暢的中文回覆問題，並**嚴格依據提供的記憶跟上下文內容**回答。\n"
    "請勿臆測、補充或創造任何上下文和記憶中未出現的資訊。\n"
    "若上下文和記憶中皆未提供相關內容，請回答:「很抱歉，這個問題不屬於TOYSUB的範圍!」\n"
    "<|im_end|>"
    )

    user_prompt = (
        f"<|im_start|>user\n"
        f"問題: {query}\n\n"
        f"記憶內容:\n{memory_chunks}\n\n"
        f"文件內容:\n{context}\n"
        f"請僅依據上述記憶和文件內容回答問題，不要提供額外建議或假設性的說法。\n"
        f"<|im_end|>"
    )
    assistant_prompt = "<|im_start|>assistant\n"
    full_prompt = f"{system_prompt}\n{user_prompt}\n{assistant_prompt}"

    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=1024)
    tokenizer_kwargs = {k: v.to("cuda") for k, v in inputs.items()}

    output_ids = model.generate(
        input_ids=tokenizer_kwargs["input_ids"],
        max_new_tokens=80,
        temperature=0.0,
        top_p=None,
        do_sample=False,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        early_stopping=True 
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text.strip()

def convert_to_traditional(text: str) -> str: #簡體中文轉繁體中文
    cc = OpenCC('s2twp')
    return cc.convert(text)

def get_page_content(chunk): #取得文件內容
    if isinstance(chunk, dict):
        return chunk.get("page_content", chunk.get("text", ""))
    elif hasattr(chunk, "page_content"):
        return chunk.page_content
    else:
        return str(chunk)

def rerank(translated_query,matched_chunks): #重新排序相關性    
    pairs = [[translated_query, get_page_content(chunk)] for chunk in matched_chunks] #將每個chunk都跟query組成一個list
    inputs = RERANK_TOKENIZER(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to("cuda") #組成tokenizer input
    with torch.no_grad():
        scores = RERANK_MODEL(**inputs, return_dict=True).logits.view(-1,).cpu().numpy() #計算分數
    reranked = sorted(zip(matched_chunks, scores), key=lambda x: x[1], reverse=True) #將分數排序
    top_reranked_chunks = [chunk for chunk, _ in reranked[:1]]  #保留分數最高的10個chunks
    
    return top_reranked_chunks

"""記憶庫初始化"""
def init_memory_vectorstore(user_id,path="./memory_vectorstore"):
    embedding_model = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cuda"})
    user_path = os.path.join(path, user_id)
    if os.path.exists(user_path):
        return FAISS.load_local(user_path, embedding_model, allow_dangerous_deserialization=True), embedding_model
    else:
        os.makedirs(user_path, exist_ok=True) 
        empty_doc = Document(page_content="",metadata = {"type": "memory", "timestamp": str(datetime.now()), "user_id": user_id})
        memory_vectorstore = FAISS.from_documents([empty_doc],embedding_model)
        memory_vectorstore.save_local(user_path)
        return memory_vectorstore, embedding_model

"""將不同使用者對話紀錄加入資料庫"""
def add_memory_vectorstore(ans, query, user_id, embedding_model, memory_vectorstore, memory_index_path="./memory_vectorstore"):
    memory_text = f"User Query: {query}\nAssistant Response: {ans}"
    memory_doc = Document(page_content=memory_text,metadata={"type": "memory", "timestamp": str(datetime.now()), "user_id": user_id})
    
    memory_vectorstore.add_documents([memory_doc])
    user_path = os.path.join(memory_index_path, user_id)  
    memory_vectorstore.save_local(user_path)
    return memory_vectorstore

def clean_answer(text: str) -> str: #後處理保留回覆
    # 1. 嘗試從常見開頭標記擷取回答
    for tag in [ "A:", "Answer:", "【Answer】"]:
        if tag in text:
            text = text.split(tag, 1)[-1].strip()
            break

    # 2. 將文字依中英文句號分段
    lines = re.findall(r'[^。！？!?\.]+[。！？!?\.]?', text)
    clean_lines = []
    

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if re.search(r"<[^>]+>", line):
            continue

        # Step 3: 找出「被引號包住」的 email 或電話
        email_matches = re.findall(r'"([^"]+@[^"]+)"', line)
        for email in email_matches:
            line = line.replace(f'"{email}"', '"{{staff@toysub.tw}}"')  

        # 提取並標記電話
        phone_matches = re.findall(r'"(\d{2,4}[- ]?\d{3,4}[- ]?\d{3,4})"', line)
        for phone in phone_matches: 
            line = line.replace(f'"{phone}"', '"{{(04)2329-9528}}"')  

        clean_lines.append(line)
    cleaned_text = " ".join(clean_lines)
    return cleaned_text.strip()

"""使用Helsinki模型翻譯英文輸出為中文""" 
# def translate_english(text: str) -> str: 
#     inputs = zh_en_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
#     with torch.no_grad():
#         translated_tokens = zh_en_model.generate(
#             **inputs,
#             max_length=100,
#             num_beams=1,
#             top_p=0.9,
#             no_repeat_ngram_size=5,
#             repetition_penalty=1.2,
#             length_penalty=1.2,
#             early_stopping=True,
#             do_sample=True
#         )
#     translations = zh_en_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True) 
#     text =translations[0]
    
#     #後處理避免重複句
#     lines = re.split(r"[。！？]", text)
#     seen = set()
#     result = []
#     for line in lines:
#         line = line.strip()
#         if line and line not in seen:
#             seen.add(line)
#             result.append(line)
#     return "。".join(result).strip("。") + "。"

"""翻譯輸入問題"""
def translate_text(query):
    if not re.search(r'[\u4e00-\u9fff]',query):
        return query
    else:
        translated = GoogleTranslator(source='zh-TW', target='en').translate(query)
        return translated

"""清洗並合併檢索內容"""
def join_context(chunks):  #刪除TOPIC欄位內容並組合，確保輸入只有OUTPUT欄位內容
    output = []
    for i,chunk in enumerate(chunks):
        text = chunk.page_content

        text = re.sub(r"TOPIC：.*?(?=A：)", "", text, flags=re.DOTALL)
        content = text.split("A:", 1)[-1].strip() if "A:" in text else text.strip()
        output.append(f"【資料{i+1}】\n{content}")
    context = "\n\n".join(output)
    return context

def main():
    """資料引入"""
    manual_documents = extract_all(r"C:\Users\USER\toysub\dataset\chinese_data\chinese_toys_data.json")
    scratch_documents_tw = extract_all(r"C:\Users\USER\toysub\dataset\chinese_data\toysub_version1.json")
    toys = extract_all(r"C:\Users\USER\toysub\dataset\chinese_data\toysub_web_data.json")
    combined_documents = manual_documents + scratch_documents_tw + toys

    """初始化模型、向量庫"""
    vectorstore ,documents= vectorize(combined_documents, save_path="faiss_index") #建立dense向量庫
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5}) #檢索器
    bm25_index(documents)  #建立sparse 向量庫
    load_finetuning_model()
    
    while True:
        query = input("輸入問題:\n")
        if query.lower() == "exit":
            break
        # refined_query = translate_text(query)  #翻譯query成英文
        refined_query = query
        print(f"翻譯問題:{refined_query}")
        start = time.time()
        
        """記憶搜尋"""
        user_id = "tseng"
        memory_index_path = "./memory_vectorstore"
        memory_vectorstore, embedding_model = init_memory_vectorstore(user_id,memory_index_path)

        memory_results = memory_vectorstore.similarity_search(refined_query, k=3)
        memory_chunk = "\n".join([doc.page_content for doc in memory_results])

        
        """檢索階段"""
        dense_query = f"Represent this sentence for retrieval: {refined_query}" #問題前綴
        dense_chunks = retriever.invoke(dense_query) #相似度計算檢索出相關內容
        sparse_chunks = bm25_search(refined_query,top_k=1) #BM25關鍵字搜尋
        combined_chunks = {get_page_content(c): c for c in dense_chunks + sparse_chunks } #合併相似度計算跟關鍵字檢索  
        combined_list = list(combined_chunks.values())

        """重新排序相關性，輸出唯一chunk"""
        reranked_chunks = rerank(refined_query,combined_list) #重新排序chunks跟query間的相關性
        context = join_context(reranked_chunks) #檢索出的內容合成上下文
        print(context)
        print("檢索耗時：", time.time() - start); start = time.time()

        answer_result = generate_chinese_answer(context, refined_query,memory_chunk) #用新query生成的上下文、原始query、向量庫產生回答
        
        print("generate_answer 耗時：", time.time() - start); start = time.time()
        
        """後處理"""
        # answer_result = clean_answer(answer_result)
        # answer = translate_english(answer_result)
        ans = convert_to_traditional(answer_result)
        print(f"客服回覆:{ans}")
        print("後處理耗時：", time.time() - start)

        """對話紀錄寫入記憶庫"""
        memory_vectorstore = add_memory_vectorstore(ans, query, user_id, embedding_model, memory_vectorstore)

if __name__ == "__main__":
    main()
    






