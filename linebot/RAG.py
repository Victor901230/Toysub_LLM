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
import unicodedata
import requests
import deepl
from PIL import Image
from io import BytesIO

auth_key = "ff8e4ae7-be74-4928-9b0f-a043b5ccccd8:fx"
deep_translator = deepl.Translator(auth_key)
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
zh_model_name ="Helsinki-NLP/opus-mt-en-zh"
zh_en_tokenizer = AutoTokenizer.from_pretrained(zh_model_name)
zh_en_model = AutoModelForSeq2SeqLM.from_pretrained(zh_model_name).to(device)

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
        image_url = item.get("images_url", "").strip()
        text = f"{output}"
        extracted.append({
            "TOPIC":topic,
            "question":question,
            "translated_text": text,
            "images_url": image_url
        })
    return extracted

"""載入微調模型並和mistralai/Mistral-7B-v0.1合併"""
def load_finetuning_model():  
    global model, tokenizer
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float32,  # float16
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True
    )
    max_memory = {
            0: "14GiB",        # ✅ 用整數表示 GPU ID 0
            "cpu": "32GiB"
        }

    
    base_model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1",
            device_map="auto",
            quantization_config=bnb_config,
            max_memory=max_memory
        )
    model = PeftModel.from_pretrained(base_model, r"C:\Users\USER\toysub\final_623_model")
    model = model.merge_and_unload()
    model.to("cuda")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    print(model.hf_device_map)
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
                        "type": "faq",
                        "image_url": doc.get("images_url","")
                    }
                )
                for doc in documents
            ]

    embedding_model_name = "BAAI/bge-large-en-v1.5"
    model_kwargs = {"device": "cuda"}
    embeddings = HuggingFaceBgeEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)

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

"""用微調量化4bits mistral產生回答"""
def generate_answer(context, query,memory_chunks): 
    rental_keywords = ["rental plan", "rental service", "subscription plan"]
    intro_keywords = ["introduction","introduce", "overview", "description", "guide","what"]
    contact_keywords = ["phone", "telephone", "call", "contact number", "how can I contact","phone number"]
    query_lower = query.lower()
    if any(k in query_lower for k in rental_keywords) and any(k in query_lower for k in intro_keywords):
        return ("Monthly Plan: NT$688 per month, with regular monthly toy exchanges. (Minimum 5 months).Annual Plan: NT$8,256 per year, with monthly exchanges. (Cannot be cancelled mid-term).Billing is based on your subscription date. For full details, please refer to our official website.")
    
    if any(k in query_lower for k in contact_keywords):
        return ("You can call 04-2329-9528 to contact live customer service, who will provide you with professional advice.")
    if any(k in query_lower for k in ["can i","i want","can you"]) and any(kw in query_lower for kw in ["choose toys", "pick toys", "select toys"]):
        return ("Hello, Toysub's toys are all selected by the system according to the baby's age and preferences. Parents can also tell us the baby's preferences and help you make adjustments! Thank you")
    if any(k in query_lower for k in ["How do I return the toys?","How to return the toys?"]) :
        return ("Hello, please put the toys in the original bags and boxes, and put the completed toy rental form in the box. After the box is sealed, Black Cat Logistics will come to your home to collect the old toy box.\n"
                ">For the return and new delivery dates, please refer to the return and new delivery dates on the toy rental form\n"
                ">After Toysub receives the recycled toys, the new batch of toys will be sent out!")
    if any(k in query_lower for k in ["hello, i would like to know","i would like to know","i want to know","want to know"]) :
        return ("Hello, Toysub has thousands of toys, and each issue will have 6 to 8 toys."
                "Toysub's toys are designed for children aged 0 to 6, providing age-appropriate and diversified matching services, and providing diverse and non-repetitive toys."
                "The types of toys are also rich and diverse, and a large number of toys from major brands in various countries are purchased from abroad many times a month, and the toy items are updated at any time."
                "There are cognitive, puzzle, construction, stacking, language, mathematics, logic, fine motor... and so on, diversified stimulation of development."
                "Parents can also tell us the baby's preferences and help you make adjustments! Thank you")
    if any(k in query_lower for k in ["will you","will i"]) and any(kw in query_lower for kw in ["provide a list of toys in advance", "toy list"]):
        return ("Hello, Toysub's toys are all selected by the system based on the baby's age and preferences."
                "After selection, they will be sent directly, and no toy catalog will be provided in advance!"
                "Parents can also tell us the baby's preferences in advance to help you make adjustments! Thank you")
    if any(k in query_lower for k in ["what should i do ","what to do "]) and any(kw in query_lower for kw in ["can't find the accessories", "accessories not found"]):
        return ("Hello, once Toysub receives your toy recycling package, if there are any missing items, we will include a large picture of the missing toy in the next toy box to help you find it. Once you have found it, please return it with your next collection. Thank you.")
    if any(k in query_lower for k in ["what should i do ","what to do "]) and any(kw in query_lower for kw in ["toys are broken", "toy is broken","broken toy"]):
        return ("Hello, please exit the customer service robot page and enter the message directly to contact real LINE customer service. Thank you.")
    
    if context == "":
        return ("This question is outside the scope of TOYSUB! I cannot answer it.You can ask your question again in another way or call 04-2329-9528 to contact live customer service, who will provide you with professional advice.")
     
    system_prompt = """<|im_start|>system
        You are an expert toy consultant at TOYSUB. Answer all questions in fluent, professional English, regardless of input language.
        Your job is to provide precise, informative, and age-appropriate toy suggestions.
        Use only the provided context. First select the most relevant section, then answer clearly based on that section.
        If only one section is provided, base your answer on that directly.
        Avoid generic phrases, emotional appeals, or closing remarks.
        Only focus on the toy’s features, recommended age, benefits, and how it helps development.
        When possible, answer in bullet points or structured explanations.
        <|im_end|>"""

    user_prompt = (
        f"<|im_start|>user\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n"
        f"Only answer based on the context above. Do not make assumptions or provide general advice.\n"
        f"<|im_end|>"
    )

    assistant_prompt = "<|im_start|>assistant\n"

    full_prompt = f"{system_prompt}\n{user_prompt}\n{assistant_prompt}"

    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=1024)
    tokenizer_kwargs = {k: v.to("cuda") for k, v in inputs.items()}
    
    output_ids = model.generate(
        input_ids=tokenizer_kwargs["input_ids"],
        max_new_tokens=256,
        temperature=0.6,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2,
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

def rerank(translated_query,matched_chunks, threshold=0.6,top_k = 2): #重新排序相關性    
    pairs = [[translated_query, get_page_content(chunk)] for chunk in matched_chunks] #將每個chunk都跟query組成一個list
    inputs = RERANK_TOKENIZER(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to("cuda") #組成tokenizer input
    with torch.no_grad():
        scores = RERANK_MODEL(**inputs, return_dict=True).logits.view(-1,).cpu().numpy() #計算分數
    reranked = sorted(zip(matched_chunks, scores), key=lambda x: x[1], reverse=True) #將分數排序
    filtered = [chunk for chunk, score in reranked if score >= threshold][:top_k]
    if not filtered:
        return []
    return filtered

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
    for tag in [ "<|im_start|>assistant"]:
        if tag in text:
            text = text.split(tag, 1)[-1].strip()
            break
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"【Information1】", "", text)
    text = re.sub(r"【Information2】", "", text)
    text = re.sub(r"\btoysa[ub]\b", "Toysub童益趣", text, flags=re.IGNORECASE)

    # 2. 將文字依中英文句號分段
    lines = re.findall(r'[^。！？!?\.]+[。！？!?\.]?', text)
    clean_lines = []
    

    for idx,line in enumerate (lines):
        line = line.strip()
        if not line:
            continue
        if idx == 0:
            line = re.sub(r"^(1\.?|1、)", "", line).strip()
            
        line = re.sub(r"#", "", line)
        line = re.sub(r"^#\d{6,}\s*", "", line)
        # if re.match(r"^(?:A\d*|Q\d*|--|==)", line):
        #     continue

        # Step 3: 找出「被引號包住」的 email 或電話
        email_matches = re.findall(r'"([^"]+@[^"]+)"', line)
        for email in email_matches:
            line = line.replace(f'"{email}"', '"{{staff@toysub.tw}}"')  

        # 提取並標記電話
        phone_matches = re.findall(r"\(?\d{2,4}\)?[ -]?\d{3,4}[ -]?\d{3,4}", line)
        for phone in phone_matches: 
            line = line.replace(f'"{phone}"', '"{{(04)2329-9528}}"')  

        clean_lines.append(line)
    cleaned_text = " ".join(clean_lines)
    cleaned_text = ''.join(
        c for c in cleaned_text
        if unicodedata.category(c)[0] not in {'So', 'Sk', 'Sm'}
        and not ('①' <= c <= '⑳')      # 圈圈數字 ①~⑳
        and not ('⒈' <= c <= '⒛')      # 小圈圈數字 ⒈~⒛
        and c not in {'✦', '★', '●', '※'}  # 額外移除的符號
    )
    cleaned_text = re.sub(
        r"^Based on the provided memory and [Cc]ontext,\s*", "", cleaned_text
    )

    return cleaned_text.strip()

"""使用deepl API翻譯英文輸出為中文""" 
def translate_english(text: str) -> str: 
    result = deep_translator.translate_text(text,source_lang="EN",target_lang="ZH")
    translated_text = result.text
    #後處理避免重複句
    lines = re.split(r"[。！？]", translated_text)
    seen = set()
    result = []
    for line in lines:
        line = line.strip()
        if line and line not in seen:
            seen.add(line)
            result.append(line)
    return "。".join(result).strip("。") + "。"

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
        output.append(f"【Information{i+1}】\n{content}")
    context = "\n\n".join(output)
    return context

def test_image_url(url):
    try:
        print(f"測試圖片連結：{url}")
        response = requests.get(url)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content))
        img.show()  # 使用預設圖片工具開啟預覽
        print("✅ 成功載入並顯示圖片")

    except Exception as e:
        print(f"❌ 圖片讀取失敗：{e}")

def main():
    """資料引入"""
    manual_documents = extract_all(r"C:\Users\USER\toysub\dataset\english_data\toysub_data.json")
    scratch_documents_tw = extract_all(r"C:\Users\USER\toysub\dataset\english_data\translated_web_data.json")
    toys = load_json_file(r"C:\Users\USER\toysub\dataset\english_data\translated_toys_data.json")
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
        refined_query = translate_text(query)  #翻譯query成英文
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
        
        image_url = ""
        if reranked_chunks and hasattr(reranked_chunks[0] , "metadata"):
            image_url = reranked_chunks[0].metadata.get("image_url" , "")
        
        context = join_context(reranked_chunks) #檢索出的內容合成上下文
        print(context)
        print("檢索耗時：", time.time() - start); start = time.time()

        answer_result = generate_answer(context, refined_query,memory_chunk) #用新query生成的上下文、原始query、向量庫產生回答
        
        print("generate_answer 耗時：", time.time() - start); start = time.time()
        
        """後處理"""
        answer_result = clean_answer(answer_result)
        print(answer_result)
        answer = answer_result
        # answer = translate_english(answer_result)
        ans = convert_to_traditional(answer)
        print(f"客服回覆:{ans}")
        print(f"\n 圖片網址：{image_url if image_url else '（無）'}")
        if image_url:
            test_image_url(image_url)
        print("後處理耗時：", time.time() - start)

        """對話紀錄寫入記憶庫"""
        memory_vectorstore = add_memory_vectorstore(ans, query, user_id, embedding_model, memory_vectorstore)

if __name__ == "__main__":
    main()
    






