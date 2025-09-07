from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage ,FlexSendMessage
from transformers import AutoTokenizer, AutoModelForSequenceClassification,AutoModelForSeq2SeqLM
from fastapi import FastAPI, Request, HTTPException
import asyncio
import time
import torch
from concurrent.futures import ThreadPoolExecutor
from RAG import (
    init_memory_vectorstore,
    add_memory_vectorstore,
    load_finetuning_model,
    load_json_file,
    extract_all,
    vectorize,
    translate_text,
    translate_english,
    get_page_content,
    join_context,
    rerank,
    generate_answer,
    clean_answer,
    convert_to_traditional,
    bm25_index,
    bm25_search
)

memory_cache = {} 

# 預載向量庫和 embedding 模型
manual_documents = extract_all(r"C:\Users\USER\TOYSUB\dataset\english_data\toysub_data.json")
scratch_documents_tw = extract_all(r"C:\Users\USER\TOYSUB\dataset\english_data\translated_web_data.json")
toys = load_json_file(r"C:\Users\USER\TOYSUB\dataset\english_data\translated_toys_data.json")
combined_documents = manual_documents+scratch_documents_tw +toys
vectorstore,document = vectorize(combined_documents, save_path="faiss_index")

"""關鍵字搜索庫"""
bm25_index(document)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})

RERANK_TOKENIZER = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_MODEL = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2").to("cuda").eval()

bm25_model = None
bm25_docs = []
bm25_tokenized_corpus = []

"""初始化翻譯模型"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
zh_model_name ="Helsinki-NLP/opus-mt-en-zh"
zh_en_tokenizer = AutoTokenizer.from_pretrained(zh_model_name)
zh_en_model = AutoModelForSeq2SeqLM.from_pretrained(zh_model_name).to(device)

async def rag_process(user_msg: str, user_id: str = "default",threshold = 0.75) -> str:
    global memory_vectorstore
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        start = time.time()
        loop = asyncio.get_event_loop()
        refined_query = await loop.run_in_executor(None, translate_text, user_msg)
        """記憶庫搜尋"""
        memory_index_path = "./memory_vectorstore"
        if user_id not in memory_cache:
            memory_vectorstore, embedding_model = await loop.run_in_executor(
                None, init_memory_vectorstore, user_id, memory_index_path
            )
            memory_cache[user_id] = (memory_vectorstore, embedding_model)
        else:
            memory_vectorstore, embedding_model = memory_cache[user_id]
        query_embedding = await loop.run_in_executor(None, embedding_model.embed_query, refined_query)
        memory_results = await loop.run_in_executor(
            None, memory_vectorstore.similarity_search_with_score_by_vector, query_embedding, 1
        )
        if memory_results and memory_results[0][1] >= threshold:
            memory_chunk = memory_results[0][0].page_content
        else:
            memory_chunk = ""

        """翻譯跟密集檢索"""
        dense_query = f"Represent this sentence for retrieval: {refined_query}"
        dense_chunks = await loop.run_in_executor(None, retriever.invoke, dense_query)

        """稀疏檢索(關鍵字搜尋)"""
        sparse_chunks = await loop.run_in_executor(None,bm25_search,refined_query)
        combined_chunks = {get_page_content(c): c for c in dense_chunks + sparse_chunks } 
        combined_list = list(combined_chunks.values())

        reranked_chunks = await loop.run_in_executor(None, rerank, refined_query, combined_list)
        print("rerank 耗時：", time.time() - start); start = time.time()
     
        image_url = ""
        if reranked_chunks and hasattr(reranked_chunks[0] , "metadata"):
            image_url = reranked_chunks[0].metadata.get("image_url" , "")

        context = await loop.run_in_executor(None,join_context,reranked_chunks)

        answer_result = await loop.run_in_executor(executor, generate_answer, context, refined_query,memory_chunk)
        print("generate_answer 耗時：", time.time() - start)
        answer = await loop.run_in_executor(None,clean_answer,answer_result)
        print(answer)
        answer = await loop.run_in_executor(None,translate_english,answer)
        ans = await loop.run_in_executor(None,convert_to_traditional,answer)
        
        """儲存回覆"""
        memory_vectorstore = await loop.run_in_executor(None,add_memory_vectorstore,answer,refined_query,user_id, embedding_model, memory_vectorstore)

        
        return {
            "text" : ans ,
            "image" : image_url
        }
    

    except Exception as e:
        return f"伺服器錯誤：{str(e)}"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

line_bot_api = LineBotApi('u9aYybfIWbpPF/EohpE1V8toS/Ay0PHaJXybQUSHBLJOXWxWhDtz5CzGsy5sdlYrlc9wBWI2NyWPiR51UAYxUdncW47XWooHB3SQOOSrY/cd839CdcZYBx5DtXc0yhZDyfkAvZaExveaqeRdoVphuwdB04t89/1O/w1cDnyilFU=')
handler = WebhookHandler('9ae3564f7100d1db0348c45cbfba343f')
user_state = {}

"""FASTAPI初始載入微調、BASE模型"""
@app.on_event("startup")
async def load_model():
    load_finetuning_model()

@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("x-line-signature")
    body = await request.body()

    print("收到 LINE Webhook 呼叫")
    print("RAW Body:", body.decode("utf-8"))

    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        print("Signature 驗證失敗")
        raise HTTPException(status_code=400, detail="Invalid signature")

    return "OK"      

@handler.add(MessageEvent,message=TextMessage)
def handle_message(event):
    msg = event.message.text
    user_id = event.source.user_id
    print("使用者輸入內容:",msg)
    if msg == "客服機器人":
        user_state[user_id] = {"mode": "rag"}
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="您好，請問有什麼需要幫忙？請直接輸入您的問題。")
        )
        return
    
    if user_state.get(user_id,{}).get("mode") == "rag":
        line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text="請稍候，正在為您整理回答...")
        )

        async def process_and_push():
            try:  
                reply_text = await rag_process(msg, user_id)
                if isinstance(reply_text, dict) and reply_text.get("image"):
                    line_bot_api.push_message(
                            user_id,
                            TextSendMessage(text=reply_text.get("text"))
                        )
                    flex_msg = {
                                    "type": "bubble",
                                    "hero": {
                                        "type": "image",
                                        "url": reply_text["image"],
                                        "size": "full",
                                        "aspectRatio": "1.51:1",
                                        "aspectMode": "cover",
                                        "aspectMode" : "fit"
                                    },
                                    "body": {
                                        "type": "box",
                                        "layout": "vertical",
                                        "contents": []
                                    }
                                }

                    line_bot_api.push_message(
                            user_id,
                            FlexSendMessage(alt_text="推薦玩具圖片", contents=flex_msg)
                        )
                else:
                    line_bot_api.push_message(
                            user_id,
                            TextSendMessage(text=reply_text.get("text"))
                        )    

                
            except Exception as e:
                line_bot_api.push_message(
                user_id,
                TextSendMessage(text=f"客服系統暫時無法使用，請稍後再試 🙇\n錯誤訊息：{e}")
            )
        asyncio.create_task(process_and_push())
        return
    
    # ==== 其他一般訊息 ====
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="感謝您的訊息，我們會盡快回覆。")
    )

@app.post("/api/chat")
async def web_chat(request: Request):
    data = await request.json()
    user_msg = data.get("message", "")


    reply = await rag_process(user_msg)
    return {"reply": reply} 


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)