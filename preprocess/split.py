from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import re
import numpy as np
import json
import tqdm

def split_to_sentences(chunks):
    all_sentences = []
    sentence_end_pattern = re.compile(r'(?<=[.!?])\s+')

    for chunk in chunks:
        text = chunk.get("translated_text", "")
        sentences = sentence_end_pattern.split(text)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 0 and not sentence.isspace():
                all_sentences.append({
                    "sentence": sentence if sentence.endswith(('.', '?', '!')) else sentence + '.',
                    "url": chunk.get("url", "")
                })
    return all_sentences

def semantic_chunk(sentences, source_url=None, similarity_threshold=0.6):
    if len(sentences) < 2:
        return [{'text': sentences[0]['sentence'], 'url': source_url}]
    
    # 嵌入模型設定
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda"}
    embedding_model = HuggingFaceBgeEmbeddings(model_name=embedding_model_name, model_kwargs=model_kwargs)

    # 加入 embedding
    for sentence in sentences:
        sentence['combined_sentence'] = sentence['sentence']
    embeddings = embedding_model.embed_documents([s['combined_sentence'] for s in sentences])
    for i, s in enumerate(sentences):
        s['embedding'] = embeddings[i]

    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        current = sentences[i]
        prev_chunk_embedding = np.mean([s['embedding'] for s in chunks[-1]], axis=0)

        sim_prev = 0.0
        sim_next = 0.0

        if current.get("url") == chunks[-1][-1].get("url"):
            sim_prev = cosine_similarity([current['embedding']], [prev_chunk_embedding])[0][0]

        if i + 1 < len(sentences):
            next_sent = sentences[i + 1]
            if current.get("url") == next_sent.get("url"):
                sim_next = cosine_similarity([current['embedding']], [next_sent['embedding']])[0][0]

        # 真正的合併條件加入 threshold
        
        if sim_prev >= sim_next:
            chunks[-1].append(current)
        else:
            chunks.append([current])

    # 合併句子文本
    final_chunks = []
    for group in chunks:
        combined = ''.join([s['sentence'] for s in group])
        if len(combined.strip()) >= 50:
            final_chunks.append({'translated_text': combined, 'url': source_url})
    return final_chunks

def combine_sentences(sentences):
    for sentence in sentences:
        sentence['combined_sentence'] = sentence['sentence']
    return sentences

def calculate_cosine_distances(sentences): #計算CHUNKS間的cos距離
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0] # Calculate cosine similarity
        
        distance = 1 - similarity # Convert to cosine distance       
        distances.append(distance)# Append cosine distance to the list      
        sentences[i]['distance_to_next'] = distance # Store distance in the dictionary
    return distances, sentences

def main():
    with open(r"C:\Users\USER\toysub\dataset\clean_data.json", "r", encoding="utf-8") as f:
        json_documents = json.load(f)
    all_chunks = []
    for chunk in tqdm.tqdm(json_documents, desc="逐 chunk 處理"):
        url = chunk.get("url", None)
        sentences = split_to_sentences([chunk])  # 僅對單一 chunk 做斷句
        semantic_chunks = semantic_chunk(sentences, source_url=url)  # 僅針對該 chunk 的句子做 semantic cut
        all_chunks.extend(semantic_chunks)  # 將結果加入總清單
    with open(r"C:\Users\USER\toysub\dataset\toysub_jap_data.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)
    print("切分完成")

if __name__ == "__main__":
    main()