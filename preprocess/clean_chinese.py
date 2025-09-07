import json
import re
ALLOWED_CHARS_PATTERN = re.compile(
    r"[^\u4E00-\u9FFF\u3000-\u303F\uFF00-\uFFEFa-zA-Z0-9。，、！？.?!:;「」『』【】［］()（）・ー\-\'\"&/%~\s]"
)

def normalize_text(text):
    text = text.replace("\n", "")
    text = re.sub(r"\s+", "", text)
    return text.strip()

def remove_weird_symbols(text):
    cleaned = ""
    prev_char = ""
    for char in text:
        if ALLOWED_CHARS_PATTERN.match(char):  # 是奇怪符號
            # 如果前後都是文字，且都不是標點，就補一個句號
            if prev_char and prev_char not in "。，、！？.?!:;「」『』":
                cleaned += "。"
            continue  # 移除這個奇怪符號
        cleaned += char
        prev_char = char
    return cleaned

def clean_chunk(text):
    text = normalize_text(text)
    text = remove_weird_symbols(text)
    return text.strip()


def split_sentence(text, max_len=200, min_len=50):
    import re
    sentences = re.split(r'(?<=[。！？!?])', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) <= max_len:
            current_chunk += sentence
        else:
            if len(current_chunk) < min_len:
                current_chunk += sentence
                chunks.append(current_chunk)
                current_chunk = ""
            else:
                chunks.append(current_chunk)
                current_chunk = sentence

    if current_chunk.strip():
        chunks.append(current_chunk)

    return [c.strip() for c in chunks if len(c.strip()) >= min_len]

def main():
    input_path = r"C:\Users\USER\toysub\dataset\toysub_site_data.json"

    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    cleaned_chunks = []
    for chunk in chunks:
        original_text = chunk["text"]
        cleaned_text = clean_chunk(original_text)
        split_chunks = split_sentence(cleaned_text, max_len=200)
        for piece in split_chunks:
            cleaned_chunks.append({
                "url": chunk["url"],
                "text": piece
            })

    with open(r"C:\Users\USER\toysub\dataset\clean_chinese_data.json", "w", encoding="utf-8") as f:
        json.dump(cleaned_chunks, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()