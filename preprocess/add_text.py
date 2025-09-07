import json
import re

# 讀取原始資料
with open(r"C:\Users\USER\TOYSUB\dataset\cleaned_generated_questions.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 處理每一筆資料
for item in data:
    # 刪除 text 中 "年齡類別:" 後的內容
    if "question" in item and isinstance(item["question"], str):
        item["question"] = re.sub(r"年齡類別:.*", "", item["question"]).strip()
        item["question"] = item["question"].replace("適合", "").strip()
    # 若 topic 結尾不是 "玩具"，則加上
    if "topic" in item and isinstance(item["topic"], str):
        if not item["topic"].endswith("玩具"):
            item["topic"] = item["topic"].strip() + "玩具"

    

# 儲存處理後的資料
with open(r"C:\Users\USER\TOYSUB\dataset\cleaned_combined_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("✅ 處理完成，已輸出 cleaned_combined_data.json")
