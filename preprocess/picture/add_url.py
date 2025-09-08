import json

# === 設定路徑 ===
input_json_path = r"C:\Users\USER\toysub\dataset\english_data\translated_toys_data.json"
output_json_path = r"C:\Users\USER\toysub\dataset\english_data\translated_toys_data_1.json"
base_url = "https://toysub-toy-images.s3.us-east-1.amazonaws.com/images"

# === 讀取 JSON 資料 ===
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# === 加入圖片 URL ===
for idx, item in enumerate(data, start=1):  # 從 1 開始對應 toy_image_1.png
    image_url = f"{base_url}/toy_image_{idx}.png"
    item["images_url"] = image_url

# === 輸出新 JSON ===
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ 已成功為 {len(data)} 筆資料加入圖片網址欄位！")