from openpyxl import load_workbook
from PIL import Image
import os
import io

# === 設定 ===
excel_path = r"C:\Users\USER\toysub\dataset\玩具列表分類.xlsx"
output_folder = r"C:\Users\USER\toysub\dataset\images"

# 建立輸出資料夾
os.makedirs(output_folder, exist_ok=True)

# 載入 Excel 檔案
wb = load_workbook(excel_path)
ws = wb.active

# 統計用
count = 0

# 取得所有嵌入圖片（openpyxl 圖片物件）
for idx, image in enumerate(ws._images):
    # 擷取圖片資料 (bytes)
    if hasattr(image, "_data"):  # openpyxl 3.1+
        image_bytes = image._data()
    elif hasattr(image, "ref") and hasattr(image, "image"):
        image_bytes = image.image
    else:
        print(f"⚠️ 無法讀取第 {idx+1} 張圖片資料")
        continue

    # 儲存圖片
    img = Image.open(io.BytesIO(image_bytes))
    output_path = os.path.join(output_folder, f"toy_image_{idx+1}.png")
    img.save(output_path)
    count += 1
    print(f"✅ 已儲存圖片: {output_path}")

print(f"\n🎉 完成！共儲存 {count} 張圖片至資料夾：{output_folder}")