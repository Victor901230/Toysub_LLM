import pandas as pd
import json
from pathlib import Path

# === 路徑設定 ===
xlsx_path = r"C:\Users\USER\Downloads\0906玩具分類_fixed.xlsx"
out_path  = r"C:\Users\USER\toysub\dataset\toysub_package\toys.json"


INPUT_SEP = " 。 "  # 第1欄與第3欄合併時的分隔符

xls = pd.ExcelFile(xlsx_path)
records = []

for sheet_name in xls.sheet_names:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name,header=None)

    col1 = df.iloc[:, 0].astype(str).fillna("").str.strip()
    col3 = df.iloc[:, 2].astype(str).fillna("").str.strip()
    col4 = df.iloc[:, 3].astype(str).fillna("").str.strip()

    combined_input = (col1 + INPUT_SEP + col3).str.strip(INPUT_SEP + " ")
    topic = col4

    temp = pd.DataFrame({
        "input": combined_input,
        "TOPIC": topic,
    })

    temp = temp[temp["input"].str.len() > 0]

    records.extend(temp.to_dict(orient="records"))

# === 輸出 JSON ===
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print(f"Done. Wrote {len(records)} records to: {out_path}")
