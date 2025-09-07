import pandas as pd
import json
csv_path = r"C:\Users\USER\Downloads\0901玩具分類_fixed.xlsx"
xls = pd.ExcelFile(csv_path)    


# 印出欄位名稱
all=[]
for sheet_name in xls.sheet_names:
    df = pd.read_excel(csv_path,sheet_name=sheet_name)
    df["年齡層"] = sheet_name
    all.extend(df.to_dict(orient="records"))

out_path = r"C:\Users\USER\toysub\dataset\toys.json"

with open(out_path,"w",encoding="utf-8") as f:
    json.dump(all,f,ensure_ascii=False,indent=2)