import re
import json
from tqdm import tqdm
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import torch
from langchain_ollama import OllamaLLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "staka/fugumt-ja-en"
jap_en_tokenizer = AutoTokenizer.from_pretrained(model_name)
jap_en_model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16
).to(device)

semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

FIXED_TRANSLATIONS = {
    "トイサブ": "TOYSUB","toysub": "TOYSUB","TPC": "TPC","スマイルコース": "Smile Course","キッズ":"kids", "キッズミニ":"Kids mini","キラキラ":"Sparkle","キャラクタ":"Character","ケベック":"Quebec","ケアレスミス":"Careless mistake","コンタクトナビ":"Contact Navi",
    "プチコース": "Petit Course","スタンダードコース": "Standard Course","アイコン":"icon","ガイドボード":"Guide Board","ガイドピックアップ":"Guide Pickup","キャッチー":"Catchy","クリエイト":"Create","コストゼロ": "Zero cost", "コロンビア": "Colombia",  "チラシ":"Flyer",
    "アイテム":"item","メニュー":"Menu","アイデア":"idea","アクセス":"access","アシスト":"assist","ガイドスマートレーダー":"Guide Smart Radar","カラーバリエーション":"Color Variations","グループリーダー":"Group Leader","コロナ":"Corona","チャイムボール":"Chime Ball",
    "アナログ":"analog","アニマル":"animal","ワンダーボックス":"WonderBox","モンテッソーリ":"Montessori", "キャッシング・ローン":"Cashing Loan","キッズミニキーボードスクイグズ":"Kids Mini Keyboard Squigs","コリドール・ジュニア": "Corridor Junior","チャレンジカード":"Challenge Card",
    "ポピー":"Popy","タブレット": "tablet","コース": "course","コラム": "column","コミ": "communication","キッズミニキーボードトイローヤル":"Kids Mini Keyboard Toy Royal","クライアント":"Client","コピーライト":"Copyright","コースター":"coaster","チームメンバー": "Team Members",
    "サブスク": "subscription","ランキング": "ranking","ゼミ": "seminar","ワーク": "work","コミュニケーション": "communication","キッズミニキーボードメーカー":"Kids Mini Keyboard Manufacturer","コツ・ポイント":"Tips","コースパネル":"Course Panel",
    "デジタル": "digital","タッチ": "touch","パズル": "puzzle", "ピース": "piece","ブロック": "block","サポート": "support","ギフトファッションレディースファッションメンズファッションコンタクトレンズ":"Gift Fashion Women's Fashion Men's Fashion Contact Lenses",
    "キャンペーン": "campaign","レベル": "level","プランナー": "planner","プラン": "plan","ランキング": "ranking","タイプ": "type","ギミックパーツ":"Gimmick Parts","クリエイティブ・チェスト":"Creative Chest", "コラージュ": "Collador Junior","テアトルアカデミー":"Theatre Academy", 
    "ボール": "ball","プレゼント": "present","アカチャン":"baby","アヒル":"duck","アフィリエイトサービス":"Affiliate Services","クリエイティブシリーズ":"Creative Series","クリスマスキャンペーン":"Christmas Campaign","コードクラフターズ":"Code Crafters","トイサブグロー":"Toysub Glow", 
    "アメリカ・ニューヨーク":"New York, USA","アメリカ・ミシガン":"Michigan, USA","アルゴリズム":"algorithm","アクティブユーザー":"Active Users","クリスマスイブ":"Christmas Eve","キャンペーン":"Campaign","コラムイラストレーション":"Column Illustration","トイサブコロンブス":"Toysub Columbus", 
    "アクティブラーニング":"Active Learning","アルミ":"Aluminum","アワード":"Award","イヤー":"year","インスタ":"Instagram","インタビューアンケート":"Interview survey","キャンペーンオリジナル":"Campaign Original","コンピュータプログラム":"Computer Program", "トイサブシリンダーブロック":"Toysub Cylinder Block",
    "ウィリアム":"William","ウエディングプランナー":"Wedding planner","ウォーカーメーカー":"Walker maker","エッセイ":"Essay", "エビデンス":"Evidence","クロノゲート":"Chronogate","グッドデザイン":"Good Design","コーポレートサイト":"Corporate site", "ティーパーティセット":"Tea Party Set", 
    "エピソード":"Episode","エレキギター":"Electric guitar","オチ":"Punchline","イマジネーションシリーズ":"Imagination Series","イベントプランナー":"Event Planner","クーポンスイーツクルマ・バイク":"Coupon Sweets Cars & Bikes","コーポレートビジョン":"Corporate vision",
    "アンバサダープロジェクト":"Ambassador Project","アカデミーエルカミノグノーブル": "Academy El Camino Noble","アップタカラトミー":"Up Takara Tomy", "オリジナルコース":"Original Course","グリーンフライデー":"Green Friday", "サイズ カスタネット":"Size Castanet",
    "オンリーワン":"Only One","カウンティング": "Counting","カスタネット":"Castanet","カスタマーオペレーション":"Customer Operation","カスタマーサポートグループ":"Customer Support Group","コチラフェバリットコレクション":"Favorite Collection","ショッピング・モール":"Shopping mall",
    "カスタマーサポートスタッフ": "Customer Support Staff","カテゴリーカード":"Category Card","カフェ":"Cafe", "カラフルキャッスルビークルタウン":"Colorful Castle Vehicle Town","カラーハンド":"Color Hand","サイコロ":"Dice","サイトポリシーサイトマップ":"Site Policy Site Map",
    "サイトマップ": "Site Map","サンプルコスメダイエットエステメンズエステ": "Sample Cosmetics Diet Salon Men's Salon","サイレンクラウンパトカー":"Siren Clown Police Car", "シェイク・パル":"Shake Pal", "シカゴ":"Chicago","シフトレバー":"Shift lever","シリンダーブロックシリンダー":"Cylinder block cylinder",
    "カラーハンドベビーオルゴール":"Color Hand Baby Music Box","カラーパレット":"Color Palette","カンブリア":"Cambria","サタデーウォッチ":"Saturday watch","サービスインターネットレンタルサーバーインターネットプロバイダー":"Service Internet Rental Server Internet Provider",
    "シリンダーブロックテディメモリースティッキー":"Cylinder block teddy memory sticky","シリンダーブロックメーカー": "Cylinder Block Manufacturer","ジェフ・ベゾス":"Jeff Bezos","ジェームズ・ヘックマン":"James Heckman", "ジュースマシン":"Juice Machine","ジャンル・カテゴリ":"Genre/Category",
    "スウェーデン": "Sweden", "スクープ":"Scoop","スタジオマグブロック":"Studio Magblock", "スタジオメーカー":"Studio Maker", "スタンダードコロコロコースター":"Standard Rolling Coaster","スタンダードセット":"Standard Set","スタンダードセットコロンブス":"Standard Set Columbus",
    "スターターセットメーカー":"Starter set manufacturer","スポーツモデル":"Sports Model","スマイルゼミサービス":"Smile Seminar Service","スマートアイス":"Smart Ice","スマートレーダーブログホーム":"Smart Radar Blog Home", "スモック":"Smock","スロープカタカタ":"Slope wierd sounds",
    "スロープ":"Slope","セットドッグギターメガブロック":"Set Dog Guitar Megablock","セットメガブロック":"Set Megablock","セットメーカー":"Set manufacturer","セルゲイ・ブリン":"Sergey Brin","ゼンマイ":"Wind-up","ソフトバンク":"Softbank","ソフトビニール":"Soft Vinyl","タイムサービス":"Time Service",
    "タイヤパーツ":"Tire Parts","センサリーソフトボールセットメーカー":"Sensory Softball Set Manufacturer", "タブレットコース":"Tablet Course","ターミナル":"Terminal","ターンテーブル":"Turntable","ダイアトニックスケール":"Diatonic scale","チェイサービーズ":"Chaser Beads",
    "テクニカルサポートセンター":"Technical Support Center", "テクロン":"Techron","テネシー":"Tennessee","ディスカバー・ライトバーメーカー":"Discover Light Bar Manufacturer","ディスカヴァー・トゥエンティワン":"Discover 21", "ディスペンサー":"Dispenser","デュケーション":"Education",
    "デジタルギフトクーポンプレゼント":"Digital Gift Coupon Present","デメリットオンライン":"Disadvantages Online","デラックスセット":"Deluxe Set","トイサブカタミノ":"Toysub Katamino","トイサブカシャカシャ":"Toysub Clack Clack", "トイサブスティッキー":"Toysub Sticky","トイサブスクイグズ":"Toysub Squigs"       
}

def split_japanese_sentences(text):
    sentences = re.split(r'(?<=[。！？、])\s*|\n', text)
    return [s for s in sentences if s.strip()]

def protect_terms(text, term_map=None):
    term_map = term_map or {}
    for original, fixed_translation in FIXED_TRANSLATIONS.items():
        placeholder = f"⟦{fixed_translation}⟧"
        text = text.replace(original, placeholder)
        term_map[placeholder] = fixed_translation  # 不會再還原，只保留格式一致
    return text, term_map

def restore_terms(text, term_map):
    for placeholder, final in term_map.items():
        text = text.replace(placeholder, final)
    return text

def clean_repetitions(text):
    words = text.split()
    new_words = []
    prev = None
    for word in words:
        if word != prev:
            new_words.append(word)
        prev = word
    return ' '.join(new_words)

def remove_duplicate_sentences(text):
    sentences = re.split(r'[.!?]', text)
    seen = set()
    filtered = []
    for s in sentences:
        s = s.strip()
        if s and s not in seen:
            filtered.append(s)
            seen.add(s)
    return '. '.join(filtered)

def translate_sentences(texts, batch_size=16):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = jap_en_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        translated_tokens = jap_en_model.generate(
                **inputs,max_length=300,do_sample=False,num_beams=3, early_stopping=True               
            )
        output = jap_en_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        results.extend(output)
    return results

def compute_similarity(jp_sentences, en_sentences, threshold=0.6):
    scores = []
    for jp, en in zip(jp_sentences, en_sentences):
        jp_vec = semantic_model.encode(jp, convert_to_tensor=True)
        en_vec = semantic_model.encode(en, convert_to_tensor=True)
        sim = util.cos_sim(jp_vec, en_vec).item()
        scores.append((sim >= threshold))
    return scores

def translate_texts(texts, batch_size=16, keep_threshold=0.6, chunk_pass_ratio=0.6):
    results = []
    for text in texts:
        protected_text, term_map = protect_terms(text)
        sentences = split_japanese_sentences(protected_text)
        translations = translate_sentences(sentences, batch_size=batch_size)
        ok_flags = compute_similarity(sentences, translations, threshold=keep_threshold)
        
        if len(ok_flags) == 0:
            continue

        pass_ratio = sum(ok_flags) / len(ok_flags)

        if pass_ratio >= chunk_pass_ratio:
            filtered_translations = []
            seen = set()
            for t, ok in zip(translations, ok_flags):
                if ok and t not in seen:
                    seen.add(t)
                    filtered_translations.append(t)
            combined = " ".join(filtered_translations).replace("\n", " ").strip()
            cleaned = clean_repetitions(combined)
            deduped = remove_duplicate_sentences(cleaned)
            final = restore_terms(deduped, term_map)
            refined = refine_translation_with_llm(final)
            results.append({
                "translated_text": refined
            })
    return results

def split_by_length(text, max_len=512):
    segments = []
    while len(text) > max_len:
        split_index = text.rfind("。", 0, max_len)# 找出 max_len 左邊最後一個句號  
        if split_index == -1: # 若找不到句號，只好強制在 max_len 切    
            split_index = max_len
        else:
            split_index += 1  # 包含句號本身

        segment = text[:split_index].strip()
        if segment:
            segments.append(segment)
        text = text[split_index:].strip()

    if text:
        segments.append(text)
    return segments


def refine_translation_with_llm(raw_translation):
    llm = OllamaLLM(model="mistral", temperature=0.2)
    prompt = f"Please correct and improve the following translation for grammar, fluency, and clarity:\n\n{raw_translation}"
    improved = llm.invoke(prompt)
    return improved.strip()

def main():
    with open(r"C:\Users\USER\toysub\dataset\split_data.json", "r", encoding="utf-8") as f:
        raw_documents = json.load(f)
    translated_output = []
    dropped = 0
    for i in tqdm(range(0, len(raw_documents), 16), desc="批次翻譯中"):
        batch_docs = raw_documents[i:i + 16]
        texts = [doc.get("text", "") for doc in batch_docs]
        translations = translate_texts(texts, batch_size=16)
        for doc, trans_info in zip(batch_docs, translations):
            new_doc = dict(doc)
            new_doc.update(trans_info)  # 包含translated_text, similarities等欄位
            translated_output.append(new_doc)
            
        dropped += (len(batch_docs) - len(translations))
    print(f"因翻譯品質不佳而捨棄的段落數量: {dropped}")
        
    with open(r"C:\Users\USER\toysub\dataset\translated_data.json", "w", encoding="utf-8") as f:
        json.dump( translated_output, f, ensure_ascii=False, indent=4)
   
    
if __name__ == "__main__":
    main()

    