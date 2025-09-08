import json
from collections import Counter
import re

COMMON_BLOCKS =[
    "LINEで\nおもちゃ診断",
    "トイサブ！LINEで\nおもちゃ診断","トップ\nよみもの\nサインイン\nお申込み\n",
    "トップ\nトイサブ！の特徴\nお届けするおもちゃについて\nおもちゃの選定ポイント",
    "年齢別おもちゃ一覧\n0歳ごろのおもちゃ一覧\n1歳ごろのおもちゃ一覧\n2歳ごろのおもちゃ一覧\n3歳ごろのおもちゃ一覧\n4歳ごろのおもちゃ一覧\n5歳以上のおもちゃ一覧",
    "ご利用の流れ\nコース一覧・料金\nお客様の声\nよくあるご質問\nお申込み\nLINEでおもちゃ診断\n知育のハンドブック\nToysub! Times 育児メディア\nマイページにサインイン",
    "トップ\nよみもの\nサインイン\nお申込み",
    "当サイトでは、利便性、品質維持・向上を目的に、Cookieを使用しております。\n閲覧を継続されることで、Cookieの利用に同意するものといたします。 詳しくは\nこちら\nをご参照ください。",
    "トイサブ！タイムス",
    "トイサブ︕タイムス",
    "トイサブ!\nお申込み",
    "OK",
    "LINEでおもちゃ診断\nおすすめ記事",
    "関連記事おもちゃの選び方！選ぶポイントと年齢別おすすめタイプを紹介3か月4か月5か月6か月7か月8か月9か月10か月11か月1歳～1歳半1歳半～2歳2歳3歳4歳5歳以上おもちゃコラム",
    "0歳から6歳までの年齢別長く遊べるおもちゃ8選！おもちゃ選びのポイントや注意点も\n1か月\n2か月\n3か月\n4か月\n5か月\n6か月\n7か月\n8か月\n9か月\n10か月\n11か月\n1歳～1歳半\n1歳半～2歳\n2歳\n3歳\n4歳\n5歳以上\nおもちゃコラム",
    "おもちゃをレンタルするメリット6つ｜サブスクの魅力も紹介！\n3か月\n4か月\n5か月\n6か月\n7か月\n8か月\n9か月\n10か月\n11か月\n1歳～1歳半\n1歳半～2歳\n2歳\n3歳\n4歳\n5歳以上\nおもちゃコラム",
    "常に高品質なおもちゃを届ける 4人子を持つ発送グループリーダー\nトイサブ！スタッフ\nトイサブ！を支えているのは私\n",
    "いま注目の教育経済学の考え方と幼児教育における重要性とは？ | 教育経済学から見る幼児教育①\n中室牧子\n教育経済学\n知育・教育\n教育経済学から見る幼児教育",
    "【幼児向け】おもちゃの種類＆男の子・女の子におすすめの玩具\nブロック・積み木\nパズル\nおもちゃコラム\n",
    "知育玩具とは？年齢別知育玩具の選び方・おすすめアイテムを紹介\n1か月\n2か月\n3か月\n4か月\n5か月\n6か月\n7か月\n8か月\n9か月\nブロック・積み木\nパズル\n布のおもちゃ・ぬいぐるみ\n音の出るおもちゃ\n英語のおもちゃ\n木のおもちゃ\n10か月\n11か月\n1歳～1歳半\n1歳半～2歳\n2歳\n3歳\n4歳\n5歳以上\n知育・教育\nおもちゃコラム\n",
    "子どもと一緒に、わくわくする時間を。\n育児に寄り添うWebメディア\n子どもの成長ってわくわくする。\n大変なことや不安なことはたくさんあるけれど\n楽しくて、短くて、大切な時間を\n私たちトイサブ！と一緒に過ごしませんか。\n\nは\n一人ひとりに合った知育おもちゃが届く\n定額制知育サービス「トイサブ！」が運営する\nWebメディアサイトです。\n",
    "年齢別おもちゃ一覧\n0歳ごろ\n1歳ごろ\n2歳ごろ\n3歳ごろ\n4歳ごろ\n5歳以上",
    "トイサブ\n0歳ごろ\n1歳ごろ\n2歳ごろ\n3歳ごろ\n4歳ごろ\n5歳以上",
    "おもちゃ一覧\n0歳ごろ\n1歳ごろ\n2歳ごろ\n3歳ごろ\n4歳ごろ\n5歳以上",
    "\n0歳ごろ\n1歳ごろ\n2歳ごろ\n3歳ごろ\n4歳ごろ\n5歳以上",
    "トイサブ！について\nトイサブ！の特徴\nご利用の流れ\nお客さまの声\nサービス一覧・料金\nお届けするおもちゃについて\nおもちゃの選定ポイント\nToysub! Times\nお知らせ\nアンバサダープロジェクト",
    "サービス一覧\nトイサブ！ファーストセレクション\n法人向けサービス\nToysub!Store",
    "おもちゃの選定ポイント\n年齢別おすすめおもちゃ\nおもちゃプラン診断\nメーカー一覧\n0歳ごろの知育おもちゃの選定ポイント\n1歳ごろの知育おもちゃの選定ポイント\n2・3歳ごろの知育おもちゃの選定ポイント\n4・5・6歳ごろの知育おもちゃの選定ポイント\nおもちゃ一覧\n0歳ごろのおもちゃ一覧\n1歳ごろのおもちゃ一覧\n2歳ごろのおもちゃ一覧\n3歳ごろのおもちゃ一覧\n4歳ごろのおもちゃ一覧\n5歳以上のおもちゃ一覧\n生後3か月向けのおもちゃ10選\n生後4か月向けのおもちゃ10選\n生後5か月向けのおもちゃ10選\n生後6か月向けのおもちゃ10選\n生後7か月向けのおもちゃ10選\n生後8か月向けのおもちゃ10選\n生後9か月向けのおもちゃ10選\n生後10か月向けのおもちゃ10選\n生後11か月向けのおもちゃ10選",
    "© 知育玩具・おもちゃのサブスク定額レンタル【トイサブ！】",
    "お申込み・お問い合わせ\nお申込み\nお問い合わせ\nよくあるご質問\nプレゼント利用のご案内\nこども商品券のご利用について\nトイサブ！への広告出稿について\n運営会社\n代表メッセージ\n会社概要\n採用情報\nこのサイトについて\n利用規約\nプライバシーポリシー\nサイトポリシー\n特定商取引法及び古物営業法に基づく表記"
]
TOYSUB_PATTERN = re.compile(r'(?<!\w)(トイサブ！)(?!\w)')
ALLOWED_CHARS_PATTERN = re.compile(r"[^\u3040-\u30FF\u4E00-\u9FFFa-zA-Z0-9。，、！？.?!:;「」『』【】［］［］()（）・ー\-\'\"&/&%~\s]")

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

def remove_isolated_toysub(text):
    # 將滿足條件的トイサブ！替換為空字串
    return TOYSUB_PATTERN.sub("", text)

def clean_chunk_by_patterns(text, patterns):
    removed = []
    cleaned_text = text

    for pattern in patterns:
        if pattern in cleaned_text:
            cleaned_text = cleaned_text.replace(pattern, "")
            removed.append(pattern)
    
    before = cleaned_text
    cleaned_text = remove_isolated_toysub(cleaned_text)
    if before != cleaned_text:
        removed.append("トイサブ！（邊界匹配）")
        
    toys_index = cleaned_text.find("TOYS LIST")
    if toys_index != -1:
        cleaned_text = cleaned_text[:toys_index]
        removed.append("TOYS LIST 後內容已移除")
    clean_text = cleaned_text.replace("\n", "")
    cleaned_text = remove_weird_symbols(cleaned_text)
    return clean_text.strip(), removed

def normalize_text(text):
    text = text.replace("\n", "")
    text = re.sub(r"\s+", "", text)
    return text.strip()

def main():
    with open(r"C:\Users\USER\toysub\dataset\toysub_articles.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    cleaned_chunks = []

    for chunk in chunks:
        original_text = chunk["text"]
        cleaned_text, removed_blocks = clean_chunk_by_patterns(original_text, COMMON_BLOCKS)
        cleaned_chunks.append({
            "url": chunk["url"],
            "text": cleaned_text
        })

    with open(r"C:\Users\USER\toysub\dataset\clean_data.json", "w", encoding="utf-8") as f:
        json.dump(cleaned_chunks, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()