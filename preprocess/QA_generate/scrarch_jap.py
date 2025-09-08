import json
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from urllib.parse import urljoin, urlparse
from urllib.parse import urljoin, urlparse, urldefrag
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

chrome_options = Options()
chrome_options.add_argument("--enable-gpu")
chrome_options.add_argument('--headless=new')
chrome_options.add_argument('--headless')
service = Service(r"C:\Users\USER\chromedriver-win64\chromedriver.exe")  
driver = webdriver.Chrome(service=service, options=chrome_options)

START_URL = "https://toysub.net/times/"
ARTICLE_PREFIX = "https://toysub.net/times/article/"

def article_url(url):
    try:
        return url.startswith(ARTICLE_PREFIX) and url != ARTICLE_PREFIX
    except:
        return False

def relevant_article(title_text):
    keywords = [
        "育児", "子育て", "成長", "発達", "しつけ", "教育", "知育", "年齢別", "赤ちゃん", "幼児"
    ]
    excluded = [
        "おもちゃ", "定額", "サブスク", "プラン", "キャンペーン", "レンタル", "商品", "料金", "サービス"
    ]
    return (
        any(k in title_text for k in keywords) and
        not any(x in title_text for x in excluded)
    )

def extract_article_content(url):
    print(f"⬇ 擷取文章: {url}")
    driver.get(url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "a")))

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    body = soup.body
    if not body:
        return "", ""

    # 擷取 <title> 或第一個 <h1> 作為標題
    title_tag = soup.find("title") or soup.find("h1")
    title_text = title_tag.get_text(strip=True) if title_tag else ""

    # 移除 footer, nav, header, aside 等
    for tag in body.find_all(['footer', 'nav', 'header', 'aside', 'script', 'style']):
        tag.decompose()

    # 移除 <p class="layout-toc-heading">目次</p>
    for tag in body.find_all("p", class_="layout-toc-heading"):
        if "目次" in tag.get_text():
            tag.decompose()

    # 擷取純文字
    text = body.get_text(separator="\n", strip=True)

    return title_text, text

def crawl(start_url, max_pages=200):
    visited = set()
    to_visit = set([start_url])
    articles_data = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop()
        if url in visited:
            continue
        visited.add(url)

        try:
            driver.get(url)
            WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "a")))
            soup = BeautifulSoup(driver.page_source, "html.parser")

            # 取得頁面中的所有連結
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href.startswith(("mailto:", "javascript:", "#")) or not href:
                    continue

                full_url = urldefrag(urljoin(url, href)).url
                if article_url(full_url) and full_url not in visited:
                    to_visit.add(full_url)

            # 若是文章頁面，就擷取文章內容
            if article_url(url):
                title, article_text = extract_article_content(url)
                if relevant_article(title) and len(article_text.strip()) > 50:
                    articles_data.append({
                        "url": url,
                        "title": title,
                        "text": article_text
                    })

        except Exception as e:
            print(f"錯誤擷取: {url}\n{e}")

    return articles_data

def main():
    try:
        articles = crawl(START_URL, max_pages=100)
        with open(r"C:\Users\USER\toysub\dataset\toysub_articles.json", "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
    finally:
        driver.quit()
if __name__ == "__main__":
    main()
    driver.quit()