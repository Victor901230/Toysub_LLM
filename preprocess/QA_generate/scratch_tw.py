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
chrome_options.add_argument('--headless')
service = Service(r"C:\Users\USER\chromedriver-win64\chromedriver.exe")  
driver = webdriver.Chrome(service=service, options=chrome_options)
BASE_URL = "https://toysub.tw/"
ALLOWED_DOMAINS = ["toysub.tw", "www.toysub.tw"]
headers = {
    "User-Agent": "Mozilla/5.0"
}

def is_toysub_domain(url):
    try:
        domain = urlparse(url).netloc.lower()
        return domain in ALLOWED_DOMAINS
    except:
        return False

def delete_japanese(text, threshold=0.3):
    if not text.strip():
        return False
    total_chars = len(text)
    japanese_chars = len(re.findall(r'[\u3040-\u30ff\u31f0-\u31ff]', text))
    
    
    return (japanese_chars / total_chars) > threshold

def extract_page_info(url):
    print(f"⬇ 擷取中: {url}")
    driver.get(url)
    WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.TAG_NAME, "a"))
    )  # 等待渲染完成

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    # 🔽 僅擷取 <body> 中的主要內容
    body = soup.body
    if not body:
        return "", set(), set()
    
    internal_links = set()
    external_links = set()

    
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("mailto:", "javascript:", "#")) or not href:
            continue

        full_url = urljoin(url, href)
        full_url = urldefrag(full_url).url  # 移除 fragment (#...)

        if is_toysub_domain(full_url):
            internal_links.add(full_url)
        else:
            external_links.add(full_url)

    # 🔽 移除 footer, nav, header, aside 等非主要內容
    for tag in body.find_all(['footer', 'nav', 'header', 'aside', 'script', 'style']):
        tag.decompose()
    
    for div in body.find_all("div"):
        div_text = div.get_text(strip=True)
        if len(div_text) < 20 or delete_japanese(div_text, threshold=0.3):
            div.decompose()
        # 🔽 擷取純文字
    text = body.get_text(separator="\n", strip=True)
    
    return text, internal_links, external_links

def crawl(start_url, max_pages=500):
    visited = set()
    to_visit = set([start_url])
    pages_data = []
    external_links = set()

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop()
        if url in visited:
            continue
        visited.add(url)

        try:
            text, internal_links, found_external_links = extract_page_info(url)
            if len(text.strip()) > 50:
                pages_data.append({
                    "url": url,
                    "text": text
                })
            external_links.update(found_external_links)
            to_visit.update(internal_links - visited)
        except Exception as e:
            print(f"錯誤擷取: {url}\n{e}")
    
    return pages_data, sorted(external_links)

def main():
    all_pages, external_urls = crawl(BASE_URL, max_pages=500)
    with open(r"C:\Users\USER\toysub\dataset\toysub_site_data.json", "w", encoding="utf-8") as f:
        json.dump(all_pages, f, ensure_ascii=False, indent=2)
    with open(r"C:\Users\USER\toysub\dataset\external_links.json", "w", encoding="utf-8") as f:
        json.dump(external_urls, f, ensure_ascii=False, indent=2)
if __name__ == "__main__":
    main()
    driver.quit()