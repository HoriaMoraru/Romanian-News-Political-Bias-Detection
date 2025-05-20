import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "python")))
from ArticleDatasetExporter import ArticleDatasetExporter
from newsplease import NewsPlease
import random
import time
import undetected_chromedriver as uc

def parse_urls_from_file(file_path):
    """
    Reads a file containing one URL per line and returns a list of URLs.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        urls = [line.strip() for line in file if line.strip()]
    return urls

# def get_html_with_selenium(url):
#     options = uc.ChromeOptions()
#     options.add_argument("--no-sandbox")
#     options.add_argument("--disable-blink-features=AutomationControlled")
#     options.add_argument("--start-maximized")

#     driver = uc.Chrome(options=options, headless=False)  # headless=True can still get blocked sometimes

#     try:
#         driver.get(url)
#         time.sleep(7)  # wait for JS challenge and full page load
#         html = driver.page_source
#     except Exception as e:
#         print(f"❌ Error loading page: {e}")
#         html = ""
#     finally:
#         driver.quit()

#     return html


if __name__ == "__main__":
    url_file = "sites_b1_2.txt"

    urls = parse_urls_from_file(url_file)
    articles = []

    for i, url in enumerate(urls, start=1):
        print(f"📥 [{i}/{len(urls)}] Scraping article: {url}")
        try:
            # html = get_html_with_selenium(url)

            # article = NewsPlease.from_html(html)
            article = NewsPlease.from_url(url)
            articles.append(article)
        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")

    print("Exporting articles to CSV...")
    exporter = ArticleDatasetExporter("romanian_political_articles_b1_1.csv")
    exporter.save(articles)
