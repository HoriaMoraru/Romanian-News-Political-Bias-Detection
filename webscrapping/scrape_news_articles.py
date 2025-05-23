import os
import sys
import logging
from ..dataset.python.ArticleDatasetExporter import ArticleDatasetExporter
from newsplease import NewsPlease
import undetected_chromedriver as uc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

URLS_FILE = "sites_v2.txt"
OUTPUT_FILE = "romanian_political_articles_v2.csv"

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

def scrape_articles(urls):
    articles = []
    for i, url in enumerate(urls, start=1):
        logging.info(f"[{i}/{len(urls)}] Scraping article: {url}")
        try:
            # html = get_html_with_selenium(url)
            # article = NewsPlease.from_html(html)
            article = NewsPlease.from_url(url)
            articles.append(article)
        except Exception as e:
            logging.warning(f"Error scraping {url}: {e}")
    return articles


if __name__ == "__main__":
    logging.info("Reading URLs...")
    urls = parse_urls_from_file(URLS_FILE)

    logging.info(f"Found {len(urls)} URLs to scrape.")

    articles = scrape_articles(urls)
    logging.info(f"Scraped {len(articles)} articles.")

    print("Exporting articles to CSV...")
    exporter = ArticleDatasetExporter(OUTPUT_FILE)
    exporter.save(articles)

    logging.info(f"Export completed. Saved to {OUTPUT_FILE}")
