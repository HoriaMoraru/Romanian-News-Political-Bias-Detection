from newsplease import NewsPlease
from dataset.python.ArticleDatasetExporter import ArticleDatasetExporter

def parse_urls_from_file(file_path):
    """
    Reads a file containing one URL per line and returns a list of URLs.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        urls = [line.strip() for line in file if line.strip()]
    return urls

if __name__ == "__main__":
    url_file = "sites2.txt"

    urls = parse_urls_from_file(url_file)
    articles = []

    for url in urls:
        print(f"📥 Scraping article: {url}")
        article = NewsPlease.from_url(url)
        articles.append(article)

    exporter = ArticleDatasetExporter("romanian_political_articles2.csv")
    exporter.save(articles)
