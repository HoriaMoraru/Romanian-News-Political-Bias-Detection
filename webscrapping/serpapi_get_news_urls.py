import os
import sys
import logging
from serpapi import GoogleSearch
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

load_dotenv()
API_KEY = os.getenv("SERPAPI_KEY")
QUERY_FILE = "query.txt"
OUTPUT_FILE = "sites_v2.txt"

def read_query(query_file):
    with open(query_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def fetch_search_results(query, api_key, pages=10):
    urls = []

    for page in range(pages):
        start_val = page * 100
        logging.info(f"🔍 Fetching results {start_val + 1}–{start_val + 100}...")

        params = {
            "q": query,
            "location": "Austin, Texas, United States",
            "google_domain": "google.com",
            "api_key": api_key,
            "engine": "google",
            "num": 100,
            "start": start_val
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        organic = results.get("organic_results", [])

        if not organic:
            logging.warning(f"⚠️ No more results at start={start_val}. Ending early.")
            break

        for result in organic:
            url = result.get("link")
            if url:
                urls.append(url)

    return list(set(urls))

def save_urls(urls, output_file):
    with open(output_file, "a", encoding="utf-8") as f:
        for u in urls:
            f.write(u + "\n")

    logging.info(f"✅ Saved {len(urls)} new URLs to {output_file}")


if __name__ == "__main__":
    if not API_KEY:
        logging.error("❌ SERPAPI_KEY not found in environment. Check the .env file.")
        sys.exit(1)

    logging.info("Reading query from file...")
    query = read_query(QUERY_FILE)

    logging.info("Fetching search results...")
    urls = fetch_search_results(query, API_KEY)

    save_urls(urls, OUTPUT_FILE)
