import argparse
from urllib.parse import urlparse
from collections import Counter
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')
stop_words = set(stopwords.words("romanian"))

def extract_words_from_urls(urls):
    words = []
    for url in urls:
        path = urlparse(url).path
        parts = re.split(r'[\/\-]', path)

        for word in parts:
            clean_word = word.strip().lower()
            if (
                len(clean_word) > 2 and
                clean_word.isalpha() and
                clean_word not in stop_words
            ):
                words.append(clean_word)
    return words

def generate_query(base_path, common_words, blacklist=None):
    query = f'site:{base_path} inurl:{base_path}**********'

    for word in common_words:
        query += f' inurl:{word}'

    if blacklist:
        for b in blacklist:
            query += f' -inurl:{b}'

    return query

def main(input_file, output_file, min_count, base_path):
    with open(input_file, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]

    all_words = extract_words_from_urls(urls)
    word_counts = Counter(all_words)
    frequent_words = [word for word, count in word_counts.items() if count >= min_count]

    blacklist = ["/pagina", "/page", "/cookie", "/confidentialitate"]

    query = generate_query(base_path, frequent_words, blacklist)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(query)

    print(f"✅ Generated query with {len(frequent_words)} words saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate search query from URL words.")
    parser.add_argument("--base_path", help="Base path to generate the query for")
    parser.add_argument("--min_count", type=int, default=10, help="Minimum word frequency (default: 10)")
    args = parser.parse_args()

    input_file = "sites.txt"
    output_file = "query.txt"

    main(input_file, output_file, args.min_count, args.base_path)
