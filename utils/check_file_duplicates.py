import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_for_duplicate_urls(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    seen = set()
    duplicates = set()

    for url in lines:
        if url in seen:
            duplicates.add(url)
        else:
            seen.add(url)

    if duplicates:
        logging.warning("❗ Duplicate URLs found:")
        for dup in duplicates:
            logging.warning(f"Duplciated URL: {dup}")
    else:
        logging.info("✅ No duplicate URLs found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for duplicate URLs in a text file.")
    parser.add_argument("file_path", help="Path to the .txt file with one URL per line")
    args = parser.parse_args()

    check_for_duplicate_urls(args.file_path)
