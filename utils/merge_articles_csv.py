import csv
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_csv_files(input_files, output_file):
    merged_rows = []
    seen_urls = set()

    def read_csv(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get('url')
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    merged_rows.append(row)

    for file in input_files:
        read_csv(file)

    if not merged_rows:
        logging.warning("⚠️ No rows to write.")
        return

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=merged_rows[0].keys())
        writer.writeheader()
        writer.writerows(merged_rows)

    logging.info(f"✅ Merged {len(merged_rows)} unique articles into '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple CSV files into one (removing duplicates by URL).")
    parser.add_argument("inputs", nargs='+', help="Paths to CSV input files")
    parser.add_argument("output", help="Path to output merged CSV file")
    args = parser.parse_args()

    merge_csv_files(args.inputs, args.output)
