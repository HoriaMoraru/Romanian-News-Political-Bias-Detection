import csv
import argparse

def merge_csv_files(file1, file2, output_file):
    merged_rows = []
    seen_urls = set()

    def read_csv(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row['url']
                if url not in seen_urls:
                    seen_urls.add(url)
                    merged_rows.append(row)

    # Read both input files
    read_csv(file1)
    read_csv(file2)

    # Write merged output
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=merged_rows[0].keys())
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"✅ Merged {len(merged_rows)} unique articles into '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two CSV files of articles into one (no duplicate URLs).")
    parser.add_argument("input1", help="Path to first CSV file")
    parser.add_argument("input2", help="Path to second CSV file")
    parser.add_argument("output", help="Path to output merged CSV file")
    args = parser.parse_args()

    merge_csv_files(args.input1, args.input2, args.output)
