import argparse

def remove_duplicate_urls(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    seen = set()
    unique_urls = []
    duplicates = set()

    for url in lines:
        if url in seen:
            duplicates.add(url)
        else:
            seen.add(url)
            unique_urls.append(url)

    # Overwrite the file with only unique URLs
    with open(file_path, 'w', encoding='utf-8') as f:
        for url in unique_urls:
            f.write(url + '\n')

    print(f"✅ Cleaned file saved: {file_path}")
    if duplicates:
        print(f"❗ Removed {len(duplicates)} duplicate URLs:")
        for dup in duplicates:
            print(dup)
    else:
        print("✅ No duplicates found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove duplicate URLs from a text file.")
    parser.add_argument("file_path", help="Path to the .txt file with one URL per line")
    args = parser.parse_args()

    remove_duplicate_urls(args.file_path)
