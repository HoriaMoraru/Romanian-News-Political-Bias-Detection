import os
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

INPUT_FILE = "dataset/nlp/phrase_source_important_words.csv"
OUTPUT_DIR = "visualization/wordclouds"
MAX_WORDS = 200


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the word-frequency CSV with 'source' as the index.
    """
    df = pd.read_csv(filepath, index_col=0)
    return df


def generate_and_save_wordcloud(freqs: dict, source: str, output_dir: str):
    """
    Given a dict of word->frequency and a source name, generates and saves a word cloud.
    """
    wc = WordCloud(
        width=800,
        height=400,
        max_words=MAX_WORDS,
        background_color="white"
    ).generate_from_frequencies(freqs)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Unique words in {source}", fontsize=16)

    filename = os.path.join(output_dir, f"wordcloud_{source.replace('.', '_')}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved word cloud for '{source}' to {filename}")


def main():
    ensure_dir(OUTPUT_DIR)

    df = load_data(INPUT_FILE)

    for source in df.index:
        row = df.loc[source]
        freqs = row[row > 0].to_dict()
        if not freqs:
            print(f"No words to plot for source '{source}'")
            continue
        generate_and_save_wordcloud(freqs, source, OUTPUT_DIR)


if __name__ == '__main__':
    main()
