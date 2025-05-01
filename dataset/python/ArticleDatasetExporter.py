import csv

class ArticleDatasetExporter:
    """
    A utility class to export a list of scraped article dictionaries to a CSV dataset,
    skipping articles with empty maintext.
    """

    def __init__(self, output_file):
        self.output_file = output_file
        self.fieldnames = [
            "url", "title", "date_publish", "description",
            "maintext", "source_domain", "authors"
        ]

    def save(self, articles):
        """
        Saves the list of article dictionaries to a CSV file.

        Parameters:
            articles (list): List of dictionaries representing news articles.
        """
        valid_count = 0

        with open(self.output_file, "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()

            for article in articles:
                if not article:
                    print("⚠️ Article was not scrapped properly.")
                    continue
                print(f"💾 Saving article to CSV: {article.url}")
                maintext = article.maintext
                if not maintext or len(maintext.strip()) == 0:
                    print("⚠️ Skipping article (no main text).")
                    continue

                authors = article.authors
                if isinstance(authors, list):
                    authors = "; ".join(authors)

                row = {
                    "url": article.url,
                    "title": article.title,
                    "date_publish": article.date_publish,
                    "description": article.description,
                    "maintext": maintext,
                    "source_domain": article.source_domain,
                    "authors": authors,
                }

                writer.writerow(row)
                valid_count += 1

        print(f"✅ Saved {valid_count} valid articles to {self.output_file}")
