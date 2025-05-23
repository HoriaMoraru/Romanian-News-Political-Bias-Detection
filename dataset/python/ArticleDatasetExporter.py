import csv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ArticleDatasetExporter:
    """
    A utility class to export a list of scraped article dictionaries to a CSV dataset,
    skipping articles with empty maintext or improperly scraped data.
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
            articles (list): List of dictionaries or NewsPlease article objects.
        """
        total = len(articles)
        skipped_empty = 0
        skipped_invalid = 0
        valid_count = 0

        logging.info(f"📁 Starting export to '{self.output_file}'")
        logging.info(f"📊 Total articles to process: {total}")

        with open(self.output_file, "w", newline='', encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()

            for i, article in enumerate(articles, start=1):
                if not article:
                    logging.warning(f"❌ [{i}/{total}] Skipped: Article object is None or malformed.")
                    skipped_invalid += 1
                    continue

                try:
                    maintext = article.maintext
                    if not maintext or len(maintext.strip()) == 0:
                        logging.warning(f"⚠️ [{i}/{total}] Skipped: Empty maintext.")
                        skipped_empty += 1
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
                    logging.info(f"✅ [{i}/{total}] Saved: {article.url}")
                    valid_count += 1

                except Exception as e:
                    logging.exception(f"🔥 [{i}/{total}] Error while writing article: {e}")
                    skipped_invalid += 1

        logging.info("📦 Export complete.")
        logging.info(f"✔️ Articles saved: {valid_count}")
        logging.info(f"⚠️ Articles skipped (empty): {skipped_empty}")
        logging.info(f"❌ Articles skipped (invalid): {skipped_invalid}")
        logging.info(f"📄 Output file: {self.output_file}")
