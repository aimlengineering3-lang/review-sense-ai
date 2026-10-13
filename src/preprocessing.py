"""
============================================================
Review Sense AI
Preprocessing Pipeline
============================================================

Purpose:
    • Load processed dataset
    • Clean review text
    • Remove HTML
    • Remove URLs
    • Normalize whitespace
    • Convert to lowercase
    • Remove empty reviews
    • Save cleaned dataset
============================================================
"""

import re
import pandas as pd

from config.config import cfg


class TextPreprocessor:

    def __init__(self):

        self.input_file = cfg.TRAIN_PROCESSED
        self.output_file = cfg.TRAIN_CLEANED

    # -----------------------------------------------------

    def load_dataset(self):

        print("=" * 60)
        print("Loading Processed Dataset")
        print("=" * 60)

        df = pd.read_csv(self.input_file)

        print(f"Rows Loaded : {len(df):,}")

        return df

    # -----------------------------------------------------

    @staticmethod
    def remove_html(text):

        return re.sub(r"<.*?>", " ", str(text))

    # -----------------------------------------------------

    @staticmethod
    def remove_urls(text):

        return re.sub(r"http\S+|www\S+", " ", text)

    # -----------------------------------------------------

    @staticmethod
    def normalize_spaces(text):

        return re.sub(r"\s+", " ", text).strip()

    # -----------------------------------------------------

    def clean_text(self, text):

        text = str(text)

        text = text.lower()

        text = self.remove_html(text)

        text = self.remove_urls(text)

        text = self.normalize_spaces(text)

        return text

    # -----------------------------------------------------

    def preprocess(self, df):

        print("\nCleaning Text...")

        df["text"] = df["text"].astype(str).apply(self.clean_text)

        return df

    # -----------------------------------------------------

    def remove_empty_reviews(self, df):

        print("Removing Empty Reviews...")

        df = df[df["text"].str.strip() != ""]

        df = df.drop_duplicates()

        df = df.reset_index(drop=True)

        return df

    # -----------------------------------------------------

    def print_statistics(self, df):

        print("\nDataset Statistics")

        print("-" * 40)

        print(f"Total Samples : {len(df):,}")

        print(df["label"].value_counts())

        print("-" * 40)

    # -----------------------------------------------------

    def save_dataset(self, df):

        df.to_csv(

            self.output_file,

            index=False

        )

        print("\nSaved Successfully")

        print(self.output_file)

    # -----------------------------------------------------

    def run(self):

        df = self.load_dataset()

        df = self.preprocess(df)

        df = self.remove_empty_reviews(df)

        self.print_statistics(df)

        self.save_dataset(df)


def main():

    processor = TextPreprocessor()

    processor.run()


if __name__ == "__main__":

    main()